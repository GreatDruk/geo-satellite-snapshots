"""Evaluate frozen checkpoints with TTA/soft ensembles and replay a selected recipe."""
import argparse
import hashlib
import json
import re
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import load_test, load_validation
from src.experiment import PROJECT_ROOT, load_checkpoint
from src.training.predict import (ensemble_probabilities, probability_predictions,
                                  probability_predictions_many, load_probabilities, save_probabilities, save_submission)
from src.utils import get_device, set_seed

VIEWS = {"none": 1, "flips": 4, "d4": 8}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def _paths(root, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Use a unique run_id of 1–80 letters/digits/underscores/hyphens")
    paths = {kind: root / f"artifacts/{kind}" / run_id for kind in ("logs", "predictions")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Inference run already exists: {run_id}; choose a new ID")
    return paths


def _data_config(checkpoint, device, batch_size, num_workers):
    config = dict(checkpoint["config"]["data"])
    config.update(batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == "cuda")
    if num_workers == 0:
        config.pop("prefetch_factor", None)
        config["persistent_workers"] = False
    return config


def _metric(predictions):
    indices = predictions.probabilities.argmax(1)
    truth_indices = pd.Index(predictions.class_labels).get_indexer(predictions.true_labels)
    correct = int((indices == truth_indices).sum())
    nll = -np.log(np.clip(predictions.probabilities[np.arange(len(indices)), truth_indices], 1e-30, 1)).mean()
    return {"correct": correct, "errors": len(indices) - correct,
            "val_accuracy": correct / len(indices), "val_nll": float(nll)}


def run_validation_suite(checkpoints, run_id, *, project_root=PROJECT_ROOT, raw_dir="data/raw",
                         device="auto", batch_size=16, num_workers=0,
                         tta_modes=("none", "flips", "d4"), cache_from=None):
    """One pass per model shares TTA forwards; compare singles and equal-weight ensemble.

    No train/test reading, no gradients, no optimizer, no experiments.csv training row.
    Selection: Accuracy, then NLL, then view count. All measured candidates stay visible.
    """
    root = Path(project_root).resolve()
    paths = _paths(root, run_id)
    if not checkpoints or "ensemble" in checkpoints or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in checkpoints):
        raise ValueError("Provide named checkpoints with safe member names")
    if batch_size <= 0 or num_workers < 0 or not tta_modes or set(tta_modes) - set(VIEWS) or len(set(tta_modes)) != len(tta_modes):
        raise ValueError("Invalid loader settings or TTA modes")
    sources = {name: _resolve(root, path).resolve() for name, path in checkpoints.items()}
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    raw = _resolve(root, raw_dir).resolve()
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    def emit(message):
        print(message, flush=True)
        with (paths["logs"] / "inference.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
    started = time.perf_counter()
    device = get_device(device)
    set_seed(42, deterministic=True, warn_only=True)
    def portable(path):
        return str(path.relative_to(root)).replace("\\", "/") if path.is_relative_to(root) else str(path)
    source_specs = {name: {"checkpoint": portable(path), "sha256": file_hash(path)}
                    for name, path in sources.items()}
    metadata = {"kind": "validation_inference", "status": "running", "device": str(device),
                "torch": str(torch.__version__), "tta_modes": list(tta_modes),
                "batch_size": batch_size, "num_workers": num_workers, "sources": source_specs,
                "val_csv_sha256": file_hash(raw / "val.csv"),
                "classes_csv_sha256": file_hash(raw / "classes.csv"), "probabilities_sha256": {}}
    cache_metadata = {}
    if cache_from is not None:
        cache_logs = root / "artifacts/logs" / cache_from
        cache_metadata = json.loads((cache_logs / "metadata.json").read_text(encoding="utf-8"))
        if (cache_metadata.get("status") != "complete"
                or cache_metadata.get("val_csv_sha256") != metadata["val_csv_sha256"]
                or cache_metadata.get("classes_csv_sha256") != metadata["classes_csv_sha256"]):
            raise ValueError("Cache must be complete and match current validation/classes CSV")
    raw_truth = pd.read_csv(raw / "val.csv", dtype={"id": str}).set_index("id").label
    metadata_path = paths["logs"] / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    members, candidates, rows = {}, {}, []
    def record(name, prediction, specs):
        save_probabilities(prediction, paths["predictions"] / f"{name}.npz")
        metadata["probabilities_sha256"][f"{name}.npz"] = file_hash(paths["predictions"] / f"{name}.npz")
        prediction.table(include_scores=True).to_csv(paths["predictions"] / f"{name}.csv", index=False)
        values = _metric(prediction)
        rows.append({"recipe": name, **values, "models": len(specs),
                     "forward_views_per_image": sum(VIEWS[item["tta"]] for item in specs)})
        candidates[name] = {"members": specs, "class_labels": prediction.class_labels.tolist()}
        emit(f"{name}: Accuracy={values['val_accuracy']:.4%}, errors={values['errors']}, NLL={values['val_nll']:.4f}")
    try:
        for name, source in sources.items():
            reusable = cache_metadata.get("sources", {}).get(name, {}).get("sha256") == source_specs[name]["sha256"]
            cached = {}
            if reusable:
                for mode in tta_modes:
                    filename = f"{name}__{mode}.npz"
                    candidate = root / "artifacts/predictions" / cache_from / filename
                    expected = cache_metadata.get("probabilities_sha256", {}).get(filename)
                    if candidate.is_file() and expected and file_hash(candidate) == expected:
                        prediction = load_probabilities(candidate)
                        if (set(prediction.ids) != set(raw_truth.index) or prediction.true_labels is None
                                or not np.array_equal(prediction.true_labels, raw_truth.loc[prediction.ids].to_numpy())):
                            raise ValueError("Cached probabilities do not match current validation targets")
                        cached[mode] = prediction
            if set(cached) == set(tta_modes):
                emit(f"{name}: verified probability cache reused from {cache_from}; no model forwards")
                members[name] = cached
                for mode, prediction in cached.items():
                    record(f"{name}__{mode}", prediction,
                           [{**source_specs[name], "name": name, "tta": mode, "weight": 1.0}])
                continue
            model, checkpoint = load_checkpoint(source, device, inference_only=True)
            data_config = _data_config(checkpoint, device, batch_size, num_workers)
            loader, classes = load_validation(raw, data_config, checkpoint["config"]["seed"])
            if set(classes.label) != set(checkpoint["class_labels"]):
                raise ValueError("Raw classes do not match saved checkpoint classes")
            def progress(index, total):
                if index == 1 or index % 5 == 0 or index == total:
                    emit(f"{name}: batch {index}/{total}, shared TTA={list(tta_modes)}")
            members[name] = probability_predictions_many(model, loader, device, checkpoint["class_labels"],
                                                        tta_modes=tta_modes, progress=progress)
            for mode, prediction in members[name].items():
                spec = {**source_specs[name], "name": name, "tta": mode, "weight": 1.0}
                record(f"{name}__{mode}", prediction, [spec])
            del model, checkpoint, loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if len(members) > 1:
            groups = [("ensemble", list(members))]
            if len(members) > 2:
                groups.extend(("ensemble_" + "_".join(pair), list(pair))
                              for pair in combinations(members, 2))
            for group_name, names in groups:
                for mode in tta_modes:
                    specs = [{**source_specs[name], "name": name, "tta": mode, "weight": 1 / len(names)}
                             for name in names]
                    ensemble = ensemble_probabilities([members[name][mode] for name in names])
                    record(f"{group_name}__{mode}", ensemble, specs)
        comparison = pd.DataFrame(rows).sort_values(
            ["val_accuracy", "val_nll", "forward_views_per_image"], ascending=[False, True, True]).reset_index(drop=True)
        comparison.to_csv(paths["logs"] / "comparison.csv", index=False)
        best = comparison.iloc[0]
        selection = {"format_version": 1, "selected_on": "val", "recipe": best.recipe,
                     "val_accuracy": float(best.val_accuracy), "val_nll": float(best.val_nll),
                     "val_csv_sha256": metadata["val_csv_sha256"],
                     **candidates[best.recipe]}
        (paths["logs"] / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        metadata.update(status="complete", inference_time_min=(time.perf_counter() - started) / 60)
        emit(f"Selected: {best.recipe}, {best.val_accuracy:.4%}; {metadata['inference_time_min']:.2f} min")
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"comparison": comparison, "selection": selection, "paths": paths}


def run_selected_inference(selection, run_id, *, split="val", project_root=PROJECT_ROOT,
                           raw_dir="data/raw", device="auto", batch_size=16, num_workers=0):
    """Replay a fixed val-selected manifest. Test is explicit and never used to select."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    root = Path(project_root).resolve()
    if not isinstance(selection, dict):
        selection = json.loads(_resolve(root, selection).read_text(encoding="utf-8"))
    if selection.get("format_version") not in {1, 2} or selection.get("selected_on") != "val" or not selection.get("members"):
        raise ValueError("Expected a supported recipe selected on val")
    local_recipe = selection.get("format_version") == 2
    if local_recipe and (selection.get("method") != "global_local" or len(selection["members"]) != 1
                         or selection.get("local_pool") not in {"mean", "max"}
                         or not 0 <= selection.get("global_weight", -1) <= 1
                         or not isinstance(selection.get("crop_size"), int) or selection["crop_size"] <= 0):
        raise ValueError("Invalid global/local recipe")
    paths = _paths(root, run_id)
    submission = root / "submissions" / f"{run_id}.csv"
    if split == "test" and submission.exists():
        raise FileExistsError(submission)
    device = get_device(device)
    raw = _resolve(root, raw_dir)
    if split == "val" and file_hash(raw / "val.csv") != selection["val_csv_sha256"]:
        raise ValueError("Validation split changed since recipe selection")
    set_seed(42, deterministic=True, warn_only=True)
    predictions = []
    for member in selection["members"]:
        checkpoint_path = _resolve(root, member["checkpoint"])
        if file_hash(checkpoint_path) != member["sha256"]:
            raise ValueError(f"Selected checkpoint changed: {checkpoint_path}")
        model, checkpoint = load_checkpoint(checkpoint_path, device, inference_only=True)
        if set(checkpoint["class_labels"]) != set(selection["class_labels"]):
            raise ValueError("Checkpoint classes do not match selection")
        data_config = _data_config(checkpoint, device, batch_size, num_workers)
        if local_recipe:
            from src.local_inference import local_loader, local_probability_predictions
            loader = local_loader(raw, data_config, checkpoint["class_labels"], split=split,
                                  crop_size=selection["crop_size"], batch_size=batch_size,
                                  num_workers=num_workers, device=device)
        elif split == "val":
            loader, _ = load_validation(raw, data_config, checkpoint["config"]["seed"])
        else:
            loader = load_test(raw, data_config, checkpoint["config"]["seed"])
        if local_recipe:
            predictions.append(local_probability_predictions(model, loader, device, checkpoint["class_labels"],
                               method=selection["local_pool"], global_weight=selection["global_weight"]))
        else:
            predictions.append(probability_predictions(model, loader, device, checkpoint["class_labels"], tta=member["tta"]))
        del model, loader, checkpoint
    result = ensemble_probabilities(predictions, [member["weight"] for member in selection["members"]])
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    save_probabilities(result, paths["predictions"] / f"{split}_probabilities.npz")
    result.table(include_scores=True).to_csv(paths["predictions"] / f"{split}.csv", index=False)
    summary = {"split": split, "recipe": selection["recipe"], "device": str(device), "selection": selection}
    if split == "val":
        summary.update(_metric(result))
    else:
        save_submission(result.table(), raw / "sample_submission_example.csv", submission)
        paths["submission"] = submission
    (paths["logs"] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"predictions": result, "paths": paths, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path pairs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tta", nargs="+", default=["none", "flips", "d4"])
    parser.add_argument("--cache-from")
    args = parser.parse_args()
    checkpoints = dict(item.split("=", 1) for item in args.checkpoints)
    run_validation_suite(checkpoints, args.run_id, device=args.device, raw_dir=args.raw_dir,
                         batch_size=args.batch_size, tta_modes=args.tta, cache_from=args.cache_from)


if __name__ == "__main__":
    main()
