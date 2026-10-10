from archive.src.utils import resolve_raw_dir
"""Separate clean+hard+JS experiment loop; existing run_experiment is untouched."""
import json
import math
import os
import platform
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from archive.src.data import ImageDataset, make_loader, read_classes
from archive.src.experiment import ExperimentResult, load_config, make_optimizer
from archive.src.hard_target_shift import HardTargetDataset, PROFILES
from archive.src.models import build_model
from archive.src.models.initialization import initialize_from_checkpoint
from archive.src.training.evaluate import evaluate_model
from archive.src.training.predict import probability_predictions, save_probabilities
from archive.src.training.schedulers import make_scheduler, step_scheduler
from archive.src.transforms import ImageTransform
from archive.src.utils import append_experiment, get_device, set_seed, validate_journal


def paired_loss(clean_logits, hard_logits, labels):
    """0.5 CE(clean) + 0.5 CE(hard) + 3 JS; T=1, gradients through BOTH views."""
    clean_ce = F.cross_entropy(clean_logits, labels)
    hard_ce = F.cross_entropy(hard_logits, labels)
    log_p, log_q = F.log_softmax(clean_logits.float(), 1), F.log_softmax(hard_logits.float(), 1)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2)
    js = .5 * ((log_p.exp()*(log_p-log_m)).sum(1)
               + (log_q.exp()*(log_q-log_m)).sum(1)).mean()
    return .5*clean_ce + .5*hard_ce + 3*js, {"clean_ce": clean_ce, "hard_ce": hard_ce, "js_loss": js}


def train_paired_epoch(model, loader, optimizer, device):
    model.train()
    totals = {name: 0. for name in ("train_loss", "clean_ce", "hard_ce", "js_loss", "clean_accuracy", "hard_accuracy")}
    counts = np.zeros(6, dtype=int)
    samples = 0
    for clean, hard, labels, profiles, indices in loader:
        non_blocking = device.type == "cuda" and loader.pin_memory
        clean, hard, labels = [value.to(device, non_blocking=non_blocking) for value in (clean, hard, labels)]
        optimizer.zero_grad(set_to_none=True)
        # B denotes ORIGINAL objects, not views. B=16 -> one forward of 32 images.
        logits = model(torch.cat((clean, hard), dim=0))
        clean_logits, hard_logits = logits.split(len(labels), dim=0)
        loss, terms = paired_loss(clean_logits, hard_logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite paired training loss")
        loss.backward()
        optimizer.step()
        size = len(labels)
        samples += size
        totals["train_loss"] += loss.item()*size
        for key, value in terms.items():
            totals[key] += value.item()*size
        totals["clean_accuracy"] += (clean_logits.argmax(1) == labels).sum().item()
        totals["hard_accuracy"] += (hard_logits.argmax(1) == labels).sum().item()
        counts += np.bincount(profiles.numpy(), minlength=6)
    if not samples:
        raise ValueError("Empty train dataset")
    return {key: value/samples for key, value in totals.items()}, counts


def validate_recipe(config):
    if config["data"]["image_size"] != 256 or config["data"].get("augmentation", {}):
        raise ValueError("Use full256 and empty augmentation; paired profiles replace old transforms")
    training, model = config["training"], config["model"]
    if training.get("hard_target_shift") != {"clean_ce_weight": .5, "hard_ce_weight": .5, "js_weight": 3., "temperature": 1.}:
        raise ValueError("This experiment requires the fixed clean+hard+JS objective")
    if any(name in training for name in ("batch_augmentation", "hard_mining", "final_refit")) or config.get("final_refit"):
        raise ValueError("No mixing, mining, or train+val fitting in this pipeline")
    if training.get("loss") != {"name": "cross_entropy", "label_smoothing": 0.}:
        raise ValueError("Use ordinary CE in paired loss")
    if model.get("mixstyle") is not None or model.get("auxiliary_cosine_head"):
        raise ValueError("MixStyle and auxiliary losses are excluded")
    if model["name"] not in {"resnet50", "resnet50_agmfa"}:
        raise ValueError("Only the requested ResNet50/AGMFA controls are supported")
    if model["name"] == "resnet50_agmfa" and model["agmfa"].get("attention_source") != "conv2":
        raise ValueError("Use corrected AGMFA conv2 taps")
    if not model.get("freeze_bn") or model.get("mode") != "partial":
        raise ValueError("Require partial fine-tuning and frozen backbone BN")


def build_shift_model(config, classes, root, device, *, model_factory=None):
    initializer = config["training"].get("initialize_from")
    model = build_model(config["model"], 256, len(classes), model_factory,
                        initialize_pretrained=not bool(initializer)).to(device)
    metadata = None
    if initializer:
        path = Path(initializer)
        if not path.is_absolute():
            path = Path(root) / path
        if not path.is_file():
            raise FileNotFoundError(f"Copy corrected AGMFA checkpoint before run C: {path}")
        # Explicitly reject historical 2048-channel attention initialization.
        source = torch.load(path, map_location="cpu", weights_only=True)
        if source["config"]["model"].get("agmfa", {}).get("attention_source") != "conv2":
            raise ValueError("Run C must initialize from corrected conv2 AGMFA")
        del source
        metadata = initialize_from_checkpoint(model, path, classes, config["data"])
    return model, metadata


def run_hard_target_shift(config, exp_id=None, *, project_root=None, show_plots=True, model_factory=None):
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    cfg = load_config(config, root)
    validate_recipe(cfg)
    exp_id = exp_id or cfg["exp_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", exp_id):
        raise ValueError("Invalid exp_id")
    cfg["exp_id"] = exp_id
    validate_journal(root.parent / "experiments.csv" if root.name == "archive" else root / "experiments.csv", exp_id)
    paths = {name: root / "artifacts" / name / exp_id for name in ("models", "logs", "predictions")}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Preserve existing run; choose a new ID: {path}")
    training, data = cfg["training"], cfg["data"]
    epochs = training["epochs"]
    if epochs <= model_warmup(cfg) or epochs <= 0 or training.get("log_every", 1) <= 0:
        raise ValueError("Invalid epochs/warmup/log_every")
    set_seed(cfg["seed"], training.get("deterministic", True), training.get("deterministic_warn_only", True))
    device = get_device(training.get("device", "auto"))
    if device.type != "cuda":
        data["pin_memory"] = False
    if os.name == "nt":
        data["num_workers"] = 0
    raw = Path(data["raw_dir"])
    raw = resolve_raw_dir(root, raw)
    classes = read_classes(raw)
    labels = classes.label.tolist()
    mapping = {label: index for index, label in enumerate(labels)}
    # Test is intentionally absent from imports, configuration and data access.
    train = HardTargetDataset(raw / "train.csv", raw / "train", data, mapping, cfg["seed"])
    val = ImageDataset(raw / "val.csv", raw / "val", ImageTransform(data, False), mapping)
    train_loader, val_loader = make_loader(train, data, cfg["seed"], True), make_loader(val, data, cfg["seed"])
    model, initialization = build_shift_model(cfg, labels, root, device, model_factory=model_factory)
    optimizer = make_optimizer(model, training["optimizer"])
    scheduler, monitor = make_scheduler(optimizer, training["scheduler"], epochs)
    assert not optimizer.state, "Initialization must not resume an old optimizer"
    model.eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 256, 256, device=device)).shape == (2, len(labels))
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    (paths["logs"] / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    metadata = {"initialization": initialization, "python": platform.python_version(), "torch": str(torch.__version__),
                "device": str(device), "seed": cfg["seed"], "profiles": list(PROFILES),
                "profile_rule": "(sample_index + epoch) % 6; epoch starts at 1",
                "batch_size_original_objects": data["batch_size"], "forward_images": 2*data["batch_size"],
                "best_pt_alias": "best_accuracy.pt", "test_read": False, "val_tta": "none"}
    (paths["logs"] / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows, profile_rows = [], []
    best_accuracy, best_loss, best_epoch, best_loss_epoch = -1., float("inf"), 0, 0
    started = time.perf_counter()
    def emit(message):
        print(message, flush=True)
        with (paths["logs"] / "training.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
    def checkpoint(epoch, val_loss, val_accuracy):
        return {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
                "val_accuracy": val_accuracy, "val_loss": val_loss, "config": cfg,
                "class_labels": labels, "class_names": classes.class_name.tolist()}
    emit(f"{exp_id} | device={device} | paired batch={data['batch_size']} objects / {2*data['batch_size']} views")
    try:
        for epoch in range(1, epochs+1):
            epoch_start = time.perf_counter()
            train.set_epoch(epoch)
            # Restarting at the same epoch produces the same shuffle and views.
            train_loader.generator.manual_seed(cfg["seed"] + epoch)
            model.set_epoch(epoch)
            rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
            metrics, counts = train_paired_epoch(model, train_loader, optimizer, device)
            expected = np.bincount([(i+epoch) % 6 for i in range(len(train))], minlength=6)
            np.testing.assert_array_equal(counts, expected)
            val_loss, val_accuracy = evaluate_model(model, val_loader, torch.nn.CrossEntropyLoss(), device)
            step_scheduler(scheduler, monitor, {"val_loss": val_loss, "val_accuracy": val_accuracy})
            rows.append({"epoch": epoch, **metrics, "val_loss": val_loss, "val_accuracy": val_accuracy,
                         "training_mode": model.active_mode, "train_samples": len(train),
                         "forward_images": 2*len(train), "learning_rate": optimizer.param_groups[0]["lr"] if not rates else rates["backbone"],
                         "learning_rates": json.dumps(rates), "next_learning_rates": json.dumps({g["name"]: g["lr"] for g in optimizer.param_groups}),
                         "epoch_time_sec": time.perf_counter()-epoch_start})
            profile_rows.extend({"epoch": epoch, "profile": name, "count": int(count)} for name, count in zip(PROFILES, counts))
            pd.DataFrame(rows).to_csv(paths["logs"] / "history.csv", index=False)
            pd.DataFrame(profile_rows).to_csv(paths["logs"] / "profile_counts.csv", index=False)
            if val_accuracy > best_accuracy:
                best_accuracy, best_epoch = val_accuracy, epoch
                torch.save(checkpoint(epoch, val_loss, val_accuracy), paths["models"] / "best_accuracy.pt")
            if val_loss < best_loss:
                best_loss, best_loss_epoch = val_loss, epoch
                torch.save(checkpoint(epoch, val_loss, val_accuracy), paths["models"] / "best_loss.pt")
            if epoch == 1 or epoch % training.get("log_every", 2) == 0 or epoch == epochs:
                emit(f"Epoch {epoch}/{epochs} | loss={metrics['train_loss']:.4f} | clean_ce={metrics['clean_ce']:.4f} "
                     f"hard_ce={metrics['hard_ce']:.4f} JS={metrics['js_loss']:.4f} | "
                     f"clean_acc={metrics['clean_accuracy']:.4f} hard_acc={metrics['hard_accuracy']:.4f} | "
                     f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} | LR={rates} | {model.active_mode}")
        torch.save(checkpoint(epochs, val_loss, val_accuracy), paths["models"] / "last.pt")
        shutil.copyfile(paths["models"] / "best_accuracy.pt", paths["models"] / "best.pt")
        for selection in ("accuracy", "loss"):
            state = torch.load(paths["models"] / f"best_{selection}.pt", map_location=device, weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            prediction = probability_predictions(model, val_loader, device, labels, tta="none")
            prediction.table().to_csv(paths["predictions"] / f"val_best_{selection}.csv", index=False)
            save_probabilities(prediction, paths["predictions"] / f"val_best_{selection}_probabilities.npz")
            if selection == "accuracy":
                prediction.table().to_csv(paths["predictions"] / "val.csv", index=False)
                save_probabilities(prediction, paths["predictions"] / "val_probabilities.npz")
        state = torch.load(paths["models"] / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        model.set_epoch(best_epoch)
        model.eval()
        elapsed = (time.perf_counter()-started)/60
        summary = {"best_epoch": best_epoch, "val_accuracy": best_accuracy, "best_loss_epoch": best_loss_epoch,
                   "best_val_loss": best_loss, "train_time_min": elapsed, "best_pt_alias": "best_accuracy.pt"}
        (paths["logs"] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        plot_shift_history(pd.DataFrame(rows), paths["logs"] / "learning_curves.png", show=show_plots)
        append_experiment(root.parent / "experiments.csv" if root.name == "archive" else root / "experiments.csv", {"exp_id": exp_id, "model": cfg["model"]["name"],
            "image_size": 256, "augmentation": "HardTargetShift: paired D4+six mandatory profiles; CE0.5/0.5+JS3",
            "pretrained": True, "val_accuracy": best_accuracy, "train_time_min": elapsed, "seed": cfg["seed"],
            "submission_id": "", "public_score": "", "notes": cfg["notes"]})
    except BaseException as error:
        emit(f"Run failed: {type(error).__name__}: {error}")
        raise
    return ExperimentResult(exp_id, model, pd.DataFrame(rows), best_epoch, best_accuracy, paths)


def model_warmup(config):
    return config["model"].get("warmup_epochs", 0)


def plot_shift_history(history, save_path=None, show=True):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    groups = [("Total paired loss", ["train_loss"]), ("Clean/hard CE", ["clean_ce", "hard_ce"]),
              ("JS (unweighted)", ["js_loss"]), ("Accuracy", ["clean_accuracy", "hard_accuracy", "val_accuracy"]),
              ("Validation loss", ["val_loss"])]
    for axis, (title, keys) in zip(axes.flat, groups):
        for key in keys:
            axis.plot(history.epoch, history[key], label=key)
        axis.set_title(title)
        axis.legend()
    rates = [json.loads(row) for row in history.learning_rates]
    for group in rates[0]:
        axes.flat[-1].plot(history.epoch, [row[group] for row in rates], label=group)
    axes.flat[-1].set_title("Learning rates used")
    axes.flat[-1].legend()
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig
