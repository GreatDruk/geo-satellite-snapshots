from archive.src.utils import raw_data_dir
"""Compare single ResNet50 runs on saved val probabilities, without inference."""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from archive.src.training.predict import load_probabilities


@dataclass
class StrategyComparison:
    comparison: pd.DataFrame
    per_class: pd.DataFrame
    changed_cases: pd.DataFrame


def compare_resnet50_runs(run_ids, *, project_root, baseline="resnet50_partial_domain_shift_stronger_001", data_dir=None):
    """Use original/full-frame probabilities only; no TTA, ensembles or test.

    Refuse other architectures, mismatched ground truth and different classes.
    Missing candidates raise explicitly so a missing run is never a zero score.
    """
    root = Path(project_root)
    run_ids = list(dict.fromkeys([baseline, *run_ids]))
    raw = Path(data_dir) if data_dir is not None else raw_data_dir(root)
    if data_dir is not None and not raw.is_absolute():
        raw = root / raw
    truth = pd.read_csv(raw / "val.csv", dtype={"id": str}).set_index("id").label
    classes = pd.read_csv(raw / "classes.csv")
    labels = classes.label.to_numpy()
    names = classes.set_index("label").class_name.to_dict()
    probabilities = {}
    for run_id in run_ids:
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("Run IDs must be directory names")
        config = json.loads((root / "artifacts/logs" / run_id / "config.json").read_text(encoding="utf-8"))
        if config["model"].get("architecture", config["model"]["name"]) != "resnet50":
            raise ValueError("Training-strategy comparisons require single ResNet50 runs")
        saved = load_probabilities(root / "artifacts/predictions" / run_id / "val_probabilities.npz")
        if saved.true_labels is None or set(saved.ids) != set(truth.index) or set(saved.class_labels) != set(labels):
            raise ValueError("Saved validation must match all current IDs and classes")
        order = pd.Index(saved.ids).get_indexer(truth.index)
        if not np.array_equal(saved.true_labels[order], truth.to_numpy()):
            raise ValueError("Saved validation ground truth mismatch")
        probabilities[run_id] = saved.probabilities[order][:, pd.Index(saved.class_labels).get_indexer(labels)]
    y = truth.to_numpy()
    indices = pd.Index(labels).get_indexer(y)
    base_pred = labels[probabilities[baseline].argmax(1)]
    base_ok = base_pred == y
    rows, class_rows, changes = [], [], []
    for run_id, p in probabilities.items():
        pred = labels[p.argmax(1)]
        ok = pred == y
        rows.append({"exp_id": run_id, "accuracy": float(ok.mean()), "correct": int(ok.sum()),
                     "errors": int((~ok).sum()), "delta_pp": float(100 * (ok.mean() - base_ok.mean())),
                     "fixes": int((ok & ~base_ok).sum()), "breaks": int((~ok & base_ok).sum()),
                     "nll": float(-np.log(np.clip(p[np.arange(len(y)), indices], 1e-30, 1)).mean())})
        for label in labels:
            mask = y == label
            class_rows.append({"exp_id": run_id, "class_name": names[label], "support": int(mask.sum()),
                               "correct": int((mask & ok).sum()), "errors": int((mask & ~ok).sum()),
                               "delta_correct": int((mask & ok).sum() - (mask & base_ok).sum())})
        for i in np.flatnonzero(ok != base_ok):
            changes.append({"exp_id": run_id, "id": truth.index[i], "true_class": names[y[i]],
                            "baseline_class": names[base_pred[i]], "candidate_class": names[pred[i]],
                            "change": "fix" if ok[i] else "break", "confidence": float(p[i].max())})
    return StrategyComparison(pd.DataFrame(rows), pd.DataFrame(class_rows),
                              pd.DataFrame(changes, columns=["exp_id", "id", "true_class", "baseline_class",
                                                            "candidate_class", "change", "confidence"]))
