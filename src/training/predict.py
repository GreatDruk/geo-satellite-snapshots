"""Inference, ordered prediction tables and submission export."""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import SequentialSampler


@dataclass
class ProbabilityPredictions:
    """Explicit image/class order; portable NPZ storage never needs pickle."""
    ids: np.ndarray
    class_labels: np.ndarray
    probabilities: np.ndarray
    true_labels: np.ndarray | None = None

    def validate(self):
        ids, labels, values = np.asarray(self.ids), np.asarray(self.class_labels), np.asarray(self.probabilities)
        if ids.ndim != 1 or labels.ndim != 1 or not len(ids) or len(labels) < 2:
            raise ValueError("Nonempty one-dimensional IDs and at least two classes required")
        if len(set(ids.tolist())) != len(ids) or len(set(labels.tolist())) != len(labels):
            raise ValueError("IDs and class labels must be unique")
        if any(not isinstance(item, str) or not item for item in ids.tolist()):
            raise ValueError("Image IDs must be nonempty strings")
        if values.shape != (len(ids), len(labels)) or not np.isfinite(values).all():
            raise ValueError("Probabilities must be finite [images, classes]")
        if (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(1), 1, atol=1e-5):
            raise ValueError("Probabilities must be nonnegative and sum to one")
        if self.true_labels is not None:
            truth = np.asarray(self.true_labels)
            if truth.shape != ids.shape or not np.isin(truth, labels).all():
                raise ValueError("Ground truth must align with IDs and class labels")
        if any(array.dtype.hasobject for array in [ids, labels, values] +
               ([] if self.true_labels is None else [np.asarray(self.true_labels)])):
            raise ValueError("Object arrays are not supported")
        return self

    def table(self, include_scores=False):
        self.validate()
        indices = self.probabilities.argmax(1)
        frame = pd.DataFrame({"id": self.ids, "label": self.class_labels[indices]})
        if self.true_labels is not None:
            frame["true_label"] = self.true_labels
        if include_scores:
            order = np.argsort(-self.probabilities, axis=1, kind="stable")
            rows = np.arange(len(frame))
            frame["confidence"] = self.probabilities[rows, order[:, 0]]
            frame["second_label"] = self.class_labels[order[:, 1]]
            frame["margin"] = frame.confidence.to_numpy() - self.probabilities[rows, order[:, 1]]
            frame["entropy"] = -(self.probabilities * np.log(np.clip(self.probabilities, 1e-30, 1))).sum(1)
        return frame


def save_probabilities(predictions, path):
    predictions.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {"ids": predictions.ids, "class_labels": predictions.class_labels,
              "probabilities": predictions.probabilities}
    if predictions.true_labels is not None:
        values["true_labels"] = predictions.true_labels
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
    return path


def load_probabilities(path):
    with np.load(path, allow_pickle=False) as arrays:
        return ProbabilityPredictions(arrays["ids"], arrays["class_labels"], arrays["probabilities"],
                                      arrays["true_labels"] if "true_labels" in arrays else None).validate()


def tta_views(images, mode):
    if mode not in {"none", "flips", "d4"}:
        raise ValueError("TTA must be none, flips or d4")
    if mode == "none":
        yield images
    elif mode == "flips":
        for dimensions in ((), (-1,), (-2,), (-2, -1)):
            yield images.flip(dimensions) if dimensions else images
    else:
        if images.shape[-2] != images.shape[-1]:
            raise ValueError("D4 TTA requires square images")
        for rotation in range(4):
            rotated = images.rot90(rotation, (-2, -1))
            yield rotated
            yield rotated.flip(-1)


@torch.inference_mode()
def probability_predictions(model, loader, device, class_labels, *, tta="none", progress=None):
    """Average softmax probabilities per view, never logits or argmax votes."""
    return probability_predictions_many(model, loader, device, class_labels,
                                        tta_modes=[tta], progress=progress)[tta]


@torch.inference_mode()
def probability_predictions_many(model, loader, device, class_labels, *,
                                 tta_modes=("none", "flips"), progress=None):
    """Share forwards between original, flips and D4; at most eight per batch."""
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Probability tables require sequential order and drop_last=false")
    if not tta_modes or len(set(tta_modes)) != len(tta_modes) or set(tta_modes) - {"none", "flips", "d4"}:
        raise ValueError("TTA must be none, flips or d4")
    device = torch.device(device)
    model.eval()
    non_blocking = device.type == "cuda" and getattr(loader, "pin_memory", False)
    values = {mode: [] for mode in tta_modes}
    full_mode = "d4" if "d4" in tta_modes else "flips" if "flips" in tta_modes else "none"
    selected = {"none": {0}, "flips": {0, 1, 4, 5} if full_mode == "d4" else {0, 1, 2, 3},
                "d4": set(range(8))}
    for batch_index, (images, _) in enumerate(loader, 1):
        images = images.to(device, non_blocking=non_blocking)
        totals = {mode: None for mode in tta_modes}
        for view_index, view in enumerate(tta_views(images, full_mode)):
            logits = model(view.contiguous())
            if not isinstance(logits, torch.Tensor) or logits.shape != (len(images), len(class_labels)):
                raise ValueError("Expected a [batch, classes] logits tensor")
            probabilities = logits.float().softmax(1)
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("Nonfinite inference probabilities")
            for mode in tta_modes:
                if view_index in selected[mode]:
                    totals[mode] = probabilities if totals[mode] is None else totals[mode] + probabilities
        for mode in tta_modes:
            values[mode].append((totals[mode] / len(selected[mode])).cpu().numpy())
        if progress is not None:
            progress(batch_index, len(loader))
    if not values[tta_modes[0]]:
        raise ValueError("Empty prediction loader")
    data = loader.dataset.data
    truth = (data.label.to_numpy() if "label" in data
             and getattr(loader.dataset, "label_to_index", {}) is not None else None)
    return {mode: ProbabilityPredictions(data.id.to_numpy(dtype=str), np.asarray(class_labels),
                                        np.concatenate(parts),
                                        truth).validate()
            for mode, parts in values.items()}


def ensemble_probabilities(members, weights=None):
    """Align both IDs and class columns; reject inconsistent validation targets."""
    if not members:
        raise ValueError("At least one ensemble member required")
    for member in members:
        member.validate()
    weights = np.asarray(weights if weights is not None else [1] * len(members), dtype=float)
    if weights.shape != (len(members),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Ensemble weights must be finite, nonnegative, with positive sum")
    weights /= weights.sum()
    first = members[0]
    total = np.zeros_like(first.probabilities, dtype=np.float64)
    for member, weight in zip(members, weights):
        if set(member.ids) != set(first.ids) or set(member.class_labels) != set(first.class_labels):
            raise ValueError("Ensemble members must share image IDs and classes")
        rows = pd.Index(member.ids).get_indexer(first.ids)
        columns = pd.Index(member.class_labels).get_indexer(first.class_labels)
        if (member.true_labels is None) != (first.true_labels is None):
            raise ValueError("Do not mix labelled and unlabelled predictions")
        if first.true_labels is not None and not np.array_equal(member.true_labels[rows], first.true_labels):
            raise ValueError("Ensemble ground truth does not agree")
        total += weight * member.probabilities[rows][:, columns]
    return ProbabilityPredictions(first.ids.copy(), first.class_labels.copy(), total.astype(np.float32),
                                  None if first.true_labels is None else first.true_labels.copy()).validate()


@torch.inference_mode()
def predict(model, data_loader, device):
    model.eval()
    non_blocking = device.type == "cuda" and getattr(data_loader, "pin_memory", False)
    predictions = [model(images.to(device, non_blocking=non_blocking)).argmax(1).cpu() for images, _ in data_loader]
    return torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.long)


def prediction_table(model, loader, device, class_labels):
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Prediction tables require sequential order and drop_last=false")
    indices = predict(model, loader, device).tolist()
    frame = pd.DataFrame({"id": loader.dataset.data["id"].tolist(),
                          "label": [class_labels[index] for index in indices]})
    if "label" in loader.dataset.data:
        frame["true_label"] = loader.dataset.data["label"].tolist()
    return frame


def save_submission(predictions, sample_path, output_path):
    sample = pd.read_csv(sample_path, dtype={"id": str})
    if list(sample.columns) != ["id", "label"]:
        raise ValueError("Expected sample submission columns: id,label")
    if sample.id.duplicated().any() or predictions.id.duplicated().any() or set(sample.id) != set(predictions.id):
        raise ValueError("Submission IDs must exactly match sample IDs")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.set_index("id").loc[sample.id, ["label"]].reset_index().to_csv(output_path, index=False)
    return output_path
