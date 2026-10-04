"""Inference, ordered prediction tables and submission export."""
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import SequentialSampler


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
