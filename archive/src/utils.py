"""Seeds, devices, experiment journal and training plots."""
import csv
import json
import os
import random
from pathlib import Path
import numpy as np
import torch

EXPERIMENT_COLUMNS = [
    "exp_id", "model", "image_size", "augmentation", "pretrained", "val_accuracy",
    "train_time_min", "seed", "submission_id", "public_score", "notes",
]


def set_seed(seed=42, deterministic=True, warn_only=False):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def get_device(name="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def get_answers(y_pred):
    return y_pred.argmax(dim=1)


def calculate_accuracy(y_true, y_pred):
    if y_true.numel() == 0:
        raise ValueError("Empty target")
    return (y_true == y_pred).float().mean().item()


def validate_journal(path, exp_id):
    path = Path(path)
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != EXPERIMENT_COLUMNS:
                raise ValueError("experiments.csv has unexpected headers")
            if any(row["exp_id"] == exp_id for row in reader):
                raise FileExistsError(f"Experiment {exp_id} already exists")


def append_experiment(path, record):
    path = Path(path)
    validate_journal(path, record["exp_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPERIMENT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def plot_history(history, save_path=None, show=True):
    import matplotlib.pyplot as plt
    has_lr = "learning_rate" in history
    columns = 3 if has_lr else 2
    fig, axes = plt.subplots(1, columns, figsize=(6 * columns, 4))
    for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        for split in ("train", "val"):
            axis.plot(history["epoch"], history[f"{split}_{metric}"], label=split)
        axis.set(xlabel="Epoch", ylabel=title, title=title)
        axis.grid(alpha=0.3)
        axis.legend()
    if has_lr:
        if "learning_rates" in history:
            rates = [json.loads(value) for value in history["learning_rates"]]
            for name in rates[0]:
                axes[2].plot(history["epoch"], [row[name] for row in rates], label=name)
            axes[2].legend()
        else:
            axes[2].plot(history["epoch"], history["learning_rate"])
        axes[2].set(xlabel="Epoch", ylabel="Learning rate", title="Learning rate")
        axes[2].grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def raw_data_dir(root):
    root = Path(root)
    return root.parent / "data/raw" if root.name == "archive" else root / "data/raw"


def resolve_raw_dir(root, value="data/raw"):
    path = Path(value)
    if path.is_absolute():
        return path
    return raw_data_dir(root) if path.as_posix() == "data/raw" else Path(root) / path
