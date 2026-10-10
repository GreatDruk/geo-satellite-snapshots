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

