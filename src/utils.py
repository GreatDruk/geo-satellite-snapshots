"""Shared utilities for reproducible experiments."""

import numpy as np
import random
import torch

def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Get the available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate the accuracy of predictions."""
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    return correct / total

def get_answers(y_pred: torch.Tensor) -> torch.Tensor:
    """Get the predicted class labels from model outputs."""
    return torch.argmax(y_pred, dim=1)
