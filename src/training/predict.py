"""Prediction and submission utilities."""

import torch
from torch.utils.data import DataLoader

def predict(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Make predictions using the trained model."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return torch.tensor(predictions)
