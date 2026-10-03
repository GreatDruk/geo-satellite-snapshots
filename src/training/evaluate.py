"""Model evaluation utilities."""

import torch
import torch.nn as nn

from src.models.models import SimpleCNN

def evaluate_model(
        model: nn.Module, test_loader: torch.utils.data.DataLoader, 
        criterion: nn.Module, device: torch.device) -> float:
    """Evaluate the model on the test dataset."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return running_loss / len(test_loader.dataset), accuracy
