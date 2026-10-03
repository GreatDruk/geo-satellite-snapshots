"""Model training utilities."""

import torch
import torch.nn as nn

from src.models.models import SimpleCNN

def train_model(
        model: nn.Module, train_loader: torch.utils.data.DataLoader, 
        criterion: nn.Module, optimizer: torch.optim.Optimizer, 
        device: torch.device) -> float:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)
