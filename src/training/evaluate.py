"""Validation without gradients or parameter updates."""
import math
import torch


@torch.inference_mode()
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, count = 0.0, 0, 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        if not math.isfinite(loss.item()):
            raise FloatingPointError("Nonfinite validation loss")
        count += labels.size(0)
        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    if not count:
        raise ValueError("Empty validation loader")
    return running_loss / count, correct / count
