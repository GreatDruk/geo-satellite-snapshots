"""Validation without gradients or parameter updates."""
import math
import torch


@torch.inference_mode()
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, count = 0.0, 0, 0
    non_blocking = device.type == "cuda" and getattr(val_loader, "pin_memory", False)
    for images, labels in val_loader:
        images, labels = images.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError("Nonfinite validation loss")
        count += labels.size(0)
        running_loss += loss_value * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    if not count:
        raise ValueError("Empty validation loader")
    return running_loss / count, correct / count
