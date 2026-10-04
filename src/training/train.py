"""One epoch for any classifier producing logits."""
import math


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, count = 0.0, 0, 0
    non_blocking = device.type == "cuda" and getattr(train_loader, "pin_memory", False)
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        count += labels.size(0)
        running_loss += loss_value * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    if not count:
        raise ValueError("Empty training loader")
    return running_loss / count, correct / count
