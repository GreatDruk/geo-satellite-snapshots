"""One epoch for any classifier producing logits."""
import math


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, count = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not math.isfinite(loss.item()):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        count += labels.size(0)
        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    if not count:
        raise ValueError("Empty training loader")
    return running_loss / count, correct / count
