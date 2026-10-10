"""One epoch for any classifier producing logits."""
import math
from .losses import CEWithCosFace
from .mining import HardExampleSampler


def train_model(model, train_loader, criterion, optimizer, device, *, batch_augmentation=None):
    model.train()
    running_loss, correct, count = 0.0, 0, 0
    non_blocking = device.type == "cuda" and getattr(train_loader, "pin_memory", False)
    mining = getattr(train_loader, "sampler", None)
    mining = mining if isinstance(mining, HardExampleSampler) else None
    if batch_augmentation is not None and batch_augmentation.p > 0 and (isinstance(criterion, CEWithCosFace) or mining is not None):
        raise ValueError("CutMix cannot be combined with CosFace or hard mining")
    for batch in train_loader:
        images, labels = batch[:2]
        indices = batch[2] if len(batch) == 3 else None
        images, labels = images.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        secondary, retained = labels, 1.0
        if batch_augmentation is not None:
            images, labels, secondary, retained = batch_augmentation(images, labels)
        if isinstance(criterion, CEWithCosFace):
            logits, cosine = model.forward_with_auxiliary(images)
            loss = criterion(logits, cosine, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            if retained < 1:
                loss = retained * loss + (1 - retained) * criterion(logits, secondary)
        if mining is not None:
            if indices is None:
                raise ValueError("Mining requires indexed training batches")
            mining.record(indices, logits, labels)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        count += labels.size(0)
        running_loss += loss_value * labels.size(0)
        predicted = logits.argmax(1)
        correct += retained * (predicted == labels).sum().item() + (1 - retained) * (predicted == secondary).sum().item()
    if not count:
        raise ValueError("Empty training loader")
    return running_loss / count, correct / count
