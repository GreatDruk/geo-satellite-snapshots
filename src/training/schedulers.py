"""Optional epoch-level learning-rate schedules configured independently of optimizer."""
from torch.optim import lr_scheduler


def make_scheduler(optimizer, config, epochs):
    if config is None:
        return None, None
    options = dict(config)
    name = options.pop("name").lower()
    if name in {"step", "steplr"}:
        if options.get("step_size", 0) <= 0:
            raise ValueError("StepLR requires positive step_size")
        return lr_scheduler.StepLR(optimizer, **options), None
    if name in {"cosine", "cosineannealinglr"}:
        options.setdefault("T_max", epochs)
        if options["T_max"] <= 0:
            raise ValueError("CosineAnnealingLR requires positive T_max")
        return lr_scheduler.CosineAnnealingLR(optimizer, **options), None
    if name in {"plateau", "reducelronplateau"}:
        monitor = options.pop("monitor", "val_loss")
        if monitor not in {"val_loss", "val_accuracy"}:
            raise ValueError("Plateau monitor must be val_loss or val_accuracy")
        mode = "min" if monitor == "val_loss" else "max"
        if options.setdefault("mode", mode) != mode:
            raise ValueError(f"{monitor} requires mode={mode}")
        return lr_scheduler.ReduceLROnPlateau(optimizer, **options), monitor
    raise ValueError(f"Unknown scheduler: {name}")


def step_scheduler(scheduler, monitor, metrics):
    """Called after training and validation, for the next epoch's LR."""
    if scheduler is not None:
        if monitor is None:
            scheduler.step()
        else:
            scheduler.step(metrics[monitor])
