"""Run a configured experiment and preserve its metrics and artifacts."""

import argparse
import copy
import json
import platform
import re
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.data import load_data, load_test
from src.models import build_model
from src.training.evaluate import evaluate_model
from src.training.predict import prediction_table, save_submission
from src.training.schedulers import make_scheduler, step_scheduler
from src.training.train import train_model
from src.utils import append_experiment, get_device, plot_history, set_seed, validate_journal

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExperimentResult:
    exp_id: str
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    val_accuracy: float
    paths: dict


def load_config(config, project_root=PROJECT_ROOT):
    if isinstance(config, dict):
        return copy.deepcopy(config)
    path = Path(config)
    if not path.is_absolute():
        path = Path(project_root) / path
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def make_optimizer(model, config):
    options = dict(config)
    name = options.pop("name").lower()
    constructors = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    if name not in constructors:
        raise ValueError(f"Unknown optimizer: {name}")
    return constructors[name](model.parameters(), **options)


def load_checkpoint(checkpoint_path, device="auto", model_factory=None):
    """Restore the model and saved class order, without reading/fitting any dataset."""
    device = get_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = build_model(config["model"], config["data"]["image_size"],
                        len(checkpoint["class_labels"]), model_factory)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def run_experiment(config, exp_id=None, *, project_root=PROJECT_ROOT,
                   model_factory=None, show_plots=True, predict_test=False):
    """Train on train, select by val accuracy, save best/last and append one journal row.

    Model factories must produce a single [batch, num_classes] logits tensor.
    Runs have unique IDs; run them sequentially when sharing experiments.csv.
    Test inference and submission creation are opt-in.
    """
    root = Path(project_root).resolve()
    config = load_config(config, root)
    exp_id = exp_id or config.get("exp_id")
    if not isinstance(exp_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", exp_id):
        raise ValueError("exp_id: 1-80 letters, digits, underscores or hyphens")
    config["exp_id"] = exp_id
    training = config["training"]
    epochs, log_every = training["epochs"], training.get("log_every", 1)
    if not isinstance(epochs, int) or epochs <= 0 or not isinstance(log_every, int) or log_every <= 0:
        raise ValueError("epochs and log_every must be positive integers")
    if config["data"]["batch_size"] <= 0 or config["data"].get("num_workers", 0) < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    journal = root / "experiments.csv"
    validate_journal(journal, exp_id)
    paths = {
        "models": root / "artifacts/models" / exp_id,
        "predictions": root / "artifacts/predictions" / exp_id,
        "logs": root / "artifacts/logs" / exp_id,
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Run directory already exists: {path}; use a new exp_id")
    submission = root / "submissions" / f"{exp_id}.csv"
    if predict_test and submission.exists():
        raise FileExistsError(submission)
    set_seed(config["seed"], training.get("deterministic", True))
    device = get_device(training.get("device", "auto"))
    data_dir = Path(config["data"].get("raw_dir", "data/raw"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    train_loader, val_loader, classes = load_data(data_dir, config["data"], config["seed"])
    class_labels = classes.label.tolist()
    model = build_model(config["model"], config["data"]["image_size"], len(classes), model_factory).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, training["optimizer"])
    scheduler, scheduler_monitor = make_scheduler(optimizer, training.get("scheduler"), epochs)
    # Catch factory/shape errors before creating any run artifacts.
    model.eval()
    height_width = config["data"]["image_size"]
    height_width = (height_width, height_width) if isinstance(height_width, int) else tuple(height_width)
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, *height_width, device=device))
        if not isinstance(output, torch.Tensor) or output.shape != (2, len(classes)):
            raise ValueError("Model must return [batch, num_classes] logits")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    (paths["logs"] / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {"python": platform.python_version(), "torch": str(torch.__version__),
                "device": str(device), "cuda": torch.version.cuda,
                "packages": {name: version(name) for name in ("numpy", "pandas", "pillow", "matplotlib")},
                "seed": config["seed"], "deterministic": training.get("deterministic", True),
                "class_labels": class_labels, "class_names": classes.class_name.tolist()}
    (paths["logs"] / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows, best_accuracy, best_epoch = [], -1.0, 0
    started = time.perf_counter()

    def checkpoint(epoch):
        return {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch, "best_epoch": best_epoch, "val_accuracy": best_accuracy,
                "config": config, "class_labels": class_labels,
                "class_names": classes.class_name.tolist()}

    def emit(message):
        print(message, flush=True)
        with (paths["logs"] / "training.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    emit(f"{exp_id} | {config['model']['name']} | device={device} | seed={config['seed']}")
    try:
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
            step_scheduler(scheduler, scheduler_monitor, {"val_loss": val_loss, "val_accuracy": val_accuracy})
            rows.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                         "val_loss": val_loss, "val_accuracy": val_accuracy,
                         "learning_rate": learning_rate,
                         "next_learning_rate": optimizer.param_groups[0]["lr"],
                         "epoch_time_sec": time.perf_counter() - epoch_started})
            pd.DataFrame(rows).to_csv(paths["logs"] / "history.csv", index=False)
            if val_accuracy > best_accuracy:
                best_accuracy, best_epoch = val_accuracy, epoch
                torch.save(checkpoint(epoch), paths["models"] / "best.pt")
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                emit(f"Epoch {epoch}/{epochs} | train loss={train_loss:.4f} acc={train_accuracy:.4f}"
                     f" | val loss={val_loss:.4f} acc={val_accuracy:.4f}"
                     f" | lr={learning_rate:.3g} -> {optimizer.param_groups[0]['lr']:.3g}"
                     f" | {rows[-1]['epoch_time_sec']:.1f}s")
        elapsed = (time.perf_counter() - started) / 60
        torch.save(checkpoint(epochs), paths["models"] / "last.pt")
        best = torch.load(paths["models"] / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(best["model_state_dict"])
        model.eval()
        prediction_table(model, val_loader, device, class_labels).to_csv(paths["predictions"] / "val.csv", index=False)
        submission_id = ""
        if predict_test:
            test_loader = load_test(data_dir, config["data"], config["seed"])
            predictions = prediction_table(model, test_loader, device, class_labels)
            predictions.to_csv(paths["predictions"] / "test.csv", index=False)
            save_submission(predictions, data_dir / "sample_submission_example.csv", submission)
            submission_id = exp_id
            paths["submission"] = submission
        history = pd.DataFrame(rows)
        plot_history(history, paths["logs"] / "learning_curves.png", show=False)
        summary = {"best_epoch": best_epoch, "val_accuracy": best_accuracy, "train_time_min": elapsed}
        (paths["logs"] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        append_experiment(journal, {
            "exp_id": exp_id, "model": config["model"]["name"],
            "image_size": json.dumps(config["data"]["image_size"]),
            "augmentation": json.dumps(config["data"].get("augmentation", {}), sort_keys=True),
            "pretrained": config["model"].get("pretrained", False), "val_accuracy": best_accuracy,
            "train_time_min": elapsed, "seed": config["seed"], "submission_id": submission_id,
            "public_score": "", "notes": config.get("notes", ""),
        })
    except BaseException as error:
        emit(f"Run failed: {type(error).__name__}: {error}")
        raise
    emit(f"Best epoch={best_epoch} | val accuracy={best_accuracy:.4f} | artifacts={paths['logs']}")
    if show_plots:
        plot_history(history, show=True)
    return ExperimentResult(exp_id, model, history, best_epoch, best_accuracy, paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--predict-test", action="store_true")
    args = parser.parse_args()
    run_experiment(args.config, args.exp_id, show_plots=False, predict_test=args.predict_test)


if __name__ == "__main__":
    main()
