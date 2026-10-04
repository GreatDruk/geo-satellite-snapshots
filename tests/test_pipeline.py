"""Synthetic CPU integration checks; never train on competition data."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data import ImageDataset, load_data
from src.experiment import PROJECT_ROOT, load_checkpoint, load_config, run_experiment
from src.models import build_model
from src.training.predict import prediction_table, save_submission
from src.training.schedulers import make_scheduler, step_scheduler
from src.utils import get_device


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "data/raw"
        self.raw.mkdir(parents=True)
        pd.DataFrame({"label": [4, 11], "class_name": ["a", "b"]}).to_csv(self.raw / "classes.csv", index=False)
        for split, count in (("train", 7), ("val", 3), ("test", 2)):
            directory = self.raw / split
            directory.mkdir()
            rows = []
            for i in range(count):
                name = f"{split}_{i}.jpg"
                Image.fromarray(np.full((12, 10, 3), 20 + i * 25, dtype=np.uint8)).save(directory / name)
                row = {"id": name}
                if split != "test":
                    row["label"] = [4, 11][i % 2]
                rows.append(row)
            pd.DataFrame(rows).to_csv(self.raw / f"{split}.csv", index=False)
        pd.DataFrame({"id": ["test_1.jpg", "test_0.jpg"], "label": [4, 4]}).to_csv(
            self.raw / "sample_submission_example.csv", index=False)
        self.config = load_config(PROJECT_ROOT / "configs/baseline_mlp.json")
        self.config["data"].update(image_size=8, batch_size=3)
        self.config["model"]["layers"] = [
            {"type": "flatten"}, {"type": "linear", "out_features": 8},
            {"type": "relu"}, {"type": "dropout", "p": 0.2},
            {"type": "linear", "out_features": "num_classes"},
        ]
        self.config["training"].update(epochs=3, log_every=2, device="cpu")

    def tearDown(self):
        self.temporary.cleanup()

    def run_config(self, config, exp_id, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            result = run_experiment(config, exp_id, project_root=self.root, show_plots=False, **kwargs)
        return result, captured.getvalue()

    def test_full_run_artifacts_journal_restore_and_log(self):
        (self.raw / "test.csv").unlink()
        self.config["training"].update(epochs=4, log_every=3)
        result, output = self.run_config(self.config, "smoke_mlp")
        self.assertIn("Epoch 3/4", output)
        self.assertNotIn("Epoch 2/4", output)
        self.assertEqual(len(result.history), 4)
        for relative in ("learning_curves.png", "history.csv", "config.json", "metadata.json", "summary.json"):
            self.assertTrue((result.paths["logs"] / relative).is_file())
        self.assertTrue((result.paths["models"] / "last.pt").is_file())
        journal = pd.read_csv(self.root / "experiments.csv")
        self.assertEqual(journal.exp_id.tolist(), ["smoke_mlp"])
        self.assertAlmostEqual(journal.val_accuracy.iloc[0], result.history.val_accuracy.max())
        restored, checkpoint = load_checkpoint(result.paths["models"] / "best.pt", "cpu")
        for key, tensor in result.model.state_dict().items():
            torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(checkpoint["class_labels"], [4, 11])
        predictions = pd.read_csv(result.paths["predictions"] / "val.csv")
        self.assertEqual(predictions.id.tolist(), [f"val_{i}.jpg" for i in range(3)])
        self.assertTrue(predictions.label.isin([4, 11]).all())
        self.assertEqual(result.best_epoch, int(result.history.loc[result.history.val_accuracy.idxmax(), "epoch"]))
        with self.assertRaises(FileExistsError):
            self.run_config(self.config, "smoke_mlp")
        self.assertEqual(len(pd.read_csv(self.root / "experiments.csv")), 1)

    def test_seed_repeats_weights_and_metrics_with_augmentation(self):
        self.config["data"]["augmentation"] = {"horizontal_flip": 0.5, "rotation_degrees": 10}
        first, _ = self.run_config(self.config, "repeat_a")
        second, _ = self.run_config(self.config, "repeat_b")
        pd.testing.assert_frame_equal(first.history.drop(columns="epoch_time_sec"), second.history.drop(columns="epoch_time_sec"))
        for key, tensor in first.model.state_dict().items():
            torch.testing.assert_close(tensor, second.model.state_dict()[key], rtol=0, atol=0)

    def test_cnn_variable_image_size_and_training(self):
        config = load_config(PROJECT_ROOT / "configs/baseline_cnn.json")
        config["data"].update(image_size=[16, 20], batch_size=3)
        config["training"].update(epochs=1, device="cpu")
        result, _ = self.run_config(config, "smoke_cnn")
        self.assertEqual(result.model(torch.zeros(2, 3, 16, 20)).shape, (2, 2))
        self.assertFalse(any(isinstance(layer, torch.nn.Softmax) for layer in result.model.modules()))

    def test_opt_in_test_predictions_and_sample_order(self):
        result, _ = self.run_config(self.config, "submission", predict_test=True)
        submission = pd.read_csv(result.paths["submission"])
        self.assertEqual(submission.id.tolist(), ["test_1.jpg", "test_0.jpg"])
        self.assertEqual(submission.columns.tolist(), ["id", "label"])
        with self.assertRaises(FileExistsError):
            save_submission(submission, self.raw / "sample_submission_example.csv", result.paths["submission"])

    def test_external_model_factory(self):
        self.config["model"] = {"name": "custom", "pretrained": False}
        factory = lambda config, size, count: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * size * size, count))
        result, _ = self.run_config(self.config, "factory", model_factory=factory)
        model, _ = load_checkpoint(result.paths["models"] / "best.pt", "cpu", model_factory=factory)
        self.assertEqual(model(torch.zeros(2, 3, 8, 8)).shape, (2, 2))

    def test_prediction_refuses_shuffled_order_and_unknown_labels(self):
        train, val, _ = load_data(self.raw, self.config["data"])
        model = build_model(self.config["model"], 8, 2)
        with self.assertRaises(ValueError):
            prediction_table(model, train, torch.device("cpu"), [4, 11])
        with self.assertRaises(ValueError):
            ImageDataset(self.raw / "train.csv", self.raw / "train", label_to_index={4: 0})

    def test_transfer_run_warmup_journal_groups_and_checkpoint(self):
        from unittest.mock import patch
        from test_transfer import TinyResNet
        self.config["model"] = {"name": "resnet18", "pretrained": True, "weights": "IMAGENET1K_V1",
                                "mode": "partial", "unfreeze_blocks": ["layer4"], "warmup_epochs": 1}
        self.config["training"]["optimizer"].update(backbone_lr=0.0001, head_lr=0.001)
        with patch("src.models.pretrained.load_backbone", return_value=(TinyResNet(), "IMAGENET1K_V1")):
            result, _ = self.run_config(self.config, "transfer_mock")
        self.assertEqual(result.history.training_mode.tolist(), ["head_only", "partial", "partial"])
        self.assertEqual(json.loads(result.history.learning_rates.iloc[0]), {"backbone": 0.0001, "head": 0.001})
        journal = pd.read_csv(self.root / "experiments.csv")
        self.assertEqual(journal.model.iloc[0], "resnet18")
        self.assertTrue(journal.pretrained.iloc[0])
        saved = torch.load(result.paths["models"] / "last.pt", weights_only=True)
        self.assertEqual(saved["config"]["model"]["mode"], "partial")

    def test_auto_device_runs_on_cpu_without_cuda(self):
        from unittest.mock import patch
        self.config["training"]["device"] = "auto"
        self.config["data"].update(pin_memory=True, num_workers=2)
        with patch("torch.cuda.is_available", return_value=False), patch("src.experiment.os.name", "nt"):
            self.assertEqual(get_device("auto").type, "cpu")
            result, _ = self.run_config(self.config, "auto_cpu")
        saved = json.loads((result.paths["logs"] / "config.json").read_text())
        metadata = json.loads((result.paths["logs"] / "metadata.json").read_text())
        self.assertFalse(saved["data"]["pin_memory"])
        self.assertEqual(saved["data"]["num_workers"], 0)
        self.assertEqual(metadata["device"], "cpu")

    def test_auto_device_selects_cuda_when_available(self):
        from unittest.mock import patch
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(get_device("auto").type, "cuda")

    def test_pretrained_policy_and_notebooks(self):
        with self.assertRaises(ValueError):
            build_model({"name": "densenet", "pretrained": True}, 8, 2)
        model = build_model({"name": "resnet", "pretrained": True}, 8, 2,
                            model_factory=lambda config, size, count: torch.nn.Linear(3, count))
        self.assertIsInstance(model, torch.nn.Module)
        notebook_paths = list((PROJECT_ROOT / "notebooks").glob("0[12]_baseline_*.ipynb"))
        notebook_paths.append(PROJECT_ROOT / "notebooks/03_pretrained.ipynb")
        for path in notebook_paths:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), str(path), "exec")

    def test_step_scheduler_history_and_checkpoint_state(self):
        self.config["training"]["scheduler"] = {"name": "step", "step_size": 1, "gamma": 0.5}
        result, output = self.run_config(self.config, "step_scheduler")
        np.testing.assert_allclose(result.history.learning_rate, [0.001, 0.0005, 0.00025])
        np.testing.assert_allclose(result.history.next_learning_rate, [0.0005, 0.00025, 0.000125])
        checkpoint = torch.load(result.paths["models"] / "last.pt", weights_only=True)
        self.assertEqual(checkpoint["scheduler_state_dict"]["last_epoch"], 3)
        self.assertAlmostEqual(checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"], 0.000125)
        self.assertIn("lr=", output)
        optimizer = torch.optim.Adam(result.model.parameters(), lr=0.1)
        scheduler, _ = make_scheduler(optimizer, self.config["training"]["scheduler"], 3)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0000625)

    def test_cosine_and_plateau_monitor_direction(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler, monitor = make_scheduler(optimizer, {"name": "cosine", "eta_min": 0.01}, 2)
        rates = []
        for _ in range(2):
            optimizer.step()
            step_scheduler(scheduler, monitor, {})
            rates.append(optimizer.param_groups[0]["lr"])
        np.testing.assert_allclose(rates, [0.055, 0.01])
        for metric in ("val_loss", "val_accuracy"):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler, monitor = make_scheduler(optimizer, {"name": "plateau", "monitor": metric,
                                                           "patience": 0, "factor": 0.5}, 3)
            for value in ([1.0, 0.8, 0.8] if metric == "val_loss" else [0.5, 0.6, 0.6]):
                optimizer.step()
                step_scheduler(scheduler, monitor, {metric: value})
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)
        with self.assertRaises(ValueError):
            make_scheduler(optimizer, {"name": "plateau", "monitor": "val_accuracy", "mode": "min"}, 3)


if __name__ == "__main__":
    unittest.main()
