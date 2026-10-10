"""Probability alignment, TTA and frozen checkpoint replay on synthetic CPU inputs."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from archive.src.models import build_model
from archive.src.inference import run_selected_inference, run_validation_suite
from archive.src.training.predict import (ProbabilityPredictions, ensemble_probabilities, load_probabilities,
                                  probability_predictions, probability_predictions_many,
                                  save_probabilities, tta_views)


class Images(Dataset):
    def __init__(self):
        self.data = pd.DataFrame({"id": ["a", "b", "c"], "label": [4, 11, 4]})
        self.images = torch.arange(3 * 3 * 4 * 4).reshape(3, 3, 4, 4).float() / 100
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.images[index], self.data.label.iloc[index]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shared_views_equal_independent_probability_averages_and_keep_bn_frozen(self):
        torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Flatten(), torch.nn.Linear(48, 2))
        loader = DataLoader(Images(), batch_size=2)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        actual = probability_predictions_many(model, loader, "cpu", [4, 11], tta_modes=["none", "flips", "d4"])
        for mode in actual:
            with torch.inference_mode():
                manual = torch.stack([model(x).softmax(1) for x in tta_views(loader.dataset.images, mode)]).mean(0)
            np.testing.assert_allclose(actual[mode].probabilities, manual.numpy(), atol=1e-6)
            self.assertEqual(actual[mode].ids.tolist(), ["a", "b", "c"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_d4_has_eight_distinct_views_and_rejects_rectangles(self):
        x = torch.arange(16).reshape(1, 1, 4, 4)
        self.assertEqual(len({tuple(view.flatten().tolist()) for view in tta_views(x, "d4")}), 8)
        with self.assertRaises(ValueError):
            list(tta_views(torch.zeros(1, 1, 3, 4), "d4"))

    def test_ensemble_aligns_ids_classes_and_targets_and_averages_probabilities(self):
        first = ProbabilityPredictions(np.array(["a", "b"]), np.array([4, 11]),
                                       np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([4, 11]))
        second = ProbabilityPredictions(np.array(["b", "a"]), np.array([11, 4]),
                                        np.array([[0.6, 0.4], [0.3, 0.7]]), np.array([11, 4]))
        result = ensemble_probabilities([first, second], [1, 3])
        np.testing.assert_allclose(result.probabilities, [[0.75, 0.25], [0.35, 0.65]])
        self.assertEqual(result.class_labels.tolist(), [4, 11])
        second.true_labels = np.array([4, 4])
        with self.assertRaises(ValueError):
            ensemble_probabilities([first, second])
        with self.assertRaises(ValueError):
            ensemble_probabilities([first], [-1])

    def test_portable_npz_and_scores_no_overwrite_invalid_probabilities(self):
        result = ProbabilityPredictions(np.array(["a"]), np.array([4, 11]), np.array([[0.2, 0.8]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probs.npz"
            save_probabilities(result, path)
            restored = load_probabilities(path)
            self.assertEqual(restored.table(include_scores=True).second_label.tolist(), [4])
            self.assertAlmostEqual(restored.table(include_scores=True).margin.iloc[0], 0.6)
            with self.assertRaises(FileExistsError):
                save_probabilities(result, path)
        result.probabilities[0, 0] = np.nan
        with self.assertRaises(ValueError):
            result.validate()

    def test_shuffled_loader_is_rejected(self):
        with self.assertRaises(ValueError):
            probability_predictions(torch.nn.Flatten(), DataLoader(Images(), shuffle=True), "cpu", [4, 11])

    def test_unlabelled_dataset_does_not_treat_placeholder_column_as_ground_truth(self):
        dataset = Images()
        dataset.label_to_index = None
        dataset.data["label"] = -1
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))
        result = probability_predictions(model, DataLoader(dataset), "cpu", [4, 11])
        self.assertIsNone(result.true_labels)
        self.assertNotIn("true_label", result.table())

    def test_suite_selects_on_val_without_train_test_or_training_journal_and_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data/raw"
            (raw / "val").mkdir(parents=True)
            pd.DataFrame({"label": [4, 11], "class_name": ["a", "b"]}).to_csv(raw / "classes.csv", index=False)
            pd.DataFrame({"id": ["a.png", "b.png"], "label": [4, 11]}).to_csv(raw / "val.csv", index=False)
            for name, color in [("a.png", 0), ("b.png", 255)]:
                Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8)).save(raw / "val" / name)
            config = {"seed": 42, "data": {"image_size": 8, "batch_size": 2},
                      "model": {"name": "mlp", "layers": [{"type": "flatten"},
                                  {"type": "linear", "out_features": "num_classes"}]}}
            model = build_model(config["model"], 8, 2)
            with torch.no_grad():
                model[1].weight[0].fill_(-1)
                model[1].weight[1].fill_(1)
                model[1].bias.zero_()
            checkpoint = root / "best.pt"
            torch.save({"config": config, "class_labels": [4, 11], "epoch": 1,
                        "model_state_dict": model.state_dict()}, checkpoint)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_validation_suite({"a": checkpoint, "b": checkpoint}, "suite",
                                              project_root=root, device="cpu", batch_size=2)
            self.assertEqual(result["comparison"].correct.tolist(), [2] * 9)
            self.assertFalse((root / "experiments.csv").exists())
            replay = run_selected_inference(result["selection"], "replay", project_root=root, device="cpu")
            self.assertEqual(replay["summary"]["correct"], 2)
            self.assertTrue((replay["paths"]["predictions"] / "val_probabilities.npz").exists())
            with patch("archive.src.inference.load_checkpoint", side_effect=AssertionError("cache must avoid forwards")):
                with contextlib.redirect_stdout(io.StringIO()):
                    cached = run_validation_suite({"a": checkpoint, "b": checkpoint}, "cached",
                                                  project_root=root, device="cpu", cache_from="suite")
            pd.testing.assert_frame_equal(result["comparison"], cached["comparison"])
            with self.assertRaises(FileExistsError):
                run_validation_suite({"a": checkpoint}, "suite", project_root=root)


if __name__ == "__main__":
    unittest.main()
