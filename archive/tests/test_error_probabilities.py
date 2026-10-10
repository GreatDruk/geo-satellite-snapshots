"""Validation-only fallback and ranking/visualization of probability cases."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from archive.src.error_analysis import validation_probabilities, probability_cases, plot_probability_cases
from archive.src.training.predict import ProbabilityPredictions, save_probabilities


class ErrorProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.classes = pd.DataFrame({"label": [11, 4, 8], "class_name": ["A", "B", "C"]})
        # Different class order; one very confident error, one close correct pair.
        self.predictions = ProbabilityPredictions(
            np.array(["wrong.png", "close.png", "right.png"]), np.array([8, 11, 4]),
            np.array([[.99, .005, .005], [.02, .49, .49], [.01, .09, .9]]), np.array([4, 11, 4]))

    def test_wrong_and_uncertain_rankings_use_probability_class_order(self):
        wrong = probability_cases(self.predictions, self.classes, selection="errors", top_k=2)
        self.assertEqual(wrong.id.tolist(), ["wrong.png"])
        self.assertEqual(wrong.iloc[0].predicted_class, "C")
        self.assertEqual(wrong.iloc[0].top_classes, ["C", "A"])
        self.assertEqual(wrong.iloc[0].true_rank, 3)
        self.assertAlmostEqual(wrong.iloc[0].true_probability, .005)
        uncertain = probability_cases(self.predictions, self.classes, selection="uncertain", limit=2, top_k=3)
        self.assertEqual(uncertain.id.tolist(), ["close.png", "right.png"])
        self.assertTrue(uncertain.iloc[0].correct)
        self.assertEqual(uncertain.iloc[0].margin, 0)

    def test_missing_cache_infers_only_validation_and_reuses_saved_probabilities(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "data/raw"
            (raw / "val").mkdir(parents=True)
            self.classes.to_csv(raw / "classes.csv", index=False)
            truth = pd.DataFrame({"id": ["x.png", "y.png"], "label": [4, 11]})
            truth.to_csv(raw / "val.csv", index=False)
            for name in truth.id:
                Image.new("RGB", (4, 4), "green").save(raw / "val" / name)
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 3))
            checkpoint = {"config": {"data": {"image_size": 4, "pin_memory": True}, "seed": 42},
                          "class_labels": [8, 11, 4]}
            with patch("archive.src.experiment.load_checkpoint", return_value=(model, checkpoint)) as restore:
                result, _ = validation_probabilities(root, "tiny", device="cpu")
                self.assertEqual(restore.call_count, 1)
                self.assertTrue(restore.call_args.kwargs["inference_only"])
            self.assertEqual(result.ids.tolist(), truth.id.tolist())
            self.assertEqual(result.true_labels.tolist(), truth.label.tolist())
            self.assertFalse(model.training)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            with patch("archive.src.experiment.load_checkpoint", side_effect=AssertionError("must use cache")):
                cached, _ = validation_probabilities(root, "tiny")
            np.testing.assert_array_equal(cached.probabilities, result.probabilities)
            self.assertFalse((root / "experiments.csv").exists())
            self.assertFalse((raw / "train.csv").exists())
            self.assertFalse((raw / "test.csv").exists())

    def test_invalid_cache_is_rejected_instead_of_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "data/raw"
            raw.mkdir(parents=True)
            self.classes.to_csv(raw / "classes.csv", index=False)
            pd.DataFrame({"id": self.predictions.ids, "label": [11, 11, 4]}).to_csv(raw / "val.csv", index=False)
            cache = root / "artifacts/predictions/run/val_probabilities.npz"
            save_probabilities(self.predictions, cache)
            before = cache.read_bytes()
            with patch("archive.src.experiment.load_checkpoint", side_effect=AssertionError("invalid cache must not infer")):
                with self.assertRaisesRegex(ValueError, "targets"):
                    validation_probabilities(root, "run")
            self.assertEqual(before, cache.read_bytes())

    def test_gallery_shows_true_probability_outside_top_k_and_full_image(self):
        with tempfile.TemporaryDirectory() as folder:
            Image.new("RGB", (256, 256), "red").save(Path(folder) / "wrong.png")
            cases = probability_cases(self.predictions, self.classes, top_k=2)
            with patch("matplotlib.pyplot.show"):
                figs = plot_probability_cases(cases, folder, "Errors")
            self.assertEqual(len(figs), 1)
            self.assertEqual(figs[0].axes[0].images[0].get_array().shape, (256, 256, 3))
            self.assertIn("rank: 3", figs[0].axes[1].get_title())
            self.assertEqual(len(figs[0].axes[1].patches), 2)
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
