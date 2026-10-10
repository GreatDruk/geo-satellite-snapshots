"""Augmentation invariants, config parity and native-resolution CNN gradients."""
import copy
import json
import random
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from archive.src.models import build_model
from archive.src.transforms import ImageTransform, preview_augmentations

ROOT = Path(__file__).resolve().parents[1]


class AugmentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = json.loads((ROOT / 'configs/cnn_256_domain_shift.json').read_text())['data']
        y, x = np.indices((256, 256))
        self.image = Image.fromarray(np.stack((x, y, (x + y) % 256), axis=-1).astype(np.uint8))

    def test_seeded_sequence_is_reproducible_and_source_immutable(self):
        original = self.image.tobytes()
        transform = ImageTransform(self.config, True)
        def sequence():
            random.seed(42)
            np.random.seed(42)
            return [transform(self.image) for _ in range(6)]
        first, second = sequence(), sequence()
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertEqual(left.shape, (3, 256, 256))
            self.assertEqual(left.dtype, torch.float32)
            self.assertTrue(torch.isfinite(left).all())
            self.assertGreaterEqual(left.min().item(), -1)
            self.assertLessEqual(left.max().item(), 1)
        self.assertFalse(torch.equal(first[0], first[1]))
        self.assertEqual(self.image.tobytes(), original)

    def test_eval_disables_every_random_operation_and_preserves_rng(self):
        transform = ImageTransform(self.config)
        random.seed(7)
        np.random.seed(7)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        result = transform(self.image)
        expected = torch.from_numpy(np.array(self.image, dtype=np.float32) / 255).permute(2, 0, 1)
        torch.testing.assert_close(result, (expected - 0.5) / 0.5, rtol=0, atol=0)
        self.assertEqual(python_state, random.getstate())
        np.testing.assert_equal(numpy_state, np.random.get_state())

    def test_each_operation_and_zero_probability(self):
        specs = {
            'color_jitter': {'p': 1, 'brightness': 0.3, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.05},
            'gamma': {'p': 1, 'range': [1.4, 1.4]},
            'gaussian_noise': {'p': 1, 'std': [0.02, 0.02]},
            'gaussian_blur': {'p': 1, 'sigma': [1, 1]},
            'jpeg_compression': {'p': 1, 'quality': [60, 60]},
            'resolution_degradation': {'p': 1, 'scale': [0.65, 0.65]},
        }
        expected = ImageTransform({**self.config, 'augmentation': {}})(self.image)
        for name, spec in specs.items():
            with self.subTest(name=name):
                random.seed(123)
                np.random.seed(123)
                config = {**self.config, 'augmentation': {name: spec}}
                self.assertFalse(torch.equal(expected, ImageTransform(config, True)(self.image)))
                config['augmentation'] = {name: {**spec, 'p': 0}}
                torch.testing.assert_close(expected, ImageTransform(config, True)(self.image), rtol=0, atol=0)

    def test_quarter_turns_preserve_pixels_and_rotation_has_no_black_corners(self):
        image = Image.fromarray(np.full((256, 256, 3), 170, dtype=np.uint8))
        for aug in ({'rotation_degrees': 15}, {'horizontal_flip': 1, 'vertical_flip': 1, 'random_rotate90': 1}):
            result = ImageTransform({**self.config, 'augmentation': aug}, True)(image)
            expected = ImageTransform({**self.config, 'augmentation': {}})(image)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        result = ImageTransform({**self.config, 'augmentation': {'random_rotate90': 1}}, True)(self.image)
        expected = ImageTransform({**self.config, 'augmentation': {}})(self.image)
        torch.testing.assert_close(result.flatten().sort().values, expected.flatten().sort().values, rtol=0, atol=0)

    def test_corruption_gate_and_budget(self):
        from unittest.mock import patch
        config = {**self.config, 'augmentation': {
            'corruption_probability': 0, 'gaussian_noise': {'p': 1, 'std': [0.1, 0.1]}}}
        clean = ImageTransform({**self.config, 'augmentation': {}})(self.image)
        torch.testing.assert_close(clean, ImageTransform(config, True)(self.image), rtol=0, atol=0)
        config['augmentation'] = {'max_corruptions': 1, 'gaussian_noise': {'p': 1}, 'gaussian_blur': {'p': 1},
                                  'jpeg_compression': {'p': 1}, 'resolution_degradation': {'p': 1}}
        # All four pass their probability gates; only one can reach its parameter sampling.
        transform = ImageTransform(config, True)
        with patch.object(transform, '_range', wraps=transform._range) as sampler:
            transform(self.image)
            self.assertEqual(sampler.call_count, 1)

    def test_invalid_settings_rejected_on_train_and_eval(self):
        invalid = [{'unknown': 1}, {'gaussian_noise': {'p': 1, 'std': [-1, 0.1]}},
                   {'gamma': {'range': [1.1, 0.9]}}, {'jpeg_compression': {'quality': [40.5, 90]}},
                   {'gaussian_blur': {'sigma': [0, float('nan')]}}, {'color_jitter': {'hue': 0.6}},
                   {'resolution_degradation': {'scale': [0.5, 1.1]}}, {'max_corruptions': True},
                   {'gaussian_noise': {'probability': 0.5}}, {'random_rotate90': 2}]
        for aug in invalid:
            for training in (True, False):
                with self.subTest(aug=aug, training=training), self.assertRaises(ValueError):
                    ImageTransform({**self.config, 'augmentation': aug}, training)

    def test_preview_preserves_rng(self):
        import matplotlib.pyplot as plt
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        fig = preview_augmentations(self.image, self.config, count=2)
        self.assertEqual(len(fig.axes), 3)
        plt.close(fig)
        self.assertEqual(python_state, random.getstate())
        np.testing.assert_equal(numpy_state, np.random.get_state())
        torch.testing.assert_close(torch_state, torch.get_rng_state(), rtol=0, atol=0)

    def test_matched_configs_and_all_active_cnn_gradients(self):
        matched = []
        for name in ('no_aug', 'geometry', 'photo', 'domain_shift'):
            config = json.loads((ROOT / f'configs/cnn_256_{name}.json').read_text())
            config.pop('notes')
            config['data'].pop('augmentation')
            matched.append(config)
        self.assertTrue(all(config == matched[0] for config in matched))
        for path in (ROOT / 'configs').glob('*.json'):
            config = json.loads(path.read_text())
            if config['model']['name'] != 'cnn':
                continue
            with self.subTest(path=path.name):
                self.assertEqual(config['data']['image_size'], 256)
                ImageTransform(config['data'], True)
                model = build_model(config['model'], 256, 20)
                logits = model(torch.randn(2, 3, 256, 256))
                self.assertEqual(logits.shape, (2, 20))
                torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
