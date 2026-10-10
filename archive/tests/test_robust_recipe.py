"""AugMix convex mixing, corruption effects, train-only MixStyle and integration."""
import copy
import csv
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from archive.src.experiment import load_checkpoint, make_optimizer, run_experiment
from archive.src.models import build_model
from archive.src.models.lgffe_checks import check_lgffe
from archive.src.models.mixstyle import MixStyle
from archive.src.robust_augmix import DEFAULT_OPERATIONS, RobustAugMix, motion_kernel, preview_robust_recipe
from archive.src.training.predict import load_probabilities, probability_predictions
from archive.src.transforms import ImageTransform
from archive.src.utils import set_seed
from archive.tests.test_lgffe import TinySpatialResNet

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/resnet50_lgffe_robust_augmix_mixstyle.json'


class RobustRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        y, x = np.indices((256, 256))
        self.image = Image.fromarray(np.stack((x, y, (x + y) % 256), -1).astype(np.uint8))
        self.config = json.loads(CONFIG.read_text(encoding='utf-8'))

    def test_single_recipe_settings_and_no_legacy_pipeline(self):
        config = self.config
        self.assertEqual(config['seed'], 42)
        self.assertEqual(config['data']['image_size'], 256)
        self.assertEqual(config['data']['batch_size'], 32)
        self.assertEqual(set(config['data']['augmentation']), {'random_d4', 'robust_augmix'})
        self.assertTrue(config['data']['augmentation']['random_d4'])
        augmentation = config['data']['augmentation']['robust_augmix']
        self.assertEqual((augmentation['width'], augmentation['depth'], augmentation['alpha']), (3, [1, 3], 1.0))
        self.assertEqual(set(augmentation['operations']), set(DEFAULT_OPERATIONS))
        old = json.loads((ROOT / 'configs/resnet50_lgffe_domain_shift_stronger.json').read_text())
        self.assertEqual(config['training'], old['training'])
        model = copy.deepcopy(config['model'])
        model.pop('mixstyle')
        self.assertEqual(model, old['model'])
        self.assertEqual(config['model']['mixstyle'], {'p': 0.5, 'alpha': 0.1, 'after_blocks': ['layer1', 'layer2']})

    def test_augmix_has_dirichlet_beta_three_branches_and_random_depth(self):
        transform = RobustAugMix({'width': 3, 'depth': [1, 3], 'alpha': 1})
        white = Image.fromarray(np.full((256, 256, 3), 255, dtype=np.uint8))
        black = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        depths = iter([1, 2, 3])
        calls = []
        def operation(image, name):
            calls.append(name)
            return black
        with patch('archive.src.robust_augmix.np.random.dirichlet', return_value=np.array([0.2, 0.3, 0.5])) as dirichlet, \
             patch('archive.src.robust_augmix.np.random.beta', return_value=0.25) as beta, \
             patch('archive.src.robust_augmix.np.random.randint', side_effect=lambda *args: next(depths)), \
             patch.object(transform, 'apply_operation', side_effect=operation):
            output = transform(white)
        dirichlet.assert_called_once_with([1, 1, 1])
        beta.assert_called_once_with(1, 1)
        self.assertEqual(len(calls), 6)
        self.assertEqual(output.mode, 'RGB')
        self.assertEqual(output.size, (256, 256))
        np.testing.assert_array_equal(np.asarray(output), np.full((256, 256, 3), 191, dtype=np.uint8))

    def test_operations_act_and_keep_shape_rgb_pixels(self):
        specs = copy.deepcopy(DEFAULT_OPERATIONS)
        specs['brightness']['range'] = [0.35, 0.35]
        specs['gamma']['range'] = [2.2, 2.2]
        specs['grayscale']['range'] = [1, 1]
        specs['resolution_degradation']['range'] = [0.25, 0.25]
        specs['motion_blur']['range'] = [13, 13]
        transform = RobustAugMix({'operations': specs})
        original = self.image.tobytes()
        for name in specs:
            with self.subTest(operation=name):
                np.random.seed(123)
                # A uniform full-range ramp is already equalized/autocontrasted.
                source = self.image
                if name in {'autocontrast', 'equalize'}:
                    ramp = np.asarray(self.image, dtype=np.float32)/255
                    source = Image.fromarray(np.rint(40+120*ramp**2).astype(np.uint8))
                output = transform.apply_operation(source, name)
                self.assertEqual(output.mode, 'RGB')
                self.assertEqual(output.size, (256, 256))
                values = np.asarray(output)
                self.assertEqual(values.dtype, np.uint8)
                self.assertGreaterEqual(values.min(), 0)
                self.assertLessEqual(values.max(), 255)
                self.assertFalse(np.array_equal(values, np.asarray(source)), name)
        for name in ['brightness', 'gamma']:
            self.assertLess(np.asarray(transform.apply_operation(self.image, name)).mean(), np.asarray(self.image).mean() * 0.7)
        gray = np.asarray(transform.apply_operation(self.image, 'grayscale'))
        np.testing.assert_array_equal(gray[..., 0], gray[..., 1])
        np.testing.assert_array_equal(gray[..., 0], gray[..., 2])
        # Independent per-channel gains on neutral pixels yield color casts.
        np.random.seed(123)
        neutral = Image.fromarray(np.full((256, 256, 3), 100, dtype=np.uint8))
        cast = np.asarray(transform.apply_operation(neutral, 'white_balance'))[0, 0]
        self.assertGreater(np.ptp(cast.astype(float)), 1)
        self.assertEqual(self.image.tobytes(), original)

    def test_motion_psf_is_directional_not_gaussian(self):
        horizontal, vertical = motion_kernel(13, 0), motion_kernel(13, 90)
        np.testing.assert_allclose(horizontal.sum(), 1, atol=1e-6)
        np.testing.assert_allclose(vertical.sum(), 1, atol=1e-6)
        self.assertGreater(horizontal[6].sum(), 0.99)
        self.assertGreater(vertical[:, 6].sum(), 0.99)
        np.testing.assert_allclose(horizontal.T, vertical, atol=1e-6)

    def test_augmix_seeded_sequence_and_validation_exact_no_rng(self):
        data = self.config['data']
        def sequence():
            set_seed(42, True, True)
            transform = ImageTransform(data, True)
            return [transform(self.image) for _ in range(8)]
        first, second = sequence(), sequence()
        for left, right in zip(first, second):
            self.assertEqual(left.shape, (3, 256, 256))
            self.assertTrue(torch.isfinite(left).all())
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        validation = ImageTransform(data)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        with patch.object(RobustAugMix, '__call__', side_effect=AssertionError('No AugMix on val')):
            actual = validation(self.image)
        expected = torch.from_numpy(np.array(self.image, dtype=np.float32)/255).permute(2, 0, 1)
        torch.testing.assert_close(actual, (expected-validation.mean)/validation.std, rtol=0, atol=0)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        self.assertIsNone(validation.robust_augmix)
        with self.assertRaises(ValueError):
            ImageTransform({**data, 'augmentation': {**data['augmentation'], 'color_jitter': {'p': 0}}})

    def test_mixstyle_train_changes_stats_preserves_spatial_pattern_and_grad(self):
        set_seed(42, True, True)
        features = torch.randn(4, 3, 8, 8)
        features = (features * torch.tensor([1, 2, 3, 4]).view(4, 1, 1, 1)
                    + torch.tensor([0, 10, 20, 30]).view(4, 1, 1, 1)).requires_grad_()
        layer = MixStyle(p=1, alpha=0.1).train()
        with patch('archive.src.models.mixstyle.torch.randperm', return_value=torch.tensor([3, 2, 1, 0])):
            output = layer(features)
        self.assertEqual(output.shape, features.shape)
        self.assertFalse(torch.equal(output, features))
        def content(value):
            return (value-value.mean((2, 3), keepdim=True))/value.std((2, 3), keepdim=True)
        torch.testing.assert_close(content(output), content(features), rtol=1e-5, atol=1e-5)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertEqual(list(layer.parameters()), [])
        self.assertEqual(dict(layer.state_dict()), {})
        layer.eval()
        state, python_state = torch.get_rng_state(), random.getstate()
        self.assertIs(layer(features), features)
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        self.assertEqual(random.getstate(), python_state)
        singleton = features[:1]
        self.assertIs(MixStyle(p=1).train()(singleton), singleton)

    def test_mixstyle_seed_reproducible_cpu(self):
        features = torch.arange(4*3*4*4).float().reshape(4, 3, 4, 4)
        layer = MixStyle(p=1).train()
        set_seed(42, True, True)
        first = layer(features)
        set_seed(42, True, True)
        torch.testing.assert_close(first, layer(features), rtol=0, atol=0)

    def test_preview_preserves_rng(self):
        import matplotlib.pyplot as plt
        python_state, numpy_state = random.getstate(), np.random.get_state()
        figure = preview_robust_recipe(self.image, self.config['data'], count=11, seed=42)
        self.assertEqual(len(figure.axes), 12)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        plt.close(figure)

    def test_mixstyle_stays_active_outside_frozen_blocks_and_eval_is_legacy(self):
        model_config = copy.deepcopy(self.config['model'])
        model_config['lgffe'] = {'gru_hidden_size': 8, 'head_hidden_size': 16}
        with patch('archive.src.models.lgffe.load_backbone', side_effect=lambda *args: (TinySpatialResNet(), 'IMAGENET1K_V2')):
            styled = build_model(model_config, 256, 20, initialize_pretrained=False)
            original_config = copy.deepcopy(model_config)
            original_config.pop('mixstyle')
            original = build_model(original_config, 256, 20, initialize_pretrained=False)
        original.load_state_dict(styled.state_dict(), strict=True)
        images = torch.randn(4, 3, 256, 256)
        styled.set_epoch(1)
        styled.train()
        self.assertFalse(styled.backbone.layer1.training)
        self.assertFalse(styled.backbone.layer2.training)
        for module in styled.mixstyle.values():
            self.assertTrue(module.training)
        with patch.object(styled.mixstyle['layer1'], 'forward', wraps=styled.mixstyle['layer1'].forward) as one, \
             patch.object(styled.mixstyle['layer2'], 'forward', wraps=styled.mixstyle['layer2'].forward) as two:
            styled(images)
            one.assert_called_once()
            two.assert_called_once()
        styled.eval()
        original.eval()
        with torch.inference_mode():
            torch.testing.assert_close(styled(images), original(images), rtol=0, atol=0)
        self.assertTrue(all(not module.training for module in styled.mixstyle.values()))
        groups = make_optimizer(styled, self.config['training']['optimizer']).param_groups
        self.assertEqual([group['name'] for group in groups], ['backbone', 'head'])

    def test_real_lgffe_mixstyle_forward_backward_frozen_bn(self):
        report = check_lgffe(self.config, 'cpu', batch_size=4)
        self.assertEqual(report['shapes']['logits'], [4, 20])
        self.assertTrue(report['bn_buffers_unchanged'])
        self.assertTrue(report['optimizer_groups_stable'])

    def test_full_recipe_synthetic_training_smoke_and_checkpoint(self):
        config = copy.deepcopy(self.config)
        config['training'].update(epochs=4, device='cpu')
        config['model']['lgffe'] = {'gru_hidden_size': 8, 'head_hidden_size': 16}
        transform = ImageTransform(config['data'], training=True)
        train_images = torch.stack([transform(self.image) for _ in range(4)])
        val_transform = ImageTransform(config['data'])
        val_images = torch.stack([val_transform(self.image) for _ in range(4)])
        def loader(images):
            dataset = TensorDataset(images, torch.tensor([0, 1, 0, 1]))
            dataset.data = pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'label': [3, 9, 3, 9]})
            dataset.label_to_index = {3: 0, 9: 1}
            return DataLoader(dataset, batch_size=4)
        train, val = loader(train_images), loader(val_images)
        classes = pd.DataFrame({'label': [3, 9], 'class_name': ['church', 'palace']})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('archive.src.models.lgffe.load_backbone', side_effect=lambda *args: (TinySpatialResNet(), 'IMAGENET1K_V2')), \
                 patch('archive.src.experiment.load_data', return_value=(train, val, classes)), \
                 patch('archive.src.experiment.load_test', side_effect=AssertionError('No test reading')):
                result = run_experiment(config, 'robust_smoke', project_root=root, show_plots=False, predict_test=False)
                restored, _ = load_checkpoint(result.paths['models']/'best.pt', 'cpu')
            self.assertEqual(result.history.training_mode.tolist(), ['head_only']*3+['partial'])
            self.assertTrue((result.paths['logs']/'summary.json').is_file())
            self.assertTrue((result.paths['models']/'last.pt').is_file())
            expected = load_probabilities(result.paths['predictions']/'val_probabilities.npz')
            actual = probability_predictions(restored, val, 'cpu', [3, 9])
            np.testing.assert_allclose(actual.probabilities, expected.probabilities, rtol=0, atol=0)
            with (root/'experiments.csv').open(encoding='utf-8', newline='') as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['submission_id'], '')


if __name__ == '__main__':
    unittest.main()
