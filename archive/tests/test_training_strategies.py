"""Synthetic CPU checks of augmentation targets, margins and full artifacts."""
import contextlib
import copy
import io
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from archive.src.data import ImageDataset
from archive.src.experiment import PROJECT_ROOT, load_checkpoint, load_config, make_optimizer, run_experiment
from archive.src.models import build_model
from archive.src.models.initialization import initialize_from_checkpoint
from archive.src.strategy_analysis import compare_resnet50_runs
from archive.src.training.losses import CEWithCosFace, make_training_loss
from archive.src.training.mining import HardExampleSampler
from archive.src.training.mixing import CutMix
from archive.src.transforms import ImageTransform
from archive.tests.test_transfer import TinyResNet


class TrainingStrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_crop_area_shape_seed_source_and_eval_invariance(self):
        image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (128, 128, 3), dtype=np.uint8))
        original = image.tobytes()
        cfg = {"image_size": [64, 80], "augmentation": {"random_resized_crop": {"p": 1, "scale": [0.65, 1], "ratio": [0.9, 1.1]}}}
        transform = ImageTransform(cfg, True)
        def sequence():
            random.seed(42)
            return [transform(image) for _ in range(5)]
        a, b = sequence(), sequence()
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertEqual(left.shape, (3, 64, 80))
        self.assertFalse(torch.equal(a[0], a[1]))
        self.assertEqual(image.tobytes(), original)
        # Feasible square crops must obey sampled area, allowing pixel rounding.
        square = Image.new("RGB", (256, 256))
        for _ in range(30):
            width, height = transform._crop(square).size
            self.assertGreaterEqual(width * height / 256**2, 0.645)
            self.assertLessEqual(width * height / 256**2, 1.005)
            self.assertTrue(0.89 <= width / height <= 1.11)
        state = random.getstate()
        plain = ImageTransform({"image_size": [64, 80]})(image)
        torch.testing.assert_close(ImageTransform(cfg)(image), plain, rtol=0, atol=0)
        self.assertEqual(state, random.getstate())
        for bad in ({"scale": [0, 1]}, {"ratio": [1.2, 0.9]}, {"p": 1.2}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                ImageTransform({"image_size": 64, "augmentation": {"random_resized_crop": bad}})

    def test_cutmix_actual_clipped_area_targets_and_immutability(self):
        images = torch.stack([torch.zeros(3, 12, 20), torch.ones(3, 12, 20)])
        original = images.clone()
        labels = torch.tensor([0, 1])
        cutmix = CutMix({"p": 1, "alpha": 1, "max_area": 0.35})
        with patch("numpy.random.beta", return_value=0.1), patch("numpy.random.randint", return_value=0), \
             patch("torch.randperm", return_value=torch.tensor([1, 0])):
            mixed, first, second, retained = cutmix(images, labels)
        changed = (mixed[0, 0] != images[0, 0]).sum().item()
        self.assertGreater(changed, 0)
        self.assertLessEqual(1-retained, 0.35)
        self.assertAlmostEqual(1-retained, changed / (12*20))
        torch.testing.assert_close(first, labels)
        torch.testing.assert_close(second, labels.flip(0))
        torch.testing.assert_close(images, original, rtol=0, atol=0)
        identity = CutMix({"p": 0})(images, labels)
        self.assertIs(identity[0], images)
        self.assertEqual(identity[3], 1)
        for bad in ({"alpha": 0}, {"max_area": 1.1}, {"p": -0.1}, {"name": "mixup"}):
            with self.assertRaises(ValueError):
                CutMix(bad)

    def test_cosface_formula_gradient_and_inference_no_labels(self):
        criterion = make_training_loss({"name": "ce_cosface", "margin": 0.15, "scale": 16, "weight": 0.2})
        logits = torch.tensor([[1.0, 2.0], [3.0, -1.0]], requires_grad=True)
        scores = torch.tensor([[0.1, 0.3], [0.5, -0.1]], requires_grad=True)
        labels = torch.tensor([0, 1])
        shifted = scores.clone()
        shifted[torch.arange(2), labels] -= 0.15
        expected = nn.functional.cross_entropy(logits, labels) + 0.2 * nn.functional.cross_entropy(16*shifted, labels)
        actual = criterion(logits, scores, labels)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(scores.grad.abs().sum().item(), 0)
        config = {"name": "resnet50", "pretrained": False, "mode": "partial", "unfreeze_blocks": ["layer4"],
                  "warmup_epochs": 1, "freeze_bn": True, "auxiliary_cosine_head": True}
        with patch("archive.src.models.pretrained.load_backbone", return_value=(TinyResNet(), "DEFAULT")):
            model = build_model(config, 8, 2)
        model.eval()
        x = torch.randn(3, 3, 8, 8)
        main, cosine = model.forward_with_auxiliary(x)
        torch.testing.assert_close(main, model(x))
        self.assertTrue((cosine.abs() <= 1.000001).all())
        optimizer = make_optimizer(model, {"name": "adamw", "lr": 0.001})
        self.assertIn(id(model.auxiliary_head.weight), [id(p) for g in optimizer.param_groups for p in g["params"]])
        model.set_epoch(2)
        model.train()
        main, cosine = model.forward_with_auxiliary(x)
        criterion(main, cosine, torch.tensor([0, 1, 0])).backward()
        self.assertIsNotNone(model.auxiliary_head.weight.grad)
        self.assertIsNotNone(model.backbone.layer4[0].weight.grad)
        self.assertIsNone(model.backbone.layer1[0].weight.grad)

    def test_mining_keeps_all_samples_repeats_only_previous_hard_train(self):
        dataset = type("Data", (), {"__len__": lambda self: 20, "data": pd.DataFrame({"id": [str(i) for i in range(20)], "label": [0]*20})})()
        sampler = HardExampleSampler(dataset, {"warmup_epochs": 1, "repeat_fraction": 0.1}, 42)
        self.assertEqual(set(iter(sampler)), set(range(20)))
        logits = torch.stack([torch.arange(20).float(), torch.zeros(20)], dim=1)
        sampler.record(torch.arange(20), logits, torch.zeros(20, dtype=torch.long))
        report = sampler.finish_epoch()
        self.assertEqual(report.loc[report.repeat_next_epoch, "id"].tolist(), ["0", "1"])
        sampler.set_epoch(2)
        indices = list(iter(sampler))
        self.assertEqual(indices, list(iter(sampler)))
        self.assertEqual(len(indices), 22)
        self.assertEqual(sorted(set(indices)), list(range(20)))
        self.assertEqual(indices.count(0), 2)
        self.assertEqual(indices.count(1), 2)
        with self.assertRaises(ValueError):
            sampler.finish_epoch()  # no observations yet

    def test_real_resnet50_auxiliary_forward_matches_main_without_downloads(self):
        config = {"name": "resnet50", "pretrained": False, "mode": "partial", "unfreeze_blocks": ["layer4"],
                  "freeze_bn": True, "auxiliary_cosine_head": True}
        model = build_model(config, 32, 20).eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            main, cosine = model.forward_with_auxiliary(x)
            torch.testing.assert_close(main, model(x))
        self.assertEqual(cosine.shape, (2, 20))
        model.train()
        main, cosine = model.forward_with_auxiliary(x)
        CEWithCosFace()(main, cosine, torch.tensor([0, 1])).backward()
        self.assertIsNotNone(model.backbone.layer4[0].conv1.weight.grad)
        self.assertIsNone(model.backbone.layer1[0].conv1.weight.grad)

    def test_new_configs_have_matched_controls_and_no_source_checkpoint(self):
        base = load_config("configs/resnet50_partial_domain_shift_stronger.json")
        rrc = load_config("configs/resnet50_partial_domain_shift_rrc.json")
        stripped = copy.deepcopy(rrc)
        stripped['data']['augmentation'].pop('random_resized_crop')
        stripped['training'].pop('loss')
        for key in ['data', 'training', 'model', 'seed']:
            self.assertEqual(stripped[key], base[key])
        for name in ['rrc_smoothing', 'rrc_cutmix', 'domain_shift_cosface', 'domain_shift_hard_mining']:
            config = load_config('configs/resnet50_partial_' + name + '.json')
            self.assertEqual(config['model']['name'], 'resnet50')
            self.assertNotIn('initialize_from', config['training'])
            self.assertEqual(config['training']['epochs'], 25)
            self.assertEqual(config['seed'], 42)
            ImageTransform(config['data'], True)
            make_training_loss(config['training'].get('loss'))


class StrategyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        raw = self.root / 'data/raw'
        raw.mkdir(parents=True)
        pd.DataFrame({'label': [4, 11], 'class_name': ['a', 'b']}).to_csv(raw/'classes.csv', index=False)
        for split, count in [('train', 8), ('val', 4)]:
            (raw/split).mkdir()
            rows = []
            for i in range(count):
                image_id = f'{i}.png'
                Image.fromarray(np.random.default_rng(i).integers(0, 256, (10, 10, 3), dtype=np.uint8)).save(raw/split/image_id)
                rows.append({'id': image_id, 'label': [4, 11][i % 2]})
            pd.DataFrame(rows).to_csv(raw/f'{split}.csv', index=False)
        self.config = load_config('configs/resnet50_partial_domain_shift_rrc.json')
        self.config['data'].update(image_size=8, num_workers=0, batch_size=3)
        self.config['training'].update(epochs=3, device='cpu')
        self.config['training']['scheduler']['T_max'] = 3
        self.config['model']['warmup_epochs'] = 1

    def tearDown(self):
        self.temp.cleanup()

    def run_config(self, cfg, run_id):
        with patch('archive.src.models.pretrained.load_backbone', return_value=(TinyResNet(), 'IMAGENET1K_V2')), \
             contextlib.redirect_stdout(io.StringIO()):
            return run_experiment(cfg, run_id, project_root=self.root, show_plots=False)

    def test_smoothing_cutmix_mining_and_margin_artifacts_restore_comparison(self):
        for name in ['baseline', 'smoothing', 'cutmix', 'mining', 'cosface']:
            cfg = copy.deepcopy(self.config)
            if name == 'smoothing':
                cfg['training']['loss']['label_smoothing'] = 0.05
            elif name == 'cutmix':
                cfg['training']['batch_augmentation'] = {'name': 'cutmix', 'p': 1, 'alpha': 1, 'max_area': 0.35}
            elif name == 'mining':
                cfg['training']['hard_mining'] = {'warmup_epochs': 1, 'repeat_fraction': 0.2}
            elif name == 'cosface':
                cfg['model']['auxiliary_cosine_head'] = True
                cfg['training']['loss'] = {'name': 'ce_cosface', 'margin': 0.15, 'weight': 0.2, 'scale': 16}
            result = self.run_config(cfg, name)
            self.assertEqual(len(result.history), 3)
            saved = torch.load(result.paths['models']/'best.pt', map_location='cpu', weights_only=True)
            from archive.src.data import load_validation
            from archive.src.training.evaluate import evaluate_model
            val, _ = load_validation(self.root/'data/raw', cfg['data'])
            val_loss, val_accuracy = evaluate_model(result.model, val, nn.CrossEntropyLoss(), torch.device('cpu'))
            self.assertAlmostEqual(val_loss, result.history.loc[result.history.epoch==result.best_epoch, 'val_loss'].iloc[0], places=6)
            with patch('archive.src.models.pretrained.load_backbone', return_value=(TinyResNet(), 'IMAGENET1K_V2')):
                restored, _ = load_checkpoint(result.paths['models']/'best.pt', 'cpu')
            x = torch.randn(2, 3, 8, 8)
            torch.testing.assert_close(restored(x), result.model(x))
            self.assertTrue((result.paths['predictions']/'val_probabilities.npz').exists())
            if name == 'mining':
                self.assertEqual(result.history.hard_repeats.tolist(), [0, 1, 1])
                self.assertTrue((result.paths['logs']/'hard_examples_epoch_003.csv').exists())
            meta = json.loads((result.paths['logs']/'metadata.json').read_text())
            self.assertEqual(meta['validation_loss'], 'unsmoothed_cross_entropy_on_main_logits')
        journal = pd.read_csv(self.root/'experiments.csv')
        self.assertEqual(len(journal), 5)
        analysis = compare_resnet50_runs(['smoothing', 'cutmix', 'cosface'], project_root=self.root, baseline='baseline')
        self.assertEqual(len(analysis.comparison), 4)
        self.assertTrue((analysis.comparison.correct + analysis.comparison.errors == 4).all())
        self.assertTrue((analysis.comparison.fixes - analysis.comparison.breaks ==
                         analysis.comparison.correct - analysis.comparison.correct.iloc[0]).all())
        with self.assertRaises(ValueError):
            compare_resnet50_runs(['../bad'], project_root=self.root, baseline='baseline')

    def test_invalid_mixed_objectives_rejected_before_artifacts(self):
        cfg = copy.deepcopy(self.config)
        cfg['training']['batch_augmentation'] = {'p': 0.25}
        cfg['training']['hard_mining'] = {'warmup_epochs': 1}
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            self.run_config(cfg, 'invalid')
        self.assertFalse((self.root/'artifacts').exists())

    def test_new_auxiliary_head_checkpoint_initialization_keeps_main_logits(self):
        plain = self.run_config(self.config, 'source')
        cfg = copy.deepcopy(self.config)
        cfg['model']['auxiliary_cosine_head'] = True
        with patch('archive.src.models.pretrained.load_backbone', return_value=(TinyResNet(), 'DEFAULT')):
            new = build_model(cfg['model'], 8, 2, initialize_pretrained=False)
        meta = initialize_from_checkpoint(new, plain.paths['models']/'best.pt', [4, 11], cfg['data'])
        new.eval()
        self.assertTrue(meta['auxiliary_head_initialized'])
        x = torch.randn(2, 3, 8, 8)
        torch.testing.assert_close(new(x), plain.model(x))


if __name__ == '__main__':
    unittest.main()
