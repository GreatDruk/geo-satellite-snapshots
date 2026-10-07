"""AGMFA method, actual ResNet geometry and shared artifact compatibility."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.experiment import load_checkpoint, run_experiment
from src.models.agmfa import AGMFAHead
from src.models import build_model
from src.models.agmfa_checks import check_agmfa
from src.training.predict import load_probabilities, probability_predictions
from src.transforms import get_transforms

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/resnet50_agmfa_robust_augmix.json'


class TinyBottleneck(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.stride = stride
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(incoming, outgoing, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.conv3 = nn.Conv2d(outgoing, outgoing, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outgoing)
        self.downsample = nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False),
                                        nn.BatchNorm2d(outgoing))

    def forward(self, x):
        residual = self.downsample(x)
        x = self.bn1(self.conv1(x)).relu()
        x = self.bn2(self.conv2(x)).relu()
        return (self.bn3(self.conv3(x)) + residual).relu()


class TinyResNet(nn.Module):
    """Small residual model only for I/O integration; geometry checked on real ResNet."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer1 = nn.Sequential(TinyBottleneck(4, 4))
        self.layer2 = nn.Sequential(TinyBottleneck(4, 8, 2))
        self.layer3 = nn.Sequential(TinyBottleneck(8, 16, 2))
        self.layer4 = nn.Sequential(TinyBottleneck(16, 32, 2), TinyBottleneck(32, 32), TinyBottleneck(32, 32))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 1000)


class AGMFATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_resnet_geometry_and_training_policies(self):
        report = check_agmfa(CONFIG)
        self.assertEqual(report['shapes']['res4_3'], [2, 512, 16, 16])
        self.assertEqual(report['shapes']['res4'], [2, 2048, 16, 16])
        self.assertEqual(report['shapes']['channel_mask_3'], [2, 512, 1, 1])
        self.assertEqual(report['bottleneck_max_abs_errors'], [0, 0, 0])
        self.assertEqual(report['shapes']['fused_map'], [2, 512, 16, 16])
        self.assertTrue(report['backbone_bn_buffers_unchanged'])

    def test_config_and_deterministic_validation(self):
        config = json.loads(CONFIG.read_text())
        old = json.loads((ROOT / 'configs/resnet50_lgffe_robust_augmix_mixstyle.json').read_text())
        original = json.loads((ROOT / 'configs/archive/resnet50_agmfa_robust_augmix_original.json').read_text())
        self.assertEqual(config['data'], old['data'])
        self.assertEqual(config['training'], old['training'])
        self.assertEqual(config['data'], original['data'])
        self.assertEqual(config['training'], original['training'])
        self.assertEqual(config['seed'], original['seed'])
        expected_model = original['model']
        expected_model['agmfa']['attention_source'] = 'conv2'
        self.assertEqual(config['model'], expected_model)
        self.assertEqual(config['exp_id'], 'resnet50_agmfa_robust_augmix_fixed_001')
        self.assertNotIn('mixstyle', config['model'])
        self.assertNotIn('random_resized_crop', config['data']['augmentation'])
        train, val = get_transforms(config['data'])
        self.assertIsNotNone(train.robust_augmix)
        self.assertIsNone(val.robust_augmix)
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (256, 256, 3), dtype=np.uint8))
        with patch('src.transforms.random.randrange', side_effect=AssertionError('val randomness')):
            torch.testing.assert_close(val(image), val(image), rtol=0, atol=0)

    def test_multilayer_mask_product_and_original_features(self):
        head = AGMFAHead((4, 8, 16), 20, fusion_channels=32, norm_groups=4, dropout=0, attention_channels=4)
        details = head.forward_details(torch.randn(2, 4, 8, 8), torch.randn(2, 8, 4, 4),
                                       [torch.randn(2, 4, 4, 4) for _ in range(3)], torch.randn(2, 16, 4, 4))
        torch.testing.assert_close(details['semantic_mask'],
                                   details['spatial_mask_1'] * details['spatial_mask_2'] * details['spatial_mask_3'])
        torch.testing.assert_close(details['concatenated'], torch.cat([
            details[f'aligned_res{i}'] * details['semantic_mask'] for i in (2, 3, 4)], dim=1))
        torch.testing.assert_close(details['aligned_res4'], details['res4'])
        with self.assertRaises(ValueError):
            AGMFAHead((4, 8, 16), 20, fusion_channels=15, norm_groups=4)

    def test_pipeline_checkpoint_probabilities_and_journal(self):
        config = json.loads(CONFIG.read_text())
        config['training'].update(epochs=4, device='cpu')
        config['model']['agmfa'].update(fusion_channels=32, norm_groups=4)
        def loader():
            ds = TensorDataset(torch.randn(2, 3, 256, 256), torch.tensor([0, 1]))
            ds.data = pd.DataFrame({'id': ['first', 'second'], 'label': [3, 9]})
            ds.label_to_index = {3: 0, 9: 1}
            return DataLoader(ds, batch_size=2)
        train, val = loader(), loader()
        classes = pd.DataFrame({'label': [3, 9], 'class_name': ['church', 'palace']})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('src.models.agmfa.load_backbone', side_effect=lambda *args: (TinyResNet(), 'IMAGENET1K_V2')) as factory, \
                 patch('src.experiment.load_data', return_value=(train, val, classes)), \
                 patch('src.experiment.load_test', side_effect=AssertionError('No test access')):
                result = run_experiment(config, 'agmfa_smoke', project_root=root, show_plots=False, predict_test=False)
                self.assertTrue(factory.call_args.args[2])
                restored, checkpoint = load_checkpoint(result.paths['models'] / 'best.pt', 'cpu')
                self.assertFalse(factory.call_args.args[2])
            self.assertEqual(result.history.training_mode.tolist(), ['head_only'] * 3 + ['partial'])
            for path in (result.paths['models'] / 'last.pt', result.paths['logs'] / 'summary.json',
                         result.paths['logs'] / 'history.csv', result.paths['logs'] / 'learning_curves.png',
                         result.paths['predictions'] / 'val.csv'):
                self.assertTrue(path.is_file(), str(path))
            expected = load_probabilities(result.paths['predictions'] / 'val_probabilities.npz')
            actual = probability_predictions(restored, val, torch.device('cpu'), [3, 9])
            np.testing.assert_allclose(expected.probabilities, actual.probabilities, rtol=1e-6, atol=1e-7)
            self.assertEqual(expected.class_labels.tolist(), [3, 9])
            with (root / 'experiments.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['model'], 'resnet50_agmfa')
            self.assertEqual(rows[0]['submission_id'], '')

    def test_historical_checkpoint_still_loads_without_changing_attention(self):
        original = json.loads((ROOT / 'configs/archive/resnet50_agmfa_robust_augmix_original.json').read_text())
        original['model']['agmfa'].update(fusion_channels=32, norm_groups=4)
        with tempfile.TemporaryDirectory() as directory, \
             patch('src.models.agmfa.load_backbone', side_effect=lambda *args: (TinyResNet(), 'IMAGENET1K_V2')):
            model = build_model(original['model'], 256, 2, initialize_pretrained=False).eval()
            images = torch.randn(2, 3, 256, 256)
            with torch.inference_mode():
                details = model.forward_details(images)
                torch.testing.assert_close(details['res4_3'], details['res4'], rtol=0, atol=0)
                res4 = model.backbone.layer4(details['res3'])
                torch.testing.assert_close(details['res4'], res4, rtol=0, atol=0)
                expected = model.backbone.fc(details['res2'], details['res3'],
                                             [details[f'res4_{i}'] for i in (1, 2, 3)], res4)
                torch.testing.assert_close(details['logits'], expected, rtol=0, atol=0)
            path = Path(directory) / 'historical.pt'
            torch.save({'config': original, 'class_labels': [3, 9], 'epoch': 25,
                        'model_state_dict': model.state_dict()}, path)
            restored, _ = load_checkpoint(path, 'cpu')
            self.assertEqual(restored.attention_source, 'block_output')
            with torch.inference_mode():
                torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
