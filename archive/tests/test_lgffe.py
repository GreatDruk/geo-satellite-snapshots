"""LGFFE geometry, train policies, D4 and shared checkpoint/artifact pipeline."""
import copy
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

from archive.src.experiment import load_checkpoint, run_experiment
from archive.src.models import build_model
from archive.src.models.lgffe import LGFFEHead
from archive.src.models.lgffe_checks import check_lgffe
from archive.src.training.predict import load_probabilities, probability_predictions
from archive.src.transforms import ImageTransform, apply_d4

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/resnet50_lgffe_domain_shift_stronger.json'


class TinySpatialResNet(nn.Module):
    """Only for pipeline integration; real ResNet50 is checked separately."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, stride=32, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.maxpool = nn.Identity()
        for index in range(1, 5):
            setattr(self, f'layer{index}', nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4), nn.ReLU()))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 1000)


class LGFFETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_matches_stronger_except_requested_d4_and_extractor(self):
        base = json.loads((ROOT / 'configs/resnet50_partial_domain_shift_stronger.json').read_text())
        new = json.loads(CONFIG.read_text())
        self.assertEqual(new['training']['optimizer'], base['training']['optimizer'])
        self.assertEqual(new['training']['scheduler'], base['training']['scheduler'])
        self.assertEqual(new['training']['epochs'], 25)
        self.assertEqual(new['training']['loss'], {'name': 'cross_entropy', 'label_smoothing': 0.0})
        before = copy.deepcopy(base['data'])
        before['augmentation'].pop('horizontal_flip')
        before['augmentation'].pop('vertical_flip')
        before['augmentation']['random_d4'] = True
        self.assertEqual(new['data'], before)
        for key in ('pretrained', 'weights', 'mode', 'head_dropout', 'freeze_bn', 'warmup_epochs', 'unfreeze_blocks'):
            self.assertEqual(new['model'][key], base['model'][key])
        self.assertNotIn('random_resized_crop', new['data']['augmentation'])

    def test_exact_d4_pixel_permutations_and_eval_rng(self):
        array = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        image = Image.fromarray(array)
        original = image.tobytes()
        outputs = []
        settings = {'image_size': 8, 'normalization': {'mean': [0]*3, 'std': [1]*3},
                    'augmentation': {'random_d4': True}}
        transform = ImageTransform(settings, True)
        for index in range(8):
            expected = np.rot90(array, index % 4)
            if index // 4:
                expected = np.flip(expected, axis=1)
            np.testing.assert_array_equal(np.asarray(apply_d4(image, index)), expected)
            with patch('archive.src.transforms.random.randrange', return_value=index) as sampler:
                output = transform(image)
                sampler.assert_called_once_with(8)
            torch.testing.assert_close(output, torch.from_numpy(expected.copy()).permute(2, 0, 1).float()/255)
            outputs.append(output.numpy().tobytes())
        self.assertEqual(len(set(outputs)), 8)
        self.assertEqual(image.tobytes(), original)
        with patch('archive.src.transforms.random.randrange', side_effect=AssertionError('eval geometry must be deterministic')):
            clean = ImageTransform(settings)(image)
        torch.testing.assert_close(clean, torch.from_numpy(array.copy()).permute(2, 0, 1).float()/255)
        with self.assertRaises(ValueError):
            ImageTransform({**settings, 'augmentation': {'random_d4': True, 'horizontal_flip': 0.5}})

    def test_local_pool_uses_original_regions_not_gru_states(self):
        head = LGFFEHead(4, (2, 2), 3, gru_hidden_size=6, head_hidden_size=8, dropout=0)
        features = torch.randn(2, 4, 2, 2)
        details = head.forward_details(features)
        torch.testing.assert_close(details['local_features'],
                                   torch.bmm(details['attention'].unsqueeze(1), details['regions']).squeeze(1))
        torch.testing.assert_close(details['global_features'], features.mean((2, 3)))
        torch.testing.assert_close(details['attention'].sum(1), torch.ones(2))
        self.assertEqual(details['contextual_regions'].shape, (2, 4, 6))
        with self.assertRaises(ValueError):
            head(torch.randn(2, 4, 3, 3))

    def test_real_resnet50_synthetic_forward_backward_and_policies(self):
        report = check_lgffe(CONFIG, 'cpu')
        self.assertEqual(report['shapes']['feature_map'], [2, 2048, 8, 8])
        self.assertEqual(report['shapes']['regions'], [2, 64, 2048])
        self.assertEqual(report['shapes']['contextual_regions'], [2, 64, 512])
        self.assertEqual(report['shapes']['fused_features'], [2, 4096])
        self.assertEqual(report['shapes']['logits'], [2, 20])
        self.assertTrue(report['bn_buffers_unchanged'])

    def test_artifacts_journal_checkpoint_and_inference_without_test_access(self):
        config = json.loads(CONFIG.read_text())
        config['training'].update(epochs=4, device='cpu')
        config['model']['lgffe'] = {'gru_hidden_size': 8, 'head_hidden_size': 16}
        def synthetic_loader():
            dataset = TensorDataset(torch.randn(2, 3, 256, 256), torch.tensor([0, 1]))
            dataset.data = pd.DataFrame({'id': ['first', 'second'], 'label': [3, 9]})
            dataset.label_to_index = {3: 0, 9: 1}
            return DataLoader(dataset, batch_size=2)
        loader, validation = synthetic_loader(), synthetic_loader()
        classes = pd.DataFrame({'label': [3, 9], 'class_name': ['church', 'palace']})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def backbone(*args):
                return TinySpatialResNet(), 'IMAGENET1K_V2'
            with patch('archive.src.models.lgffe.load_backbone', side_effect=backbone) as factory, \
                 patch('archive.src.experiment.load_data', return_value=(loader, validation, classes)), \
                 patch('archive.src.experiment.load_test', side_effect=AssertionError('No test access')):
                result = run_experiment(config, 'lgffe_smoke', project_root=root, show_plots=False, predict_test=False)
                self.assertTrue(factory.call_args.args[2])
                restored, checkpoint = load_checkpoint(result.paths['models'] / 'best.pt', 'cpu')
                self.assertFalse(factory.call_args.args[2])
            self.assertEqual(result.history.training_mode.tolist(), ['head_only']*3 + ['partial'])
            self.assertTrue((result.paths['models'] / 'last.pt').is_file())
            self.assertTrue((result.paths['logs'] / 'learning_curves.png').is_file())
            expected = load_probabilities(result.paths['predictions'] / 'val_probabilities.npz')
            self.assertEqual(expected.class_labels.tolist(), [3, 9])
            # Prediction loaders return IDs; verify ordinary/TTA inference on restored logits.
            actual = probability_predictions(restored, validation, torch.device('cpu'), [3, 9])
            np.testing.assert_allclose(expected.probabilities, actual.probabilities, rtol=1e-6, atol=1e-7)
            d4 = probability_predictions(restored, validation, torch.device('cpu'), [3, 9], tta='d4')
            np.testing.assert_allclose(d4.probabilities.sum(1), 1, rtol=1e-6)
            with (root / 'experiments.csv').open(newline='', encoding='utf-8') as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['model'], 'resnet50_lgffe')
            self.assertEqual(records[0]['submission_id'], '')


if __name__ == '__main__':
    unittest.main()
