"""Synthetic checks for split isolation, fixed refit, Geometry-4 and group blending."""
import contextlib
import copy
import io
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

from final.src.ensemble import blend
from final.src.inference import export_predictions, geometry4
from final.src.training import CONFIG_DIR, load_config, train_models, validate_config


class TinyModel(nn.Module):
    def __init__(self, classes=5):
        super().__init__()
        self.fc = nn.Linear(3, classes)
        self.epochs = []
        self.active_mode = 'head_only'

    def set_epoch(self, epoch):
        self.epochs.append(epoch)
        self.active_mode = 'head_only' if epoch <= 3 else 'partial'

    def optimizer_groups(self, lr, backbone_lr, head_lr):
        return [{'params': list(self.parameters()), 'lr': head_lr, 'name': 'head'}]

    def forward(self, x):
        return self.fc(x.mean((-2, -1)))


def fixture(root):
    raw = root / 'data/raw'
    for split, count in [('train', 4), ('val', 2)]:
        folder = raw / split
        folder.mkdir(parents=True)
        rows = []
        for index in range(count):
            # Identical IDs across directories ensure no accidental split collapse.
            image_id = f'{index}.png'
            Image.fromarray(np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(folder / image_id)
            rows.append({'id': image_id, 'label': index % 5})
        pd.DataFrame(rows).to_csv(raw / f'{split}.csv', index=False)
    pd.DataFrame({'label': range(5), 'class_name': ['church', 'palace', 'commercial_area', 'railway', 'railway_station']}).to_csv(raw / 'classes.csv', index=False)
    return raw


class FinalPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_both_configs_are_fixed_25_and_reject_evaluation(self):
        for name in ('agmfa', 'resnet50'):
            config = load_config(name)
            validate_config(config)
            config['evaluation'] = {'labels_csv': 'anything.csv'}
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_25_epoch_training_reads_only_train_val_and_uses_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = fixture(root)
            model = TinyModel()
            config = load_config('resnet50')
            config['fit_counts'] = {'train': 4, 'val': 2}
            calls = []
            original_read_csv = pd.read_csv

            def guarded_csv(path, *args, **kwargs):
                calls.append(Path(path).name)
                return original_read_csv(path, *args, **kwargs)

            with patch('final.src.training.load_config', return_value=config), \
                 patch('final.src.training.build_model', return_value=model), \
                 patch('pandas.read_csv', side_effect=guarded_csv), \
                 contextlib.redirect_stdout(io.StringIO()):
                paths = train_models(raw, 'synthetic', project_root=root, names=('resnet50',), batch_size=6, device='cpu')
            self.assertEqual(set(calls), {'train.csv', 'val.csv', 'classes.csv'})
            self.assertEqual(model.epochs, list(range(1, 26)))
            saved = torch.load(paths['resnet50'], weights_only=True)
            self.assertEqual(saved['epoch'], 25)
            self.assertEqual(saved['training_splits'], ['train', 'val'])
            self.assertEqual(saved['checkpoint_selection'], 'fixed_last_epoch')
            self.assertFalse((paths['resnet50'].parent / 'best.pt').exists())
            history = pd.read_csv(root / 'artifacts/logs/synthetic_resnet50/history.csv')
            self.assertTrue((history.source_samples == 4).all())
            self.assertTrue((history.target_samples == 2).all())
            self.assertTrue((history.forward_images == 10).all())
            self.assertFalse(any(c.startswith(('eval_', 'val_')) for c in history.columns))
            journal = pd.read_csv(root / 'experiments.csv')
            self.assertTrue(journal.val_accuracy.isna().all())
            with self.assertRaises(FileExistsError):
                train_models(raw, 'synthetic', project_root=root, names=('resnet50',), device='cpu')

    def test_geometry4_averages_probabilities_for_four_rotations(self):
        class Orientation(nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def forward(self, x):
                self.inputs.append(x.clone())
                v = x[:, 0, 0, 0]
                return torch.stack([v, -v], 1)

        model = Orientation()
        x = torch.arange(4, dtype=torch.float32).view(1, 1, 2, 2)
        actual = geometry4(model, x)
        expected = torch.stack([torch.tensor([[v, -v]], dtype=torch.float32).softmax(1).double()
                                for v in [0, 1, 3, 2]]).mean(0)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(model.inputs), 4)
        self.assertEqual(len({tuple(v.flatten().tolist()) for v in model.inputs}), 4)

    def test_group_blend_preserves_group_mass_and_other_columns(self):
        config = json.loads((CONFIG_DIR / 'ensemble.json').read_text())
        names = ['railway_station', 'palace', 'other', 'commercial_area', 'church', 'railway']
        a = np.array([[.01, .55, .05, .02, .36, .01], [.12, .02, .05, .02, .01, .78], [.1, .12, .6, .08, .05, .05]])
        r = np.array([[.01, .20, .05, .02, .71, .01], [.65, .02, .05, .02, .01, .25], [.1, .12, .6, .08, .05, .05]])
        actual = blend(a, r, names, config)
        base = (a + r) / 2
        for row, columns, weight in [(0, [4, 1, 3], .75), (1, [5, 0], .25)]:
            mixed = weight * a[row, columns] + (1 - weight) * r[row, columns]
            np.testing.assert_allclose(actual[row, columns], base[row, columns].sum() * mixed / mixed.sum())
            other = [i for i in range(6) if i not in columns]
            np.testing.assert_array_equal(actual[row, other], base[row, other])
        np.testing.assert_array_equal(actual[2], base[2])
        np.testing.assert_allclose(actual.sum(1), 1)

    def test_export_uses_template_ids_only_and_preserves_order_without_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / 'sample.csv'
            pd.DataFrame({'id': ['b.png', 'a.png'], 'label': ['ignored', 'ignored']}).to_csv(sample, index=False)
            p = np.array([[.6, .2, .1, .05, .05], [.05, .05, .1, .4, .4]])
            names = ['church', 'palace', 'commercial_area', 'railway', 'railway_station']
            with patch('final.src.inference.predict_models', return_value=({'agmfa': p, 'resnet50': p}, np.arange(5), names)) as predict:
                result = export_predictions({}, sample, root / 'images', root / 'out', device='cpu')
            np.testing.assert_array_equal(predict.call_args.args[1], ['b.png', 'a.png'])
            for name, path in result.items():
                frame = pd.read_csv(path)
                self.assertEqual(list(frame.columns), ['id', 'label'])
                self.assertEqual(frame.id.tolist(), ['b.png', 'a.png'])
                with np.load(root / 'out' / f'{name}_probabilities.npz') as data:
                    self.assertEqual(set(data.files), {'ids', 'class_labels', 'probabilities'})
            with self.assertRaises(FileExistsError):
                export_predictions({}, sample, root / 'images', root / 'out')


if __name__ == '__main__':
    unittest.main()
