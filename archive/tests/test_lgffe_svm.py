"""Frozen concat extraction, train-only SVM fitting and separate artifacts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from archive.src.lgffe_svm import (compare_predictions, extract_lgffe_features, fit_linear_svm,
                           run_lgffe_svm, svm_scores, _read_features)
from archive.src.models import build_model
from archive.src.inference import file_hash
from archive.tests.test_lgffe import TinySpatialResNet

ROOT = Path(__file__).resolve().parents[1]


class LGFFESVMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_twenty_class_hinge_svm_train_standardization_and_restore(self):
        random = np.random.default_rng(42)
        labels = np.repeat(np.arange(20)*3, 4)
        features = np.repeat(np.eye(20), 4, axis=0)*4 + random.normal(0, 0.02, (80, 20))
        svm = fit_linear_svm(features, labels, np.arange(20)*3, C=1.0, max_iter=100)
        scores = svm_scores(svm, features)
        np.testing.assert_array_equal(np.asarray(svm['class_labels'])[scores.argmax(1)], labels)
        np.testing.assert_allclose(svm['mean'].numpy(), features.mean(0))
        np.testing.assert_allclose(svm['scale'].numpy(), features.std(0))
        self.assertEqual(svm['state_dict']['weight'].shape, (20, 20))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'svm.pt'
            torch.save(svm, path)
            restored = torch.load(path, weights_only=True)
            np.testing.assert_array_equal(svm_scores(restored, features), scores)
        with self.assertRaises(ValueError):
            fit_linear_svm(features[:4], labels[:4], np.arange(20)*3)

    def test_error_comparison_and_noncontiguous_class_order(self):
        summary, per_class, changed, matrices = compare_predictions(
            ['a', 'b', 'c', 'd'], [9, 3, 9, 3], [3, 3, 9, 9], [9, 9, 9, 9], [9, 3], ['palace', 'church'])
        self.assertEqual((summary['fixed'], summary['new_errors'], summary['shared_errors']), (1, 1, 1))
        self.assertEqual(changed.id.tolist(), ['a', 'b'])
        self.assertEqual(matrices['svm'].loc['palace', 'palace'], 2)
        self.assertEqual(matrices['svm'].loc['church', 'palace'], 2)
        self.assertEqual(per_class.svm_errors.sum(), 2)

    def test_extract_freezes_exact_concat_cache_and_preserves_checkpoint(self):
        config = json.loads((ROOT/'configs/resnet50_lgffe_robust_augmix_mixstyle.json').read_text())
        config['model']['lgffe'] = {'gru_hidden_size': 8, 'head_hidden_size': 16}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/'data/raw'
            raw.mkdir(parents=True)
            pd.DataFrame({'label': [3, 9], 'class_name': ['church', 'palace']}).to_csv(raw/'classes.csv', index=False)
            for split in ['train', 'val']:
                (raw/split).mkdir()
                names = [f'{split}_a.png', f'{split}_b.png']
                pd.DataFrame({'id': names, 'label': [3, 9]}).to_csv(raw/f'{split}.csv', index=False)
                for index, name in enumerate(names):
                    Image.fromarray(np.full((256, 256, 3), 50+index*80, dtype=np.uint8)).save(raw/split/name)
            factory = lambda *args: (TinySpatialResNet(), 'IMAGENET1K_V2')
            with patch('archive.src.models.lgffe.load_backbone', side_effect=factory):
                model = build_model(config['model'], 256, 2, initialize_pretrained=False)
                source = root/'best.pt'
                torch.save({'model_state_dict': model.state_dict(), 'config': config, 'class_labels': [3, 9], 'epoch': 17}, source)
                source_hash = file_hash(source)
                # Fail if train transforms or random MixStyle can enter extraction.
                with patch('archive.src.robust_augmix.RobustAugMix.__call__', side_effect=AssertionError('No AugMix in extraction')):
                    cache = extract_lgffe_features(root, source, 'cache', device='cpu', batch_size=2)
            self.assertEqual(file_hash(source), source_hash)
            train = _read_features(cache/'train.npz')
            self.assertEqual(train['features'].shape, (2, 8))  # concat=4+4, not classifier hidden16
            model.eval()
            with torch.inference_mode():
                data = config['data']
                from archive.src.transforms import ImageTransform
                images = torch.stack([ImageTransform(data)(Image.open(raw/'train'/name)) for name in train['ids']])
                expected = model.forward_details(images)['fused_features'].numpy()
            np.testing.assert_allclose(train['features'], expected, rtol=1e-6, atol=1e-7)
            with patch('archive.src.lgffe_svm.load_checkpoint', side_effect=AssertionError('Cache should skip extraction')):
                self.assertEqual(extract_lgffe_features(root, source, 'cache', device='cpu'), cache)
            paths = run_lgffe_svm(root, cache, 'frozen_svm', max_iter=50)
            self.assertTrue((paths['models']/'linear_svm.pt').is_file())
            self.assertTrue((paths['predictions']/'changed_objects.csv').is_file())
            self.assertFalse((root/'experiments.csv').exists())
            self.assertEqual(file_hash(source), source_hash)
            with self.assertRaises(FileExistsError):
                run_lgffe_svm(root, cache, 'frozen_svm', max_iter=50)

    def test_fitter_receives_only_train_and_val_is_never_standardized_into_train(self):
        random = np.random.default_rng(4)
        train = np.concatenate([random.normal(-2, 0.1, (8, 4)), random.normal(2, 0.1, (8, 4))])
        labels = np.array([3]*8+[9]*8)
        svm = fit_linear_svm(train, labels, [3, 9], max_iter=50)
        before = svm['mean'].clone()
        svm_scores(svm, np.full((3, 4), 1e6))
        torch.testing.assert_close(before, svm['mean'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
