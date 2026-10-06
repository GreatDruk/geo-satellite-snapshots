"""Train-only fitting and label-free routing on synthetic frozen features."""
import unittest
import numpy as np
import torch
from src.pairwise import fit_logistic, binary_scores, rerank_labels
from src.training.predict import ProbabilityPredictions


class PairwiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_train_pair_scaling_and_solver_ignore_other_classes(self):
        features = np.array([[-3,1],[-2,1],[-1,1],[1,1],[2,1],[3,1],[999,999]], dtype=float)
        labels = np.array([3,3,3,9,9,9,5])
        model = fit_logistic(features, labels, (3,9))
        clean = fit_logistic(features[:-1], labels[:-1], (3,9))
        np.testing.assert_allclose(model['weight'], clean['weight'])
        np.testing.assert_allclose(model['mean'], [0,1])
        self.assertEqual(model['training_samples'], 6)
        self.assertEqual(model['training_accuracy'], 1)
        self.assertTrue(model['converged'])
        self.assertLess(model['gradient_max'], 1e-4)
        scores = binary_scores(model, [[-2,1],[2,1]])
        self.assertLess(scores[0],0); self.assertGreater(scores[1],0)

    def test_top_two_routing_including_confident_predictions_without_truth(self):
        # Column order differs from pair order; feature rows also reordered.
        p = ProbabilityPredictions(np.array(['a','b','c']), np.array([9,5,3]),
            np.array([[.999999,0,.000001],[.1,.8,.1],[.6,.3,.1]]), np.array([9,5,9]))
        classifier = {'pair':[9,3],'mean':[0.],'scale':[1.],'weight':[1.],'bias':0.}
        labels, audit = rerank_labels(p, ['c','a','b'], [[10],[2],[10]], [classifier])
        self.assertEqual(labels.tolist(), [3,5,9])
        self.assertEqual(audit.id.tolist(), ['a'])
        p.true_labels = np.array([5,3,3])
        alternate, _ = rerank_labels(p, ['c','a','b'], [[10],[2],[10]], [classifier])
        np.testing.assert_array_equal(labels, alternate)
        p.true_labels = None
        unlabeled, _ = rerank_labels(p, ['c','a','b'], [[10],[2],[10]], [classifier])
        np.testing.assert_array_equal(labels, unlabeled)

    def test_reject_duplicate_or_missing_features_and_duplicate_pairs(self):
        p = ProbabilityPredictions(np.array(['a','b']), np.array([3,9]), np.array([[.6,.4],[.2,.8]]))
        classifier = {'pair':[3,9],'mean':[0.],'scale':[1.],'weight':[1.],'bias':0.}
        for ids in [['a','a'],['a','c']]:
            with self.assertRaises(ValueError): rerank_labels(p, ids, [[0],[1]], [classifier])
        with self.assertRaises(ValueError): rerank_labels(p,['a','b'],[[0],[1]],[classifier,classifier])


if __name__ == '__main__': unittest.main()
