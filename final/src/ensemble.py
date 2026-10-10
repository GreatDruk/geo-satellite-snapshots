"""Fixed probability ensemble with conditional weights for two class groups."""
import numpy as np


def probabilities(values):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or not p.size or not np.isfinite(p).all() or (p < 0).any() or (p > 1 + 1e-8).any():
        raise ValueError('Invalid probability matrix')
    if not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Rows must sum to one')
    return p


def blend(agmfa, resnet, class_names, config):
    a, r = probabilities(agmfa), probabilities(resnet)
    if a.shape != r.shape or a.shape[1] != len(class_names) or len(set(class_names)) != len(class_names):
        raise ValueError('Model shapes/class names differ')
    if config['base_agmfa_weight'] != .5 or config['eligibility'] != 'top2':
        raise ValueError('Use the fixed 50/50, top-2 ensemble recipe')
    base = (a + r) / 2
    output = base.copy()
    order = np.argsort(-base, axis=1, kind='stable')
    used = set()
    for group in config['groups'].values():
        columns = np.array([class_names.index(name) for name in group['classes']], dtype=int)
        weight = group['agmfa_weight']
        if len(set(columns)) != len(columns) or used.intersection(columns) or not 0 <= weight <= 1:
            raise ValueError('Invalid/overlapping groups')
        used.update(columns)
        rows = np.flatnonzero(np.isin(order[:, :2], columns).all(1))
        if weight != .5 and len(rows):
            mixed = weight * a[np.ix_(rows, columns)] + (1 - weight) * r[np.ix_(rows, columns)]
            mass = base[np.ix_(rows, columns)].sum(1, keepdims=True)
            denominator = mixed.sum(1, keepdims=True)
            output[np.ix_(rows, columns)] = mass * np.divide(mixed, denominator, out=np.zeros_like(mixed), where=denominator > 0)
    return probabilities(output)
