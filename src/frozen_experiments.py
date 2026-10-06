"""Small fixed experiments on saved probabilities / frozen building features."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.inference import file_hash, _metric
from src.pairwise import BUILDING_PAIRS, fit_logistic, binary_scores, rerank_labels
from src.training.predict import load_probabilities, save_probabilities, ensemble_probabilities
from src.utils import set_seed

SOURCE_FILES = {
    'stronger': 'resnet50_partial_domain_shift_stronger_001/val_probabilities.npz',
    'smoothing': 'resnet50_partial_domain_shift_smoothing_001/val_probabilities.npz',
    'cosface': 'resnet50_partial_domain_shift_cosface_001/val_probabilities.npz',
    'mining': 'resnet50_partial_domain_shift_hard_mining_001/val_probabilities.npz',
    'resnet101': 'resnet101_partial_domain_shift_stronger_001/val_probabilities.npz',
    'stronger_d4': 'quality_inference_001/stronger__d4.npz',
}
FIXED_ENSEMBLES = {
    'stronger_cosface_mining': ('stronger', 'cosface', 'mining'),
    'stronger_smoothing_cosface_mining': ('stronger', 'smoothing', 'cosface', 'mining'),
    'resnet101_stronger_d4_cosface': ('resnet101', 'stronger_d4', 'cosface'),
    'resnet101_stronger_d4_mining': ('resnet101', 'stronger_d4', 'mining'),
}


def load_sources(root):
    root = Path(root)
    truth = pd.read_csv(root/'data/raw/val.csv', dtype={'id': str}).set_index('id').label
    classes = pd.read_csv(root/'data/raw/classes.csv')
    predictions = {}
    for name, relative in SOURCE_FILES.items():
        p = load_probabilities(root/'artifacts/predictions'/relative)
        if (set(p.ids) != set(truth.index) or set(p.class_labels) != set(classes.label)
            or p.true_labels is None or not np.array_equal(p.true_labels, truth.loc[p.ids].to_numpy())):
            raise ValueError(f'Invalid source alignment: {name}')
        predictions[name] = p
    return predictions, classes


def evaluate_labels(ids, labels, reference, name):
    order = pd.Index(ids).get_indexer(reference.ids)
    if (order < 0).any() or len(set(ids)) != len(reference.ids):
        raise ValueError('Result IDs must align exactly')
    actual = np.asarray(labels)[order]
    base = reference.class_labels[reference.probabilities.argmax(1)]
    truth = reference.true_labels
    ok, was_ok = actual == truth, base == truth
    return {'recipe': name, 'correct': int(ok.sum()), 'errors': int((~ok).sum()),
            'val_accuracy': float(ok.mean()), 'fixes': int((ok & ~was_ok).sum()),
            'breaks': int((~ok & was_ok).sum())}


def _change_tables(ids, labels, reference, classes):
    actual = np.asarray(labels)[pd.Index(ids).get_indexer(reference.ids)]
    base = reference.class_labels[reference.probabilities.argmax(1)]
    names = classes.set_index('label').class_name
    frame = pd.DataFrame({'id': reference.ids, 'true_label': reference.true_labels,
                          'reference_label': base, 'label': actual})
    frame['true_class'] = frame.true_label.map(names)
    frame['reference_class'] = frame.reference_label.map(names)
    frame['predicted_class'] = frame.label.map(names)
    frame['change'] = np.where((base!=reference.true_labels)&(actual==reference.true_labels), 'fix',
                              np.where((base==reference.true_labels)&(actual!=reference.true_labels), 'break', 'wrong_to_wrong'))
    changes = frame.loc[actual != base]
    per_class = classes.copy()
    per_class['support'] = [int((reference.true_labels==label).sum()) for label in classes.label]
    per_class['errors'] = [int(((reference.true_labels==label)&(actual!=reference.true_labels)).sum()) for label in classes.label]
    return frame, changes, per_class


def _source_hashes(root):
    return {relative: file_hash(root/'artifacts/predictions'/relative) for relative in SOURCE_FILES.values()}


def run_fixed_ensembles(root, run_id='frozen_probability_ensembles_001'):
    root = Path(root).resolve()
    if Path(run_id).name != run_id or run_id in {'', '.', '..'}:
        raise ValueError('Use a directory name as run ID')
    logs, outputs = root/'artifacts/logs'/run_id, root/'artifacts/predictions'/run_id
    protocol = {'version': 1, 'sources': _source_hashes(root), 'ensembles': FIXED_ENSEMBLES,
                'weights': 'equal per original model; D4 is one model',
                'val_csv_sha256': file_hash(root/'data/raw/val.csv')}
    protocol = json.loads(json.dumps(protocol))
    metadata_path = logs/'metadata.json'
    if logs.exists() or outputs.exists():
        if not metadata_path.exists():
            raise FileExistsError('Existing incomplete run; choose a new ID')
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        if meta['protocol'] != protocol or meta['status'] != 'complete':
            raise ValueError('Existing run does not match protocol')
        return pd.read_csv(logs/'comparison.csv')
    members, classes = load_sources(root)
    reference = ensemble_probabilities([members['resnet101'], members['stronger_d4']])
    assert _metric(reference)['correct'] == 679, 'Reference differs from recorded best'
    logs.mkdir(parents=True); outputs.mkdir(parents=True)
    metadata_path.write_text(json.dumps({'status': 'running', 'protocol': protocol}, indent=2), encoding='utf-8')
    candidates = {'stronger_single': members['stronger'], 'resnet101_single': members['resnet101'],
                  'reference_101_50d4': reference}
    candidates.update({name: ensemble_probabilities([members[key] for key in keys])
                       for name, keys in FIXED_ENSEMBLES.items()})
    rows = []
    for name, prediction in candidates.items():
        labels = prediction.class_labels[prediction.probabilities.argmax(1)]
        row = evaluate_labels(prediction.ids, labels, reference, name)
        row['val_nll'] = _metric(prediction)['val_nll']
        row['reference'] = 'reference_101_50d4'
        rows.append(row)
        save_probabilities(prediction, outputs/f'{name}.npz')
        frame, changes, per_class = _change_tables(prediction.ids, labels, reference, classes)
        frame.to_csv(outputs/f'{name}.csv', index=False)
        changes.to_csv(logs/f'{name}_changes.csv', index=False)
        per_class.to_csv(logs/f'{name}_classes.csv', index=False)
    table = pd.DataFrame(rows)
    table.to_csv(logs/'comparison.csv', index=False)
    metadata_path.write_text(json.dumps({'status': 'complete', 'protocol': protocol,
                                         'neural_training': False, 'test_read': False}, indent=2), encoding='utf-8')
    return table


def run_pairwise(root, feature_dir='artifacts/predictions/building_pairwise_features_001',
                 run_id='frozen_building_reranker_001'):
    root = Path(root).resolve()
    feature_dir = Path(feature_dir)
    feature_dir = feature_dir if feature_dir.is_absolute() else root/feature_dir
    if Path(run_id).name != run_id or run_id in {'', '.', '..'}:
        raise ValueError('Use a directory name as run ID')
    feature_meta = json.loads((feature_dir/'metadata.json').read_text(encoding='utf-8'))
    if feature_meta['status'] != 'complete':
        raise ValueError('Feature cache must be complete')
    for filename, sha in feature_meta['hashes'].items():
        if file_hash(feature_dir/filename) != sha:
            raise ValueError('Feature cache corrupted')
    if (feature_meta['protocol']['train_csv_sha256'] != file_hash(root/'data/raw/train.csv')
        or feature_meta['protocol']['val_csv_sha256'] != file_hash(root/'data/raw/val.csv')):
        raise ValueError('Feature source labels changed')
    logs, outputs, models = [root/f'artifacts/{kind}'/run_id for kind in ('logs','predictions','models')]
    protocol = {'version': 1, 'features': feature_meta['hashes'], 'feature_protocol': feature_meta['protocol'],
                'sources': _source_hashes(root), 'pairs': list(BUILDING_PAIRS), 'C': 1.0,
                'solver': 'torch.LBFGS sum BCE + L2/(2*C), standardize pair train only',
                'max_iter': 500, 'routing': 'unordered original top2, no confidence threshold',
                'bases': ['resnet101_single', 'reference_101_50d4'], 'seed': 42}
    protocol = json.loads(json.dumps(protocol))
    metadata_path = logs/'metadata.json'
    if any(path.exists() for path in (logs, outputs, models)):
        if not metadata_path.exists():
            raise FileExistsError('Existing incomplete run; choose a new ID')
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        if meta['protocol'] != protocol or meta['status'] != 'complete':
            raise ValueError('Existing run does not match protocol')
        return pd.read_csv(logs/'comparison.csv')
    for path in (logs, outputs, models): path.mkdir(parents=True)
    metadata_path.write_text(json.dumps({'status':'running','protocol':protocol}, indent=2), encoding='utf-8')
    with np.load(feature_dir/'train.npz', allow_pickle=False) as data:
        train_x, train_y = data['features'], data['labels']
    with np.load(feature_dir/'val.npz', allow_pickle=False) as data:
        val_x, val_y, val_ids = data['features'], data['labels'], data['ids']
        cpu_p, cpu_classes = data['probabilities'], data['class_labels']
    train_raw = pd.read_csv(root/'data/raw/train.csv', dtype={'id':str})
    val_raw = pd.read_csv(root/'data/raw/val.csv', dtype={'id':str})
    with np.load(feature_dir/'train.npz', allow_pickle=False) as data:
        expected = train_raw.loc[train_raw.label.isin([3,5,9])]
        if not np.array_equal(data['ids'], expected.id.to_numpy()) or not np.array_equal(train_y, expected.label.to_numpy()):
            raise ValueError('Train features/labels not aligned')
    if not np.array_equal(val_ids, val_raw.id.to_numpy()) or not np.array_equal(val_y, val_raw.label.to_numpy()):
        raise ValueError('Val features/labels not aligned')
    members, classes = load_sources(root)
    p101 = members['resnet101']
    order = pd.Index(val_ids).get_indexer(p101.ids)
    reproduced = cpu_classes[cpu_p[order].argmax(1)]
    if not np.array_equal(reproduced, p101.class_labels[p101.probabilities.argmax(1)]):
        raise ValueError('Frozen checkpoint does not reproduce saved ResNet101 decisions')
    torch.set_num_threads(4)
    set_seed(42, deterministic=True, warn_only=True)
    started = time.perf_counter()
    classifiers = [fit_logistic(train_x, train_y, pair) for pair in BUILDING_PAIRS]
    diagnostics = []
    for classifier in classifiers:
        a,b = classifier['pair']
        (models/f'pair_{a}_{b}.json').write_text(json.dumps(classifier, indent=2), encoding='utf-8')
        selected = np.isin(val_y, (a,b))
        predicted = np.where(binary_scores(classifier, val_x[selected]) >= 0, b, a)
        diagnostics.append({'pair':f'{a}_{b}', 'train_samples':classifier['training_samples'],
                            'train_accuracy':classifier['training_accuracy'], 'val_pair_samples':int(selected.sum()),
                            'val_pair_correct':int((predicted==val_y[selected]).sum()),
                            'val_pair_accuracy':float((predicted==val_y[selected]).mean()),
                            'iterations':classifier['iterations'], 'gradient_max':classifier['gradient_max'],
                            'converged':classifier['converged']})
    pd.DataFrame(diagnostics).to_csv(logs/'binary_diagnostics.csv', index=False)
    bases = {'resnet101_single':p101,
             'reference_101_50d4':ensemble_probabilities([p101,members['stronger_d4']])}
    rows, pair_rows = [], []
    for name, base in bases.items():
        labels, routes = rerank_labels(base, val_ids, val_x, classifiers)
        base_labels = base.class_labels[base.probabilities.argmax(1)]
        row = evaluate_labels(base.ids, labels, base, name+'_pairwise')
        row.update(reference=name, routed=len(routes), changed=int(routes.changed.sum()),
                   elapsed_fit_seconds=time.perf_counter()-started)
        rows.append(row)
        frame, changes, per_class = _change_tables(base.ids, labels, base, classes)
        frame.to_csv(outputs/f'{name}_pairwise.csv', index=False)
        changes.to_csv(logs/f'{name}_changes.csv', index=False)
        per_class.to_csv(logs/f'{name}_classes.csv', index=False)
        truth_by_id = pd.Series(base.true_labels, index=base.ids)
        routes['true_label'] = routes.id.map(truth_by_id)
        routes['base_correct'] = routes.base_label == routes.true_label
        routes['reranked_correct'] = routes.reranked_label == routes.true_label
        routes.to_csv(logs/f'{name}_routing.csv', index=False)
        for pair, group in routes.groupby('pair'):
            pair_rows.append({'base':name,'pair':pair,'routed':len(group),'changed':int(group.changed.sum()),
                              'correct_before':int(group.base_correct.sum()), 'correct_after':int(group.reranked_correct.sum()),
                              'fixes':int((~group.base_correct & group.reranked_correct).sum()),
                              'breaks':int((group.base_correct & ~group.reranked_correct).sum()),
                              'outside_true_pair':int((~group.true_label.isin([int(i) for i in pair.split('_')])).sum())})
    pd.DataFrame(pair_rows).to_csv(logs/'routing_summary.csv', index=False)
    result = pd.DataFrame(rows)
    result.to_csv(logs/'comparison.csv', index=False)
    metadata_path.write_text(json.dumps({'status':'complete','protocol':protocol,
                                         'neural_training':False,'test_read':False,
                                         'fitted_on':'train only','val_cpu_argmax_matches':len(val_ids),
                                         'elapsed_fit_seconds':time.perf_counter()-started}, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['ensembles','pairwise'])
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    print((run_fixed_ensembles if args.mode=='ensembles' else run_pairwise)(args.root).to_string(index=False))
