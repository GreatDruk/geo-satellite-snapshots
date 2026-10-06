"""One train-only multiclass linear SVM on frozen pre-classifier LGFFE concat."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import ImageDataset, make_loader, read_classes
from src.experiment import load_checkpoint
from src.inference import file_hash
from src.transforms import ImageTransform
from src.utils import get_device, set_seed


def _path(root, value):
    value = Path(value)
    return value if value.is_absolute() else root/value


def _read_features(path):
    with np.load(path, allow_pickle=False) as arrays:
        values = {name: arrays[name] for name in arrays.files}
    features, labels, ids, classes = (values[name] for name in ('features', 'labels', 'ids', 'class_labels'))
    if (features.ndim != 2 or not len(features) or not np.isfinite(features).all()
            or labels.shape != ids.shape or features.shape[0] != len(ids)
            or len(set(ids.tolist())) != len(ids) or not np.isin(labels, classes).all()
            or values['softmax_probabilities'].shape != (len(ids), len(classes))
            or not np.isfinite(values['softmax_probabilities']).all()
            or not np.allclose(values['softmax_probabilities'].sum(1), 1, atol=1e-5)):
        raise ValueError('Invalid frozen LGFFE feature cache')
    return values


@torch.inference_mode()
def extract_lgffe_features(root, checkpoint_path, cache_dir, *, raw_dir='data/raw',
                           device='auto', batch_size=32, num_workers=0, cpu_threads=4):
    """Clean, ordered train+val extraction; eval/no gradients, no test access.

Capture model.forward_details()['fused_features'], not classifier hidden units.
Cache is reusable only with matching source and output hashes.
"""
    root = Path(root).resolve()
    source, raw, output = (_path(root, value).resolve() for value in (checkpoint_path, raw_dir, cache_dir))
    if not source.is_file():
        raise FileNotFoundError(f'Frozen LGFFE best.pt is required: {source}')
    if batch_size < 1 or num_workers < 0 or cpu_threads < 1:
        raise ValueError('Invalid feature loader settings')
    protocol = {'version': 1, 'checkpoint_sha256': file_hash(source),
                'train_csv_sha256': file_hash(raw/'train.csv'), 'val_csv_sha256': file_hash(raw/'val.csv'),
                'classes_csv_sha256': file_hash(raw/'classes.csv'),
                'representation': 'concat(global_features, local_features)_before_entire_classifier',
                'transforms': 'deterministic_clean_full_frame', 'tta': 'none'}
    metadata_path = output/'metadata.json'
    metadata = {'protocol': protocol, 'status': 'running', 'hashes': {},
                'test_read': False, 'neural_training': False}
    if output.exists():
        if not metadata_path.is_file():
            raise FileExistsError('Feature cache directory has no provenance')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata['protocol'] != protocol:
            raise ValueError('Feature cache protocol/source mismatch; use a new cache directory')
        for name, digest in metadata['hashes'].items():
            if file_hash(output/name) != digest:
                raise ValueError(f'Feature cache hash mismatch: {name}')
        if metadata['status'] == 'complete':
            for split in ('train', 'val'):
                _read_features(output/f'{split}.npz')
            return output
    else:
        output.mkdir(parents=True)
    selected = get_device(device)
    if selected.type == 'cpu':
        torch.set_num_threads(cpu_threads)
    set_seed(42, deterministic=True, warn_only=True)
    model, checkpoint = load_checkpoint(source, selected, inference_only=True)
    if checkpoint['config']['model']['name'] != 'resnet50_lgffe' or not hasattr(model, 'forward_details'):
        raise ValueError('Expected a trained LGFFE checkpoint')
    if checkpoint['config'].get('final_refit'):
        raise ValueError('Train+val refit checkpoint cannot use val as held-out evaluation')
    model.requires_grad_(False)
    model.eval()
    classes = read_classes(raw)
    class_labels = np.asarray(checkpoint['class_labels'])
    if not np.array_equal(class_labels, classes.label.to_numpy()):
        raise ValueError('Checkpoint classes differ from classes.csv')
    channels = model.backbone.fc.channels
    data_config = dict(checkpoint['config']['data'])
    data_config.update(batch_size=batch_size, num_workers=num_workers, pin_memory=selected.type == 'cuda')
    metadata.update(device=str(selected), feature_dim=2*channels, class_labels=class_labels.tolist(),
                    class_names=classes.class_name.tolist())
    def write_meta():
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    write_meta()
    started = time.perf_counter()
    for split in ('train', 'val'):
        dataset = ImageDataset(raw/f'{split}.csv', raw/split, ImageTransform(data_config, training=False),
                               {label: index for index, label in enumerate(class_labels)})
        target = output/f'{split}.npz'
        if target.exists():
            cached = _read_features(target)
            if (not np.array_equal(cached['ids'], dataset.data.id.to_numpy(dtype=str))
                    or not np.array_equal(cached['labels'], dataset.data.label.to_numpy())
                    or cached['features'].shape[1] != 2*channels):
                raise ValueError('Partial feature cache does not align with split')
        else:
            loader = make_loader(dataset, data_config, 42, shuffle=False)
            features, probabilities = [], []
            for index, (images, _) in enumerate(loader, 1):
                details = model.forward_details(images.to(selected, non_blocking=selected.type == 'cuda'))
                fused = details['fused_features']
                if fused.shape != (len(images), 2*channels):
                    raise ValueError('Expected local+global concat before classifier')
                features.append(fused.float().cpu().numpy())
                probabilities.append(details['logits'].float().softmax(1).cpu().numpy())
                if index == 1 or index % 20 == 0 or index == len(loader):
                    print(f'Frozen LGFFE {split}: {index}/{len(loader)} batches', flush=True)
            with target.open('xb') as stream:
                np.savez_compressed(stream, ids=dataset.data.id.to_numpy(dtype=str),
                                    labels=dataset.data.label.to_numpy(), class_labels=class_labels,
                                    features=np.concatenate(features), softmax_probabilities=np.concatenate(probabilities))
        metadata['hashes'][target.name] = file_hash(target)
        write_meta()
    if file_hash(source) != protocol['checkpoint_sha256']:
        raise RuntimeError('Source checkpoint changed during extraction')
    metadata.update(status='complete', elapsed_seconds=time.perf_counter()-started)
    write_meta()
    return output


def fit_linear_svm(features, labels, class_labels, *, C=1.0, max_iter=200, device='cpu'):
    """Fixed-C Weston-Watkins hinge SVM, train standardization only.

Objective: 0.5*||W||² + C*sum_i sum_{j!=y_i} max(0,1+s_j-s_y).
For numerical scale divide objective by C*N. Bias is unpenalized. Full-batch
LBFGS sees no validation tensors, no early stopping and no parameter search.
"""
    x, y, classes = np.asarray(features, dtype=np.float64), np.asarray(labels), np.asarray(class_labels)
    if (x.ndim != 2 or not len(x) or not np.isfinite(x).all() or y.shape != (len(x),)
            or classes.ndim != 1 or len(classes) < 2 or len(set(classes.tolist())) != len(classes)
            or set(y.tolist()) != set(classes.tolist())):
        raise ValueError('Finite train features and all configured classes are required')
    if isinstance(C, bool) or not isinstance(C, (int, float)) or not math.isfinite(C) or C <= 0:
        raise ValueError('C must be finite and positive')
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError('max_iter must be positive')
    set_seed(42, deterministic=True, warn_only=True)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1
    selected = get_device(device)
    tx = torch.as_tensor((x-mean)/scale, dtype=torch.float64, device=selected)
    ty = torch.as_tensor(pd.Index(classes).get_indexer(y), dtype=torch.long, device=selected)
    classifier = nn.Linear(x.shape[1], len(classes), dtype=torch.float64, device=selected)
    nn.init.zeros_(classifier.weight)
    nn.init.zeros_(classifier.bias)
    hinge = nn.MultiMarginLoss(p=1, margin=1, reduction='mean')
    optimizer = torch.optim.LBFGS(classifier.parameters(), max_iter=max_iter, max_eval=max_iter*2,
                                 history_size=20, tolerance_grad=1e-6, tolerance_change=1e-10,
                                 line_search_fn='strong_wolfe')
    history = []
    def objective():
        # MultiMarginLoss divides each sample's margin sum by class count.
        return len(classes)*hinge(classifier(tx), ty) + classifier.weight.square().sum()/(2*C*len(x))
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite SVM objective')
        loss.backward()
        history.append({'closure': len(history)+1, 'train_objective': loss.item()})
        if len(history) == 1 or len(history) % 25 == 0:
            print(f'SVM optimizer evaluation {len(history)}: train objective={loss.item():.6f}', flush=True)
        return loss
    started = time.perf_counter()
    optimizer.step(closure)
    final_loss = objective()
    gradients = torch.autograd.grad(final_loss, tuple(classifier.parameters()))
    gradient_max = max(value.abs().max().item() for value in gradients)
    state = optimizer.state[classifier.weight]
    with torch.no_grad():
        train_accuracy = (classifier(tx).argmax(1) == ty).double().mean().item()
    return {'state_dict': {name: value.detach().cpu() for name, value in classifier.state_dict().items()},
            'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),
            'class_labels': classes.tolist(), 'feature_dim': x.shape[1], 'C': C,
            'objective': 'sum_multiclass_hinge_plus_half_weight_l2', 'train_samples': len(x),
            'train_accuracy': train_accuracy, 'final_train_objective_scaled': final_loss.item(),
            'gradient_max': gradient_max, 'gradient_tolerance_reached': gradient_max <= 1e-6,
            'iterations': int(state['n_iter']), 'max_iter': max_iter,
            'closure_evaluations': len(history), 'elapsed_seconds': time.perf_counter()-started,
            'history': history}


def svm_scores(svm, features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != svm['feature_dim'] or not np.isfinite(features).all():
        raise ValueError('Feature dimensions must match fitted SVM')
    values = (torch.from_numpy(features)-svm['mean'])/svm['scale']
    with torch.inference_mode():
        return (values@svm['state_dict']['weight'].T + svm['state_dict']['bias']).numpy()


def compare_predictions(ids, truth, softmax_pred, svm_pred, class_labels, class_names):
    """True-label error statistics; SVM scores are not probabilities."""
    ids, truth, old, new, labels = map(np.asarray, (ids, truth, softmax_pred, svm_pred, class_labels))
    if (any(value.shape != ids.shape for value in (truth, old, new))
            or any(not np.isin(value, labels).all() for value in (truth, old, new))):
        raise ValueError('Prediction arrays/classes do not align')
    index = pd.Index(labels)
    old_ok, new_ok = old == truth, new == truth
    matrices = {}
    for name, prediction in [('softmax', old), ('svm', new)]:
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        np.add.at(matrix, (index.get_indexer(truth), index.get_indexer(prediction)), 1)
        matrices[name] = pd.DataFrame(matrix, index=class_names, columns=class_names)
    per_class = []
    for label, name in zip(labels, class_names):
        mask = truth == label
        per_class.append({'label': label, 'class_name': name, 'samples': int(mask.sum()),
                          'softmax_errors': int((mask & ~old_ok).sum()), 'svm_errors': int((mask & ~new_ok).sum()),
                          'fixed': int((mask & ~old_ok & new_ok).sum()), 'new_errors': int((mask & old_ok & ~new_ok).sum())})
    changed = pd.DataFrame({'id': ids, 'true_label': truth, 'softmax_pred': old, 'svm_pred': new})
    changed['fixed'] = ~old_ok & new_ok
    changed['new_error'] = old_ok & ~new_ok
    summary = {'samples': len(ids), 'softmax_correct': int(old_ok.sum()), 'svm_correct': int(new_ok.sum()),
               'softmax_accuracy': float(old_ok.mean()), 'svm_accuracy': float(new_ok.mean()),
               'fixed': int((~old_ok & new_ok).sum()), 'new_errors': int((old_ok & ~new_ok).sum()),
               'shared_errors': int((~old_ok & ~new_ok).sum()), 'changed_predictions': int((old != new).sum())}
    return summary, pd.DataFrame(per_class), changed[old != new].reset_index(drop=True), matrices


def run_lgffe_svm(root, cache_dir, run_id, *, C=1.0, max_iter=200, device='cpu'):
    """Fit once on train, then evaluate val; save a separate classifier and reports."""
    root = Path(root).resolve()
    cache = _path(root, cache_dir)
    if not run_id or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in run_id):
        raise ValueError('Use a safe unique SVM run_id')
    metadata = json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    if metadata['status'] != 'complete':
        raise ValueError('Complete frozen feature cache required')
    for name, digest in metadata['hashes'].items():
        if file_hash(cache/name) != digest:
            raise ValueError('Frozen feature cache hash mismatch')
    paths = {kind: root/'artifacts'/kind/run_id for kind in ('models', 'logs', 'predictions')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f'SVM run exists: {run_id}; view saved reports instead')
    train = _read_features(cache/'train.npz')
    # Neither val features nor targets enter fitting or train standardization.
    svm = fit_linear_svm(train['features'], train['labels'], train['class_labels'], C=C, max_iter=max_iter, device=device)
    validation = _read_features(cache/'val.npz')
    if not np.array_equal(train['class_labels'], validation['class_labels']):
        raise ValueError('Train/val class columns differ')
    scores = svm_scores(svm, validation['features'])
    labels = train['class_labels']
    prediction = labels[scores.argmax(1)]
    old = labels[validation['softmax_probabilities'].argmax(1)]
    summary, per_class, changed, matrices = compare_predictions(
        validation['ids'], validation['labels'], old, prediction, labels, metadata['class_names'])
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    history = svm.pop('history')
    svm.update(source_protocol=metadata['protocol'], feature_location=metadata['protocol']['representation'])
    torch.save(svm, paths['models']/'linear_svm.pt')
    parameters = {name: value for name, value in svm.items() if name not in ('state_dict', 'mean', 'scale')}
    parameters.update(kind='frozen_linear_svm', seed=42, neural_training=False, test_read=False,
                      val_used_for_fitting=False, feature_cache=str(cache), class_names=metadata['class_names'])
    (paths['logs']/'parameters.json').write_text(json.dumps(parameters, indent=2), encoding='utf-8')
    (paths['logs']/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame(history).to_csv(paths['logs']/'optimizer_history.csv', index=False)
    per_class.to_csv(paths['logs']/'per_class_errors.csv', index=False)
    changed.to_csv(paths['predictions']/'changed_objects.csv', index=False)
    pd.DataFrame({'id': validation['ids'], 'true_label': validation['labels'], 'softmax_pred': old,
                  'svm_pred': prediction}).to_csv(paths['predictions']/'val.csv', index=False)
    with (paths['predictions']/'val_scores.npz').open('xb') as stream:
        np.savez_compressed(stream, ids=validation['ids'], class_labels=labels, scores=scores, labels=validation['labels'])
    for name, matrix in matrices.items():
        matrix.to_csv(paths['logs']/f'{name}_confusion_matrix.csv', index_label='true_class')
    return paths


def show_svm_results(root, run_id):
    """Display saved summaries/tables and paired confusion matrices in notebook13."""
    import matplotlib.pyplot as plt
    from IPython.display import display
    root = Path(root)
    logs, predictions = root/'artifacts/logs'/run_id, root/'artifacts/predictions'/run_id
    summary = json.loads((logs/'summary.json').read_text(encoding='utf-8'))
    display(pd.DataFrame([summary]))
    display(pd.read_csv(logs/'per_class_errors.csv').sort_values('svm_errors', ascending=False))
    changed = pd.read_csv(predictions/'changed_objects.csv', dtype={'id': str})
    display(changed)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for axis, name in zip(axes, ('softmax', 'svm')):
        matrix = pd.read_csv(logs/f'{name}_confusion_matrix.csv', index_col=0)
        axis.imshow(matrix.to_numpy(), cmap='Blues')
        axis.set(xticks=range(len(matrix)), yticks=range(len(matrix)), xticklabels=matrix.columns,
                 yticklabels=matrix.index, xlabel='Predicted', ylabel='True', title=name)
        axis.tick_params(axis='x', labelrotation=90)
        for row, column in np.argwhere(matrix.to_numpy() > 0):
            axis.text(column, row, str(matrix.iat[row, column]), ha='center', va='center', fontsize=7)
    fig.tight_layout()
    fig.savefig(logs/'confusion_matrices.png', dpi=130, bbox_inches='tight')
    plt.show()
    return changed
