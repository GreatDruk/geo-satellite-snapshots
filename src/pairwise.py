"""Frozen ResNet features and train-only binary logistic reranking.

No new dependencies: solve the convex L2 logistic objective using torch LBFGS.
Routing uses the original top two labels, never ground truth or confidence gates.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import ImageDataset
from src.experiment import load_checkpoint
from src.inference import file_hash
from src.transforms import ImageTransform
from src.utils import get_device, set_seed

BUILDING_PAIRS = ((3, 9), (3, 5), (9, 5))


@torch.inference_mode()
def extract_building_features(root, checkpoint_path, output_dir, *, raw_dir=None,
                              device="auto", batch_size=8, cpu_threads=4):
    """Cache clean train-building features + all val; no fitting or test access.

    Completed cache is reused only when source hashes and array hashes match.
    Partial split files are reusable after interruption; validate IDs and shapes.
    """
    root = Path(root).resolve()
    raw = Path(raw_dir).resolve() if raw_dir else root/'data/raw'
    source = Path(checkpoint_path)
    source = source if source.is_absolute() else root/source
    output = Path(output_dir)
    output = output if output.is_absolute() else root/output
    if batch_size < 1 or cpu_threads < 1:
        raise ValueError("Positive batch size and CPU threads required")
    protocol = {'version': 1, 'checkpoint_sha256': file_hash(source),
                'train_csv_sha256': file_hash(raw/'train.csv'),
                'val_csv_sha256': file_hash(raw/'val.csv'),
                'classes_csv_sha256': file_hash(raw/'classes.csv'),
                'train_classes': [3, 5, 9], 'views': 'clean_full_frame',
                'feature_location': 'backbone.fc input before dropout'}
    meta_path = output/'metadata.json'
    if output.exists():
        if not meta_path.exists():
            raise FileExistsError("Feature directory has no provenance")
        previous = json.loads(meta_path.read_text(encoding='utf-8'))
        if previous['protocol'] != protocol:
            raise ValueError("Feature cache source/protocol mismatch")
        for name, digest in previous.get('hashes', {}).items():
            if file_hash(output/name) != digest:
                raise ValueError("Feature cache hash mismatch")
        if previous['status'] == 'complete':
            return output
    else:
        output.mkdir(parents=True)
    selected_device = get_device(device)
    if selected_device.type == 'cpu':
        torch.set_num_threads(cpu_threads)
    set_seed(42, deterministic=True, warn_only=True)
    model, checkpoint = load_checkpoint(source, selected_device, inference_only=True)
    if not hasattr(model, 'backbone') or model.backbone.fc[1].in_features != 2048:
        raise ValueError("Expected average-pool ResNet with 2048 features")
    metadata = {'protocol': protocol, 'status': 'running', 'test_read': False,
                'neural_training': False, 'device': str(selected_device),
                'torch': str(torch.__version__), 'hashes': {}}
    def write_meta():
        meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    write_meta()
    captured = []
    hook = model.backbone.fc.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().cpu().numpy().copy()))
    started = time.perf_counter()
    try:
        for split in ('train', 'val'):
            dataset = ImageDataset(raw/f'{split}.csv', raw/split,
                                   ImageTransform(checkpoint['config']['data'], training=False),
                                   {label: i for i, label in enumerate(checkpoint['class_labels'])})
            if split == 'train':
                dataset.data = dataset.data.loc[dataset.data.label.isin([3, 5, 9])].reset_index(drop=True)
            target_path = output/f'{split}.npz'
            if target_path.exists():
                with np.load(target_path, allow_pickle=False) as arrays:
                    if (not np.array_equal(arrays['ids'], dataset.data.id.to_numpy(dtype=str))
                        or not np.array_equal(arrays['labels'], dataset.data.label.to_numpy())
                        or arrays['features'].shape != (len(dataset), 2048)
                        or not np.isfinite(arrays['features']).all()):
                        raise ValueError("Invalid partial feature cache")
                metadata['hashes'][target_path.name] = file_hash(target_path)
                write_meta()
                continue
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                pin_memory=selected_device.type == 'cuda')
            features, probabilities = [], []
            for i, (images, _) in enumerate(loader, 1):
                captured.clear()
                logits = model(images.to(selected_device).contiguous())
                if len(captured) != 1:
                    raise ValueError("Expected one pooled feature tensor per forward")
                features.append(captured[0])
                probabilities.append(logits.softmax(1).cpu().numpy())
                if i == 1 or i % 10 == 0 or i == len(loader):
                    print(f'Frozen {split} features: {i}/{len(loader)} batches', flush=True)
            with target_path.open('xb') as stream:
                np.savez_compressed(stream, ids=dataset.data.id.to_numpy(dtype=str),
                                    labels=dataset.data.label.to_numpy(),
                                    class_labels=np.asarray(checkpoint['class_labels']),
                                    features=np.concatenate(features), probabilities=np.concatenate(probabilities))
            metadata['hashes'][target_path.name] = file_hash(target_path)
            write_meta()
    finally:
        hook.remove()
    metadata.update(status='complete', elapsed_seconds=time.perf_counter()-started)
    write_meta()
    return output


def fit_logistic(features, labels, pair, *, C=1.0, max_iter=500, tolerance=1e-6):
    """Standardize pair's train only, minimize sum BCE + ||w||²/(2*C).

    Bias unpenalized; no balancing, feature selection or validation tuning.
    This is an ordinary binary linear classifier, not backbone optimization.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    pair = tuple(pair)
    if (x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all()
        or len(pair) != 2 or pair[0] == pair[1] or C <= 0):
        raise ValueError('Invalid features/labels/pair/C')
    select = np.isin(y, pair)
    x, y = x[select], y[select]
    if set(y.tolist()) != set(pair):
        raise ValueError('Both classes must be present in train')
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    tx = torch.from_numpy((x-mean)/scale)
    ty = torch.from_numpy((y == pair[1]).astype(np.float64))
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=max_iter, max_eval=max_iter*2,
                                 tolerance_grad=tolerance, tolerance_change=1e-12,
                                 line_search_fn='strong_wolfe')
    def objective():
        return torch.nn.functional.binary_cross_entropy_with_logits(tx@weight+bias, ty, reduction='sum') + weight.square().sum()/(2*C)
    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss
    optimizer.step(closure)
    loss = objective()
    grad = torch.autograd.grad(loss, (weight, bias))
    grad_max = max(g.abs().max().item() for g in grad)
    state = optimizer.state[weight]
    train_pred = np.where((tx@weight+bias).detach().numpy() >= 0, pair[1], pair[0])
    return {'pair': list(pair), 'C': C, 'mean': mean.tolist(), 'scale': scale.tolist(),
            'weight': weight.detach().numpy().tolist(), 'bias': bias.item(),
            'training_samples': len(x), 'training_accuracy': float((train_pred==y).mean()),
            'objective': loss.item(), 'gradient_max': grad_max,
            'iterations': state['n_iter'], 'max_iter': max_iter,
            'converged': bool(state['n_iter'] < max_iter or grad_max <= tolerance)}


def binary_scores(model, features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or x.shape[1] != len(model['weight']):
        raise ValueError('Feature dimension/values mismatch')
    return ((x-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['weight']) + model['bias']


def rerank_labels(predictions, feature_ids, features, classifiers):
    """Return labels and audit rows; keep every non-routed label unchanged.

    Does not access true_labels. Do not treat binary conditional probabilities
    as a calibrated 20-way distribution or use labels to choose which to route.
    """
    predictions.validate()
    ids = np.asarray(feature_ids)
    if len(set(ids)) != len(ids) or set(ids) != set(predictions.ids):
        raise ValueError('Feature IDs must exactly match prediction IDs')
    x = np.asarray(features)[pd.Index(ids).get_indexer(predictions.ids)]
    order = np.argsort(-predictions.probabilities, axis=1, kind='stable')
    top = predictions.class_labels[order[:, :2]]
    result = top[:, 0].copy()
    audit = []
    seen = set()
    for model in classifiers:
        pair = tuple(model['pair'])
        if frozenset(pair) in seen:
            raise ValueError('Duplicate classifier pair')
        seen.add(frozenset(pair))
        routed = np.isin(top, pair).all(1)
        scores = binary_scores(model, x[routed])
        binary = np.where(scores >= 0, pair[1], pair[0])
        result[routed] = binary
        for index, score, predicted in zip(np.flatnonzero(routed), scores, binary):
            p_second = 1/(1+np.exp(-np.clip(score, -700, 700)))
            audit.append({'id': predictions.ids[index], 'pair': f'{pair[0]}_{pair[1]}',
                          'base_label': int(top[index, 0]), 'second_label': int(top[index, 1]),
                          'reranked_label': int(predicted), 'binary_second_probability': float(p_second),
                          'changed': bool(predicted != top[index, 0])})
    return result, pd.DataFrame(audit)
