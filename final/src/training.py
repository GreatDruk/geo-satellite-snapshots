"""Fresh ImageNet initialization and fixed train+val refit, without held-out evaluation."""
import copy
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import read_classes
from .hard_target_shift import HardTargetDataset, PROFILES
from .models.agmfa import build_agmfa
from .models.pretrained import build_pretrained
from .origin_training import TargetDomainDataset, OriginDataset, origin_collate, train_origin_epoch
from .utils import append_experiment, get_device, seed_worker, set_seed, validate_journal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configs'


def load_config(name):
    if name not in ('agmfa', 'resnet50'):
        raise ValueError('Choose agmfa or resnet50')
    return json.loads((CONFIG_DIR / f'{name}.json').read_text(encoding='utf-8'))


def validate_config(config):
    training = config['training']
    if training['epochs'] != 25 or training['selection'] != 'fixed_last_epoch':
        raise ValueError('Final recipe requires exactly 25 epochs and last-epoch selection')
    if training['scheduler']['name'] != 'cosine' or training['scheduler']['T_max'] != 25:
        raise ValueError('Cosine schedule must span 25 epochs')
    if 'evaluation' in config or training.get('initialize_from'):
        raise ValueError('Final refit accepts no evaluation set or fitted task checkpoint')
    if training['hard_target_shift'] != dict(clean_ce_weight=.5, hard_ce_weight=.5,
                                             js_weight=3., temperature=1.):
        raise ValueError('HardShift loss must match the fixed training mechanics')
    if training['target_weight'] != 2. or training['loss'] != dict(name='cross_entropy', label_smoothing=0.):
        raise ValueError('Final recipe uses CE and former-val weight 2')


def build_model(name, config, num_classes, *, imagenet_weights=None):
    factory = build_agmfa if name == 'agmfa' else build_pretrained
    kwargs = {'initialize_pretrained': imagenet_weights is None}
    model = (factory(config['model'], 256, num_classes, **kwargs) if name == 'agmfa'
             else factory(config['model'], num_classes, **kwargs))
    if imagenet_weights is not None:
        state = torch.load(imagenet_weights, map_location='cpu', weights_only=True)
        result = model.backbone.load_state_dict({k: v for k, v in state.items() if not k.startswith('fc.')}, strict=False)
        expected = {k for k in model.backbone.state_dict() if k.startswith('fc.')}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError('Supply original torchvision ResNet50 ImageNet V2 weights')
    return model


def make_fit_loader(config, raw, classes, device):
    data, seed = config['data'], config['seed']
    mapping = {label: index for index, label in enumerate(classes.label)}
    source = HardTargetDataset(raw / 'train.csv', raw / 'train', data, mapping, seed)
    target = TargetDomainDataset(raw, data, mapping, seed)
    if (len(source), len(target)) != (config['fit_counts']['train'], config['fit_counts']['val']):
        raise ValueError('Unexpected train/val sizes')
    dataset = OriginDataset(source, target)
    workers = 0 if os.name == 'nt' else data['num_workers']
    options = ({'persistent_workers': data['persistent_workers'], 'prefetch_factor': data['prefetch_factor']}
               if workers else {})
    return DataLoader(dataset, batch_size=data['batch_size'], shuffle=True, num_workers=workers,
                      pin_memory=device.type == 'cuda', collate_fn=origin_collate,
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed), **options)


def fit_fixed(model, loader, config, device, model_dir, log_dir):
    training = config['training']
    options = dict(training['optimizer'])
    if options.pop('name') != 'adamw':
        raise ValueError('Final optimizer is AdamW')
    backbone_lr, head_lr = options.pop('backbone_lr'), options.pop('head_lr')
    optimizer = torch.optim.AdamW(model.optimizer_groups(options['lr'], backbone_lr, head_lr), **options)
    schedule = dict(training['scheduler'])
    schedule.pop('name')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **schedule)
    model_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    metadata = {'status': 'running', 'training_splits': ['train', 'val'],
                'validation_performed': False, 'checkpoint_selection': 'fixed_last_epoch'}
    rows, profiles = [], []
    started = time.perf_counter()
    try:
        for epoch in range(1, 26):
            epoch_start = time.perf_counter()
            loader.dataset.set_epoch(epoch)
            loader.generator.manual_seed(config['seed'] + epoch)
            model.set_epoch(epoch)
            rates = {g['name']: g['lr'] for g in optimizer.param_groups}
            metrics, counts = train_origin_epoch(model, loader, optimizer, device, training['target_weight'])
            scheduler.step()
            rows.append({'epoch': epoch, **metrics, 'learning_rates': json.dumps(rates),
                         'epoch_time_sec': time.perf_counter() - epoch_start, 'training_mode': model.active_mode})
            profiles.extend({'epoch': epoch, 'profile': name, 'count': int(count)} for name, count in zip(PROFILES, counts))
            pd.DataFrame(rows).to_csv(log_dir / 'history.csv', index=False)
            pd.DataFrame(profiles).to_csv(log_dir / 'profile_counts.csv', index=False)
            checkpoint = {'model_state_dict': model.state_dict(), 'config': config, 'epoch': epoch,
                          'class_labels': config['class_labels'], 'class_names': config['class_names'],
                          'training_splits': ['train', 'val'], 'checkpoint_selection': 'fixed_last_epoch'}
            torch.save(checkpoint, model_dir / 'last.pt')
            message = (f"{config['run_name']} {epoch}/25 | train_loss={metrics['train_loss']:.4f} "
                       f"clean/hard/val-fit={metrics['clean_accuracy']:.4f}/"
                       f"{metrics['hard_accuracy']:.4f}/{metrics['target_accuracy']:.4f} | LR={rates}")
            print(message, flush=True)
            with (log_dir / 'training.log').open('a', encoding='utf-8') as stream:
                stream.write(message + '\n')
        summary = {**metadata, 'status': 'complete', 'epochs': 25,
                   'train_time_min': (time.perf_counter() - started) / 60, 'checkpoint': 'last.pt'}
        (log_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (log_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return model_dir / 'last.pt', summary


def train_models(raw_dir, run_id, *, project_root=PROJECT_ROOT, names=('agmfa', 'resnet50'),
                 batch_size=16, device='auto', imagenet_weights=None):
    root, raw = Path(project_root).resolve(), Path(raw_dir).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,60}', run_id):
        raise ValueError('Invalid run_id')
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if not names or len(set(names)) != len(names) or not set(names) <= {'agmfa', 'resnet50'}:
        raise ValueError('Select unique final model names')
    classes = read_classes(raw).sort_values('label').reset_index(drop=True)
    configs = {name: load_config(name) for name in names}
    paths = {}
    for name, config in configs.items():
        validate_config(config)
        exp_id = f'{run_id}_{name}'
        validate_journal(root / 'experiments.csv', exp_id)
        paths[name] = {kind: root / 'artifacts' / kind / exp_id for kind in ('models', 'logs')}
        if any(p.exists() for p in paths[name].values()):
            raise FileExistsError(f'Run {exp_id} already exists; choose a new run_id')
    device = get_device(device)
    if device.type == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    checkpoints = {}
    for name, config in configs.items():
        config = copy.deepcopy(config)
        config.update(run_name=name, class_labels=classes.label.tolist(), class_names=classes.class_name.tolist())
        config['data'].update(raw_dir=str(raw), batch_size=batch_size)
        config['training']['device'] = str(device)
        set_seed(config['seed'], config['training']['deterministic'], config['training']['deterministic_warn_only'])
        loader = make_fit_loader(config, raw, classes, device)
        model = build_model(name, config, len(classes), imagenet_weights=imagenet_weights).to(device)
        checkpoint, summary = fit_fixed(model, loader, config, device, paths[name]['models'], paths[name]['logs'])
        checkpoints[name] = checkpoint
        append_experiment(root / 'experiments.csv', {
            'exp_id': f'{run_id}_{name}', 'model': config['model']['name'], 'image_size': 256,
            'augmentation': 'HardShift paired D4; former val D4, CE weight 2', 'pretrained': True,
            'val_accuracy': '', 'train_time_min': summary['train_time_min'], 'seed': config['seed'],
            'submission_id': '', 'public_score': '', 'notes': 'Fresh ImageNet V2; fixed epoch 25 on train+val; no held-out evaluation.'})
        del model, loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return checkpoints
