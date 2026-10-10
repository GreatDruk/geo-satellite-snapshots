"""Geometry-4 inference and fixed ensemble export; no target labels or scoring."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .ensemble import blend, probabilities
from .models.agmfa import build_agmfa
from .models.pretrained import build_pretrained
from .transforms import ImageTransform
from .training import CONFIG_DIR
from .utils import get_device, set_seed


class Images(Dataset):
    def __init__(self, ids, image_dir, config):
        self.ids = np.asarray(ids, dtype=str)
        self.directory = Path(image_dir).resolve()
        self.transform = ImageTransform(config)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        path = (self.directory / self.ids[index]).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError('Image ID escapes its directory')
        with Image.open(path) as image:
            return self.transform(image)


def load_model(name, checkpoint_path, device):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (saved['epoch'] != 25 or saved.get('training_splits') != ['train', 'val']
            or saved.get('checkpoint_selection') != 'fixed_last_epoch'):
        raise ValueError('Inference requires a completed fixed-epoch-25 train+val checkpoint')
    config = saved['config']
    expected = 'resnet50_agmfa' if name == 'agmfa' else 'resnet50'
    if config['model']['name'] != expected:
        raise ValueError('Checkpoint architecture mismatch')
    labels, names = saved['class_labels'], saved['class_names']
    if labels != config['class_labels'] or names != config['class_names']:
        raise ValueError('Checkpoint class order mismatch')
    factory = build_agmfa if name == 'agmfa' else build_pretrained
    model = (factory(config['model'], 256, len(labels), initialize_pretrained=False) if name == 'agmfa'
             else factory(config['model'], len(labels), initialize_pretrained=False))
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model.to(device).requires_grad_(False).eval(), saved


def geometry4(model, inputs):
    """Mean of softmax probabilities at 0, 90, 180 and 270 degrees."""
    return torch.stack([model(torch.rot90(inputs, k, (-2, -1))).softmax(1).double()
                        for k in range(4)]).mean(0)


def predict_models(checkpoints, ids, image_dir, *, batch_size=8, device='auto'):
    device = get_device(device)
    set_seed(42, deterministic=True, warn_only=True)
    outputs, labels, names = {}, None, None
    for name in ('agmfa', 'resnet50'):
        model, saved = load_model(name, checkpoints[name], device)
        current_labels, current_names = saved['class_labels'], saved['class_names']
        if labels is None:
            labels, names = current_labels, current_names
        elif current_labels != labels or current_names != names:
            raise ValueError('Checkpoints must use identical class order')
        loader = DataLoader(Images(ids, image_dir, saved['config']['data']), batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
        chunks = []
        with torch.inference_mode():
            for inputs in loader:
                chunks.append(geometry4(model, inputs.to(device)).cpu().numpy())
        outputs[name] = probabilities(np.concatenate(chunks))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return outputs, np.asarray(labels), names


def export_predictions(checkpoints, sample_csv, image_dir, output_dir, *, batch_size=8, device='auto'):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Output directory exists; choose a new directory')
    # Read IDs only. A label column in the template is never used as ground truth.
    ids = pd.read_csv(sample_csv, usecols=['id'], dtype={'id': str}).id.to_numpy()
    if not len(ids) or pd.isna(ids).any() or len(set(ids)) != len(ids):
        raise ValueError('Sample requires nonempty unique IDs')
    values, labels, names = predict_models(checkpoints, ids, image_dir, batch_size=batch_size, device=device)
    config = json.loads((CONFIG_DIR / 'ensemble.json').read_text(encoding='utf-8'))
    values['ensemble'] = blend(values['agmfa'], values['resnet50'], names, config)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {}
    for name, p in values.items():
        np.savez_compressed(output_dir / f'{name}_probabilities.npz', ids=ids, class_labels=labels, probabilities=p)
        paths[name] = output_dir / f'{name}.csv'
        pd.DataFrame({'id': ids, 'label': labels[p.argmax(1)]}).to_csv(paths[name], index=False)
    (output_dir / 'recipe.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    return paths
