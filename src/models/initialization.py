"""Initialize a new fine-tuning run from trained weights, without optimizer resume."""
import hashlib
from pathlib import Path
import torch


def initialize_from_checkpoint(model, path, class_labels, data_config):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint.pop('optimizer_state_dict',None);checkpoint.pop('scheduler_state_dict',None)
    config = checkpoint['config']
    architecture = config['model'].get('architecture',config['model']['name'])
    if architecture != getattr(model,'architecture',None):
        raise ValueError("Initialization requires the same backbone architecture")
    if list(checkpoint['class_labels']) != list(class_labels):
        raise ValueError("Initialization requires identical ordered classes")
    for key in ('image_size','normalization'):
        if config['data'].get(key) != data_config.get(key):
            raise ValueError(f"Initialization data mismatch: {key}")
    source=checkpoint['model_state_dict']
    target=model.state_dict()
    key='backbone.fc.1.weight'
    expanded=False
    if key in source and key in target and source[key].shape!=target[key].shape:
        old,new=source[key],target[key]
        if new.shape[0]!=old.shape[0] or new.shape[1]!=2*old.shape[1] or config['model'].get('pooling','average')!='average':
            raise ValueError("Only average -> avgmax head expansion is supported")
        new=torch.zeros_like(new)
        new[:,:old.shape[1]]=old
        source[key]=new;expanded=True
    model.load_state_dict(source,strict=True)
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return {'checkpoint':str(path),'sha256':digest.hexdigest(),'source_epoch':checkpoint['epoch'],
            'head_expansion':expanded,'optimizer_resumed':False}
