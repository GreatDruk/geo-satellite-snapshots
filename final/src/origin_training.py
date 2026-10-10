"""Original HardShift training mechanics."""
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from .data import ImageDataset
from .transforms import ImageTransform,apply_d4


class TargetDomainDataset(ImageDataset):
    """Former val: shared epoch, uniform D4 ONLY, no synthetic corruptions."""
    def __init__(self, raw, data, mapping, seed):
        super().__init__(Path(raw)/'val.csv', Path(raw)/'val', label_to_index=mapping)
        self.tensor = ImageTransform(data, training=False)
        self.seed = seed
        self._epoch = torch.tensor(1, dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch):
        self._epoch.fill_(epoch)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(self._epoch), index, 1]))
        image = image.resize((256,256), Image.Resampling.BILINEAR)
        return self.tensor(apply_d4(image, int(rng.integers(8)))), label

class OriginDataset(Dataset):
    def __init__(self, source, target=None):
        self.source, self.target = source, target

    def __len__(self):
        return len(self.source) + (len(self.target) if self.target is not None else 0)

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)
        if self.target is not None:
            self.target.set_epoch(epoch)

    def __getitem__(self, index):
        if index < len(self.source):
            clean, hard, label, profile, local_index = self.source[index]
            return {'origin':'source', 'clean':clean, 'hard':hard, 'label':label,
                    'profile':profile, 'index':local_index}
        local_index = index-len(self.source)
        image, label = self.target[local_index]
        return {'origin':'target', 'clean':image, 'label':label, 'profile':-1, 'index':local_index}

def origin_collate(items):
    source = [item for item in items if item['origin']=='source']
    target = [item for item in items if item['origin']=='target']
    # No placeholder hard image for target: it receives exactly ONE forward.
    result = {}
    for name, rows in (('source',source), ('target',target)):
        result[name+'_labels'] = torch.tensor([r['label'] for r in rows],dtype=torch.long)
        result[name+'_indices'] = torch.tensor([r['index'] for r in rows],dtype=torch.long)
        result[name+'_images'] = torch.stack([r['clean'] for r in rows]) if rows else None
    result['source_hard'] = torch.stack([r['hard'] for r in source]) if source else None
    result['source_profiles'] = torch.tensor([r['profile'] for r in source],dtype=torch.long)
    return result

def source_object_losses(clean, hard, labels):
    clean_ce, hard_ce = F.cross_entropy(clean,labels,reduction='none'), F.cross_entropy(hard,labels,reduction='none')
    log_p, log_q = F.log_softmax(clean.float(),1), F.log_softmax(hard.float(),1)
    log_m = torch.logaddexp(log_p,log_q)-math.log(2)
    js = .5*((log_p.exp()*(log_p-log_m)).sum(1)+(log_q.exp()*(log_q-log_m)).sum(1))
    return .5*clean_ce+.5*hard_ce+3*js, {'clean_ce':clean_ce,'hard_ce':hard_ce,'js_loss':js}

def mixed_objective(clean, hard, source_labels, target_logits, target_labels, target_weight=2.):
    """Exact weighted per-object numerator, with original-object denominator."""
    n_source, n_target = len(source_labels), len(target_labels)
    if not n_source and not n_target:
        raise ValueError('Empty batch')
    terms = {}
    numerator = None
    if n_source:
        losses, terms = source_object_losses(clean,hard,source_labels)
        numerator = losses.sum()
    if n_target:
        terms['target_ce'] = F.cross_entropy(target_logits,target_labels,reduction='none')
        target_sum = target_weight*terms['target_ce'].sum()
        numerator = target_sum if numerator is None else numerator+target_sum
    mass = n_source+target_weight*n_target
    return numerator/mass, terms, mass

def train_origin_epoch(model, loader, optimizer, device, target_weight=2.):
    model.train()
    totals = dict(clean_ce=0.,hard_ce=0.,js_loss=0.,clean_accuracy=0.,hard_accuracy=0.,target_ce=0.,target_accuracy=0.)
    source_seen = np.zeros(len(loader.dataset.source),dtype=int)
    target_seen = np.zeros(len(loader.dataset.target) if loader.dataset.target is not None else 0,dtype=int)
    profile_counts = np.zeros(6,dtype=int)
    numerator_sum, mass_sum, forward_count = 0.,0.,0
    for batch in loader:
        ns, nt = len(batch['source_labels']),len(batch['target_labels'])
        non_blocking = device.type=='cuda' and loader.pin_memory
        images = [batch[key] for key in ('source_images','source_hard','target_images') if batch[key] is not None]
        inputs = torch.cat(images).to(device,non_blocking=non_blocking)
        sy = batch['source_labels'].to(device,non_blocking=non_blocking)
        ty = batch['target_labels'].to(device,non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        clean,hard,target = logits[:ns],logits[ns:2*ns],logits[2*ns:]
        loss,terms,mass = mixed_objective(clean,hard,sy,target,ty,target_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite final hardshift loss')
        loss.backward()
        optimizer.step()
        numerator_sum += loss.item()*mass
        mass_sum += mass
        forward_count += len(inputs)
        for key,value in terms.items():
            totals[key] += value.detach().sum().item()
        if ns:
            totals['clean_accuracy'] += (clean.argmax(1)==sy).sum().item()
            totals['hard_accuracy'] += (hard.argmax(1)==sy).sum().item()
            np.add.at(source_seen,batch['source_indices'].numpy(),1)
            profile_counts += np.bincount(batch['source_profiles'].numpy(),minlength=6)
        if nt:
            totals['target_accuracy'] += (target.argmax(1)==ty).sum().item()
            np.add.at(target_seen,batch['target_indices'].numpy(),1)
    if not len(source_seen) or not (source_seen==1).all() or not (target_seen==1).all():
        raise AssertionError('Every source/target object must appear exactly once per epoch')
    source_count,target_count = len(source_seen),len(target_seen)
    assert forward_count==2*source_count+target_count
    metrics = {key:totals[key]/source_count for key in ('clean_ce','hard_ce','js_loss','clean_accuracy','hard_accuracy')}
    metrics.update(train_loss=numerator_sum/mass_sum,source_samples=source_count,target_samples=target_count,
                   forward_images=forward_count,target_ce=totals['target_ce']/target_count if target_count else None,
                   target_accuracy=totals['target_accuracy']/target_count if target_count else None,
                   weighted_target_fraction=target_weight*target_count/(source_count+target_weight*target_count))
    return metrics,profile_counts
