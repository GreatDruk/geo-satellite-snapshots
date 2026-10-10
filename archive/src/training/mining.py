"""Previous-epoch train margins; each original sample stays in the next epoch."""
import math

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .losses import finite_number


class IndexedDataset(Dataset):
    """Add stable dataset indices only to the training loader used for mining."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = dataset.data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return image, target, index


class HardExampleSampler(Sampler):
    """Shuffle all N examples plus at most repeat_fraction*N hard repetitions.

    Difficulty is average p(true)-max p(other) over augmented train views in
    the previous epoch. No val/test probabilities or labels are accepted.
    """
    def __init__(self, dataset, config, seed):
        options = dict(config)
        if set(options) - {"warmup_epochs", "repeat_fraction"}:
            raise ValueError("hard_mining supports warmup_epochs,repeat_fraction")
        warmup = options.get("warmup_epochs", 5)
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
            raise ValueError("hard_mining.warmup_epochs must be a positive integer")
        self.fraction = finite_number(options.get("repeat_fraction", 0.05), "hard_mining.repeat_fraction", 0.001, 0.25)
        self.dataset, self.seed, self.warmup = dataset, seed, warmup
        self.repeat_count = math.floor(len(dataset) * self.fraction)
        self.hard_indices = np.array([], dtype=np.int64)
        self.set_epoch(1)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.sums = np.zeros(len(self.dataset), dtype=np.float64)
        self.counts = np.zeros(len(self.dataset), dtype=np.int64)

    def __len__(self):
        return len(self.dataset) + (len(self.hard_indices) if self.epoch > self.warmup else 0)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.epoch > self.warmup:
            indices.extend(self.hard_indices.tolist())
        order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(self.seed + self.epoch))
        return iter([indices[position] for position in order.tolist()])

    def record(self, indices, logits, labels):
        probabilities = logits.detach().float().softmax(1)
        true = probabilities.gather(1, labels[:, None]).squeeze(1)
        other = probabilities.clone()
        other.scatter_(1, labels[:, None], -1)
        margins = (true - other.max(1).values).cpu().numpy()
        indices = indices.cpu().numpy()
        np.add.at(self.sums, indices, margins)
        np.add.at(self.counts, indices, 1)

    def finish_epoch(self):
        if (self.counts == 0).any():
            raise ValueError("Mining must observe every original train sample")
        margins = self.sums / self.counts
        self.hard_indices = np.argsort(margins, kind="stable")[:self.repeat_count]
        result = self.dataset.data[["id", "label"]].copy()
        result["margin"] = margins
        result["observations"] = self.counts
        result["repeat_next_epoch"] = np.isin(np.arange(len(result)), self.hard_indices) & (self.epoch >= self.warmup)
        return result
