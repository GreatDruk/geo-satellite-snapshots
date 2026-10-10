"""Train-only CutMix with targets weighted by the actual clipped patch area."""
import math

import numpy as np
import torch

from .losses import finite_number


class CutMix:
    def __init__(self, config=None):
        config = dict(config or {})
        if set(config) - {"name", "p", "alpha", "max_area"} or config.get("name", "cutmix") != "cutmix":
            raise ValueError("batch_augmentation supports only cutmix: name,p,alpha,max_area")
        self.p = finite_number(config.get("p", 0), "cutmix.p", 0, 1)
        self.alpha = finite_number(config.get("alpha", 1), "cutmix.alpha", 0.001, 100)
        self.max_area = finite_number(config.get("max_area", 1), "cutmix.max_area", 0.001, 1)

    def __call__(self, images, labels):
        if self.p == 0 or len(images) < 2 or np.random.random() >= self.p:
            return images, labels, labels, 1.0
        height, width = images.shape[-2:]
        requested_area = min(1 - float(np.random.beta(self.alpha, self.alpha)), self.max_area)
        crop_height = int(height * math.sqrt(requested_area))
        crop_width = int(width * math.sqrt(requested_area))
        cy, cx = np.random.randint(height), np.random.randint(width)
        y1, y2 = max(0, cy - crop_height // 2), min(height, cy + (crop_height + 1) // 2)
        x1, x2 = max(0, cx - crop_width // 2), min(width, cx + (crop_width + 1) // 2)
        permutation = torch.randperm(len(images), device=images.device)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        retained = 1 - (y2 - y1) * (x2 - x1) / (height * width)
        return mixed, labels, labels[permutation], retained
