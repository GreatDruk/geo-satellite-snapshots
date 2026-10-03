"""Configured PIL/tensor transforms; fixed normalization, no fitting on test."""
import random
import numpy as np
import torch
from PIL import Image


class ImageTransform:
    def __init__(self, config, training=False):
        size = config["image_size"]
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if len(self.size) != 2 or any(x <= 0 for x in self.size):
            raise ValueError("image_size must be positive or [height, width]")
        normalization = config.get("normalization", {})
        self.mean = torch.tensor(normalization.get("mean", [0.5] * 3)).view(3, 1, 1)
        self.std = torch.tensor(normalization.get("std", [0.5] * 3)).view(3, 1, 1)
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Normalization needs finite mean and positive std")
        self.augmentation = config.get("augmentation", {}) if training else {}
        unknown = set(self.augmentation) - {"horizontal_flip", "vertical_flip", "rotation_degrees"}
        if unknown:
            raise ValueError(f"Unsupported augmentations: {unknown}")
        for key in ("horizontal_flip", "vertical_flip"):
            if not 0 <= self.augmentation.get(key, 0) <= 1:
                raise ValueError(f"{key} must be a probability")
        if self.augmentation.get("rotation_degrees", 0) < 0:
            raise ValueError("rotation_degrees must be nonnegative")

    def __call__(self, image):
        target_size = (self.size[1], self.size[0])
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.BILINEAR)
        for key, operation in (("horizontal_flip", Image.Transpose.FLIP_LEFT_RIGHT),
                               ("vertical_flip", Image.Transpose.FLIP_TOP_BOTTOM)):
            probability = self.augmentation.get(key, 0)
            if probability and random.random() < probability:
                image = image.transpose(operation)
        degrees = self.augmentation.get("rotation_degrees", 0)
        if degrees:
            image = image.rotate(random.uniform(-degrees, degrees), resample=Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
        return (tensor - self.mean) / self.std


def get_transforms(config):
    return ImageTransform(config, True), ImageTransform(config)
