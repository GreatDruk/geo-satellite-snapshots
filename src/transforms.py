"""Train-only PIL/NumPy augmentations and fixed normalization for every split."""
import io
import math
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


def _number(value, name, low=0, high=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return value


def _interval(value, name, low=0, high=math.inf, integer=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [minimum, maximum]")
    for item in value:
        _number(item, name, low, high)
        if integer and not isinstance(item, int):
            raise ValueError(f"{name} needs integers")
    if value[0] > value[1]:
        raise ValueError(f"{name} minimum exceeds maximum")
    return tuple(value)


class ImageTransform:
    """Preserve the full frame; stochastic operations are enabled only on train."""

    OPERATIONS = {
        "color_jitter": {"p", "brightness", "contrast", "saturation", "hue"},
        "gamma": {"p", "range"},
        "gaussian_noise": {"p", "std"},
        "gaussian_blur": {"p", "sigma"},
        "jpeg_compression": {"p", "quality"},
        "resolution_degradation": {"p", "scale"},
    }
    RANGES = {
        "gamma": ("range", [0.85, 1.15], 0.01, 10, False),
        "gaussian_noise": ("std", [0.005, 0.025], 0, 1, False),
        "gaussian_blur": ("sigma", [0.2, 1.0], 0, 20, False),
        "jpeg_compression": ("quality", [60, 95], 1, 100, True),
        "resolution_degradation": ("scale", [0.65, 1.0], 0.01, 1, False),
    }
    CORRUPTIONS = ("resolution_degradation", "gaussian_blur", "jpeg_compression", "gaussian_noise")

    def __init__(self, config, training=False):
        size = config["image_size"]
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if len(self.size) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in self.size):
            raise ValueError("image_size must be a positive integer or [height, width]")
        normalization = config.get("normalization", {})
        self.mean = torch.tensor(normalization.get("mean", [0.5] * 3)).view(3, 1, 1)
        self.std = torch.tensor(normalization.get("std", [0.5] * 3)).view(3, 1, 1)
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Normalization needs finite mean and positive std")
        augmentation = config.get("augmentation", {})
        if not isinstance(augmentation, dict):
            raise ValueError("augmentation must be an object")
        allowed = set(self.OPERATIONS) | {"horizontal_flip", "vertical_flip", "rotation_degrees", "random_rotate90", "corruption_probability", "max_corruptions"}
        if set(augmentation) - allowed:
            raise ValueError(f"Unsupported augmentations: {set(augmentation) - allowed}")
        for key in ("horizontal_flip", "vertical_flip", "random_rotate90", "corruption_probability"):
            _number(augmentation.get(key, 1 if key == "corruption_probability" else 0), key, 0, 1)
        _number(augmentation.get("rotation_degrees", 0), "rotation_degrees", 0, 180)
        if augmentation.get("random_rotate90", 0) and self.size[0] != self.size[1]:
            raise ValueError("random_rotate90 requires square image_size")
        budget = augmentation.get("max_corruptions", len(self.CORRUPTIONS))
        if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= len(self.CORRUPTIONS):
            raise ValueError("max_corruptions must be an integer from 1 to 4")
        for name, keys in self.OPERATIONS.items():
            if name not in augmentation:
                continue
            spec = augmentation[name]
            if not isinstance(spec, dict) or set(spec) - keys:
                raise ValueError(f"Invalid settings for {name}; supported: {sorted(keys)}")
            _number(spec.get("p", 0), f"{name}.p", 0, 1)
            if name == "color_jitter":
                for key in ("brightness", "contrast", "saturation", "hue"):
                    _number(spec.get(key, 0), key, 0, 0.5 if key == "hue" else 1)
            else:
                key, default, low, high, integer = self.RANGES[name]
                _interval(spec.get(key, default), f"{name}.{key}", low, high, integer)
        self.augmentation = augmentation if training else {}

    def _enabled(self, name):
        probability = self.augmentation.get(name, {}).get("p", 0)
        return probability > 0 and random.random() < probability

    def _range(self, name):
        key, default, *_ = self.RANGES[name]
        return self.augmentation[name].get(key, default)

    def _color(self, image):
        spec = self.augmentation["color_jitter"]
        operations = [key for key in ("brightness", "contrast", "saturation", "hue") if spec.get(key, 0)]
        random.shuffle(operations)
        for key in operations:
            amount = spec[key]
            if key == "hue":
                hsv = np.array(image.convert("HSV"), dtype=np.uint8, copy=True)
                shift = round(random.uniform(-amount, amount) * 255)
                hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
                image = Image.fromarray(hsv, mode="HSV").convert("RGB")
            else:
                enhancer = {"brightness": ImageEnhance.Brightness, "contrast": ImageEnhance.Contrast, "saturation": ImageEnhance.Color}[key]
                image = enhancer(image).enhance(random.uniform(1 - amount, 1 + amount))
        return image

    def __call__(self, image):
        image = image.convert("RGB")
        target_size = (self.size[1], self.size[0])
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.BILINEAR)
        for key, operation in (("horizontal_flip", Image.Transpose.FLIP_LEFT_RIGHT), ("vertical_flip", Image.Transpose.FLIP_TOP_BOTTOM)):
            if self.augmentation.get(key, 0) and random.random() < self.augmentation[key]:
                image = image.transpose(operation)
        if self.augmentation.get("random_rotate90", 0) and random.random() < self.augmentation["random_rotate90"]:
            turns = random.randrange(4)
            if turns:
                image = image.transpose((Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)[turns - 1])
        degrees = self.augmentation.get("rotation_degrees", 0)
        if degrees:
            # Reflection padding avoids black corners in legacy arbitrary rotations.
            width, height = image.size
            border = math.ceil(math.hypot(width, height) / 2)
            padded = Image.fromarray(np.pad(np.asarray(image), ((border, border), (border, border), (0, 0)), mode="reflect"))
            padded = padded.rotate(random.uniform(-degrees, degrees), resample=Image.Resampling.BILINEAR)
            image = padded.crop((border, border, border + width, border + height))
        if self._enabled("color_jitter"):
            image = self._color(image)
        if self._enabled("gamma"):
            gamma = random.uniform(*self._range("gamma"))
            image = image.point([round(255 * (value / 255) ** gamma) for value in range(256)] * 3)
        candidates = []
        if any(name in self.augmentation for name in self.CORRUPTIONS) and random.random() < self.augmentation.get("corruption_probability", 1):
            candidates = [name for name in self.CORRUPTIONS if self._enabled(name)]
            random.shuffle(candidates)
            candidates = candidates[:self.augmentation.get("max_corruptions", len(self.CORRUPTIONS))]
        if "resolution_degradation" in candidates:
            scale = random.uniform(*self._range("resolution_degradation"))
            reduced = tuple(max(1, round(length * scale)) for length in image.size)
            image = image.resize(reduced, Image.Resampling.BILINEAR).resize(target_size, Image.Resampling.BILINEAR)
        if "gaussian_blur" in candidates:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(*self._range("gaussian_blur"))))
        if "jpeg_compression" in candidates:
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG", quality=random.randint(*self._range("jpeg_compression")))
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    image = decoded.convert("RGB")
        array = np.array(image, dtype=np.float32, copy=True) / 255
        if "gaussian_noise" in candidates:
            std = random.uniform(*self._range("gaussian_noise"))
            array = np.clip(array + np.random.normal(0, std, array.shape).astype(np.float32), 0, 1)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


def get_transforms(config):
    return ImageTransform(config, True), ImageTransform(config)


def preview_augmentations(image, config, count=5, seed=42):
    """Show a PIL image and train variants without changing Python/NumPy RNG state."""
    import matplotlib.pyplot as plt

    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 12:
        raise ValueError("count must be an integer from 1 to 12")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        transform = ImageTransform(config, training=True)
        fig, axes = plt.subplots(1, count + 1, figsize=(3 * (count + 1), 3))
        axes[0].imshow(image)
        axes[0].set_title("Original train")
        for index in range(count):
            tensor = transform(image) * transform.std + transform.mean
            axes[index + 1].imshow(tensor.permute(1, 2, 0).numpy().clip(0, 1))
            axes[index + 1].set_title(f"Train variant {index + 1}")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        return fig
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
