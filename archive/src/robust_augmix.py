"""Full-frame AugMix mixing with a configurable photometric/corruption pool.

Dirichlet branch weights and Beta clean/augmented mixing follow AugMix
(Hendrycks et al., 2020). This pool deliberately differs from the paper's
original operators. No JSD, geometric operations or extra model forwards.
"""
import copy
import io
import math

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


DEFAULT_OPERATIONS = {
    "brightness": {"range": [0.35, 1.6]},
    "contrast": {"range": [0.4, 1.6]},
    "gamma": {"range": [0.45, 2.2]},
    "saturation": {"range": [0.0, 1.8]},
    "hue": {"range": [-0.08, 0.08]},
    "grayscale": {"range": [0.4, 1.0]},
    "white_balance": {"range": [0.6, 1.4]},
    "autocontrast": {}, "equalize": {},
    "solarize": {"range": [64, 192], "weight": 0.15},
    "invert": {"weight": 0.1},
    "gaussian_blur": {"range": [0.4, 2.5]},
    "motion_blur": {"range": [3, 13]},
    "defocus_blur": {"range": [1, 4]},
    "resolution_degradation": {"range": [0.25, 0.8]},
    "jpeg_compression": {"range": [25, 90]},
    "gaussian_noise": {"range": [0.005, 0.06]},
    "shot_noise": {"range": [20, 120]},
}
# Integer ranges: thresholds, kernel sizes, disk radii, JPEG quality.
LIMITS = {
    "brightness": (0.01, 4, False), "contrast": (0.01, 4, False),
    "gamma": (0.01, 5, False), "saturation": (0, 4, False),
    "hue": (-0.5, 0.5, False), "grayscale": (0, 1, False),
    "white_balance": (0.01, 4, False), "solarize": (0, 255, True),
    "gaussian_blur": (0, 10, False), "motion_blur": (3, 13, True),
    "defocus_blur": (1, 4, True), "resolution_degradation": (0.01, 1, False),
    "jpeg_compression": (1, 100, True), "gaussian_noise": (0, 1, False),
    "shot_noise": (1, 10000, False),
}


def _image(array):
    return Image.fromarray(np.rint(np.clip(array, 0, 255)).astype(np.uint8))


def motion_kernel(size, angle):
    """Normalized line PSF at any direction, bilinearly rasterized on an odd grid."""
    if size not in range(3, 14, 2):
        raise ValueError("Motion kernel must be odd and between 3 and 13")
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    radians = math.radians(angle)
    for distance in np.linspace(-center, center, 4 * size + 1):
        x, y = center + distance * math.cos(radians), center + distance * math.sin(radians)
        x0, y0 = math.floor(x), math.floor(y)
        for yy in (y0, y0 + 1):
            for xx in (x0, x0 + 1):
                if 0 <= yy < size and 0 <= xx < size:
                    kernel[yy, xx] += max(0, 1 - abs(y - yy)) * max(0, 1 - abs(x - xx))
    return kernel / kernel.sum()


def _filter_psf(image, kernel):
    """Small RGB convolution using reflected boundaries; no scipy dependency."""
    array = np.asarray(image, dtype=np.float32)
    radius = kernel.shape[0] // 2
    padded = np.pad(array, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    result = np.zeros_like(array)
    height, width = array.shape[:2]
    for y, x in np.argwhere(kernel > 0):
        result += kernel[y, x] * padded[y:y + height, x:x + width]
    return _image(result)


class RobustAugMix:
    """Return same-size RGB PIL image; parameters are fixed independently of test."""

    def __init__(self, config):
        if not isinstance(config, dict) or set(config) - {"width", "depth", "alpha", "operations"}:
            raise ValueError("Robust AugMix settings: width, depth, alpha, operations")
        self.width = config.get("width", 3)
        self.depth = config.get("depth", [1, 3])
        self.alpha = config.get("alpha", 1.0)
        if isinstance(self.width, bool) or not isinstance(self.width, int) or not 1 <= self.width <= 8:
            raise ValueError("AugMix width must be an integer from 1 to 8")
        if (not isinstance(self.depth, (list, tuple)) or len(self.depth) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in self.depth)
                or not 1 <= self.depth[0] <= self.depth[1] <= 3):
            raise ValueError("AugMix depth must be [min,max] within 1..3")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("AugMix alpha must be finite and positive")
        self.operations = copy.deepcopy(config.get("operations", DEFAULT_OPERATIONS))
        if not isinstance(self.operations, dict) or not self.operations or set(self.operations) - set(DEFAULT_OPERATIONS):
            raise ValueError("Unknown or empty Robust AugMix operation pool")
        for name, spec in self.operations.items():
            allowed = {"weight", "range"} if name in LIMITS else {"weight"}
            if not isinstance(spec, dict) or set(spec) - allowed:
                raise ValueError(f"Invalid AugMix operation settings: {name}")
            weight = spec.get("weight", 1.0)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"{name}.weight must be positive")
            if name in LIMITS:
                low, high, integer = LIMITS[name]
                interval = spec.get("range", DEFAULT_OPERATIONS[name]["range"])
                if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                        or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                               or not low <= x <= high or (integer and not isinstance(x, int)) for x in interval)
                        or interval[0] > interval[1]):
                    raise ValueError(f"Invalid {name}.range")
                if name == "motion_blur" and any(x % 2 == 0 for x in interval):
                    raise ValueError("motion_blur.range endpoints must be odd")
                spec["range"] = list(interval)
        self.names = list(self.operations)
        weights = np.array([self.operations[name].get("weight", 1.0) for name in self.names], dtype=np.float64)
        self.weights = weights / weights.sum()

    def apply_operation(self, image, name):
        """Single operation for diagnostics; no geometry, label or size changes."""
        if name not in self.operations:
            raise ValueError(f"Operation is not enabled: {name}")
        image = image.convert("RGB")
        interval = self.operations[name].get("range")
        continuous = {"brightness", "contrast", "saturation", "gamma", "hue", "grayscale",
                      "gaussian_blur", "resolution_degradation", "gaussian_noise", "shot_noise"}
        value = float(np.random.uniform(*interval)) if name in continuous else None
        if name in {"brightness", "contrast", "saturation"}:
            enhancer = {"brightness": ImageEnhance.Brightness, "contrast": ImageEnhance.Contrast,
                        "saturation": ImageEnhance.Color}[name]
            return enhancer(image).enhance(value)
        if name == "gamma":
            lut = np.rint(255 * (np.arange(256) / 255) ** value).astype(np.uint8).tolist()
            return image.point(lut * 3)
        if name == "hue":
            hsv = np.array(image.convert("HSV"), copy=True)
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + round(value * 255)) % 256
            return Image.fromarray(hsv, mode="HSV").convert("RGB")
        if name == "grayscale":
            return Image.blend(image, ImageOps.grayscale(image).convert("RGB"), value)
        if name == "white_balance":
            gains = np.random.uniform(*interval, size=3).astype(np.float32)
            return _image(np.asarray(image, dtype=np.float32) * gains)
        if name == "autocontrast":
            return ImageOps.autocontrast(image)
        if name == "equalize":
            return ImageOps.equalize(image)
        if name == "solarize":
            return ImageOps.solarize(image, int(np.random.randint(interval[0], interval[1] + 1)))
        if name == "invert":
            return ImageOps.invert(image)
        if name == "gaussian_blur":
            return image.filter(ImageFilter.GaussianBlur(value))
        if name == "motion_blur":
            size = int(np.random.choice(np.arange(interval[0], interval[1] + 1, 2)))
            return _filter_psf(image, motion_kernel(size, float(np.random.uniform(0, 180))))
        if name == "defocus_blur":
            radius = int(np.random.randint(interval[0], interval[1] + 1))
            y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
            kernel = (x * x + y * y <= radius * radius).astype(np.float32)
            return _filter_psf(image, kernel / kernel.sum())
        if name == "resolution_degradation":
            small = tuple(max(1, round(length * value)) for length in image.size)
            return image.resize(small, Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
        if name == "jpeg_compression":
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG", quality=int(np.random.randint(interval[0], interval[1] + 1)))
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    return decoded.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255
        if name == "gaussian_noise":
            array = array + np.random.normal(0, value, array.shape).astype(np.float32)
        elif name == "shot_noise":
            array = np.random.poisson(array * value).astype(np.float32) / value
        else:
            raise ValueError(f"Unimplemented AugMix operation: {name}")
        return _image(array * 255)

    def __call__(self, image):
        image = image.convert("RGB")
        clean = np.asarray(image, dtype=np.float32) / 255
        weights = np.random.dirichlet([self.alpha] * self.width).astype(np.float32)
        coefficient = float(np.random.beta(self.alpha, self.alpha))
        mixed = np.zeros_like(clean)
        for weight in weights:
            branch = image.copy()
            depth = int(np.random.randint(self.depth[0], self.depth[1] + 1))
            for _ in range(depth):
                name = str(np.random.choice(self.names, p=self.weights))
                branch = self.apply_operation(branch, name)
            mixed += weight * (np.asarray(branch, dtype=np.float32) / 255)
        return _image(255 * ((1 - coefficient) * clean + coefficient * mixed))


def preview_robust_recipe(image, data_config, count=11, seed=42):
    """Original + 8..12 train variants on a compact grid; preserve Python/NumPy RNG."""
    import random
    import matplotlib.pyplot as plt
    from archive.src.transforms import ImageTransform

    if isinstance(count, bool) or not isinstance(count, int) or not 8 <= count <= 12:
        raise ValueError("Preview count must be an integer from 8 to 12")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        transform = ImageTransform(data_config, True)
        rows = math.ceil((count + 1)/4)
        figure, axes = plt.subplots(rows, 4, figsize=(12, 3*rows), squeeze=False)
        axes = axes.ravel()
        axes[0].imshow(image)
        axes[0].set_title("Original train")
        for index in range(count):
            tensor = transform(image)*transform.std + transform.mean
            axes[index+1].imshow(tensor.permute(1, 2, 0).numpy().clip(0, 1))
            axes[index+1].set_title(f"D4 + AugMix #{index+1}")
        for axis in axes:
            axis.axis('off')
        figure.tight_layout()
        return figure
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
