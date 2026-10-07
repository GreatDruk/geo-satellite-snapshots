"""Deterministic paired full-frame views: one shared D4 plus a mandatory profile.

No global RNG, augmentation probability gates, or clean/augmented mixing.
Only PIL, NumPy and torch are used. Existing augmentation recipes are untouched.
"""
import io
import math

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.data import ImageDataset
from src.transforms import ImageTransform, apply_d4
from src.robust_augmix import _filter_psf

PROFILES = ("dark_gray", "blur_lowres", "dusk_color_cast", "haze_lowcontrast",
            "off_nadir", "artifact_noise")


def profile_id(index, epoch):
    return (int(index) + int(epoch)) % len(PROFILES)


def _image(array):
    return Image.fromarray(np.rint(np.clip(array, 0, 255)).astype(np.uint8))


def _jpeg(image, quality):
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


def _gamma(image, gamma):
    lut = np.rint(255 * (np.arange(256) / 255) ** gamma).astype(np.uint8).tolist()
    return image.point(lut * 3)


def _motion_kernel(size, angle):
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


def off_nadir(image, rng):
    """Fit the WHOLE warped frame inside output; reflect-pad the uncovered area.

Forward corner fit prevents object truncation. Inverse sampling is bilinear;
there is no crop, black fill, cv2, or scipy dependency.
"""
    width, height = image.size
    angle = float(rng.choice([-1, 1]) * rng.uniform(8, 20))
    shear = float(rng.choice([-1, 1]) * rng.uniform(5, 12))
    radians = math.radians(angle)
    rotation = np.array([[math.cos(radians), -math.sin(radians), 0],
                         [math.sin(radians), math.cos(radians), 0], [0, 0, 1]])
    skew = np.array([[1, math.tan(math.radians(shear)), 0], [0, 1, 0], [0, 0, 1]])
    perspective = np.eye(3)
    perspective[2, :2] = rng.choice([-1, 1], size=2) * rng.uniform(.025, .075, size=2)
    # Work in normalized coordinates, then fit all four projected corners.
    centered = np.array([[2 / (width - 1), 0, -1], [0, 2 / (height - 1), -1], [0, 0, 1]])
    mapping = perspective @ rotation @ skew @ centered
    corners = np.array([[0, 0, 1], [width-1, 0, 1], [width-1, height-1, 1], [0, height-1, 1]]).T
    warped = mapping @ corners
    projected = warped[:2] / warped[2:]
    low, high = projected.min(1), projected.max(1)
    scale = min((width - 9) / (high[0] - low[0]), (height - 9) / (high[1] - low[1]))
    center = (low + high) / 2
    fit = np.array([[scale, 0, (width-1)/2 - scale*center[0]],
                    [0, scale, (height-1)/2 - scale*center[1]], [0, 0, 1]])
    mapping = fit @ mapping
    final_corners = mapping @ corners
    final_corners = (final_corners[:2] / final_corners[2:]).T
    yy, xx = np.mgrid[:height, :width]
    sampled = np.linalg.inv(mapping) @ np.stack((xx.ravel(), yy.ravel(), np.ones(width*height)))
    x, y = (sampled[:2] / sampled[2:]).reshape(2, height, width)
    def reflected(values, size):
        values = np.mod(values, 2 * (size-1))
        return np.minimum(values, 2 * (size-1) - values)
    x, y = reflected(x, width), reflected(y, height)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    wx, wy = (x-x0)[..., None], (y-y0)[..., None]
    array = np.asarray(image, dtype=np.float32)
    result = ((1-wy)*((1-wx)*array[y0, x0] + wx*array[y0, x1])
              + wy*((1-wx)*array[y1, x0] + wx*array[y1, x1]))
    return _image(result), {"angle": angle, "shear": shear, "corners": final_corners.tolist()}


def apply_profile(image, name, rng):
    """Every listed operation is applied; stochasticity only controls parameters."""
    image = image.convert("RGB")
    trace = {"profile": name}
    def uniform(key, low, high):
        trace[key] = float(rng.uniform(low, high))
        return trace[key]
    def enhance(kind, low, high):
        nonlocal image
        classes = {"brightness": ImageEnhance.Brightness, "contrast": ImageEnhance.Contrast,
                   "saturation": ImageEnhance.Color}
        image = classes[kind](image).enhance(uniform(kind, low, high))
    if name == "dark_gray":
        image = Image.blend(image, ImageOps.grayscale(image).convert("RGB"), uniform("grayscale", .85, 1))
        enhance("brightness", .25, .60)
        image = _gamma(image, uniform("gamma", 1.4, 2.5))
        enhance("contrast", .45, .8)
        image = _image(np.asarray(image, dtype=float) + rng.normal(0, 255*uniform("noise_std", .008, .018), (*image.size[::-1], 3)))
    elif name == "blur_lowres":
        factor = uniform("downscale", .18, .45)
        size = image.size
        image = image.resize(tuple(max(1, round(s*factor)) for s in size), Image.Resampling.BILINEAR).resize(size, Image.Resampling.BILINEAR)
        if rng.integers(2):
            trace.update(blur="motion", kernel=int(rng.choice(np.arange(7, 18, 2))))
            kernel = _motion_kernel(trace["kernel"], uniform("angle", 0, 180))
        else:
            trace.update(blur="defocus", radius=int(rng.integers(2, 6)))
            radius = trace["radius"]
            y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
            kernel = (x*x+y*y <= radius*radius).astype(np.float32)
            kernel /= kernel.sum()
        image = _filter_psf(image, kernel)
        trace["jpeg_quality"] = int(rng.integers(20, 56))
        image = _jpeg(image, trace["jpeg_quality"])
    elif name == "dusk_color_cast":
        enhance("brightness", .35, .8)
        enhance("saturation", .2, .8)
        image = _gamma(image, uniform("gamma", 1, 2))
        families = {"warm": [(1.2, 1.5), (.95, 1.3), (.5, .8)],
                    "green": [(.5, .8), (1.2, 1.5), (.6, 1)],
                    "cool": [(.5, .8), (.9, 1.2), (1.2, 1.5)]}
        family = str(rng.choice(list(families)))
        gains = [float(rng.uniform(*interval)) for interval in families[family]]
        trace.update(cast_family=family, channel_gains=gains)
        image = _image(np.asarray(image, dtype=float)*np.asarray(gains))
        image = image.filter(ImageFilter.GaussianBlur(uniform("blur", .4, 1.2)))
    elif name == "haze_lowcontrast":
        enhance("contrast", .25, .65)
        enhance("saturation", .25, .8)
        enhance("brightness", .8, 1.35)
        veil = int(rng.integers(180, 241))
        trace["veil_value"] = veil
        image = Image.blend(image, Image.new("RGB", image.size, (veil,)*3), uniform("veil_alpha", .15, .4))
        image = image.filter(ImageFilter.GaussianBlur(uniform("blur", .4, 1)))
    elif name == "off_nadir":
        image, geometry = off_nadir(image, rng)
        trace.update(geometry)
    elif name == "artifact_noise":
        array = np.asarray(image, dtype=float) / 255
        counts = uniform("shot_counts", 12, 40)
        array = rng.poisson(array*counts) / counts
        array += rng.normal(0, uniform("gaussian_std", .06, .14), array.shape)
        trace["blotch_grid"] = int(rng.integers(3, 7))
        grid = trace["blotch_grid"]
        # Smooth signed RGB blotches, not a blend with the clean original.
        low = _image(128 + rng.normal(0, 55, (grid, grid, 3)))
        field = np.asarray(low.resize(image.size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(8)), dtype=float)
        array += uniform("blotch_amplitude", .18, .35) * (field-128) / 55
        image = _image(array*255)
        trace["jpeg_quality"] = int(rng.integers(15, 51))
        image = _jpeg(image, trace["jpeg_quality"])
    else:
        raise ValueError(f"Unknown profile: {name}")
    return image, trace


class HardTargetViews:
    def __init__(self, data_config, seed=42):
        if data_config["image_size"] != 256 or data_config.get("augmentation", {}):
            raise ValueError("Hard Target Shift requires image_size256 and no old augmentation pool")
        self.seed = seed
        self.tensor = ImageTransform(data_config, training=False)

    def images(self, image, index, epoch, *, forced_profile=None):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch), int(index)]))
        d4 = int(rng.integers(8))
        clean = apply_d4(image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR), d4)
        selected = profile_id(index, epoch) if forced_profile is None else int(forced_profile)
        hard, trace = apply_profile(clean.copy(), PROFILES[selected], rng)
        trace.update(profile_id=selected, d4=d4, seed=self.seed, epoch=int(epoch), sample_index=int(index))
        return clean, hard, trace

    def __call__(self, image, index, epoch):
        clean, hard, trace = self.images(image, index, epoch)
        return self.tensor(clean), self.tensor(hard), trace["profile_id"]


class HardTargetDataset(ImageDataset):
    def __init__(self, csv_path, image_dir, data_config, label_to_index, seed=42):
        super().__init__(csv_path, image_dir, label_to_index=label_to_index)
        self.views = HardTargetViews(data_config, seed)
        # Shared tensor propagates epoch changes to persistent fork/spawn workers.
        self._epoch = torch.tensor(1, dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch):
        self._epoch.fill_(epoch)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        clean, hard, profile = self.views(image, index, int(self._epoch.item()))
        return clean, hard, label, profile, index


def preview_profiles(image, data_config, seed=42, index=0, epoch=1, save_path=None):
    import matplotlib.pyplot as plt
    views = HardTargetViews(data_config, seed)
    fig, axes = plt.subplots(1, 7, figsize=(18, 3))
    clean, _, _ = views.images(image, index, epoch)
    axes[0].imshow(clean)
    axes[0].set_title("clean (shared D4)")
    for i, name in enumerate(PROFILES):
        _, hard, _ = views.images(image, index, epoch, forced_profile=i)
        axes[i+1].imshow(hard)
        axes[i+1].set_title(name, fontsize=9)
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    return fig
