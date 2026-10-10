"""Full-frame normalization and exact D4 pixel permutations."""
import numpy as np
import torch
from PIL import Image


def apply_d4(image, index):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 8:
        raise ValueError('D4 index must be an integer from 0 to 7')
    turns, flip = index % 4, index // 4
    if turns:
        image = image.transpose((Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                                 Image.Transpose.ROTATE_270)[turns - 1])
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else image


class ImageTransform:
    def __init__(self, config, training=False):
        if training or config['image_size'] != 256 or config.get('augmentation'):
            raise ValueError('Use explicit HardShift views with full-frame normalization')
        self.mean = torch.tensor(config['normalization']['mean']).view(3, 1, 1)
        self.std = torch.tensor(config['normalization']['std']).view(3, 1, 1)

    def __call__(self, image):
        image = image.convert('RGB')
        if image.size != (256, 256):
            image = image.resize((256, 256), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(image, dtype=np.float32, copy=True) / 255).permute(2, 0, 1)
        return (tensor - self.mean) / self.std
