"""Image transformations and augmentation definitions."""

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

from trochvision import transforms
from torchvision.transforms import v2

import os
import numpy as np
from PIL import Image

def get_transforms():
    """Define image transformations for training and validation datasets."""
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    train_transforms = transforms
    val_transforms = transforms

    return train_transforms, val_transforms