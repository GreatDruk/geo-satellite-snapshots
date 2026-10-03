"""Dataset loading utilities."""

import torch
import torchvision

from torchvision.datasets import ImageFolder
from torch.urils.data import DataLoader, Dataset

import os
import numpy as np
import pandas as pd
import pathlib as Path

from PIL import Image


class ImageDataset(Dataset):
    """Custom dataset for loading images and labels from a CSV file."""

    def __init__(self, path, transform=None):
        """
        Args:
            path (str): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(path)
        self.transform = transform
        self.len_dataset = len(self.data)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        image_path = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_data(data_dir: str, batch_size: int = 32):
    """Load train, validation, test and classes dataframes from CSV files."""

    train_df: pd.DataFrame = pd.read_csv(data_dir / 'train.csv')
    val_df: pd.DataFrame = pd.read_csv(data_dir / 'val.csv')
    test_df: pd.DataFrame = pd.read_csv(data_dir / 'test.csv')
    classes_df: pd.DataFrame = pd.read_csv(data_dir / 'classes.csv')

    train_loader = DataLoader(ImageDataset(train_df), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ImageDataset(val_df), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ImageDataset(test_df), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes_df

