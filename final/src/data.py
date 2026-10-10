"""Immutable CSV/image datasets."""
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, label_to_index=None):
        self.data = pd.read_csv(csv_path, dtype={"id": str})
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_to_index = label_to_index
        if "id" not in self.data or self.data.empty or self.data.id.isna().any() or self.data.id.duplicated().any():
            raise ValueError(f"Expected nonempty unique IDs in {csv_path}")
        if label_to_index is not None:
            if "label" not in self.data or not self.data.label.isin(label_to_index).all():
                raise ValueError(f"Missing or unknown labels in {csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        path = (self.image_dir / row["id"]).resolve()
        if not path.is_relative_to(self.image_dir.resolve()):
            raise ValueError("Image ID must stay inside its split directory")
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = self.label_to_index[row["label"]] if self.label_to_index is not None else row["id"]
        return image, target


def read_classes(data_dir):
    classes = pd.read_csv(Path(data_dir) / "classes.csv")
    if not {"label", "class_name"}.issubset(classes.columns):
        raise ValueError("classes.csv requires label,class_name")
    if classes.empty or classes.label.isna().any() or classes.label.duplicated().any():
        raise ValueError("Class labels must be nonempty and unique")
    return classes

