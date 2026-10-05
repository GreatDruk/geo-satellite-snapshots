"""Immutable CSV/image datasets; test is loaded only for explicit inference."""
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from src.transforms import get_transforms
from src.utils import seed_worker


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


def make_loader(dataset, config, seed, shuffle=False, sampler=None):
    pin_memory = bool(config.get("pin_memory", False) and torch.cuda.is_available())
    num_workers = config.get("num_workers", 0)
    options = {}
    if num_workers > 0:
        options["persistent_workers"] = config.get("persistent_workers", False)
        if "prefetch_factor" in config:
            prefetch = config["prefetch_factor"]
            if not isinstance(prefetch, int) or isinstance(prefetch, bool) or prefetch <= 0:
                raise ValueError("prefetch_factor must be a positive integer")
            options["prefetch_factor"] = prefetch
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=shuffle if sampler is None else False, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
        **options,
    )


def load_data(data_dir, config, seed=42, *, hard_mining=None):
    """Return train, val, classes; do not load test for training."""
    data_dir = Path(data_dir)
    classes = read_classes(data_dir)
    mapping = {label: i for i, label in enumerate(classes.label.tolist())}
    train_transform, val_transform = get_transforms(config)
    train = ImageDataset(data_dir / "train.csv", data_dir / "train", train_transform, mapping)
    val = ImageDataset(data_dir / "val.csv", data_dir / "val", val_transform, mapping)
    sampler = None
    if hard_mining is not None:
        from src.training.mining import HardExampleSampler, IndexedDataset
        sampler = HardExampleSampler(train, hard_mining, seed)
        train = IndexedDataset(train)
    return make_loader(train, config, seed, True, sampler), make_loader(val, config, seed), classes


def load_test(data_dir, config, seed=42):
    data_dir = Path(data_dir)
    _, transform = get_transforms(config)
    dataset = ImageDataset(data_dir / "test.csv", data_dir / "test", transform)
    return make_loader(dataset, config, seed)


def load_validation(data_dir, config, seed=42):
    """Inference-only validation loading: no train/test access or fitting."""
    data_dir = Path(data_dir)
    classes = read_classes(data_dir)
    _, transform = get_transforms(config)
    mapping = {label: index for index, label in enumerate(classes.label.tolist())}
    dataset = ImageDataset(data_dir / "val.csv", data_dir / "val", transform, mapping)
    return make_loader(dataset, config, seed), classes
