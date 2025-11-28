import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import Subset
from torchvision import transforms, datasets
from torchvision.datasets import EuroSAT as TV_EuroSAT

from .config import SEED


def _get_labels_array(ds):
    # Try standard attributes for labels
    if hasattr(ds, "targets"):
        return np.array(ds.targets)
    if hasattr(ds, "labels"):
        return np.array(ds.labels)
    if hasattr(ds, "samples"):
        # ImageFolder style: list of (path, label)
        return np.array([lbl for _, lbl in ds.samples])
    raise ValueError("Cannot extract labels from dataset; no .targets/.labels/.samples")


def load_mnist(img_size: int = 28):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    train_full = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_full = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    num_classes = 10
    return train_full, test_full, num_classes


def load_eurosat(img_size: int = 64):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    full = TV_EuroSAT("./data", download=True, transform=tfm)
    labels = _get_labels_array(full)
    idx = np.arange(len(full))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    train_ds = Subset(full, train_idx)
    test_ds = Subset(full, test_idx)
    num_classes = len(np.unique(labels))
    return train_ds, test_ds, num_classes


def load_ucmerced(
    root: str = "./data/UCMerced_LandUse/Images",
    img_size: int = 224,
):
    """
    root should point to the Images folder that contains 21 class subfolders.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"UC Merced 'Images' root not found at: {root} "
            "(expected folder with 21 class subdirectories)."
        )

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    full = datasets.ImageFolder(root=root, transform=tfm)
    labels = _get_labels_array(full)
    idx = np.arange(len(full))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    train_ds = Subset(full, train_idx)
    test_ds = Subset(full, test_idx)
    num_classes = len(np.unique(labels))
    return train_ds, test_ds, num_classes


def load_dataset(
    name: str,
    ucm_root: str = "./data/UCMerced_LandUse/Images",
) -> Tuple[object, object, int]:
    """
    Returns (train_dataset, test_dataset, num_classes) for:
      - "mnist"
      - "eurosat"
      - "ucmerced"
    """
    name = name.lower()
    if name == "mnist":
        return load_mnist(img_size=28)
    if name == "eurosat":
        return load_eurosat(img_size=64)
    if name == "ucmerced":
        return load_ucmerced(root=ucm_root, img_size=224)
    raise ValueError(f"Unknown dataset name: {name}")
