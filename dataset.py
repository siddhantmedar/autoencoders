#!/usr/bin/env python3
"""
MNIST dataset loading and DataLoader utilities.
"""

import tomllib
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np

CONFIG_PATH = Path(__file__).parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    cfg = tomllib.load(f)

# ==================== DataLoader ====================


class MNISTDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """MNIST transforms."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def get_dataloaders(split="all", batch_size=None, num_workers=None,gen=None):
    """
    Create DataLoaders for MNIST.

    Args:
        split: Which loader(s) to return - 'train', 'val', 'test', or 'all'
        batch_size: Batch size for training (defaults to config value)
        num_workers: Number of data loading workers (defaults to config value)

    Returns:
        Single DataLoader if split specified, or (train_loader, val_loader, test_loader) if 'all'
    """
    if split.lower() not in {"train", "val", "test", "all"}:
        raise ValueError("split not valid. should be train/val/test/all")

    batch_size = batch_size or cfg["training"]["batch_size"]
    num_workers = num_workers or cfg["training"]["num_workers"]

    print("Loading MNIST datasets...")
    train_dataset = load_dataset(cfg["dataset"]["name"], split="train")
    full_test_dataset = load_dataset(cfg["dataset"]["name"], split="test")

    indices = list(range(len(full_test_dataset)))
    labels = full_test_dataset["label"]

    val_idx, test_idx = train_test_split(
        indices, test_size=0.5, stratify=labels, random_state=313
    )

    val_dataset = full_test_dataset.select(val_idx)
    test_dataset = full_test_dataset.select(test_idx)

    train_dataset = MNISTDataset(train_dataset, get_transforms())
    val_dataset = MNISTDataset(val_dataset, get_transforms())
    test_dataset = MNISTDataset(test_dataset, get_transforms())

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=gen,
        worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

    if split.lower() == "train":
        return train_loader
    elif split.lower() == "val":
        return val_loader
    elif split.lower() == "test":
        return test_loader
    else:
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader = get_dataloaders("train")
    val_loader = get_dataloaders("val")
    test_loader = get_dataloaders("test")

    print(f"train_loader batches: {len(train_loader)}")
    print(f"val_loader batches: {len(val_loader)}")
    print(f"test_loader batches: {len(test_loader)}")

    print(next(iter(train_loader))[0].shape)
