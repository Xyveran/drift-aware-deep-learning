from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from src.utils.config import Config

#
# Dataset
#

# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html


class SensorDataset(Dataset):
    """
    Dataset for windowed sensor data.

    Expects
        features : np.ndarray of shape [N, channels, window_size]
        labels   : np.ndarrar of shape [N]
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        batch_ids: Optional[np.ndarray] = None,
    ):
        assert len(features) == len(labels), "Features and labels must align"

        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

        # Always store timestamps and batch ids, default to -1 tensors so
        # __getitem__ always returns a 4-tuple to avoid unpacking bugs in training
        self.timestamps = (
            torch.tensor(timestamps, dtype=torch.long)
            if timestamps is not None
            else torch.full((len(labels),), -1, dtype=torch.long)
        )

        self.batch_ids = (
            torch.tensor(batch_ids, dtype=torch.long)
            if batch_ids is not None
            else torch.full((len(labels),), -1, dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.timestamps[idx], self.batch_ids[idx]


#
# Loading from disk
#


def load_numpy_dataset(data_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load all numpy artifacts from preprocessing.

    Raises
        FileNotFoundError
            If the processed data dir or any expected artifact is missing
            with a message pointing to the preprocessing step.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: '{data_dir}'. "
            "Run scripts/preprocess_data.py first."
        )

    expected = ["features.npy", "labels.npy", "timestamps.npy", "batch_ids.npy"]
    missing = [f for f in expected if not (data_dir / f).exists()]

    if missing:
        raise FileNotFoundError(
            f"Missing preprocessing artifacts in '{data_dir}': {missing}. "
            "Run scripts/preprocess_data.py first."
        )

    return {
        "features": np.load(data_dir / "features.npy"),
        "labels": np.load(data_dir / "labels.npy"),
        "timestamps": np.load(data_dir / "timestamps.npy"),
        "batch_ids": np.load(data_dir / "batch_ids.npy"),
    }


def load_splits(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load train/test indices produced by preprocessing.

    Raises
        FileNotFoundError
            If the splits directory or index files are missing.
    """
    splits_dir = data_dir / "splits"

    if not splits_dir.exists():
        raise FileNotFoundError(
            f"Splits directory not found: `{splits_dir}`. "
            "Run scripts/preprocess_data.py first."
        )

    return (
        np.load(splits_dir / "train_idx.npy"),
        np.load(splits_dir / "test_idx.npy"),
    )


#
# Dataset builders
#


def build_datasets(data_dir: Path) -> Tuple[SensorDataset, SensorDataset]:
    """Create train and test SensorDatasets from saved preprocessing artifacts."""
    data = load_numpy_dataset(data_dir)
    train_idx, test_idx = load_splits(data_dir)

    train_dataset = SensorDataset(
        features=data["features"][train_idx],
        labels=data["labels"][train_idx],
        timestamps=data["timestamps"][train_idx],
        batch_ids=data["batch_ids"][train_idx],
    )

    test_dataset = SensorDataset(
        features=data["features"][test_idx],
        labels=data["labels"][test_idx],
        timestamps=data["timestamps"][test_idx],
        batch_ids=data["batch_ids"][test_idx],
    )

    return train_dataset, test_dataset


#
# Dataloader builders
#


def create_dataloader(
    dataset: SensorDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """
    Wrap a SensorDataset in a PyTorch DataLoader.

    pin_memory is enabled automatically when a CUDA device is available,
    and disabled otherwise to avoid warnings on CPU-only envs.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_dataloaders(
    cfg: Config,
) -> Tuple[DataLoader, DataLoader]:
    """
    Full pipeline: preprocessing artifacts to train and test DataLoaders.

    Reads all relevant parameters from the project Config object so callers
    don't have to pass individual values through manaually.
    """
    data_dir = Path(cfg.paths.processed_dir)
    batch_size = cfg.training.batch_size
    num_workers = getattr(cfg.data, "num_workers", 0)

    train_dataset, test_dataset = build_datasets(data_dir)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
