"""PyTorch loaders for prepared HSI splits."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(data_dir: Path | str, split: str, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    data_dir = Path(data_dir)
    x_path, y_path = data_dir / f"X_{split}.npy", data_dir / f"y_{split}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Prepared {split} split is missing in {data_dir}. Run dataset_split.py first.")
    x, y = np.load(x_path), np.load(y_path)
    if x.ndim != 4 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError(f"Malformed {split} split: X={x.shape}, y={y.shape}.")
    return DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()),
                      batch_size=batch_size, shuffle=shuffle)


def make_loaders(data_dir: Path | str, batch_size: int = 64):
    return (make_loader(data_dir, "train", batch_size, True),
            make_loader(data_dir, "val", batch_size), make_loader(data_dir, "test", batch_size))
