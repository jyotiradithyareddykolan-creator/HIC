"""Pure presentation-time inference and evaluation helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from dataset_config import DatasetConfig
from preprocessing import iter_patch_batches, transform_image
from vit_model import HSI_ViT


def _classification_metrics(actual: np.ndarray, predicted: np.ndarray, count: int) -> dict[str, Any]:
    matrix = confusion_matrix(actual, predicted, labels=list(range(count)))
    per_class = np.divide(np.diag(matrix), matrix.sum(axis=1), out=np.zeros(count, dtype=float), where=matrix.sum(axis=1) != 0)
    return {"overall_accuracy": float(np.mean(actual == predicted)), "average_accuracy": float(np.mean(per_class)),
            "kappa": float(cohen_kappa_score(actual, predicted, labels=list(range(count)))),
            "per_class_accuracy": per_class, "confusion_matrix": matrix,
            "classification_report": classification_report(actual, predicted, labels=list(range(count)), output_dict=True, zero_division=0),
            "evaluated_pixels": int(len(actual))}


def evaluate_prepared_test(data_dir, checkpoint: dict[str, Any], model: HSI_ViT,
                           batch_size: int = 256) -> dict[str, Any]:
    """Evaluate on the saved held-out test patches for this checkpoint."""
    from pathlib import Path
    data_dir = Path(data_dir)
    x = np.load(data_dir / "X_test.npy", mmap_mode="r")
    actual = np.load(data_dir / "y_test.npy")
    if x.ndim != 4 or actual.ndim != 1 or len(x) != len(actual):
        raise ValueError("Prepared held-out test split is malformed.")
    predicted_batches = []
    with torch.inference_mode():
        for start in range(0, len(actual), batch_size):
            batch = torch.from_numpy(np.array(x[start:start + batch_size], copy=True)).float()
            predicted_batches.append(model(batch).argmax(1).cpu().numpy())
    predicted = np.concatenate(predicted_batches).astype(np.int64)
    return _classification_metrics(actual.astype(np.int64), predicted, int(checkpoint["model_config"]["num_classes"]))


def build_model(checkpoint: dict[str, Any]) -> HSI_ViT:
    model = HSI_ViT(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def validate_for_cube(checkpoint: dict[str, Any], dataset_key: str, bands: int) -> None:
    if checkpoint.get("dataset_key") != dataset_key:
        raise ValueError(f"This checkpoint belongs to {checkpoint.get('dataset_key')!r}, not {dataset_key!r}.")
    expected = int(checkpoint["preprocessor"]["input_bands"])
    if expected != bands:
        raise ValueError(f"Checkpoint expects {expected} original spectral bands, but this cube has {bands}.")


def classify_cube(image: np.ndarray, checkpoint: dict[str, Any], model: HSI_ViT,
                  batch_size: int = 256) -> np.ndarray:
    preprocessor = checkpoint["preprocessor"]
    cube = transform_image(image, preprocessor)
    coordinates = np.indices(image.shape[:2]).reshape(2, -1).T
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for patches in iter_patch_batches(cube, coordinates, int(preprocessor["patch_size"]), batch_size):
            predictions.append(model(torch.from_numpy(patches)).argmax(1).cpu().numpy())
    return np.concatenate(predictions).reshape(image.shape[:2]).astype(np.int64)


def false_color_image(image: np.ndarray, band_indices: tuple[int, int, int] | None = None) -> np.ndarray:
    """Create a display-only false-color composite from three real bands."""
    bands = image.shape[2]
    indices = band_indices or (max(0, int(0.75 * (bands - 1))), max(0, int(0.50 * (bands - 1))), max(0, int(0.25 * (bands - 1))))
    channels = []
    for index in indices:
        band = image[:, :, index].astype(np.float32)
        low, high = np.nanpercentile(band, (2, 98))
        channels.append(np.clip((band - low) / (high - low if high > low else 1.0), 0, 1))
    return np.stack(channels, axis=-1)


def evaluate_map(prediction: np.ndarray, labels: np.ndarray, checkpoint: dict[str, Any], background_label: int = 0) -> dict[str, Any]:
    mapping = {int(raw): int(index) for raw, index in checkpoint["class_mapping"].items()}
    valid = labels != background_label
    raw = labels[valid].astype(np.int64)
    known = np.array([value in mapping for value in raw], dtype=bool)
    if not known.any():
        raise ValueError("Ground truth contains no labels represented by this checkpoint.")
    actual = np.array([mapping[int(value)] for value in raw[known]], dtype=np.int64)
    predicted = prediction[valid][known]
    count = int(checkpoint["model_config"]["num_classes"])
    return _classification_metrics(actual, predicted, count)


def class_names(config: DatasetConfig | None, checkpoint: dict[str, Any]) -> list[str]:
    count = int(checkpoint["model_config"]["num_classes"])
    if config and len(config.class_names) == count:
        return list(config.class_names)
    return [f"Class {index + 1}" for index in range(count)]

