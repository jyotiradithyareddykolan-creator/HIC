"""Discovery, validation, and reuse of dataset-specific HSI checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class CheckpointRecord:
    dataset_key: str
    checkpoint_path: Path | None
    status: str
    message: str
    legacy: bool = False


# This is the presentation registry. Candidate names are intentionally explicit;
# discovery never assigns one dataset's checkpoint to another dataset.
CHECKPOINT_REGISTRY: dict[str, dict[str, Any]] = {
    "indian_pines": {"checkpoint_names": ("indian_pines_vit_weighted.pth", "indian_pines_vit.pth", "hsi_vit_v2.pth"),
                     "legacy_preprocessor": "hsi_preprocessor.joblib"},
    "salinas": {"checkpoint_names": ("salinas_vit.pth",)},
    "pavia_university": {"checkpoint_names": ("pavia_university_vit.pth", "paviaU_vit.pth")},
    "pavia_centre": {"checkpoint_names": ("pavia_centre_vit.pth", "paviaC_vit.pth")},
    "botswana": {"checkpoint_names": ("botswana_vit.pth",)},
}

def checkpoint_candidates(dataset_key: str, models_dir: Path = MODELS_DIR) -> tuple[Path, ...]:
    """Return existing candidate files for explicit, dataset-scoped selection."""
    entry = CHECKPOINT_REGISTRY.get(dataset_key)
    candidate_names = tuple(entry["checkpoint_names"]) if entry else (f"{dataset_key}_vit.pth",)
    candidates: list[Path] = []
    for name in candidate_names:
        for path in (models_dir / dataset_key / name, models_dir / name):
            if path.is_file() and path not in candidates:
                candidates.append(path)
    return tuple(candidates)


def checkpoint_status(dataset_key: str, models_dir: Path = MODELS_DIR) -> CheckpointRecord:
    candidates = checkpoint_candidates(dataset_key, models_dir)
    if candidates:
        path = candidates[0]
        legacy = dataset_key == "indian_pines" and path.name == "hsi_vit_v2.pth"
        return CheckpointRecord(dataset_key, path, "available", f"Found {path.name}.", legacy)
    return CheckpointRecord(dataset_key, None, "missing", "No compatible trained checkpoint is available; train this dataset first.")


def registry_status(models_dir: Path = MODELS_DIR) -> dict[str, CheckpointRecord]:
    return {key: checkpoint_status(key, models_dir) for key in CHECKPOINT_REGISTRY}


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint {path.name} is not a metadata dictionary.")
    return value


def _normalise_legacy(checkpoint: dict[str, Any], path: Path, models_dir: Path) -> dict[str, Any]:
    if path.name != "hsi_vit_v2.pth" or checkpoint.get("format_version") != 2:
        raise ValueError("This legacy checkpoint is not a supported self-describing format.")
    preprocessor_name = checkpoint.get("preprocessor_path", "hsi_preprocessor.joblib")
    preprocessor_path = path.parent / Path(preprocessor_name).name
    if not preprocessor_path.is_file():
        preprocessor_path = models_dir / Path(preprocessor_name).name
    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"Legacy checkpoint requires {preprocessor_path.name} beside it.")
    preprocessor = joblib.load(preprocessor_path)
    class_mapping = {int(raw): int(index) for raw, index in checkpoint.get("class_mapping", {}).items()}
    return {"format_version": 3, "model_state_dict": checkpoint["model_state_dict"],
            "model_config": checkpoint["model_config"], "preprocessor": preprocessor,
            "dataset_key": "indian_pines", "class_mapping": class_mapping,
            "training_config": checkpoint.get("training_config", {}), "source_format": 2,
            "checkpoint_path": str(path)}


def load_checkpoint(dataset_key: str, models_dir: Path = MODELS_DIR,
                    checkpoint_path: Path | str | None = None) -> dict[str, Any]:
    if checkpoint_path is None:
        record = checkpoint_status(dataset_key, models_dir)
        if record.checkpoint_path is None:
            raise FileNotFoundError(record.message)
        selected_path = record.checkpoint_path
        legacy = record.legacy
    else:
        selected_path = Path(checkpoint_path).resolve()
        if selected_path not in checkpoint_candidates(dataset_key, models_dir):
            raise ValueError(f"Checkpoint {selected_path.name!r} is not a registered candidate for {dataset_key!r}.")
        legacy = dataset_key == "indian_pines" and selected_path.name == "hsi_vit_v2.pth"
    checkpoint = _load_torch(selected_path)
    if legacy:
        checkpoint = _normalise_legacy(checkpoint, selected_path, models_dir)
    elif checkpoint.get("format_version") != 3:
        raise ValueError(f"Checkpoint {selected_path.name} is not a format-3 dataset checkpoint.")
    validate_checkpoint_metadata(checkpoint, dataset_key)
    checkpoint["checkpoint_path"] = str(selected_path)
    return checkpoint


def validate_checkpoint_metadata(checkpoint: dict[str, Any], dataset_key: str) -> None:
    required = ("model_state_dict", "model_config", "preprocessor", "dataset_key", "class_mapping")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing metadata: {', '.join(missing)}.")
    if checkpoint["dataset_key"] != dataset_key:
        raise ValueError(f"Checkpoint belongs to {checkpoint['dataset_key']!r}, not {dataset_key!r}.")
    model_config = checkpoint["model_config"]
    preprocessor = checkpoint["preprocessor"]
    if int(model_config["input_channels"]) != int(preprocessor["n_components"]):
        raise ValueError("Checkpoint model input channels do not match its PCA component count.")
    if int(model_config["patch_size"]) != int(preprocessor["patch_size"]):
        raise ValueError("Checkpoint patch size does not match its preprocessing artifact.")
    if int(model_config["num_classes"]) != len(checkpoint["class_mapping"]):
        raise ValueError("Checkpoint output classes do not match its class mapping.")
    return None

