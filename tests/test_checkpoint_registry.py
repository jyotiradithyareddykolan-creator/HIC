import sys
import shutil
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from checkpoint_registry import checkpoint_candidates, checkpoint_status, load_checkpoint, validate_checkpoint_metadata


def checkpoint(dataset_key="salinas"):
    return {
        "format_version": 3,
        "model_state_dict": {},
        "model_config": {"input_channels": 2, "patch_size": 3, "num_classes": 2,
                          "embed_dim": 8, "num_heads": 2, "num_layers": 1, "dropout": 0.1},
        "preprocessor": {"input_bands": 4, "n_components": 2, "patch_size": 3},
        "dataset_key": dataset_key,
        "class_mapping": {1: 0, 2: 1},
    }


def test_supported_dataset_discovery_and_reuse():
    tmp_path = Path(__file__).resolve().parents[1] / ".registry_test_tmp"
    tmp_path.mkdir(exist_ok=True)
    try:
        path = tmp_path / "salinas_vit.pth"
        torch.save(checkpoint(), path)
        status = checkpoint_status("salinas", tmp_path)
        assert status.status == "available"
        loaded_first = load_checkpoint("salinas", tmp_path)
        loaded_second = load_checkpoint("salinas", tmp_path)
        assert loaded_first["checkpoint_path"] == loaded_second["checkpoint_path"] == str(path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_explicit_checkpoint_selection_is_dataset_scoped():
    tmp_path = Path(__file__).resolve().parents[1] / ".registry_test_tmp"
    tmp_path.mkdir(exist_ok=True)
    try:
        weighted = tmp_path / "indian_pines_vit_weighted.pth"
        original = tmp_path / "indian_pines_vit.pth"
        torch.save(checkpoint("indian_pines"), weighted)
        torch.save(checkpoint("indian_pines"), original)
        assert [path.name for path in checkpoint_candidates("indian_pines", tmp_path)] == [
            "indian_pines_vit_weighted.pth", "indian_pines_vit.pth"
        ]
        loaded = load_checkpoint("indian_pines", tmp_path, checkpoint_path=weighted)
        assert loaded["checkpoint_path"] == str(weighted)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_dataset_switching_does_not_use_another_dataset_checkpoint():
    tmp_path = Path(__file__).resolve().parents[1] / ".registry_test_tmp"
    tmp_path.mkdir(exist_ok=True)
    try:
        torch.save(checkpoint("salinas"), tmp_path / "salinas_vit.pth")
        assert checkpoint_status("pavia_university", tmp_path).status == "missing"
        with pytest.raises(FileNotFoundError, match="train this dataset"):
            load_checkpoint("pavia_university", tmp_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_checkpoint_compatibility_rejects_wrong_dataset_and_dimensions():
    with pytest.raises(ValueError, match="belongs to"):
        validate_checkpoint_metadata(checkpoint("salinas"), "botswana")
    invalid = checkpoint()
    invalid["model_config"] = dict(invalid["model_config"], input_channels=3)
    with pytest.raises(ValueError, match="input channels"):
        validate_checkpoint_metadata(invalid, "salinas")
