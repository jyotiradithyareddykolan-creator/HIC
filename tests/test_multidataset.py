import io
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing import load_dataset
from vit_model import HSI_ViT


def mat_bytes(**values):
    stream = io.BytesIO()
    sio.savemat(stream, values)
    stream.seek(0)
    return stream


def test_salinas_like_cube_and_labels_are_detected_without_filename_assumptions():
    cube = np.ones((8, 9, 7), dtype=np.float32)
    labels = np.zeros((8, 9), dtype=np.int16)
    labels[:4, :4] = 1
    labels[4:, 4:] = 2
    image, ground_truth, metadata = load_dataset(mat_bytes(salinas_corrected=cube), mat_bytes(salinas_gt=labels))
    assert image.shape == (8, 9, 7)
    assert ground_truth.shape == (8, 9)
    assert metadata["image_variable"] == "salinas_corrected"


def test_ambiguous_image_variables_require_selection():
    with pytest.raises(ValueError, match="Ambiguous image"):
        load_dataset(mat_bytes(first=np.ones((2, 2, 3)), second=np.ones((2, 2, 4))))


def test_missing_ground_truth_is_valid_for_inference():
    image, labels, _ = load_dataset(mat_bytes(paviaU=np.ones((3, 4, 5))))
    assert image.shape == (3, 4, 5)
    assert labels is None


def test_invalid_ground_truth_dimensions_are_rejected():
    with pytest.raises(ValueError, match="matching"):
        load_dataset(mat_bytes(cube=np.ones((3, 4, 5))), mat_bytes(gt=np.ones((2, 2), dtype=np.int16)), "cube", "gt")


def test_embedded_ground_truth_is_auto_detected_when_unambiguous():
    labels = np.zeros((3, 4), dtype=np.int16)
    labels[0, 0] = 1
    _, detected, metadata = load_dataset(mat_bytes(paviaU=np.ones((3, 4, 5)), paviaU_gt=labels))
    assert detected is not None
    assert metadata["label_variable"] == "paviaU_gt"


def test_model_accepts_non_indian_pines_dimensions_and_class_count():
    model = HSI_ViT(input_channels=7, patch_size=3, num_classes=2, embed_dim=8, num_heads=2, num_layers=1)
    output = model(torch.randn(2, 3, 3, 7))
    assert output.shape == (2, 2)


def test_matlab_v73_hdf5_loader_transposes_matlab_dimensions():
    path = Path(__file__).resolve().parents[1] / ".v73_test.mat"
    try:
        with h5py.File(path, "w") as handle:
            handle.create_dataset("salinas_corrected", data=np.ones((5, 4, 3), dtype=np.float32))
            labels = np.zeros((3, 4), dtype=np.int16); labels[0, 0] = 1
            handle.create_dataset("salinas_gt", data=labels.T)
        image, ground_truth, _ = load_dataset(path, None, "salinas_corrected", "salinas_gt")
        assert image.shape == (3, 4, 5)
        assert ground_truth.shape == (3, 4)
    finally:
        path.unlink(missing_ok=True)
