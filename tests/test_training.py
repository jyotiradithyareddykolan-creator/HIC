import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import class_weight_vector


def test_class_weights_are_finite_and_upweight_rare_classes():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "indian_pines"
    weights = class_weight_vector(data_dir, 16).numpy()
    assert np.isfinite(weights).all()
    assert weights[8] > weights[1] > weights[10]
    assert np.isclose(weights.mean(), 1.0)
