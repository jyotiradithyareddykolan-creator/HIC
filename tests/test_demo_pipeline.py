import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demo_pipeline import evaluate_map, false_color_image


def metric_checkpoint():
    return {"model_config": {"num_classes": 2}, "class_mapping": {1: 0, 2: 1}}


def test_prediction_metrics_include_oa_aa_kappa_and_report():
    labels = np.array([[0, 1, 1], [2, 2, 0]])
    prediction = np.array([[0, 0, 1], [1, 1, 0]])
    metrics = evaluate_map(prediction, labels, metric_checkpoint())
    assert metrics["overall_accuracy"] == 0.75
    assert metrics["average_accuracy"] == 0.75
    assert "kappa" in metrics
    assert "0" in metrics["classification_report"]
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_false_color_and_prediction_dimensions_are_spatially_preserved():
    image = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    composite = false_color_image(image, (0, 2, 5))
    prediction = np.zeros(image.shape[:2], dtype=np.int64)
    assert composite.shape == (4, 5, 3)
    assert prediction.shape == image.shape[:2]
