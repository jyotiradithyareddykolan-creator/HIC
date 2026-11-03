from pathlib import Path


APP_SOURCE = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")


def test_professor_ui_hides_checkpoint_inventory_and_filenames():
    assert 'st.subheader("Model Checkpoint Management")' not in APP_SOURCE
    assert 'st.subheader("Checkpoint registry")' not in APP_SOURCE
    assert '"Checkpoint": record.checkpoint_path.name' not in APP_SOURCE
    assert '"Action": "Loaded automatically"' not in APP_SOURCE


def test_ui_exposes_active_checkpoint_and_invalidates_stale_results():
    assert 'st.selectbox("Model checkpoint"' in APP_SOURCE
    assert 'st.caption(f"Active checkpoint:' in APP_SOURCE
    assert 'st.session_state.pop("prediction_signature", None)' in APP_SOURCE
    assert 'evaluate_prepared_test' in APP_SOURCE


def test_evaluation_dashboard_removes_comparison_maps_and_keeps_metrics():
    assert 'Ground truth (model class indices)' not in APP_SOURCE
    assert 'Agreement: red=incorrect, green=correct' not in APP_SOURCE
    assert 'Macro F1-score' in APP_SOURCE
    assert 'Per-class evaluation report' in APP_SOURCE
    assert 'Confusion matrix' in APP_SOURCE
    assert 'Predicted classification map' in APP_SOURCE
