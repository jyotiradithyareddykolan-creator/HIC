from pathlib import Path


APP_SOURCE = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")


def test_professor_ui_hides_checkpoint_inventory_and_filenames():
    assert 'st.subheader("Model Checkpoint Management")' not in APP_SOURCE
    assert 'st.subheader("Checkpoint registry")' not in APP_SOURCE
    assert '"Checkpoint": record.checkpoint_path.name' not in APP_SOURCE
    assert '"Action": "Loaded automatically"' not in APP_SOURCE
