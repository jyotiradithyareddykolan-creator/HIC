"""Presentation-ready Streamlit application for verified HSI checkpoints."""
from __future__ import annotations

import io
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from checkpoint_registry import (CHECKPOINT_REGISTRY, checkpoint_status, load_checkpoint,
                                 validate_checkpoint_metadata)
from dataset_config import colors_for, get_config, infer_config
from demo_pipeline import (build_model, class_names, classify_cube, evaluate_map,
                           false_color_image, validate_for_cube)
from preprocessing import load_dataset, mat_variables

st.set_page_config(page_title="HSI-ViT Presentation", page_icon="🔬", layout="wide")
st.title("🔬 Hyperspectral Image Classification with ViT")
st.subheader("Project Overview")
st.caption("A dataset-specific Vision Transformer pipeline: MATLAB loading → train-only preprocessing → spatial patches → checkpointed training → batched inference → held-out evaluation.")


@st.cache_resource(show_spinner=False)
def cached_checkpoint(dataset_key: str, models_dir: str):
    checkpoint = load_checkpoint(dataset_key, Path(models_dir))
    return checkpoint, build_model(checkpoint)


def save_upload(uploaded, destination: Path) -> Path:
    destination.write_bytes(uploaded.getvalue())
    return destination


def training_workflow(image_file, label_file, image_variable, label_variable, dataset_key,
                      components, patch_size, epochs, batch_size, learning_rate,
                      train_fraction, validation_fraction):
    from dataset_split import build_dataset
    from train import train

    with tempfile.TemporaryDirectory(prefix="hsi_vit_train_") as temp_name:
        started = time.perf_counter()
        temp = Path(temp_name); data_dir = temp / "data"
        image_path = save_upload(image_file, temp / "image.mat")
        label_path = save_upload(label_file, temp / "labels.mat") if label_file else None
        model_dir = ROOT / "models" / dataset_key
        model_dir.mkdir(parents=True, exist_ok=True)
        preprocessor_path = model_dir / f"{dataset_key}_preprocessor.joblib"
        checkpoint_path = model_dir / f"{dataset_key}_vit.pth"
        build_dataset(image_path, label_path, image_variable, label_variable, components, patch_size,
                      output_dir=data_dir, preprocessor_path=preprocessor_path, dataset_key=dataset_key,
                      train_fraction=train_fraction, validation_fraction=validation_fraction)
        progress = st.progress(0, text="Training model..."); status = st.empty(); chart = st.empty()

        def update(epoch, total, metrics):
            history.append(metrics)
            progress.progress(epoch / total, text=f"Training epoch {epoch}/{total}")
            status.info(f"Train loss {metrics['train_loss']:.4f} · validation loss {metrics['validation_loss']:.4f} · validation accuracy {metrics['validation_accuracy']:.2%}")
            chart.line_chart({"train loss": [item["train_loss"] for item in history],
                              "validation loss": [item["validation_loss"] for item in history]})

        history = []
        checkpoint_path, metrics = train(data_dir, preprocessor_path, checkpoint_path, epochs,
                                         batch_size, learning_rate, progress_callback=update)
        from evaluate import evaluate_prepared
        trained_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        test_metrics = evaluate_prepared(data_dir, trained_checkpoint)
    st.cache_resource.clear()
    st.success("Training completed. The trained model is ready for this dataset and will be reused in future sessions.")
    st.session_state["training_metrics"] = metrics
    st.session_state["training_test_metrics"] = test_metrics
    st.session_state["training_duration_seconds"] = time.perf_counter() - started


with st.sidebar:
    demonstration_mode = st.checkbox("Demonstration Mode", value=True)
    if demonstration_mode:
        batch_size = 256
        checkpoint_override = None
    else:
        with st.expander("Advanced Settings"):
            batch_size = st.number_input("Inference batch size", min_value=32, max_value=1024, value=256, step=32)
            checkpoint_override = st.file_uploader("Optional compatible checkpoint override", type=["pth"], key="checkpoint_override")

image_file = st.file_uploader("Image cube (.mat)", type=["mat"], key="image")
label_file = st.file_uploader("Ground truth (.mat, optional)", type=["mat"], key="labels")
if image_file is None:
    st.info("Upload an image cube to begin. Supported MATLAB v5 files can contain separate or embedded ground truth.")
    st.stop()

image_bytes = image_file.getvalue(); label_bytes = label_file.getvalue() if label_file else None
try:
    variables = mat_variables(io.BytesIO(image_bytes))
    image_variables = [name for name, shape in variables.items() if len(shape) == 3]
    if not image_variables:
        raise ValueError("No 3-D image cube variable was found.")
    preferred = next((name for name in image_variables if get_config(name)), image_variables[0])
    image_variable = st.selectbox("Image cube variable", image_variables, index=image_variables.index(preferred))
    if label_file:
        label_variables = [name for name, shape in mat_variables(io.BytesIO(label_bytes)).items() if len(shape) == 2]
        if not label_variables:
            raise ValueError("No 2-D ground-truth variable was found in the label file.")
        label_variable = st.selectbox("Ground-truth variable", label_variables)
    else:
        embedded = [name for name, shape in variables.items() if len(shape) == 2]
        choice = st.selectbox("Embedded ground truth (optional)", ["(none)"] + embedded)
        label_variable = None if choice == "(none)" else choice
    image, labels, metadata = load_dataset(io.BytesIO(image_bytes), io.BytesIO(label_bytes) if label_bytes else None,
                                           image_variable, label_variable)
except ValueError as exc:
    st.error(str(exc)); st.stop()

detected_config = infer_config(image_variable, label_variable)
dataset_choices = ["Automatic detection"] + [get_config(key).name for key in CHECKPOINT_REGISTRY]
selected_dataset = st.selectbox("Dataset selection", dataset_choices)
if selected_dataset == "Automatic detection":
    config = detected_config
else:
    config = next(get_config(key) for key in CHECKPOINT_REGISTRY if get_config(key).name == selected_dataset)
dataset_key = config.key if config else "custom"
dataset_signature = hashlib.sha256(image_bytes + (label_bytes or b"") + image_variable.encode("utf-8") + dataset_key.encode("utf-8")).hexdigest()
if st.session_state.get("dataset_signature") != dataset_signature:
    st.session_state.pop("prediction", None)
    st.session_state["dataset_signature"] = dataset_signature
label_count = len(np.unique(labels)) - (1 if labels is not None and 0 in labels else 0) if labels is not None else None
st.subheader(config.name if config else "Custom compatible dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dimensions", f"{image.shape[0]} × {image.shape[1]}")
c2.metric("Spectral bands", image.shape[2])
c3.metric("Classes", label_count if label_count is not None else "unknown")
c4.metric("Dataset key", dataset_key)
st.caption(f"Variables: image={image_variable!r} · ground truth={label_variable or 'not provided'!r}")
st.subheader("Hyperspectral Image Viewer")
st.caption("False-color composite: three selected spectral bands mapped to display red, green, and blue channels; it is not a natural-color photograph.")
default_rgb = (int(0.75 * (image.shape[2] - 1)), int(0.50 * (image.shape[2] - 1)), int(0.25 * (image.shape[2] - 1)))
rgb1, rgb2, rgb3 = st.columns(3)
red_band = rgb1.number_input("Display red band", 0, image.shape[2] - 1, default_rgb[0], key="red_band")
green_band = rgb2.number_input("Display green band", 0, image.shape[2] - 1, default_rgb[1], key="green_band")
blue_band = rgb3.number_input("Display blue band", 0, image.shape[2] - 1, default_rgb[2], key="blue_band")
rgb_image = false_color_image(image, (int(red_band), int(green_band), int(blue_band)))
viewer_left, viewer_right = st.columns(2)
with viewer_left:
    fig, ax = plt.subplots(figsize=(7, 5)); ax.imshow(rgb_image); ax.set_title(f"False color (R={red_band}, G={green_band}, B={blue_band})"); ax.axis("off"); st.pyplot(fig); plt.close(fig)
with viewer_right:
    selected_band = st.slider("Spectral band", 0, image.shape[2] - 1, default_rgb[1], key="selected_band")
    fig, ax = plt.subplots(figsize=(7, 5)); ax.imshow(image[:, :, selected_band], cmap="gray"); ax.set_title(f"Band {selected_band} grayscale"); ax.axis("off"); st.pyplot(fig); plt.close(fig)
pixel_left, pixel_right = st.columns(2)
pixel_row = pixel_left.number_input("Pixel row", 0, image.shape[0] - 1, image.shape[0] // 2)
pixel_col = pixel_right.number_input("Pixel column", 0, image.shape[1] - 1, image.shape[1] // 2)
pixel_spectrum = image[int(pixel_row), int(pixel_col), :]
fig, ax = plt.subplots(figsize=(9, 3)); ax.plot(np.arange(image.shape[2]), pixel_spectrum); ax.set_xlabel("Spectral band index"); ax.set_ylabel("Intensity / reflectance"); ax.set_title(f"Spectral profile at ({pixel_row}, {pixel_col})"); st.pyplot(fig); plt.close(fig)
st.dataframe({"Band": np.arange(image.shape[2]), "Value": pixel_spectrum}, height=180, use_container_width=True)
if labels is not None:
    st.subheader("Ground Truth")
    fig, ax = plt.subplots(figsize=(7, 5)); ax.imshow(labels, cmap="tab20"); ax.axis("off"); st.pyplot(fig); plt.close(fig)

record = checkpoint_status(dataset_key) if dataset_key != "custom" else checkpoint_status("custom")
checkpoint = model = None
if record.checkpoint_path is not None:
    try:
        checkpoint, model = cached_checkpoint(dataset_key, str(ROOT / "models"))
        validate_for_cube(checkpoint, dataset_key, image.shape[2])
        st.success("A compatible trained model is loaded for this dataset.")
        st.caption(f"Architecture: {checkpoint['model_config']} · PCA: {checkpoint['preprocessor']['n_components']} components · patch: {checkpoint['preprocessor']['patch_size']}×{checkpoint['preprocessor']['patch_size']}")
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        st.error(f"A checkpoint was found but failed compatibility validation: {exc}")
if checkpoint_override is not None:
    try:
        uploaded_checkpoint = torch.load(io.BytesIO(checkpoint_override.getvalue()), map_location="cpu", weights_only=False)
        validate_checkpoint_metadata(uploaded_checkpoint, dataset_key)
        validate_for_cube(uploaded_checkpoint, dataset_key, image.shape[2])
        checkpoint, model = uploaded_checkpoint, build_model(uploaded_checkpoint)
        st.success("Compatible uploaded checkpoint loaded for this dataset.")
    except (KeyError, RuntimeError, ValueError) as exc:
        st.error(f"Uploaded checkpoint is incompatible: {exc}")
if checkpoint is None:
    st.warning(f"No trained checkpoint is available for {config.name if config else 'this custom dataset'}. Inference is disabled until it is trained.")

if checkpoint is not None and model is not None:
    if st.button("Run verified inference", type="primary"):
        with st.spinner("Classifying the full cube with the cached model..."):
            prediction = classify_cube(image, checkpoint, model, int(batch_size))
        st.session_state["prediction"] = prediction

prediction = st.session_state.get("prediction")
if prediction is not None and checkpoint is not None:
    st.subheader("Predicted classification map")
    names = class_names(config, checkpoint); colors = colors_for(config, len(names))
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(7, 5)); ax.imshow(prediction, cmap=ListedColormap(colors), vmin=0, vmax=len(names) - 1); ax.axis("off"); st.pyplot(fig); plt.close(fig)
    legend = [{"Class ID": index + 1, "Class": name, "Color": colors[index]} for index, name in enumerate(names)]
    st.dataframe(legend, hide_index=True, use_container_width=True)
    if labels is not None:
        try:
            metrics = evaluate_map(prediction, labels, checkpoint)
            st.subheader("Ground Truth vs Prediction")
            mapping = {int(raw): int(index) for raw, index in checkpoint["class_mapping"].items()}
            mapped_truth = np.full(labels.shape, -1, dtype=np.int64)
            for raw, index in mapping.items():
                mapped_truth[labels == raw] = index
            comparison_left, comparison_right = st.columns(2)
            with comparison_left:
                fig, ax = plt.subplots(figsize=(6, 4)); ax.imshow(mapped_truth, cmap=ListedColormap(colors), vmin=-1, vmax=len(names) - 1); ax.set_title("Ground truth (model class indices)"); ax.axis("off"); st.pyplot(fig); plt.close(fig)
            with comparison_right:
                agreement = np.where(mapped_truth >= 0, prediction == mapped_truth, -1)
                fig, ax = plt.subplots(figsize=(6, 4)); ax.imshow(agreement, cmap="RdYlGn", vmin=-1, vmax=1); ax.set_title("Agreement: red=incorrect, green=correct"); ax.axis("off"); st.pyplot(fig); plt.close(fig)
            st.subheader("Evaluation Results")
            metric_left, metric_mid, metric_right = st.columns(3)
            metric_left.metric("Overall Accuracy", f"{metrics['overall_accuracy']:.2%}")
            metric_mid.metric("Average Accuracy", f"{metrics['average_accuracy']:.2%}")
            metric_right.metric("Cohen's Kappa", f"{metrics['kappa']:.4f}")
            st.caption(f"Evaluated labeled pixels: {metrics['evaluated_pixels']:,}")
            st.dataframe([{"Class": names[index], "Accuracy": f"{value:.2%}"} for index, value in enumerate(metrics["per_class_accuracy"])], hide_index=True)
            st.write("Confusion matrix (rows = true class, columns = predicted class)")
            st.dataframe(metrics["confusion_matrix"], use_container_width=True)
            report_rows = []
            for index, name in enumerate(names):
                row = metrics["classification_report"].get(str(index), {})
                report_rows.append({"Class": name, "Precision": row.get("precision", 0.0), "Recall": row.get("recall", 0.0), "F1": row.get("f1-score", 0.0), "Support": row.get("support", 0)})
            st.dataframe(report_rows, hide_index=True, use_container_width=True)
        except ValueError as exc:
            st.info(f"Accuracy is unavailable: {exc}")
    buffer = io.BytesIO(); np.save(buffer, prediction)
    st.download_button("Download prediction map (.npy)", buffer.getvalue(), "prediction_map.npy")

if st.session_state.get("training_metrics"):
    st.subheader("Training Results")
    st.line_chart({"Training loss": [row["train_loss"] for row in st.session_state["training_metrics"]],
                   "Validation loss": [row["validation_loss"] for row in st.session_state["training_metrics"]]})
    test_metrics = st.session_state.get("training_test_metrics")
    if test_metrics:
        st.metric("Held-out test accuracy", f"{test_metrics['accuracy']:.2%}")
        st.caption(f"Training duration: {st.session_state.get('training_duration_seconds', 0):.1f} seconds")
        st.text(test_metrics["report"])

if checkpoint is None and record.checkpoint_path is None and labels is not None:
    with st.expander("Model Training", expanded=True):
        st.info("Use this section to create a dataset-specific model. The completed model is saved for reuse in later sessions.")
        preset = st.selectbox("Training preset", ["Quick demonstration", "Full training"])
        quick = preset == "Quick demonstration"
        components = st.number_input("PCA components", min_value=1, max_value=int(image.shape[2]), value=min(15 if quick else 30, int(image.shape[2])))
        patch_size = st.number_input("Patch size", min_value=3, max_value=31, value=7 if quick else 11, step=2)
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=1 if quick else 30)
        train_batch = st.number_input("Training batch size", min_value=4, max_value=512, value=32 if quick else 64, step=4)
        learning_rate = st.number_input("Learning rate", min_value=0.00001, max_value=1.0, value=0.001, format="%.5f")
        train_fraction = st.number_input("Training fraction", min_value=0.10, max_value=0.90, value=0.70, step=0.05)
        validation_fraction = st.number_input("Validation fraction", min_value=0.05, max_value=0.45, value=0.10, step=0.05)
        train_button = st.button("Train Model", type="primary")
        if train_button:
            try:
                training_workflow(image_file, label_file, image_variable, label_variable, dataset_key,
                                  int(components), int(patch_size), int(epochs), int(train_batch), float(learning_rate),
                                  float(train_fraction), float(validation_fraction))
                st.rerun()
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(f"Training failed: {exc}")
elif checkpoint is None:
    st.info(f"{config.name if config else 'Custom dataset'} requires a separately trained compatible checkpoint. No unrelated checkpoint will be used.")
