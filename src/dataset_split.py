"""Prepare train/validation/test patches for any compatible HSI cube."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from dataset_config import infer_config
from preprocessing import (N_COMPONENTS, PATCH_SIZE, RANDOM_SEED, ROOT, extract_patches,
                           fit_preprocessor, load_dataset, save_preprocessor, transform_image)


def build_dataset(image_path: Path | str = ROOT / "data" / "Indian_pines_corrected.mat",
                  label_path: Path | str | None = ROOT / "data" / "Indian_pines_gt.mat",
                  image_variable: str | None = None, label_variable: str | None = None,
                  n_components: int = N_COMPONENTS, patch_size: int = PATCH_SIZE,
                  output_dir: Path | str = ROOT / "data", preprocessor_path: Path | str | None = None,
                  dataset_key: str | None = None, spatial: bool = False,
                  train_fraction: float = 0.70, validation_fraction: float = 0.10) -> dict:
    if spatial:
        raise NotImplementedError("Spatially separated splitting is not implemented; use default stratified pixel split.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must be positive and leave at least one test sample fraction.")
    image, gt, metadata = load_dataset(image_path, label_path, image_variable, label_variable)
    if gt is None:
        raise ValueError("Training requires a ground-truth label array.")
    raw_labels = sorted(int(label) for label in np.unique(gt) if label != metadata["background_label"])
    if len(raw_labels) < 2:
        raise ValueError("Training requires at least two labeled classes.")
    coordinates = np.argwhere(np.isin(gt, raw_labels))
    class_mapping = {raw: index for index, raw in enumerate(raw_labels)}
    labels = np.array([class_mapping[int(gt[r, c])] for r, c in coordinates], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(raw_labels))
    if np.any(counts < 3):
        raise ValueError("Each class needs at least 3 labeled pixels for a stratified split.")
    train_coords, held_coords, y_train, y_held = train_test_split(
        coordinates, labels, test_size=1 - train_fraction, random_state=RANDOM_SEED, stratify=labels)
    val_coords, test_coords, y_val, y_test = train_test_split(
        held_coords, y_held, test_size=1 - validation_fraction / (1 - train_fraction),
        random_state=RANDOM_SEED, stratify=y_held)
    artifact = fit_preprocessor(image, train_coords, n_components, patch_size)
    cube = transform_image(image, artifact)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    for name, coords, split_labels in (("train", train_coords, y_train), ("val", val_coords, y_val), ("test", test_coords, y_test)):
        np.save(output / f"X_{name}.npy", extract_patches(cube, coords, patch_size))
        np.save(output / f"y_{name}.npy", split_labels.astype(np.int64))
    config = infer_config(metadata["image_variable"], metadata["label_variable"])
    key = dataset_key or (config.key if config else "custom")
    artifact.update({"dataset_key": key, "image_variable": metadata["image_variable"],
                     "label_variable": metadata["label_variable"], "class_mapping": class_mapping,
                     "background_label": metadata["background_label"]})
    preprocessor = Path(preprocessor_path) if preprocessor_path else ROOT / "models" / f"{key}_preprocessor.joblib"
    save_preprocessor(artifact, preprocessor)
    metadata_path = output / f"{key}_metadata.joblib"
    joblib.dump({"dataset_key": key, "class_mapping": class_mapping, "raw_labels": raw_labels,
                 "image_shape": tuple(image.shape), "preprocessor_path": str(preprocessor)}, metadata_path)
    return {"dataset_key": key, "class_mapping": class_mapping, "preprocessor_path": preprocessor,
            "metadata_path": metadata_path, "sizes": {"train": len(y_train), "val": len(y_val), "test": len(y_test)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=ROOT / "data" / "Indian_pines_corrected.mat")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "Indian_pines_gt.mat")
    parser.add_argument("--image-variable"); parser.add_argument("--label-variable")
    parser.add_argument("--components", type=int, default=N_COMPONENTS); parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--dataset-key"); parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--preprocessor", type=Path); parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.70); parser.add_argument("--validation-fraction", type=float, default=0.10)
    args = parser.parse_args()
    result = build_dataset(args.image, args.labels, args.image_variable, args.label_variable,
                           args.components, args.patch_size, output_dir=args.output_dir,
                           preprocessor_path=args.preprocessor, spatial=args.spatial, dataset_key=args.dataset_key,
                           train_fraction=args.train_fraction, validation_fraction=args.validation_fraction)
    print(f"Saved {result['dataset_key']} splits: {result['sizes']}")
