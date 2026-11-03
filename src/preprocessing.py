"""Dataset-independent MAT loading, leakage-safe preprocessing, and patches."""
from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "data" / "Indian_pines_corrected.mat"
LABEL_PATH = ROOT / "data" / "Indian_pines_gt.mat"
PREPROCESSOR_PATH = ROOT / "models" / "hsi_preprocessor.joblib"
PATCH_SIZE = 11
N_COMPONENTS = 30
RANDOM_SEED = 42


def _load_mat(source: str | Path | BinaryIO) -> dict[str, np.ndarray]:
    try:
        values = sio.loadmat(source)
    except (ValueError, OSError, NotImplementedError) as exc:
        try:
            import h5py
            if hasattr(source, "seek"):
                source.seek(0)
            with h5py.File(source, "r") as handle:
                values = {}
                def visit(name, node):
                    if isinstance(node, h5py.Dataset) and node.ndim in (2, 3):
                        array = np.asarray(node[()])
                        values[name.rsplit("/", 1)[-1]] = np.transpose(array, tuple(range(array.ndim - 1, -1, -1)))
                handle.visititems(visit)
            if values:
                return values
        except ImportError as import_exc:
            if "7.3" in str(exc) or "HDF5" in str(exc):
                raise ValueError("MATLAB v7.3 files require h5py. Install project requirements or save the file as MATLAB v5.") from import_exc
        except (OSError, ValueError):
            pass
        if "7.3" in str(exc) or "HDF5" in str(exc):
            raise ValueError("MATLAB v7.3 file contains no readable 2-D or 3-D numeric variables.") from exc
        raise ValueError(f"Could not read MATLAB file: {exc}") from exc
    return {name: value for name, value in values.items() if not name.startswith("__")}


def mat_variables(source: str | Path | BinaryIO) -> dict[str, tuple[int, ...]]:
    """Return candidate variable names and shapes from a MATLAB v5 file."""
    return {name: tuple(np.asarray(value).shape) for name, value in _load_mat(source).items()}


def _choose_variable(values: dict[str, np.ndarray], requested: str | None, kind: str) -> str:
    if requested:
        if requested not in values:
            raise ValueError(f"Selected {kind} variable {requested!r} was not found. Available: {sorted(values)}")
        return requested
    candidates = [name for name, value in values.items()
                  if (np.asarray(value).ndim == 3 if kind == "image" else np.asarray(value).ndim == 2)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"No {kind} candidate found. Available variables: {sorted(values)}")
    raise ValueError(f"Ambiguous {kind} variables: {', '.join(sorted(candidates))}; select one explicitly.")


def load_dataset(image_source: str | Path | BinaryIO, label_source: str | Path | BinaryIO | None = None,
                 image_variable: str | None = None, label_variable: str | None = None,
                 background_label: int = 0) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Load an HxWxB cube and optional HxW ground truth without filename assumptions."""
    image_values = _load_mat(image_source)
    image_name = _choose_variable(image_values, image_variable, "image")
    image = np.asarray(image_values[image_name])
    if image.ndim != 3 or not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"Image cube {image_name!r} must be a numeric HxWxB array; got {image.shape}.")
    image = image.astype(np.float32, copy=False)
    if image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError("Image cube dimensions must be positive.")
    if not np.isfinite(image).all():
        raise ValueError("Image cube contains NaN or infinite values.")

    labels = None
    label_name = None
    if label_source is not None:
        label_values = _load_mat(label_source)
        label_name = _choose_variable(label_values, label_variable, "ground-truth")
        labels = np.asarray(label_values[label_name])
    elif label_variable:
        label_name = _choose_variable(image_values, label_variable, "ground-truth")
        labels = np.asarray(image_values[label_name])
    else:
        embedded_candidates = [name for name, value in image_values.items()
                               if np.asarray(value).ndim == 2 and np.asarray(value).shape == image.shape[:2]]
        if len(embedded_candidates) == 1:
            label_name = embedded_candidates[0]
            labels = np.asarray(image_values[label_name])
        elif len(embedded_candidates) > 1:
            raise ValueError("Ambiguous ground-truth variables: " + ", ".join(sorted(embedded_candidates)) + "; select one explicitly.")
    if labels is not None:
        if labels.ndim != 2 or labels.shape != image.shape[:2]:
            raise ValueError(f"Ground truth must be HxW matching {image.shape[:2]}; got {labels.shape}.")
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < background_label):
            raise ValueError("Ground-truth labels must be integer values at or above the background label.")
        labels = labels.astype(np.int64, copy=False)
        if np.count_nonzero(labels != background_label) == 0:
            raise ValueError("Ground truth contains no labeled pixels.")
    metadata = {"image_variable": image_name, "label_variable": label_name,
                "shape": tuple(image.shape), "bands": int(image.shape[2]),
                "background_label": int(background_label)}
    return image, labels, metadata


def load_indian_pines(image_path: Path = IMAGE_PATH, label_path: Path = LABEL_PATH) -> tuple[np.ndarray, np.ndarray]:
    image, labels, _ = load_dataset(image_path, label_path, "indian_pines_corrected", "indian_pines_gt")
    assert labels is not None
    return image, labels


def fit_preprocessor(image: np.ndarray, train_coordinates: np.ndarray, n_components: int = N_COMPONENTS,
                     patch_size: int = PATCH_SIZE) -> dict[str, Any]:
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError("Image must be a finite HxWxB array.")
    if train_coordinates.ndim != 2 or train_coordinates.shape[1] != 2 or len(train_coordinates) == 0:
        raise ValueError("Training coordinates must be a non-empty Nx2 array.")
    max_components = min(len(train_coordinates), image.shape[2])
    if n_components <= 0 or n_components > max_components:
        raise ValueError(f"PCA components must be between 1 and {max_components}; got {n_components}.")
    spectra = image[train_coordinates[:, 0], train_coordinates[:, 1], :]
    scaler = StandardScaler().fit(spectra)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED).fit(scaler.transform(spectra))
    return {"scaler": scaler, "pca": pca, "input_bands": int(image.shape[2]),
            "n_components": int(n_components), "patch_size": int(patch_size),
            "padding": "reflect", "random_seed": RANDOM_SEED}


def transform_image(image: np.ndarray, artifact: dict[str, Any]) -> np.ndarray:
    expected = int(artifact["input_bands"])
    if image.ndim != 3 or image.shape[2] != expected:
        raise ValueError(f"Expected an HxWx{expected} image, got {image.shape}.")
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values.")
    h, w, bands = image.shape
    transformed = artifact["pca"].transform(artifact["scaler"].transform(image.reshape(-1, bands)))
    return transformed.reshape(h, w, int(artifact["n_components"])).astype(np.float32)


def extract_patches(cube: np.ndarray, coordinates: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    if cube.ndim != 3 or patch_size <= 0 or patch_size % 2 != 1:
        raise ValueError("Cube must be HxWxC and patch size must be a positive odd number.")
    coordinates = np.asarray(coordinates, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
        raise ValueError("Coordinates must be a non-empty Nx2 array.")
    if np.any(coordinates < 0) or np.any(coordinates[:, 0] >= cube.shape[0]) or np.any(coordinates[:, 1] >= cube.shape[1]):
        raise ValueError("Patch coordinates are outside the cube.")
    margin = patch_size // 2
    padded = np.pad(cube, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
    return np.stack([padded[r:r + patch_size, c:c + patch_size] for r, c in coordinates]).astype(np.float32)


def iter_patch_batches(cube: np.ndarray, coordinates: np.ndarray, patch_size: int = PATCH_SIZE,
                       batch_size: int = 256):
    """Yield patch batches so full-cube inference does not materialize all patches."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    coordinates = np.asarray(coordinates, dtype=np.int64)
    margin = patch_size // 2
    padded = np.pad(cube, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start:start + batch_size]
        yield np.stack([padded[r:r + patch_size, c:c + patch_size]
                         for r, c in batch_coordinates]).astype(np.float32)


def save_preprocessor(artifact: dict[str, Any], path: Path = PREPROCESSOR_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(path)


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Preprocessing artifact missing: {path}")
    artifact = joblib.load(path)
    for required in ("scaler", "pca", "input_bands", "n_components", "patch_size"):
        if required not in artifact:
            raise ValueError(f"Preprocessing artifact is missing {required!r}.")
    return artifact
