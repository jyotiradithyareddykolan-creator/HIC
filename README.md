# HSI_ViT

HSI_ViT is a lightweight Vision Transformer pipeline for hyperspectral image classification. The application supports Indian Pines, Salinas, Pavia University, Botswana, and compatible custom MATLAB v5 datasets. Dataset names are metadata hints only: image and ground-truth variables are detected by array shape and can be selected explicitly.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
python -m pip install -r requirements.txt
```

Run the tests and launch the UI:

```bash
python -m pytest -q
streamlit run app.py
```

MATLAB v5 files are read with SciPy. MATLAB v7.3 files are reported clearly and should be saved as v5 unless an HDF5 loader is added. An image cube must be `H x W x bands`; an optional ground truth must be `H x W`, integer-valued, with background label `0` by default. An unknown dataset is accepted without assigning it misleading semantic class names; its legend uses `Class 1`, `Class 2`, and so on.

## Train a dataset

Prepare train/validation/test patches. The scaler and PCA are fit only on training-center spectra, then saved with the dataset key, source variables, band count, class mapping, patch size, and PCA configuration.

```bash
python src/dataset_split.py \
  --image path/to/salinas.mat \
  --labels path/to/salinas_gt.mat \
  --image-variable salinas_corrected \
  --label-variable salinas_gt \
  --components 30 \
  --patch-size 11

python src/train.py \
  --data-dir data \
  --preprocessor models/salinas_preprocessor.joblib \
  --epochs 30 \
  --batch-size 64 \
  --learning-rate 0.001

python src/evaluate.py --data-dir data --checkpoint models/salinas_vit.pth
```

Use the same commands for Indian Pines, Pavia University, Botswana, or a custom cube. For small synthetic/custom data, choose a PCA component count no greater than both the number of training samples and original bands. Each class needs at least three labeled pixels for the default stratified split.

The checkpoint is dataset-specific and contains the model configuration, preprocessing objects, raw-to-zero-based class mapping, and training metadata. The Streamlit app rejects old or incompatible checkpoints instead of applying an Indian Pines model to another cube. A cube without labels can only be visualized or classified with a compatible pre-trained checkpoint.

## Project layout

- `src/dataset_config.py`: supported dataset hints, class names, and colors.
- `src/preprocessing.py`: MAT loading, variable detection, validation, scaler/PCA fitting, and patch extraction.
- `src/dataset_split.py`: leakage-safe stratified splits and preprocessing artifact creation.
- `src/pytorch_dataset.py`: dynamic PyTorch loaders.
- `src/vit_model.py`: reusable patch-based ViT with configurable bands, patch size, and class count.
- `src/train.py` and `src/evaluate.py`: dataset-specific checkpoint training and evaluation.
- `app.py`: upload, variable selection, metadata display, visualization, checkpoint compatibility checks, and full-cube maps.

## Limitations

The default evaluation uses stratified pixel splits, which can be optimistic because nearby patches may overlap spatially. The `--spatial` option intentionally reports that spatial holdout is not implemented yet; no spatial accuracy is fabricated. Full training is CPU-compatible but may be slow for large cubes. Existing legacy checkpoints and the old fixed preprocessor should be regenerated with the commands above.

## Presentation preparation

The application discovers checkpoints from `models/` using the registry in `src/checkpoint_registry.py`. The current repository contains a validated format-3 Indian Pines checkpoint at `models/indian_pines/indian_pines_vit.pth`. The older `models/hsi_vit_v2.pth` and raw `models/hsi_vit.pth` files are retained for backward compatibility; the raw state dict is not auto-loaded because it has no dataset/configuration metadata. Salinas, Pavia University, Pavia Centre, Botswana, and custom datasets require their own trained checkpoints.

For each dataset, prepare its checkpoint before the demonstration:

```bash
python src/dataset_split.py --image path/to/salinas.mat --labels path/to/salinas_gt.mat --image-variable salinas_corrected --label-variable salinas_gt --dataset-key salinas --output-dir data/salinas --preprocessor models/salinas/salinas_preprocessor.joblib
python src/train.py --data-dir data/salinas --preprocessor models/salinas/salinas_preprocessor.joblib --checkpoint models/salinas/salinas_vit.pth --epochs 30

python src/dataset_split.py --image path/to/pavia.mat --labels path/to/pavia_gt.mat --image-variable paviaU --label-variable paviaU_gt --dataset-key pavia_university --output-dir data/pavia_university --preprocessor models/pavia_university/pavia_university_preprocessor.joblib
python src/train.py --data-dir data/pavia_university --preprocessor models/pavia_university/pavia_university_preprocessor.joblib --checkpoint models/pavia_university/pavia_university_vit.pth --epochs 30

python src/dataset_split.py --image path/to/pavia_centre.mat --labels path/to/pavia_centre_gt.mat --image-variable paviaC --label-variable paviaC_gt --dataset-key pavia_centre --output-dir data/pavia_centre --preprocessor models/pavia_centre/pavia_centre_preprocessor.joblib
python src/train.py --data-dir data/pavia_centre --preprocessor models/pavia_centre/pavia_centre_preprocessor.joblib --checkpoint models/pavia_centre/pavia_centre_vit.pth --epochs 30

python src/dataset_split.py --image path/to/botswana.mat --labels path/to/botswana_gt.mat --image-variable Botswana --label-variable Botswana_gt --dataset-key botswana --output-dir data/botswana --preprocessor models/botswana/botswana_preprocessor.joblib
python src/train.py --data-dir data/botswana --preprocessor models/botswana/botswana_preprocessor.joblib --checkpoint models/botswana/botswana_vit.pth --epochs 30
```

Alternatively, upload a labeled dataset in the app and use the one-time `Train Model` button. Once the resulting `*_vit.pth` file exists in `models/`, later sessions load it automatically. The app never falls back to an unrelated dataset checkpoint.

The professor-facing app is organized into Dataset Upload, Dataset Information, Hyperspectral Image Viewer, Ground Truth, Model Training, Predicted Classification Map, Evaluation Results, and Advanced Settings. The viewer works before model loading and includes selectable false-color bands, individual grayscale bands, pixel spectral profiles, and the raw spectral vector. Evaluation reports OA, AA, per-class accuracy, Cohen's Kappa, a confusion matrix, and a classification report only when labeled pixels and a verified prediction are available.
