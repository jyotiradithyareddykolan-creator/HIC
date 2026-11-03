"""Train a dataset-specific lightweight HSI ViT checkpoint."""
from __future__ import annotations
import argparse, random
from pathlib import Path
import joblib, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from pytorch_dataset import make_loaders
from vit_model import HSI_ViT

ROOT = Path(__file__).resolve().parent.parent

def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None; model.train(training); loss_sum = correct = total = 0
    with (torch.enable_grad() if training else torch.no_grad()):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images); loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(labels); correct += (logits.argmax(1) == labels).sum().item(); total += len(labels)
    return loss_sum / total, correct / total

def class_weight_vector(data_dir, num_classes):
    """Return normalized inverse-square-root frequency weights from train labels."""
    labels = np.load(Path(data_dir) / "y_train.npy").astype(np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Training labels do not contain every model class.")
    weights = 1.0 / np.sqrt(counts)
    return torch.from_numpy((weights / weights.mean()).astype(np.float32))

def train(data_dir=ROOT / "data", preprocessor_path=None, checkpoint_path=None, epochs=30,
          batch_size=64, learning_rate=1e-3, embed_dim=64, num_heads=4, num_layers=2,
          progress_callback=None, class_weighted=False):
    data_dir = Path(data_dir)
    preprocessor_file = Path(preprocessor_path) if preprocessor_path else ROOT / "models" / "indian_pines_preprocessor.joblib"
    preprocessor = joblib.load(preprocessor_file)
    train_loader, val_loader, _ = make_loaders(data_dir, batch_size)
    sample_x, sample_y = next(iter(train_loader)); patch_size, input_channels = sample_x.shape[1], sample_x.shape[-1]
    num_classes = int(max(sample_y.max().item() + 1, max(int(np.load(data_dir / "y_train.npy").max()), int(np.load(data_dir / "y_val.npy").max()), int(np.load(data_dir / "y_test.npy").max())) + 1))
    config = {"input_channels": input_channels, "patch_size": patch_size, "num_classes": num_classes,
              "embed_dim": embed_dim, "num_heads": num_heads, "num_layers": num_layers, "dropout": 0.1}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = HSI_ViT(**config).to(device)
    weights = class_weight_vector(data_dir, num_classes) if class_weighted else None
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate); best = float("inf")
    dataset_key = preprocessor.get("dataset_key", "custom")
    checkpoint_path = Path(checkpoint_path or ROOT / "models" / dataset_key / f"{dataset_key}_vit.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = []
    for epoch in range(epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        metrics.append({"epoch": epoch + 1, "train_loss": tr_loss, "train_accuracy": tr_acc,
                        "validation_loss": va_loss, "validation_accuracy": va_acc})
        if progress_callback:
            progress_callback(epoch + 1, epochs, metrics[-1])
        print(f"Epoch {epoch + 1}/{epochs}: train loss={tr_loss:.4f} acc={tr_acc:.2%}; val loss={va_loss:.4f} acc={va_acc:.2%}")
        if va_loss < best:
            best = va_loss
            checkpoint = {"format_version": 3, "model_state_dict": model.state_dict(), "model_config": config,
                          "preprocessor": preprocessor, "dataset_key": dataset_key,
                          "class_mapping": preprocessor.get("class_mapping", {}),
                          "training_config": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                                               "class_weighted": class_weighted},
                          "training_history": list(metrics),
                          "class_weights": weights.tolist() if weights is not None else None,
                          "best_validation_loss": best}
            torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", type=Path, default=ROOT / "data"); p.add_argument("--preprocessor", type=Path)
    p.add_argument("--checkpoint", type=Path); p.add_argument("--epochs", type=int, default=30); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--class-weighted", action="store_true")
    a = p.parse_args(); print(f"Saved checkpoint: {train(a.data_dir, a.preprocessor, a.checkpoint, a.epochs, a.batch_size, a.learning_rate, class_weighted=a.class_weighted)}")
