"""Evaluate a checkpoint using its own model dimensions and label mapping."""
from pathlib import Path
import argparse, torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pytorch_dataset import make_loader
from vit_model import HSI_ViT

def evaluate_prepared(data_dir, checkpoint):
    model = HSI_ViT(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for images, labels in make_loader(data_dir, "test"):
            predicted.extend(model(images).argmax(1).tolist()); actual.extend(labels.tolist())
    print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")
    return {"accuracy": float(accuracy_score(actual, predicted)), "report": classification_report(actual, predicted, zero_division=0),
            "confusion_matrix": confusion_matrix(actual, predicted)}


def evaluate(data_dir, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 3: raise ValueError("Unsupported checkpoint format; train a dataset-specific checkpoint.")
    model = HSI_ViT(**checkpoint["model_config"]); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for images, labels in make_loader(data_dir, "test"):
            predicted.extend(model(images).argmax(1).tolist()); actual.extend(labels.tolist())
    print(f"Dataset: {checkpoint.get('dataset_key', 'custom')}"); print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")
    print(classification_report(actual, predicted, zero_division=0)); print(confusion_matrix(actual, predicted))
    return actual, predicted

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data"); p.add_argument("--checkpoint", type=Path, required=True); a = p.parse_args(); evaluate(a.data_dir, a.checkpoint)
