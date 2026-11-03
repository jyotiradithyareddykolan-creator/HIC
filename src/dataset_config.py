"""Dataset metadata used by loading, training, evaluation, and visualisation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class DatasetConfig:
    """Semantic metadata for a dataset; raw MATLAB labels remain authoritative."""

    name: str
    key: str
    image_variables: tuple[str, ...] = ()
    label_variables: tuple[str, ...] = ()
    class_names: tuple[str, ...] = ()
    background_label: int = 0
    colors: tuple[str, ...] = ()

    def label_name(self, class_index: int) -> str:
        if 0 <= class_index < len(self.class_names):
            return self.class_names[class_index]
        return f"Class {class_index + 1}"


_COLORS = ("#377eb8", "#ff7f00", "#4daf4a", "#e41a1c", "#984ea3", "#a65628",
           "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
           "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3")

DATASET_CONFIGS = {
    "indian_pines": DatasetConfig(
        "Indian Pines", "indian_pines", ("indian_pines_corrected", "indian_pines"),
        ("indian_pines_gt", "Indian_pines_gt"),
        ("Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees",
         "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
         "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"),
        0, _COLORS),
    "salinas": DatasetConfig(
        "Salinas", "salinas", ("salinas_corrected", "salinas"), ("salinas_gt", "Salinas_gt"),
        ("Broccoli green weeds 1", "Broccoli green weeds 2", "Fallow", "Fallow rough plow",
         "Fallow smooth", "Stubble", "Celery", "Grapes untrained", "Soil vineyard develop",
         "Corn senesced green weeds", "Lettuce romaine 4wk", "Lettuce romaine 5wk",
         "Lettuce romaine 6wk", "Lettuce romaine 7wk", "Vineyard untrained", "Vineyard vertical trellis"),
        0, _COLORS),
    "pavia_university": DatasetConfig(
        "Pavia University", "pavia_university", ("paviaU", "pavia", "PaviaU"),
        ("paviaU_gt", "pavia_gt", "PaviaU_gt"),
        ("Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets", "Bare Soil",
         "Bitumen", "Self-Blocking Bricks", "Shadows"), 0, _COLORS),
    "pavia_centre": DatasetConfig(
        "Pavia Centre", "pavia_centre", ("paviaC", "paviaC_corrected", "PaviaCentre"),
        ("paviaC_gt", "pavia_centre_gt", "PaviaCentre_gt"),
        ("Water", "Trees", "Asphalt", "Self-Blocking Bricks", "Bitumen", "Barren Land",
         "Tiles", "Meadows", "Bare Soil"), 0, _COLORS),
    "botswana": DatasetConfig(
        "Botswana", "botswana", ("Botswana", "botswana"), ("Botswana_gt", "botswana_gt"), (), 0, _COLORS),
}


def get_config(key: str | None) -> DatasetConfig | None:
    return DATASET_CONFIGS.get(key or "")


def colors_for(config: DatasetConfig | None, count: int) -> list[str]:
    colors = list(config.colors) if config else list(_COLORS)
    colors = colors or list(_COLORS)
    return [colors[index % len(colors)] for index in range(count)]


def infer_config(image_variable: str, label_variable: str | None = None) -> DatasetConfig | None:
    for config in DATASET_CONFIGS.values():
        if image_variable in config.image_variables or (label_variable and label_variable in config.label_variables):
            return config
    return None

