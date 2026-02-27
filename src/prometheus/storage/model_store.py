"""Model storage: JSON manifest, .pt state dicts, no pickle."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from prometheus.data.taxonomy import TaxonomyInfo
from prometheus.training.cascade import HierarchicalClassifier
from prometheus.training.trainer import TrainingResult, TrainingConfig


@dataclass
class ModelManifest:
    """Single source of truth for a saved model."""
    taxonomy: TaxonomyInfo
    training_config: dict
    per_level_metrics: dict
    calibration_temperatures: dict[int, float]
    threshold_config: dict
    embedding_dim: int
    created_at: str = ""
    artifacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "taxonomy": self.taxonomy.to_dict(),
            "training_config": self.training_config,
            "per_level_metrics": self.per_level_metrics,
            "calibration_temperatures": {str(k): v for k, v in self.calibration_temperatures.items()},
            "threshold_config": self.threshold_config,
            "embedding_dim": self.embedding_dim,
            "created_at": self.created_at,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ModelManifest:
        return cls(
            taxonomy=TaxonomyInfo.from_dict(data["taxonomy"]),
            training_config=data["training_config"],
            per_level_metrics=data["per_level_metrics"],
            calibration_temperatures={int(k): v for k, v in data["calibration_temperatures"].items()},
            threshold_config=data["threshold_config"],
            embedding_dim=data["embedding_dim"],
            created_at=data.get("created_at", ""),
            artifacts=data.get("artifacts", {}),
        )


def generate_thresholds(num_levels: int, margin: float = 0.15) -> dict:
    """Generate linearly decaying thresholds from 0.90 (root) to 0.75 (leaf)."""
    thresholds = {}
    for i in range(num_levels):
        if num_levels == 1:
            thresholds[i] = 0.90
        else:
            thresholds[i] = 0.90 - i * (0.90 - 0.75) / (num_levels - 1)
    thresholds["margin"] = margin
    thresholds["z_score"] = 2.5
    thresholds["novelty_conf"] = 0.70
    return thresholds


def save_model(
    path: Path,
    model: HierarchicalClassifier,
    taxonomy: TaxonomyInfo,
    result: TrainingResult,
    config: TrainingConfig,
    embedding_dim: int,
    training_embeddings: np.ndarray | None = None,
    training_labels: np.ndarray | None = None,
    confusion_profiles: dict | None = None,
    centroids_data: dict | None = None,
    thresholds: dict | None = None,
):
    """Save a complete model to disk (no pickle)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    level_names = taxonomy.level_names

    # Save classifier weights
    model.save(path, level_names=level_names)

    # Save confusion profiles
    if confusion_profiles:
        with open(path / "confusion_profiles.json", "w") as f:
            json.dump(confusion_profiles, f, indent=2)

    # Save novelty data
    if training_embeddings is not None:
        np.save(path / "novelty_embeddings.npy", training_embeddings)
    if centroids_data is not None:
        np.savez(path / "novelty_centroids.npz", **centroids_data)

    # Generate thresholds if not provided
    if thresholds is None:
        thresholds = generate_thresholds(taxonomy.num_levels)
    with open(path / "thresholds.json", "w") as f:
        json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)

    # Build per-level metrics for manifest
    per_level_metrics = {}
    for lm in result.level_metrics:
        entry = {
            "train_accuracy": round(lm.train_accuracy, 4),
            "val_accuracy": round(lm.val_accuracy, 4),
            "train_loss": round(lm.train_loss, 4),
            "val_loss": round(lm.val_loss, 4),
        }
        if lm.calibration:
            entry["calibration"] = {
                "temperature": round(lm.calibration.temperature, 4),
                "ece_before": round(lm.calibration.ece_before, 4),
                "ece_after": round(lm.calibration.ece_after, 4),
            }
        per_level_metrics[lm.name] = entry

    # Calibration temperatures
    calib_temps = {}
    for lm in result.level_metrics:
        if lm.calibration:
            calib_temps[lm.depth] = lm.calibration.temperature

    # Artifacts list
    artifacts = {
        "classifiers": [f"level_{i}_{name}.pt" for i, name in enumerate(level_names)],
    }
    if confusion_profiles:
        artifacts["confusion_profiles"] = "confusion_profiles.json"
    if training_embeddings is not None:
        artifacts["novelty_embeddings"] = "novelty_embeddings.npy"
    if centroids_data is not None:
        artifacts["novelty_centroids"] = "novelty_centroids.npz"
    artifacts["thresholds"] = "thresholds.json"

    manifest = ModelManifest(
        taxonomy=taxonomy,
        training_config={
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "noise_rate": config.noise_rate,
            "dropout": config.dropout,
            "hidden_dims": config.hidden_dims,
        },
        per_level_metrics=per_level_metrics,
        calibration_temperatures=calib_temps,
        threshold_config=thresholds,
        embedding_dim=embedding_dim,
        created_at=datetime.now(timezone.utc).isoformat(),
        artifacts=artifacts,
    )

    with open(path / "manifest.json", "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def load_model(
    path: Path,
) -> tuple[HierarchicalClassifier, ModelManifest]:
    """Load model and manifest from disk.

    Returns:
        (model, manifest)
    """
    path = Path(path)

    with open(path / "manifest.json") as f:
        manifest = ModelManifest.from_dict(json.load(f))

    taxonomy = manifest.taxonomy

    model = HierarchicalClassifier(
        embedding_dim=manifest.embedding_dim,
        num_classes_per_level=taxonomy.num_classes_per_level,
        hidden_dims=manifest.training_config.get("hidden_dims", [256, 64]),
        dropout=manifest.training_config.get("dropout", 0.3),
    )
    model.load(path, level_names=taxonomy.level_names)

    # Apply calibration temperatures
    for depth, temp in manifest.calibration_temperatures.items():
        model.get_level(depth).set_temperature(temp)

    return model, manifest


def load_thresholds(path: Path) -> dict:
    """Load thresholds from disk."""
    with open(path / "thresholds.json") as f:
        raw = json.load(f)
    result = {}
    for k, v in raw.items():
        try:
            result[int(k)] = v
        except ValueError:
            result[k] = v
    return result


def delete_model(path: Path):
    """Delete a model directory."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
