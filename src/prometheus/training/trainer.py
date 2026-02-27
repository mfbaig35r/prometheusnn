"""N-level training loop with noise injection, early stopping, and calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from prometheus.data.taxonomy import TaxonomyInfo
from prometheus.data.splits import DataSplit, BatchIterator
from prometheus.training.cascade import HierarchicalClassifier
from prometheus.training.noise import ParentNoiseInjector
from prometheus.training.calibration import TemperatureScaler, CalibrationResult


@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 0.001
    batch_size: int = 64
    noise_rate: float = 0.15
    dropout: float = 0.3
    hidden_dims: list[int] = field(default_factory=lambda: [256, 64])
    early_stopping_patience: int = 5
    device: str = "cpu"


@dataclass
class LevelMetrics:
    depth: int
    name: str
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    confusion_matrix: np.ndarray
    calibration: CalibrationResult | None = None


@dataclass
class TrainingResult:
    level_metrics: list[LevelMetrics]
    confusion_profiles: dict

    def summary(self) -> dict:
        return {
            lm.name: {
                "train_acc": round(lm.train_accuracy, 4),
                "val_acc": round(lm.val_accuracy, 4),
                "val_loss": round(lm.val_loss, 4),
                "calibration_temp": round(lm.calibration.temperature, 3) if lm.calibration else None,
            }
            for lm in self.level_metrics
        }


def train_model(
    taxonomy: TaxonomyInfo,
    train_data: DataSplit,
    val_data: DataSplit,
    embedding_dim: int,
    config: TrainingConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[HierarchicalClassifier, TrainingResult, ParentNoiseInjector]:
    """Full training pipeline: build model, train N levels, calibrate.

    Returns:
        (model, training_result, noise_injector)
    """
    config = config or TrainingConfig()
    device = torch.device(config.device)
    num_classes = taxonomy.num_classes_per_level

    model = HierarchicalClassifier(
        embedding_dim=embedding_dim,
        num_classes_per_level=num_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    noise_injector = ParentNoiseInjector(noise_rate=config.noise_rate)
    level_metrics: list[LevelMetrics] = []

    def _log(msg: str):
        if progress_callback:
            progress_callback(msg)

    for depth in range(taxonomy.num_levels):
        level = taxonomy.get_level(depth)
        parent_depths = list(range(depth))

        _log(f"Training level {depth} ({level.name}): {level.num_classes} classes")

        metrics = _train_level(
            model=model,
            depth=depth,
            level_name=level.name,
            train_data=train_data,
            val_data=val_data,
            parent_depths=parent_depths,
            num_classes=num_classes,
            noise_injector=noise_injector,
            config=config,
            device=device,
        )

        # Build confusion profile for noise injection in subsequent levels
        noise_injector.build_confusion_profile(depth, metrics.confusion_matrix)

        _log(
            f"  Level {level.name}: val_acc={metrics.val_accuracy:.4f}, "
            f"val_loss={metrics.val_loss:.4f}"
        )

        level_metrics.append(metrics)

    # Calibrate
    _log("Calibrating temperatures...")
    scaler = TemperatureScaler(model, device)
    calib_results = scaler.calibrate_all(val_data, num_classes, config.batch_size)
    for depth, cr in calib_results.items():
        level_metrics[depth].calibration = cr
        _log(
            f"  {taxonomy.get_level(depth).name}: temp={cr.temperature:.3f}, "
            f"ECE {cr.ece_before:.4f} -> {cr.ece_after:.4f}"
        )

    result = TrainingResult(
        level_metrics=level_metrics,
        confusion_profiles=noise_injector.get_serializable_profiles(),
    )
    return model, result, noise_injector


def _train_level(
    model: HierarchicalClassifier,
    depth: int,
    level_name: str,
    train_data: DataSplit,
    val_data: DataSplit,
    parent_depths: list[int],
    num_classes: list[int],
    noise_injector: ParentNoiseInjector,
    config: TrainingConfig,
    device: torch.device,
) -> LevelMetrics:
    """Train a single level classifier."""
    classifier = model.get_level(depth)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    train_loss = train_acc = val_loss = val_acc = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for epoch in range(config.epochs):
        # Training
        classifier.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for batch in BatchIterator(train_data, batch_size=config.batch_size, shuffle=True):
            embeddings = torch.tensor(batch["embeddings"], dtype=torch.float32, device=device)
            labels = torch.tensor(batch[depth], dtype=torch.long, device=device)

            parent_context = noise_injector.build_noisy_context(
                batch, parent_depths, num_classes, device, training=True
            )

            optimizer.zero_grad()
            logits = classifier(embeddings, parent_context)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * len(labels)
            t_correct += (logits.argmax(dim=-1) == labels).sum().item()
            t_total += len(labels)

        train_loss = t_loss / t_total
        train_acc = t_correct / t_total

        # Validation
        classifier.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        epoch_preds, epoch_labels = [], []

        with torch.no_grad():
            for batch in BatchIterator(val_data, batch_size=config.batch_size, shuffle=False):
                embeddings = torch.tensor(batch["embeddings"], dtype=torch.float32, device=device)
                labels = torch.tensor(batch[depth], dtype=torch.long, device=device)

                parent_context = noise_injector.build_noisy_context(
                    batch, parent_depths, num_classes, device, training=False
                )

                logits = classifier(embeddings, parent_context)
                loss = criterion(logits, labels)

                v_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=-1)
                v_correct += (preds == labels).sum().item()
                v_total += len(labels)

                epoch_preds.extend(preds.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            all_preds = epoch_preds
            all_labels = epoch_labels
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    # Restore best model
    if best_state is not None:
        classifier.load_state_dict(best_state)
        classifier.to(device)

    cm = sk_confusion_matrix(
        all_labels, all_preds, labels=list(range(num_classes[depth]))
    )

    return LevelMetrics(
        depth=depth,
        name=level_name,
        train_loss=train_loss,
        train_accuracy=train_acc,
        val_loss=val_loss,
        val_accuracy=val_acc,
        confusion_matrix=cm,
    )
