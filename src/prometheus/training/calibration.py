"""Temperature scaling calibration — grid search + ECE metric."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from prometheus.data.splits import BatchIterator, DataSplit
from prometheus.training.cascade import HierarchicalClassifier


@dataclass
class CalibrationResult:
    temperature: float
    ece_before: float
    ece_after: float


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        n = mask.sum()
        if n > 0:
            ece += (n / total) * abs(accuracies[mask].mean() - confidences[mask].mean())

    return ece


def find_optimal_temperature(
    logits: np.ndarray, labels: np.ndarray, temps: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Grid search for optimal temperature. Returns (temp, ece_before, ece_after)."""
    if temps is None:
        temps = np.linspace(0.5, 3.0, 51)

    probs_before = _softmax(logits)
    ece_before = compute_ece(probs_before, labels)

    best_temp, best_ece = 1.0, ece_before
    for temp in temps:
        probs = _softmax(logits / temp)
        ece = compute_ece(probs, labels)
        if ece < best_ece:
            best_ece = ece
            best_temp = float(temp)

    return best_temp, ece_before, best_ece


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class TemperatureScaler:
    """Calibrator for N-level hierarchical classifier."""

    def __init__(self, model: HierarchicalClassifier, device: torch.device | None = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.results: dict[int, CalibrationResult] = {}

    def calibrate_all(
        self,
        calib_data: DataSplit,
        num_classes_per_level: list[int],
        batch_size: int = 64,
    ) -> dict[int, CalibrationResult]:
        results = {}
        for depth in range(self.model.num_levels):
            result = self._calibrate_level(depth, calib_data, num_classes_per_level, batch_size)
            results[depth] = result
            self.results[depth] = result
        return results

    def _calibrate_level(
        self,
        depth: int,
        calib_data: DataSplit,
        num_classes_per_level: list[int],
        batch_size: int,
    ) -> CalibrationResult:
        classifier = self.model.get_level(depth)
        classifier.eval()

        parent_depths = list(range(depth))
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in BatchIterator(calib_data, batch_size=batch_size, shuffle=False):
                embeddings = torch.tensor(
                    batch["embeddings"], dtype=torch.float32, device=self.device
                )
                labels = batch[depth]

                parent_context = self._build_context(batch, parent_depths, num_classes_per_level)

                old_temp = classifier.temperature.data.clone()
                classifier.temperature.data = torch.ones(1, device=self.device)
                logits = classifier(embeddings, parent_context)
                classifier.temperature.data = old_temp

                all_logits.append(logits.cpu().numpy())
                all_labels.extend(labels)

        all_logits_np = np.concatenate(all_logits, axis=0)
        all_labels_np = np.array(all_labels)

        opt_temp, ece_before, ece_after = find_optimal_temperature(all_logits_np, all_labels_np)
        classifier.set_temperature(opt_temp)

        return CalibrationResult(temperature=opt_temp, ece_before=ece_before, ece_after=ece_after)

    def _build_context(
        self,
        batch: dict,
        parent_depths: list[int],
        num_classes_per_level: list[int],
    ) -> torch.Tensor | None:
        if not parent_depths:
            return None

        context_parts = []
        for depth in parent_depths:
            labels = torch.tensor(batch[depth], dtype=torch.long, device=self.device)
            one_hot = F.one_hot(labels, num_classes=num_classes_per_level[depth]).float()
            context_parts.append(one_hot)

        return torch.cat(context_parts, dim=-1)
