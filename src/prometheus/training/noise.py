"""Parent noise injection using confusion-matrix-based corruption."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class ParentNoiseInjector:
    """Injects realistic noise into parent labels during training.

    Uses confusion profiles (from prior level validation) to corrupt
    parent labels with plausible wrong predictions, making downstream
    classifiers robust to upstream errors.
    """

    def __init__(self, noise_rate: float = 0.15):
        self.noise_rate = noise_rate
        # {depth: {true_idx: {wrong_idx: count, ...}}}
        self.confusion_profiles: dict[int, dict[int, dict[int, int]]] = {}

    def set_confusion_profile(self, depth: int, profile: dict[int, dict[int, int]]):
        self.confusion_profiles[depth] = profile

    def build_confusion_profile(self, depth: int, cm: np.ndarray) -> dict[int, dict[int, int]]:
        """Build and store a confusion profile from a confusion matrix."""
        profile: dict[int, dict[int, int]] = {}
        num_classes = cm.shape[0]

        for true_idx in range(num_classes):
            row = cm[true_idx]
            if row.sum() == 0:
                continue
            confusions = {}
            for pred_idx in range(num_classes):
                if pred_idx != true_idx and row[pred_idx] > 0:
                    confusions[pred_idx] = int(row[pred_idx])
            if confusions:
                profile[true_idx] = confusions

        self.confusion_profiles[depth] = profile
        return profile

    def build_noisy_context(
        self,
        batch: dict,
        parent_depths: list[int],
        num_classes_per_level: list[int],
        device: torch.device,
        training: bool,
    ) -> torch.Tensor | None:
        """Build one-hot parent context, optionally with noise injection."""
        if not parent_depths:
            return None

        context_parts = []
        for depth in parent_depths:
            labels = torch.tensor(batch[depth], dtype=torch.long, device=device)

            if training and depth in self.confusion_profiles and self.noise_rate > 0:
                labels = self._inject_noise(labels, depth, device)

            one_hot = F.one_hot(labels, num_classes=num_classes_per_level[depth]).float()
            context_parts.append(one_hot)

        return torch.cat(context_parts, dim=-1)

    def _inject_noise(self, labels: torch.Tensor, depth: int, device: torch.device) -> torch.Tensor:
        profile = self.confusion_profiles[depth]
        noisy = labels.clone()
        mask = torch.rand(len(labels), device=device) < self.noise_rate

        for i in range(len(labels)):
            if not mask[i]:
                continue
            true_label = labels[i].item()
            confusion_dist = profile.get(true_label)
            if confusion_dist is None or len(confusion_dist) == 0:
                continue
            wrong_labels = list(confusion_dist.keys())
            weights = list(confusion_dist.values())
            total = sum(weights)
            probs = [w / total for w in weights]
            noisy[i] = int(np.random.choice(wrong_labels, p=probs))

        return noisy

    def get_serializable_profiles(self) -> dict:
        """Return JSON-serializable confusion profiles."""
        return {
            str(depth): {str(k): v for k, v in profile.items()}
            for depth, profile in self.confusion_profiles.items()
        }

    def load_profiles(self, data: dict):
        """Load confusion profiles from deserialized JSON."""
        self.confusion_profiles = {
            int(depth): {int(k): v for k, v in profile.items()} for depth, profile in data.items()
        }
