"""Generalized N-level hierarchical classifier (nn.ModuleList)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prometheus.training.classifier import LevelClassifier


class HierarchicalClassifier(nn.Module):
    """Container for N level classifiers using nn.ModuleList."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes_per_level: list[int],
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes_per_level = list(num_classes_per_level)
        hidden_dims = hidden_dims or [256, 64]

        self.classifiers = nn.ModuleList(
            [
                LevelClassifier(
                    embedding_dim=embedding_dim,
                    num_classes=num_classes_per_level[i],
                    parent_dims=num_classes_per_level[:i],
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                )
                for i in range(len(num_classes_per_level))
            ]
        )

    @property
    def num_levels(self) -> int:
        return len(self.classifiers)

    def get_level(self, depth: int) -> LevelClassifier:
        return self.classifiers[depth]

    def save(self, path: Path, level_names: list[str] | None = None):
        """Save all classifier state dicts to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        names = level_names or [str(i) for i in range(self.num_levels)]
        for i, (clf, name) in enumerate(zip(self.classifiers, names)):
            torch.save(clf.state_dict(), path / f"level_{i}_{name}.pt")

    def load(self, path: Path, level_names: list[str] | None = None):
        """Load all classifier state dicts from directory."""
        path = Path(path)
        names = level_names or [str(i) for i in range(self.num_levels)]
        for i, (clf, name) in enumerate(zip(self.classifiers, names)):
            clf.load_state_dict(torch.load(path / f"level_{i}_{name}.pt", weights_only=True))


def build_parent_context(
    predictions: dict[int, torch.Tensor],
    num_classes: list[int],
    parent_depths: list[int],
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Build one-hot parent context from predictions at given depths."""
    if not parent_depths:
        return None

    if device is None:
        device = predictions[parent_depths[0]].device

    context_parts = []
    for depth in parent_depths:
        one_hot = F.one_hot(
            predictions[depth],
            num_classes=num_classes[depth],
        ).float()
        context_parts.append(one_hot)

    return torch.cat(context_parts, dim=-1).to(device)
