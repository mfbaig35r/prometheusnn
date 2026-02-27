"""Single-level feedforward classifier (ported from parts-classifier, unchanged)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LevelClassifier(nn.Module):
    """Classifier for a single taxonomy level.

    Takes an embedding and optional parent context (one-hot encoded
    parent predictions) and outputs class logits for this level.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        parent_dims: list[int] | None = None,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.parent_dims = parent_dims or []
        hidden_dims = hidden_dims or [256, 64]

        parent_context_dim = sum(self.parent_dims)
        input_dim = embedding_dim + parent_context_dim

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(
        self,
        embedding: torch.Tensor,
        parent_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if parent_context is not None:
            x = torch.cat([embedding, parent_context], dim=-1)
        else:
            x = embedding
        logits = self.network(x)
        return logits / self.temperature

    def predict_proba(
        self,
        embedding: torch.Tensor,
        parent_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward(embedding, parent_context)
        return F.softmax(logits, dim=-1)

    def set_temperature(self, temperature: float):
        self.temperature.data = torch.tensor(
            [temperature],
            device=self.temperature.device,
            dtype=self.temperature.dtype,
        )
