"""Thread-safe SentenceTransformer singleton and encoding utilities."""

from __future__ import annotations

import threading

import numpy as np


_lock = threading.Lock()
_models: dict[str, object] = {}


def get_embedding_model(model_name: str):
    """Get or create a SentenceTransformer model (thread-safe singleton)."""
    if model_name not in _models:
        with _lock:
            if model_name not in _models:
                from sentence_transformers import SentenceTransformer
                _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode texts to embeddings using a sentence-transformer model.

    Returns:
        (n_texts, embedding_dim) float32 array
    """
    model = get_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def get_embedding_dim(model_name: str) -> int:
    """Get the embedding dimension for a model."""
    model = get_embedding_model(model_name)
    return model.get_sentence_embedding_dimension()
