"""Thread-safe SentenceTransformer LRU cache and encoding utilities."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

from prometheus.core.lru import LRUCache

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_MODEL_CACHE_SIZE = 4

_lock = threading.Lock()
_models: LRUCache[str, SentenceTransformer] = LRUCache(maxsize=_MODEL_CACHE_SIZE)


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Get or load a SentenceTransformer model (LRU cached, thread-safe)."""
    cached = _models.get(model_name)
    if cached is not None:
        return cached
    with _lock:
        cached = _models.get(model_name)
        if cached is not None:
            return cached
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _models.put(model_name, model)
        return model


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
