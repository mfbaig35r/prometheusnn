"""Shared fixtures for PrometheusNN tests."""

import numpy as np
import pytest

from prometheus.data.taxonomy import TaxonomyInfo, LevelSpec
from prometheus.data.splits import DataSplit


@pytest.fixture
def three_level_taxonomy() -> TaxonomyInfo:
    """Small 3-level taxonomy for testing."""
    return TaxonomyInfo(
        name="test_model",
        levels=[
            LevelSpec(
                depth=0,
                name="Category",
                classes=["Electronics", "Plumbing"],
                class_to_idx={"Electronics": 0, "Plumbing": 1},
                parent_dims=[],
            ),
            LevelSpec(
                depth=1,
                name="SubCategory",
                classes=["Cables", "Fittings", "Sensors"],
                class_to_idx={"Cables": 0, "Fittings": 1, "Sensors": 2},
                parent_dims=[2],
            ),
            LevelSpec(
                depth=2,
                name="Type",
                classes=["Copper", "HDMI", "Pressure", "USB"],
                class_to_idx={"Copper": 0, "HDMI": 1, "Pressure": 2, "USB": 3},
                parent_dims=[2, 3],
            ),
        ],
        text_column="description",
        embedding_model="all-MiniLM-L6-v2",
        total_samples=50,
    )


@pytest.fixture
def embedding_dim() -> int:
    return 32


@pytest.fixture
def small_datasplit(three_level_taxonomy, embedding_dim) -> DataSplit:
    """Small DataSplit with 20 samples, 3 levels."""
    rng = np.random.default_rng(42)
    n = 20
    embeddings = rng.standard_normal((n, embedding_dim)).astype(np.float32)

    # Category: 0 or 1
    depth0 = rng.integers(0, 2, size=n)
    # SubCategory: 0, 1, or 2
    depth1 = rng.integers(0, 3, size=n)
    # Type: 0, 1, 2, or 3
    depth2 = rng.integers(0, 4, size=n)

    texts = [f"item_{i}" for i in range(n)]

    return DataSplit(
        embeddings=embeddings,
        labels_by_depth={0: depth0, 1: depth1, 2: depth2},
        texts=texts,
    )
