"""Tests for training pipeline."""

import numpy as np

from prometheus.data.splits import DataSplit
from prometheus.data.taxonomy import LevelSpec, TaxonomyInfo
from prometheus.training.trainer import TrainingConfig, train_model


def _make_synthetic_data(n=60, embed_dim=16):
    """Create a simple 2-level synthetic dataset."""
    rng = np.random.default_rng(42)

    # 2 levels: 3 classes, 4 classes
    taxonomy = TaxonomyInfo(
        name="synth",
        levels=[
            LevelSpec(
                depth=0,
                name="L0",
                classes=["A", "B", "C"],
                class_to_idx={"A": 0, "B": 1, "C": 2},
                parent_dims=[],
            ),
            LevelSpec(
                depth=1,
                name="L1",
                classes=["X", "Y", "Z", "W"],
                class_to_idx={"X": 0, "Y": 1, "Z": 2, "W": 3},
                parent_dims=[3],
            ),
        ],
        text_column="text",
        embedding_model="test",
        total_samples=n,
    )

    embeddings = rng.standard_normal((n, embed_dim)).astype(np.float32)
    d0 = rng.integers(0, 3, size=n)
    d1 = rng.integers(0, 4, size=n)

    train_n = int(n * 0.8)
    train = DataSplit(
        embeddings=embeddings[:train_n],
        labels_by_depth={0: d0[:train_n], 1: d1[:train_n]},
    )
    val = DataSplit(
        embeddings=embeddings[train_n:],
        labels_by_depth={0: d0[train_n:], 1: d1[train_n:]},
    )

    return taxonomy, train, val, embed_dim


def test_train_model_runs():
    taxonomy, train, val, embed_dim = _make_synthetic_data()

    config = TrainingConfig(epochs=3, batch_size=16, noise_rate=0.1)
    model, result, noise = train_model(
        taxonomy=taxonomy,
        train_data=train,
        val_data=val,
        embedding_dim=embed_dim,
        config=config,
    )

    assert model.num_levels == 2
    assert len(result.level_metrics) == 2
    assert result.level_metrics[0].val_accuracy > 0
    assert result.level_metrics[1].val_accuracy >= 0  # may be 0 with tiny random data

    # Check confusion profiles exist
    assert len(noise.confusion_profiles) > 0


def test_train_model_with_calibration():
    taxonomy, train, val, embed_dim = _make_synthetic_data()

    config = TrainingConfig(epochs=2, batch_size=32)
    model, result, _ = train_model(
        taxonomy=taxonomy,
        train_data=train,
        val_data=val,
        embedding_dim=embed_dim,
        config=config,
    )

    # Calibration should have been applied
    for lm in result.level_metrics:
        assert lm.calibration is not None
        assert lm.calibration.temperature > 0
