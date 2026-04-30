"""Tests for model storage (save/load round-trip)."""

import json
import tempfile
from pathlib import Path

import numpy as np

from prometheus.data.splits import DataSplit
from prometheus.data.taxonomy import LevelSpec, TaxonomyInfo
from prometheus.storage.model_store import generate_thresholds, load_model, save_model
from prometheus.training.trainer import TrainingConfig, train_model


def test_generate_thresholds_4_levels():
    config = generate_thresholds(4)
    assert config.level_thresholds[0] == 0.90
    assert abs(config.level_thresholds[1] - 0.85) < 1e-10
    assert abs(config.level_thresholds[2] - 0.80) < 1e-10
    assert config.level_thresholds[3] == 0.75


def test_generate_thresholds_1_level():
    config = generate_thresholds(1)
    assert config.level_thresholds[0] == 0.90


def test_save_load_roundtrip():
    rng = np.random.default_rng(42)
    n = 40
    embed_dim = 16

    taxonomy = TaxonomyInfo(
        name="roundtrip_test",
        levels=[
            LevelSpec(
                depth=0,
                name="L0",
                classes=["A", "B"],
                class_to_idx={"A": 0, "B": 1},
                parent_dims=[],
            ),
            LevelSpec(
                depth=1,
                name="L1",
                classes=["X", "Y", "Z"],
                class_to_idx={"X": 0, "Y": 1, "Z": 2},
                parent_dims=[2],
            ),
        ],
        text_column="text",
        embedding_model="test",
        total_samples=n,
    )

    embs = rng.standard_normal((n, embed_dim)).astype(np.float32)
    d0 = rng.integers(0, 2, size=n)
    d1 = rng.integers(0, 3, size=n)

    train_n = 30
    train = DataSplit(embeddings=embs[:train_n], labels_by_depth={0: d0[:train_n], 1: d1[:train_n]})
    val = DataSplit(embeddings=embs[train_n:], labels_by_depth={0: d0[train_n:], 1: d1[train_n:]})

    config = TrainingConfig(epochs=2, batch_size=16)
    model, result, noise = train_model(
        taxonomy=taxonomy,
        train_data=train,
        val_data=val,
        embedding_dim=embed_dim,
        config=config,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model"
        save_model(
            path=path,
            model=model,
            taxonomy=taxonomy,
            result=result,
            config=config,
            embedding_dim=embed_dim,
            confusion_profiles=result.confusion_profiles,
        )

        # Verify manifest is valid JSON
        with open(path / "manifest.json") as f:
            manifest_data = json.load(f)
        assert manifest_data["taxonomy"]["name"] == "roundtrip_test"
        assert manifest_data["embedding_dim"] == 16

        # Load
        loaded_model, loaded_manifest = load_model(path)
        assert loaded_model.num_levels == 2
        assert loaded_manifest.taxonomy.num_levels == 2
        assert loaded_manifest.embedding_dim == embed_dim

        # Verify predictions match
        import torch

        test_emb = torch.randn(1, embed_dim)
        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            orig_logits = model.get_level(0)(test_emb)
            loaded_logits = loaded_model.get_level(0)(test_emb)

        np.testing.assert_allclose(orig_logits.numpy(), loaded_logits.numpy(), atol=1e-5)
