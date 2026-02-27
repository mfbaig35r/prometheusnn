"""Tests for CascadePredictor."""

import numpy as np
from prometheus.data.taxonomy import TaxonomyInfo, LevelSpec
from prometheus.training.cascade import HierarchicalClassifier
from prometheus.inference.novelty import NoveltyDetector
from prometheus.inference.router import ThresholdRouter
from prometheus.inference.predictor import CascadePredictor


def _make_predictor():
    taxonomy = TaxonomyInfo(
        name="test",
        levels=[
            LevelSpec(depth=0, name="A", classes=["a1", "a2"],
                      class_to_idx={"a1": 0, "a2": 1}, parent_dims=[]),
            LevelSpec(depth=1, name="B", classes=["b1", "b2", "b3"],
                      class_to_idx={"b1": 0, "b2": 1, "b3": 2}, parent_dims=[2]),
        ],
        text_column="text",
        embedding_model="test",
    )

    model = HierarchicalClassifier(
        embedding_dim=16,
        num_classes_per_level=[2, 3],
        hidden_dims=[8],
    )

    # Fit novelty detector
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((20, 16)).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 5 + [2] * 5)
    novelty = NoveltyDetector()
    novelty.fit(embs, labels, ["b1", "b2", "b3"])

    thresholds = {0: 0.90, 1: 0.80, "margin": 0.15, "z_score": 2.5, "novelty_conf": 0.70}
    router = ThresholdRouter(thresholds, ["A", "B"])

    return CascadePredictor(model=model, taxonomy=taxonomy,
                            novelty_detector=novelty, router=router)


def test_predict_returns_valid_result():
    predictor = _make_predictor()
    emb = np.random.randn(16).astype(np.float32)

    result = predictor.predict(emb, text="test item")

    assert len(result.path) == 2
    assert result.path[0] in ["a1", "a2"]
    assert result.path[1] in ["b1", "b2", "b3"]
    assert result.routing["decision"] in ["AUTO_ACCEPT", "HUMAN_REVIEW", "OUT_OF_TAXONOMY"]
    assert "centroid_z_score" in result.novelty


def test_predict_batch():
    predictor = _make_predictor()
    embs = np.random.randn(5, 16).astype(np.float32)

    results = predictor.predict_batch(embs)
    assert len(results) == 5
