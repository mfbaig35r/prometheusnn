"""Tests for novelty detection."""

import tempfile
from pathlib import Path

import numpy as np

from prometheus.inference.novelty import NoveltyDetector


def test_fit_and_score():
    rng = np.random.default_rng(42)
    n = 50
    embs = rng.standard_normal((n, 16)).astype(np.float32)
    labels = np.array([0] * 25 + [1] * 25)
    classes = ["classA", "classB"]

    det = NoveltyDetector(z_score_threshold=2.5)
    det.fit(embs, labels, classes)

    assert len(det.centroids) == 2

    # Score an in-distribution sample
    score = det.score(embs[0], "classA", confidence=0.9)
    assert not score.is_novel

    # Score a far-out sample
    outlier = np.ones(16, dtype=np.float32) * 100
    score = det.score(outlier, "classA", confidence=0.3)
    assert score.centroid_z_score > 2.5


def test_save_load_roundtrip():
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((30, 8)).astype(np.float32)
    labels = np.array([0] * 15 + [1] * 15)
    classes = ["X", "Y"]

    det = NoveltyDetector()
    det.fit(embs, labels, classes)

    with tempfile.TemporaryDirectory() as tmpdir:
        det.save(Path(tmpdir))

        det2 = NoveltyDetector()
        det2.load(Path(tmpdir))

        assert set(det2.centroids.keys()) == {"X", "Y"}
        assert det2.training_embeddings is not None
        assert det2.training_embeddings.shape == (30, 8)
