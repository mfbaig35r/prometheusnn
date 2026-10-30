"""Tests for novelty detection."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

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


class _Payload:
    """Unpickling this creates a file: proof that loading ran attacker code."""

    def __init__(self, marker: Path):
        self.marker = str(marker)

    def __reduce__(self):
        return (open, (self.marker, "w"))


def test_load_never_unpickles_centroids(tmp_path):
    marker = tmp_path / "executed"
    np.savez(
        tmp_path / "novelty_centroids.npz",
        centroid_0=np.zeros(4),
        class_names=np.array([_Payload(marker)], dtype=object),
        mean_dists=[1.0],
        std_dists=[1.0],
        p95_dists=[1.0],
        counts=[1],
    )
    with pytest.raises(ValueError, match="pickled objects"):
        NoveltyDetector().load(tmp_path)
    assert not marker.exists()


def test_load_never_unpickles_embeddings(tmp_path):
    marker = tmp_path / "executed"
    det = NoveltyDetector()
    det.fit(np.eye(4, dtype=np.float32), np.array([0, 0, 1, 1]), ["a", "b"])
    det.save(tmp_path)
    np.save(tmp_path / "novelty_embeddings.npy", np.array([_Payload(marker)], dtype=object))
    with pytest.raises(ValueError):
        NoveltyDetector().load(tmp_path)
    assert not marker.exists()


def test_saved_files_need_no_pickle(tmp_path):
    rng = np.random.default_rng(0)
    det = NoveltyDetector()
    det.fit(rng.standard_normal((20, 4)), np.array([0] * 10 + [1] * 10), ["Café", "naïve/class"])
    det.save(tmp_path)
    with np.load(tmp_path / "novelty_centroids.npz", allow_pickle=False) as data:
        assert all(data[k].dtype != object for k in data.files)
    det2 = NoveltyDetector()
    det2.load(tmp_path)
    assert set(det2.centroids) == {"Café", "naïve/class"}
