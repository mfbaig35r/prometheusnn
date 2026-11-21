"""Tests for novelty detection."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from prometheus.inference.novelty import NoveltyDetector, leave_one_out_knn_distances


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


def _clustered(rng, n_per=40, dim=8):
    centers = rng.standard_normal((3, dim)) * 4
    embs = np.concatenate([c + rng.standard_normal((n_per, dim)) for c in centers])
    labels = np.repeat(np.arange(3), n_per)
    return embs.astype(np.float32), labels, ["a", "b", "c"]


def test_leave_one_out_excludes_self():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    np.testing.assert_allclose(leave_one_out_knn_distances(x, k=1), [1.0, 1.0, 2.0])


def test_leave_one_out_sampled_queries_match_full():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 4))
    full = leave_one_out_knn_distances(x, k=5)
    sampled = leave_one_out_knn_distances(x, k=5, max_queries=100)
    assert len(sampled) == 100
    assert set(np.round(sampled, 8)) <= set(np.round(full, 8))


def test_knn_threshold_fit_from_training():
    embs, labels, classes = _clustered(np.random.default_rng(1))
    det = NoveltyDetector(knn_k=5, knn_percentile=99.0)
    det.fit(embs, labels, classes)
    expected = np.percentile(leave_one_out_knn_distances(embs, 5), 99.0)
    assert det.knn_threshold == pytest.approx(expected)

    far = det.score(np.full(8, 50.0, dtype=np.float32), "a", confidence=0.99)
    assert far.is_novel  # far from everything, however confident the model is
    near = det.score(embs[0] + 0.01, "a", confidence=0.1)
    assert not near.is_novel  # inside the data, however unsure the model is
    assert near.knn_threshold == pytest.approx(expected)


def test_tiny_leaf_spread_does_not_make_in_distribution_novel():
    """Regression: two near-identical rows gave a leaf a clamped 0.1 spread, so an input
    from a sibling leaf got a huge z-score and was routed OUT_OF_TAXONOMY."""
    rng = np.random.default_rng(2)
    embs, labels, classes = _clustered(rng)
    tight = embs[labels == 0][:1] + rng.standard_normal((2, 8)).astype(np.float32) * 1e-3
    embs = np.concatenate([embs, tight])
    labels = np.concatenate([labels, [3, 3]])
    det = NoveltyDetector(knn_k=5)
    det.fit(embs, labels, classes + ["tight"])

    sibling = embs[labels == 0][5]
    score = det.score(sibling, "tight", confidence=0.3)
    assert score.centroid_z_score > det.z_score_threshold  # the old rule would have fired
    assert not score.is_novel


def test_threshold_survives_save_load():
    embs, labels, classes = _clustered(np.random.default_rng(3))
    det = NoveltyDetector(knn_k=5, knn_percentile=95.0)
    det.fit(embs, labels, classes)
    with tempfile.TemporaryDirectory() as tmpdir:
        det.save(Path(tmpdir))
        loaded = NoveltyDetector()
        loaded.load(Path(tmpdir))
    assert loaded.knn_threshold == pytest.approx(det.knn_threshold)
    assert loaded.knn_percentile == 95.0


def test_model_saved_before_threshold_refits_on_load():
    embs, labels, classes = _clustered(np.random.default_rng(4))
    det = NoveltyDetector()
    det.fit(embs, labels, classes)
    with tempfile.TemporaryDirectory() as tmpdir:
        det.save(Path(tmpdir))
        npz = Path(tmpdir) / "novelty_centroids.npz"
        old = {k: v for k, v in np.load(npz).items() if not k.startswith("knn_")}
        np.savez(npz, **old)  # what 0.2.2 wrote
        loaded = NoveltyDetector()
        loaded.load(Path(tmpdir))
    assert loaded.knn_threshold == pytest.approx(det.knn_threshold)


def test_without_training_embeddings_uses_centroid_rule():
    embs, labels, classes = _clustered(np.random.default_rng(5))
    det = NoveltyDetector()
    det.fit(embs, labels, classes)
    det.training_embeddings = None
    det.knn_threshold = None
    outlier = np.full(8, 50.0, dtype=np.float32)
    assert det.score(outlier, "a", confidence=0.3).is_novel
    assert not det.score(outlier, "a", confidence=0.9).is_novel  # legacy confidence guard
