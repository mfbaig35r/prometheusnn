"""Novelty detection: is an input outside everything the model was trained on?

The decision uses the distance from the input to its k-th nearest training embedding,
against a threshold fit on the training set itself (a high percentile of leave-one-out
kNN distances). That measures distance to the whole taxonomy, so it does not depend on
the leaf prediction being right.

The per-leaf centroid z-score is still computed and reported, and is the decision rule
only for models saved without training embeddings. It is a poor decision rule on its own:
leaves with a handful of near-identical training rows have a tiny spread, so any input not
sitting on them scores as extreme even when it belongs to a sibling leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ClassCentroid:
    centroid: np.ndarray
    mean_dist: float
    std_dist: float
    p95_dist: float
    count: int


@dataclass
class NoveltyScore:
    centroid_distance: float
    centroid_z_score: float
    knn_distance: float
    is_novel: bool
    knn_threshold: float | None = None


DEFAULT_KNN_PERCENTILE = 99.0
_MAX_THRESHOLD_QUERIES = 5000


def leave_one_out_knn_distances(
    embeddings: np.ndarray, k: int, max_queries: int = _MAX_THRESHOLD_QUERIES
) -> np.ndarray:
    """Distance from training rows to their k-th nearest *other* training row.

    Uses at most ``max_queries`` rows as queries (a fixed random sample) so fitting stays
    quick on large training sets; every row is still a candidate neighbor.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    n = len(x)
    queries = np.arange(n)
    if n > max_queries:
        queries = np.sort(np.random.default_rng(0).choice(n, max_queries, replace=False))
    sq = (x**2).sum(axis=1)
    chunk = max(1, int(2e7 // max(n, 1)))
    out = np.empty(len(queries))
    for start in range(0, len(queries), chunk):
        q = queries[start : start + chunk]
        d2 = sq[q, None] + sq[None, :] - 2.0 * (x[q] @ x.T)
        d2[np.arange(len(q)), q] = np.inf  # leave the row itself out
        out[start : start + len(q)] = np.partition(d2, k - 1, axis=1)[:, k - 1]
    return np.sqrt(np.maximum(out, 0.0))


class NoveltyDetector:
    """Novelty from kNN distance to the training set; centroid z-score as a diagnostic."""

    def __init__(
        self,
        z_score_threshold: float = 2.5,
        knn_k: int = 10,
        novelty_conf_threshold: float = 0.70,
        knn_percentile: float = DEFAULT_KNN_PERCENTILE,
    ):
        self.z_score_threshold = z_score_threshold
        self.knn_k = knn_k
        self.novelty_conf_threshold = novelty_conf_threshold
        self.knn_percentile = knn_percentile
        self.centroids: dict[str, ClassCentroid] = {}
        self.training_embeddings: np.ndarray | None = None
        self.knn_threshold: float | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        leaf_labels: np.ndarray,
        leaf_classes: list[str],
    ):
        """Fit on training data using leaf-level labels."""
        self.training_embeddings = embeddings.copy()

        for class_idx, class_name in enumerate(leaf_classes):
            mask = leaf_labels == class_idx
            if mask.sum() == 0:
                continue

            class_embs = embeddings[mask]
            centroid = class_embs.mean(axis=0)
            distances = np.linalg.norm(class_embs - centroid, axis=1)

            self.centroids[class_name] = ClassCentroid(
                centroid=centroid,
                mean_dist=float(distances.mean()),
                std_dist=max(float(distances.std()), 0.1) if len(distances) > 1 else 1.0,
                p95_dist=float(np.percentile(distances, 95))
                if len(distances) > 1
                else float(distances.max()),
                count=int(mask.sum()),
            )

        self.knn_threshold = self._fit_knn_threshold()

    def _fit_knn_threshold(self) -> float | None:
        """A high percentile of the training set's own leave-one-out kNN distances."""
        x = self.training_embeddings
        if x is None or len(x) <= self.knn_k:
            return None
        return float(np.percentile(leave_one_out_knn_distances(x, self.knn_k), self.knn_percentile))

    def score(self, embedding: np.ndarray, predicted_leaf: str, confidence: float) -> NoveltyScore:
        if predicted_leaf in self.centroids:
            info = self.centroids[predicted_leaf]
            dist = float(np.linalg.norm(embedding - info.centroid))
            z_score = (dist - info.mean_dist) / info.std_dist if info.std_dist > 0 else 0.0
        else:
            dist = float("inf")
            z_score = 10.0

        knn_dist = self._knn_distance(embedding)
        if self.knn_threshold is not None:
            is_novel = knn_dist > self.knn_threshold
        else:
            # Legacy rule, for models saved without training embeddings.
            is_novel = z_score > self.z_score_threshold and confidence < self.novelty_conf_threshold

        return NoveltyScore(
            centroid_distance=dist,
            centroid_z_score=z_score,
            knn_distance=knn_dist,
            is_novel=is_novel,
            knn_threshold=self.knn_threshold,
        )

    def _knn_distance(self, embedding: np.ndarray) -> float:
        if self.training_embeddings is None:
            return 0.0
        distances = np.linalg.norm(self.training_embeddings - embedding, axis=1)
        k = min(self.knn_k, len(distances))
        return float(np.partition(distances, k - 1)[k - 1])

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.savez(path / "novelty_centroids.npz", **self.get_centroids_data())

        if self.training_embeddings is not None:
            np.save(path / "novelty_embeddings.npy", self.training_embeddings)

    def load(self, path: Path):
        path = Path(path)
        # Never unpickle: a model directory is data, and a crafted .npz with object arrays would
        # run code on load. save() writes only numeric and string arrays, which need no pickle.
        data = np.load(path / "novelty_centroids.npz", allow_pickle=False)
        try:
            class_names = data["class_names"]
            mean_dists = data["mean_dists"]
            std_dists = data["std_dists"]
            p95_dists = data["p95_dists"]
            counts = data["counts"]
            centroids = [data[f"centroid_{i}"] for i in range(len(class_names))]
        except ValueError as exc:
            raise ValueError(
                f"{path / 'novelty_centroids.npz'} contains pickled objects; refusing to load it. "
                "Retrain the model to regenerate it."
            ) from exc

        self.centroids = {}
        for i, name in enumerate(class_names):
            self.centroids[str(name)] = ClassCentroid(
                centroid=centroids[i],
                mean_dist=float(mean_dists[i]),
                std_dist=max(float(std_dists[i]), 0.1),
                p95_dist=float(p95_dists[i]),
                count=int(counts[i]),
            )

        emb_path = path / "novelty_embeddings.npy"
        if emb_path.exists():
            self.training_embeddings = np.load(emb_path, allow_pickle=False)

        if "knn_threshold" in data.files:
            self.knn_threshold = float(data["knn_threshold"])
            self.knn_percentile = float(data["knn_percentile"])
        else:
            # Saved before 0.2.3: fit the threshold now from the stored training embeddings.
            self.knn_threshold = self._fit_knn_threshold()

    def get_centroids_data(self) -> dict:
        """Return data dict suitable for np.savez."""
        data = {
            f"centroid_{i}": info.centroid for i, (_, info) in enumerate(self.centroids.items())
        }
        data["class_names"] = list(self.centroids.keys())
        data["mean_dists"] = [info.mean_dist for info in self.centroids.values()]
        data["std_dists"] = [info.std_dist for info in self.centroids.values()]
        data["p95_dists"] = [info.p95_dist for info in self.centroids.values()]
        data["counts"] = [info.count for info in self.centroids.values()]
        if self.knn_threshold is not None:
            data["knn_threshold"] = self.knn_threshold
            data["knn_percentile"] = self.knn_percentile
        return data
