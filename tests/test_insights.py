"""Tests for educational learning insights module."""

import numpy as np
import pytest

from prometheus.training.trainer import LevelMetrics
from prometheus.training.calibration import CalibrationResult
from prometheus.data.taxonomy import TaxonomyInfo, LevelSpec
from prometheus.data.splits import DataSplit
from prometheus.insights import (
    _ece_quality,
    _accuracy_assessment,
    build_training_insights,
    build_evaluation_insights,
    build_model_health,
)


# ─── Helper tests ───


class TestEceQuality:
    def test_excellent(self):
        assert _ece_quality(0.01) == "excellent"

    def test_good(self):
        assert _ece_quality(0.04) == "good"

    def test_fair(self):
        assert _ece_quality(0.08) == "fair"

    def test_poor(self):
        assert _ece_quality(0.15) == "poor"


class TestAccuracyAssessment:
    def test_strong(self):
        assert _accuracy_assessment(0.95) == "strong"

    def test_good(self):
        assert _accuracy_assessment(0.80) == "good"

    def test_moderate(self):
        assert _accuracy_assessment(0.65) == "moderate"

    def test_weak(self):
        assert _accuracy_assessment(0.45) == "weak"


# ─── Fixtures ───


def _make_taxonomy():
    return TaxonomyInfo(
        name="test",
        levels=[
            LevelSpec(depth=0, name="Category", classes=["A", "B", "C"],
                      class_to_idx={"A": 0, "B": 1, "C": 2}, parent_dims=[]),
            LevelSpec(depth=1, name="SubCategory", classes=["X", "Y", "Z", "W"],
                      class_to_idx={"X": 0, "Y": 1, "Z": 2, "W": 3}, parent_dims=[3]),
        ],
        text_column="text",
        embedding_model="test",
        total_samples=100,
    )


def _make_level_metrics(
    val_accs=(0.85, 0.70),
    train_accs=(0.90, 0.88),
    temps=(1.1, 1.5),
    ece_befores=(0.08, 0.12),
    ece_afters=(0.03, 0.05),
    best_epochs=(5, 3),
    total_epochs=(10, 8),
):
    """Create level metrics for a 2-level taxonomy."""
    metrics = []
    names = ["Category", "SubCategory"]
    num_classes = [3, 4]

    for i in range(2):
        cm = np.zeros((num_classes[i], num_classes[i]), dtype=int)
        # Diagonal: correct predictions
        for j in range(num_classes[i]):
            cm[j, j] = 10
        # Off-diagonal: some confusion
        if num_classes[i] >= 2:
            cm[0, 1] = 3
        if num_classes[i] >= 3:
            cm[1, 2] = 2

        calib = CalibrationResult(
            temperature=temps[i],
            ece_before=ece_befores[i],
            ece_after=ece_afters[i],
        )
        metrics.append(LevelMetrics(
            depth=i,
            name=names[i],
            train_loss=0.5,
            train_accuracy=train_accs[i],
            val_loss=0.6,
            val_accuracy=val_accs[i],
            confusion_matrix=cm,
            calibration=calib,
            best_epoch=best_epochs[i],
            total_epochs=total_epochs[i],
        ))
    return metrics


def _make_train_data():
    rng = np.random.default_rng(42)
    n = 100
    return DataSplit(
        embeddings=rng.standard_normal((n, 16)).astype(np.float32),
        labels_by_depth={
            0: rng.integers(0, 3, size=n),
            1: rng.integers(0, 4, size=n),
        },
    )


def _make_imbalanced_train_data():
    """Create data with severe class imbalance at level 0."""
    rng = np.random.default_rng(42)
    n = 100
    # Level 0: class 0 = 90 samples, class 1 = 5, class 2 = 5
    d0 = np.array([0] * 90 + [1] * 5 + [2] * 5)
    rng.shuffle(d0)
    return DataSplit(
        embeddings=rng.standard_normal((n, 16)).astype(np.float32),
        labels_by_depth={
            0: d0,
            1: rng.integers(0, 4, size=n),
        },
    )


# ─── Training insights tests ───


class TestBuildTrainingInsights:
    def test_confused_pairs_sorted_by_count(self):
        taxonomy = _make_taxonomy()
        metrics = _make_level_metrics()
        train_data = _make_train_data()

        result = build_training_insights(metrics, taxonomy, train_data)
        pairs = result["most_confused_pairs"]
        assert len(pairs) > 0
        # Should be sorted descending by count
        counts = [p["count"] for p in pairs]
        assert counts == sorted(counts, reverse=True)

    def test_confused_pairs_have_class_names(self):
        taxonomy = _make_taxonomy()
        metrics = _make_level_metrics()
        train_data = _make_train_data()

        result = build_training_insights(metrics, taxonomy, train_data)
        pairs = result["most_confused_pairs"]
        # Top pair at Category level should have class names (A/B/C), not indices
        cat_pairs = [p for p in pairs if p["level"] == "Category"]
        if cat_pairs:
            assert cat_pairs[0]["true_class"] in ["A", "B", "C"]
            assert cat_pairs[0]["predicted_class"] in ["A", "B", "C"]

    def test_early_stopping_reported(self):
        taxonomy = _make_taxonomy()
        metrics = _make_level_metrics(best_epochs=(3, 7), total_epochs=(10, 10))
        train_data = _make_train_data()

        result = build_training_insights(metrics, taxonomy, train_data)
        es = result["early_stopping"]
        assert len(es) == 2
        assert es[0]["converged_early"] is True  # 3 < 10-1
        assert es[1]["converged_early"] is True  # 7 < 10-1

    def test_early_stopping_none(self):
        taxonomy = _make_taxonomy()
        metrics = _make_level_metrics()
        # Override to None
        for m in metrics:
            m.best_epoch = None
            m.total_epochs = None
        train_data = _make_train_data()

        result = build_training_insights(metrics, taxonomy, train_data)
        assert result["early_stopping"] == []

    def test_class_balance_imbalanced(self):
        taxonomy = _make_taxonomy()
        metrics = _make_level_metrics()
        train_data = _make_imbalanced_train_data()

        result = build_training_insights(metrics, taxonomy, train_data)
        balances = result["class_balance"]
        cat_balance = [b for b in balances if b["level"] == "Category"][0]
        assert cat_balance["imbalance_ratio"] >= 10

    def test_interpretation_non_empty(self):
        taxonomy = _make_taxonomy()
        metrics = _make_level_metrics()
        train_data = _make_train_data()

        result = build_training_insights(metrics, taxonomy, train_data)
        assert len(result["interpretation"]) >= 1


# ─── Evaluation insights tests ───


class TestBuildEvaluationInsights:
    def _per_level_accuracy(self, accs):
        return {
            name: {"accuracy": acc, "correct": int(acc * 100), "total": 100}
            for name, acc in zip(["Category", "SubCategory"], accs)
        }

    def test_routing_healthy(self):
        result = build_evaluation_insights(
            self._per_level_accuracy([0.85, 0.80]),
            {"AUTO_ACCEPT": 80, "HUMAN_REVIEW": 15, "OUT_OF_TAXONOMY": 5},
            100,
        )
        assert result["routing_analysis"]["assessment"] == "healthy"

    def test_routing_too_aggressive(self):
        result = build_evaluation_insights(
            self._per_level_accuracy([0.85, 0.80]),
            {"AUTO_ACCEPT": 30, "HUMAN_REVIEW": 60, "OUT_OF_TAXONOMY": 10},
            100,
        )
        assert result["routing_analysis"]["assessment"] == "too_aggressive"

    def test_routing_high_novelty(self):
        result = build_evaluation_insights(
            self._per_level_accuracy([0.85, 0.80]),
            {"AUTO_ACCEPT": 50, "HUMAN_REVIEW": 20, "OUT_OF_TAXONOMY": 30},
            100,
        )
        assert result["routing_analysis"]["assessment"] == "high_novelty"

    def test_accuracy_vs_training_degraded(self):
        manifest = {
            "per_level_metrics": {
                "Category": {"val_accuracy": 0.90, "train_accuracy": 0.92},
                "SubCategory": {"val_accuracy": 0.85, "train_accuracy": 0.88},
            }
        }
        result = build_evaluation_insights(
            self._per_level_accuracy([0.88, 0.70]),  # SubCategory dropped > 10pp
            {"AUTO_ACCEPT": 80, "HUMAN_REVIEW": 20},
            100,
            manifest=manifest,
        )
        avt = result["accuracy_vs_training"]
        assert avt is not None
        sub_avt = [a for a in avt if a["level"] == "SubCategory"][0]
        assert sub_avt["assessment"] == "degraded"

    def test_accuracy_vs_training_no_manifest(self):
        result = build_evaluation_insights(
            self._per_level_accuracy([0.85, 0.75]),
            {"AUTO_ACCEPT": 90, "HUMAN_REVIEW": 10},
            100,
            manifest=None,
        )
        assert result["accuracy_vs_training"] is None

    def test_interpretation_non_empty(self):
        result = build_evaluation_insights(
            self._per_level_accuracy([0.85, 0.75]),
            {"AUTO_ACCEPT": 80, "HUMAN_REVIEW": 15, "OUT_OF_TAXONOMY": 5},
            100,
        )
        assert len(result["interpretation"]) >= 1


# ─── Model health tests ───


class TestBuildModelHealth:
    def _manifest(self, val_accs=(0.85, 0.78), train_accs=(0.88, 0.82),
                  ece_afters=(0.02, 0.04)):
        return {
            "taxonomy": {
                "levels": [
                    {"name": "Category", "classes": ["A", "B", "C"]},
                    {"name": "SubCategory", "classes": ["X", "Y", "Z", "W"]},
                ],
            },
            "per_level_metrics": {
                "Category": {
                    "val_accuracy": val_accs[0], "train_accuracy": train_accs[0],
                    "calibration": {
                        "temperature": 1.1, "ece_before": 0.08, "ece_after": ece_afters[0],
                    },
                },
                "SubCategory": {
                    "val_accuracy": val_accs[1], "train_accuracy": train_accs[1],
                    "calibration": {
                        "temperature": 1.2, "ece_before": 0.10, "ece_after": ece_afters[1],
                    },
                },
            },
        }

    def test_production_readiness_ready(self):
        result = build_model_health(self._manifest(
            val_accs=(0.88, 0.80),
            train_accs=(0.90, 0.83),
            ece_afters=(0.02, 0.03),
        ))
        assert result["production_readiness"]["overall"] == "ready"
        assert result["production_readiness"]["flags"] == []

    def test_production_readiness_caution(self):
        result = build_model_health(self._manifest(
            val_accs=(0.85, 0.62),
            train_accs=(0.88, 0.65),
            ece_afters=(0.03, 0.04),
        ))
        assert result["production_readiness"]["overall"] == "caution"

    def test_production_readiness_not_ready(self):
        result = build_model_health(self._manifest(
            val_accs=(0.55, 0.45),
            train_accs=(0.95, 0.90),
            ece_afters=(0.15, 0.12),
        ))
        assert result["production_readiness"]["overall"] == "not_ready"
        assert len(result["production_readiness"]["flags"]) >= 2

    def test_interpretation_non_empty(self):
        result = build_model_health(self._manifest())
        assert len(result["interpretation"]) >= 1
