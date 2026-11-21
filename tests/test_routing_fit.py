"""Tests for fitting routing thresholds on validation data."""

import numpy as np
import pytest

from prometheus.data.splits import DataSplit
from prometheus.inference.router import RouterConfig
from prometheus.training.routing import (
    UNREACHABLE,
    RoutingSignals,
    fit_level_thresholds,
    fit_routing,
)

BASE = RouterConfig(level_thresholds={0: 0.9, 1: 0.75}, margin=0.15)


def _signals(conf0, correct0, conf1=None, correct1=None, margin=0.5):
    n = len(conf0)
    conf1 = conf0 if conf1 is None else conf1
    correct1 = correct0 if correct1 is None else correct1
    return RoutingSignals(
        confidences=np.column_stack([conf0, conf1]).astype(float),
        margins=np.full((n, 2), margin),
        correct=np.column_stack([correct0, correct1]).astype(bool),
    )


def test_picks_smallest_threshold_reaching_target():
    # Confidence 0.91-0.95 is right 50% of the time; 0.96-0.99 is always right.
    conf = np.concatenate([np.linspace(0.91, 0.95, 40), np.linspace(0.96, 0.99, 40)])
    correct = np.concatenate([np.arange(40) % 2 == 0, np.ones(40, bool)])
    cfg, report = fit_level_thresholds(_signals(conf, correct), BASE, target_precision=0.95)
    assert 0.94 < cfg.level_thresholds[0] <= 0.96
    assert report["levels"][0]["precision"] >= 0.95


def test_never_goes_below_default():
    conf = np.linspace(0.5, 0.99, 100)
    cfg, _ = fit_level_thresholds(_signals(conf, np.ones(100, bool)), BASE, target_precision=0.9)
    assert cfg.level_thresholds[0] >= 0.9
    assert cfg.level_thresholds[1] >= 0.75


def test_unreachable_level_blocks_everything_below():
    conf = np.full(100, 0.97)
    correct = np.arange(100) % 2 == 0  # 50% precision at any threshold
    cfg, report = fit_level_thresholds(_signals(conf, correct), BASE, target_precision=0.95)
    assert cfg.level_thresholds[0] == UNREACHABLE
    assert cfg.level_thresholds[1] == UNREACHABLE
    assert report["auto_accept_rate"] == 0.0
    assert report["auto_accept_precision"] is None


def test_deeper_level_fits_on_items_that_cleared_the_level_above():
    n = 100
    conf0 = np.full(n, 0.99)
    conf1 = np.linspace(0.76, 0.99, n)
    correct1 = conf1 > 0.9  # leaf is right only above 0.9
    cfg, report = fit_level_thresholds(
        _signals(conf0, np.ones(n, bool), conf1, correct1), BASE, target_precision=0.95
    )
    assert cfg.level_thresholds[0] == pytest.approx(0.9)
    assert 0.85 < cfg.level_thresholds[1] <= 0.91
    assert report["auto_accept_precision"] >= 0.95


def test_margin_failures_are_not_auto_accepted():
    conf = np.full(60, 0.99)
    cfg, report = fit_level_thresholds(
        _signals(conf, np.ones(60, bool), margin=0.05), BASE, target_precision=0.9
    )
    assert cfg.level_thresholds[0] == UNREACHABLE
    assert report["levels"][0]["accepted"] == 0


def test_needs_min_support():
    conf = np.full(10, 0.99)
    cfg, _ = fit_level_thresholds(_signals(conf, np.ones(10, bool)), BASE, min_support=20)
    assert cfg.level_thresholds[0] == UNREACHABLE


def test_fit_routing_keeps_defaults_when_validation_is_tiny():
    tiny = DataSplit(embeddings=np.zeros((5, 4)), labels_by_depth={0: np.zeros(5), 1: np.zeros(5)})

    class _Tax:
        num_levels = 2

    cfg, report = fit_routing(None, _Tax(), None, tiny)
    assert "skipped" in report
    assert cfg.level_thresholds == {0: 0.9, 1: 0.75}
