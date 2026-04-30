"""Tests for threshold router."""

from prometheus.inference.beam_search import CascadePrediction, LevelPrediction
from prometheus.inference.novelty import NoveltyScore
from prometheus.inference.router import RouterConfig, RoutingDecision, ThresholdRouter


def _config(**overrides) -> RouterConfig:
    return RouterConfig(level_thresholds={0: 0.90, 1: 0.85, 2: 0.80}, margin=0.15, **overrides)


def _make_prediction(confs, margins=None):
    """Helper to build a CascadePrediction with given confidences."""
    names = ["Cat", "Sub", "Type"]
    classes = ["A", "B", "C"]
    margins = margins or [0.5] * len(confs)

    lps = [
        LevelPrediction(
            depth=i,
            name=names[i],
            predicted=classes[i],
            predicted_idx=i,
            confidence=confs[i],
            margin=margins[i],
            top_candidates=[],
        )
        for i in range(len(confs))
    ]
    return CascadePrediction(
        path=[lp.predicted for lp in lps],
        path_indices=[lp.predicted_idx for lp in lps],
        path_score=-1.0,
        level_predictions=lps,
        alternative_paths=[],
    )


def test_auto_accept():
    router = ThresholdRouter(_config(), level_names=["Cat", "Sub", "Type"])
    pred = _make_prediction([0.95, 0.90, 0.85])
    novelty = NoveltyScore(0.5, 0.5, 0.5, is_novel=False)

    result = router.route(pred, novelty)
    assert result.decision == RoutingDecision.AUTO_ACCEPT
    assert result.confident_depth == 2


def test_human_review_low_confidence():
    router = ThresholdRouter(_config(), level_names=["Cat", "Sub", "Type"])
    pred = _make_prediction([0.95, 0.70, 0.85])  # L1 below threshold
    novelty = NoveltyScore(0.5, 0.5, 0.5, is_novel=False)

    result = router.route(pred, novelty)
    assert result.decision == RoutingDecision.HUMAN_REVIEW
    assert result.confident_depth == 0


def test_out_of_taxonomy():
    router = ThresholdRouter(_config(), level_names=["Cat", "Sub", "Type"])
    pred = _make_prediction([0.95, 0.90, 0.85])
    novelty = NoveltyScore(10.0, 5.0, 5.0, is_novel=True)

    result = router.route(pred, novelty)
    assert result.decision == RoutingDecision.OUT_OF_TAXONOMY


def test_narrow_margin_triggers_review():
    router = ThresholdRouter(_config(), level_names=["Cat", "Sub", "Type"])
    pred = _make_prediction([0.95, 0.90, 0.85], margins=[0.05, 0.5, 0.5])
    novelty = NoveltyScore(0.5, 0.5, 0.5, is_novel=False)

    result = router.route(pred, novelty)
    assert result.decision == RoutingDecision.HUMAN_REVIEW
    assert "margin" in result.review_reason
