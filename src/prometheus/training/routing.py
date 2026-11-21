"""Fit routing thresholds so AUTO_ACCEPT is trustworthy on held-out data.

The default thresholds (0.90 at the root down to 0.75 at the leaf) are guesses. On a large
taxonomy they let through leaf predictions that are wrong a third of the time, mostly where
the true leaf had few or no training rows. ``fit_level_thresholds`` raises each level's
threshold, top-down, until the predictions it would auto-accept on validation data reach a
target precision. Thresholds never go below the defaults.

Fitting top-down on the items that already clear every level above means a level's
precision is the precision of the whole confident path down to it: a child can only be
right if its parent is, because children are masked to the predicted parent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prometheus.data.splits import DataSplit
from prometheus.inference.router import RouterConfig

# A threshold no confidence can reach: that level (and everything below it) is never
# auto-accepted, and those items go to HUMAN_REVIEW with the confident prefix above it.
UNREACHABLE = 1.01


@dataclass
class RoutingSignals:
    """Per item and level: the best path's confidence, its margin, and whether it was right."""

    confidences: np.ndarray  # (n, levels)
    margins: np.ndarray  # (n, levels)
    correct: np.ndarray  # (n, levels) bool


def collect_routing_signals(predictor, data: DataSplit, beam_width: int = 3) -> RoutingSignals:
    """Run the predictor exactly as `predict` does and record what routing sees."""
    taxonomy = predictor.taxonomy
    index = [
        {name: i for i, name in enumerate(taxonomy.get_level(d).classes)}
        for d in range(taxonomy.num_levels)
    ]
    n, levels = len(data), taxonomy.num_levels
    conf = np.zeros((n, levels))
    margin = np.zeros((n, levels))
    correct = np.zeros((n, levels), dtype=bool)
    for i in range(n):
        result = predictor.predict(
            data.embeddings[i], use_beam_search=beam_width > 1, beam_width=beam_width
        )
        for d, p in enumerate(result.predictions):
            conf[i, d] = p["confidence"]
            margin[i, d] = p["margin"]
            correct[i, d] = index[d][p["predicted"]] == int(data.labels_by_depth[d][i])
    return RoutingSignals(conf, margin, correct)


def fit_level_thresholds(
    signals: RoutingSignals,
    base: RouterConfig,
    target_precision: float = 0.95,
    min_support: int = 20,
) -> tuple[RouterConfig, dict]:
    """Smallest threshold per level whose auto-accepted predictions reach the target precision.

    A threshold only counts if at least ``min_support`` validation items would be accepted at
    it. A level that cannot reach the target gets ``UNREACHABLE``, and so does every level below.
    Returns the fitted config and a report of validation precision and coverage per level.
    """
    conf, margin, correct = signals.confidences, signals.margins, signals.correct
    n, levels = conf.shape
    passing = np.ones(n, dtype=bool)
    fitted = base.model_copy(deep=True)
    report_levels = []

    for d in range(levels):
        floor = base.threshold_for_depth(d)
        pool = passing & (margin[:, d] >= base.margin)
        chosen = None
        # Try the default first, then each observed confidence above it: only raise when needed.
        for t in [floor, *np.unique(conf[pool & (conf[:, d] > floor), d])]:
            accepted = pool & (conf[:, d] >= t)
            if accepted.sum() < min_support:
                break
            if correct[accepted, d].mean() >= target_precision:
                chosen = float(t)
                break
        threshold = UNREACHABLE if chosen is None else chosen
        fitted.level_thresholds[d] = threshold
        passing = pool & (conf[:, d] >= threshold)
        report_levels.append(
            {
                "depth": d,
                "threshold": round(threshold, 4),
                "default": round(floor, 4),
                "reachable": chosen is not None,
                "accepted": int(passing.sum()),
                "precision": round(float(correct[passing, d].mean()), 4) if passing.any() else None,
            }
        )

    report = {
        "target_precision": target_precision,
        "validation_items": n,
        "auto_accept_rate": round(float(passing.mean()), 4),
        "auto_accept_precision": report_levels[-1]["precision"],
        "levels": report_levels,
    }
    return fitted, report


def fit_routing(
    model,
    taxonomy,
    novelty,
    val_data: DataSplit,
    target_precision: float = 0.95,
    min_support: int = 20,
    beam_width: int = 3,
) -> tuple[RouterConfig, dict]:
    """Fit thresholds for a freshly trained model on its validation split.

    Keeps the defaults when the validation split is too small to measure precision.
    """
    from prometheus.inference.predictor import CascadePredictor
    from prometheus.inference.router import ThresholdRouter
    from prometheus.storage.model_store import generate_thresholds

    base = generate_thresholds(taxonomy.num_levels)
    if len(val_data) < 2 * min_support:
        return base, {
            "target_precision": target_precision,
            "skipped": (
                f"validation split has {len(val_data)} items; at least {2 * min_support} are "
                "needed to fit thresholds, so the defaults are kept"
            ),
        }
    predictor = CascadePredictor(
        model=model,
        taxonomy=taxonomy,
        novelty_detector=novelty,
        router=ThresholdRouter(base, taxonomy.level_names),
        parent_child_map=taxonomy.parent_child_map,
        adaptive_factor=base.adaptive_factor,
    )
    signals = collect_routing_signals(predictor, val_data, beam_width=beam_width)
    return fit_level_thresholds(signals, base, target_precision, min_support)
