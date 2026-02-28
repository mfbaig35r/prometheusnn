"""Educational learning insights for Prometheus training, evaluation, and model health."""

from __future__ import annotations

import numpy as np

from prometheus.data.splits import DataSplit
from prometheus.data.taxonomy import TaxonomyInfo
from prometheus.training.trainer import LevelMetrics

# ─── Shared helpers ───


def _overfit_assessment(train_acc: float, val_acc: float) -> tuple[float, str]:
    """Return (gap_pp, risk) based on train-val accuracy gap."""
    gap = train_acc - val_acc
    if gap > 0.15:
        return gap, "high"
    elif gap > 0.05:
        return gap, "moderate"
    else:
        return gap, "low"


def _ece_quality(ece: float) -> str:
    """Rate ECE quality."""
    if ece <= 0.02:
        return "excellent"
    elif ece <= 0.05:
        return "good"
    elif ece <= 0.10:
        return "fair"
    else:
        return "poor"


def _accuracy_assessment(acc: float) -> str:
    """Assess accuracy level."""
    if acc >= 0.90:
        return "strong"
    elif acc >= 0.75:
        return "good"
    elif acc >= 0.60:
        return "moderate"
    else:
        return "weak"


# ─── Training insights ───


def build_training_insights(
    level_metrics: list[LevelMetrics],
    taxonomy: TaxonomyInfo,
    train_data: DataSplit,
) -> dict:
    """Build structured learning insights from training results.

    Returns a dict with most_confused_pairs, class_balance, early_stopping,
    and interpretation sentences.
    """
    # Most confused pairs (top 5 globally from confusion matrices)
    confused_pairs = []
    for lm in level_metrics:
        level = taxonomy.get_level(lm.depth)
        cm = lm.confusion_matrix
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    true_class = level.classes[i] if i < len(level.classes) else str(i)
                    pred_class = level.classes[j] if j < len(level.classes) else str(j)
                    confused_pairs.append(
                        {
                            "level": lm.name,
                            "true_class": true_class,
                            "predicted_class": pred_class,
                            "count": int(cm[i, j]),
                        }
                    )
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    most_confused_pairs = confused_pairs[:5]

    # Early stopping
    early_stopping = []
    for lm in level_metrics:
        if lm.best_epoch is not None and lm.total_epochs is not None:
            early_stopping.append(
                {
                    "level": lm.name,
                    "best_epoch": lm.best_epoch,
                    "total_epochs": lm.total_epochs,
                    "converged_early": lm.best_epoch < lm.total_epochs - 1,
                }
            )

    # Class balance
    class_balance = []
    for lm in level_metrics:
        depth = lm.depth
        if depth in train_data.labels_by_depth:
            labels = train_data.labels_by_depth[depth]
            _, counts = np.unique(labels, return_counts=True)
            min_count = int(counts.min())
            max_count = int(counts.max())
            ratio = max_count / min_count if min_count > 0 else float("inf")
            class_balance.append(
                {
                    "level": lm.name,
                    "num_classes": len(counts),
                    "min_class_count": min_count,
                    "max_class_count": max_count,
                    "imbalance_ratio": round(ratio, 1),
                }
            )

    # Interpretation sentences
    interpretation = []

    # Overfit warnings (computed inline, not output as section)
    for lm in level_metrics:
        gap, risk = _overfit_assessment(lm.train_accuracy, lm.val_accuracy)
        if risk == "high":
            interpretation.append(
                f"{lm.name} shows overfitting (train {lm.train_accuracy:.0%} vs val {lm.val_accuracy:.0%}) "
                f"— consider more data or higher dropout."
            )

    # Top confused pair
    if most_confused_pairs:
        top = most_confused_pairs[0]
        interpretation.append(
            f"At {top['level']}, '{top['true_class']}' is most often confused with "
            f"'{top['predicted_class']}' ({top['count']} errors)."
        )

    # Class imbalance warnings
    for cb in class_balance:
        if cb["imbalance_ratio"] >= 10:
            interpretation.append(
                f"{cb['level']} has severe class imbalance "
                f"({cb['max_class_count']}/{cb['min_class_count']} = {cb['imbalance_ratio']:.0f}x) "
                f"— underrepresented classes may underperform."
            )

    # Early stopping info
    for es in early_stopping:
        if es["converged_early"]:
            interpretation.append(
                f"{es['level']} converged at epoch {es['best_epoch']}/{es['total_epochs']} "
                f"— training time could be reduced."
            )

    return {
        "most_confused_pairs": most_confused_pairs,
        "class_balance": class_balance,
        "early_stopping": early_stopping,
        "interpretation": interpretation,
    }


# ─── Evaluation insights ───


def build_evaluation_insights(
    per_level_accuracy: dict[str, dict],
    routing_distribution: dict[str, int],
    total_items: int,
    manifest: dict | None = None,
) -> dict:
    """Build structured insights from evaluation results.

    Args:
        per_level_accuracy: {level_name: {"accuracy": float, ...}}
        routing_distribution: {"AUTO_ACCEPT": n, "HUMAN_REVIEW": n, ...}
        total_items: Total items evaluated
        manifest: Optional model manifest for training comparison
    """
    level_names = list(per_level_accuracy.keys())

    # Weakest level accuracy (used by routing assessment logic)
    weakest_name = min(per_level_accuracy, key=lambda k: per_level_accuracy[k]["accuracy"])
    weakest_acc = per_level_accuracy[weakest_name]["accuracy"]

    # Routing analysis
    auto_accept = routing_distribution.get("AUTO_ACCEPT", 0)
    human_review = routing_distribution.get("HUMAN_REVIEW", 0)
    out_of_taxonomy = routing_distribution.get("OUT_OF_TAXONOMY", 0)

    auto_pct = auto_accept / total_items if total_items > 0 else 0.0
    hr_pct = human_review / total_items if total_items > 0 else 0.0
    oot_pct = out_of_taxonomy / total_items if total_items > 0 else 0.0

    if oot_pct > 0.20:
        routing_assessment = "high_novelty"
    elif hr_pct > 0.50:
        routing_assessment = "too_aggressive"
    elif auto_pct > 0.95 and weakest_acc < 0.80:
        routing_assessment = "too_lenient"
    else:
        routing_assessment = "healthy"

    routing_analysis = {
        "auto_accept_pct": round(auto_pct, 4),
        "human_review_pct": round(hr_pct, 4),
        "out_of_taxonomy_pct": round(oot_pct, 4),
        "assessment": routing_assessment,
    }

    # Accuracy vs training
    accuracy_vs_training = None
    if manifest is not None:
        per_level_metrics = manifest.get("per_level_metrics", {})
        if per_level_metrics:
            accuracy_vs_training = []
            for name in level_names:
                eval_acc = per_level_accuracy[name]["accuracy"]
                train_metrics = per_level_metrics.get(name, {})
                train_val_acc = train_metrics.get("val_accuracy", train_metrics.get("val_acc"))
                if train_val_acc is not None:
                    delta = eval_acc - train_val_acc
                    if delta < -0.10:
                        assessment = "degraded"
                    elif delta > 0.05:
                        assessment = "improved"
                    else:
                        assessment = "consistent"
                    accuracy_vs_training.append(
                        {
                            "level": name,
                            "eval_accuracy": round(eval_acc, 4),
                            "train_val_accuracy": round(train_val_acc, 4),
                            "delta": round(delta, 4),
                            "assessment": assessment,
                        }
                    )

    # Interpretation
    interpretation = []

    if routing_assessment == "too_aggressive":
        interpretation.append(
            f"Routing: {auto_pct:.0%} auto-accepted, {hr_pct:.0%} for review "
            f"— over half need review, consider loosening thresholds."
        )
    elif routing_assessment == "high_novelty":
        interpretation.append(
            f"Routing: {oot_pct:.0%} flagged as out-of-taxonomy "
            f"— evaluation data may contain categories not seen during training."
        )
    else:
        interpretation.append(
            f"Routing: {auto_pct:.0%} auto-accepted, {hr_pct:.0%} for review — {routing_assessment}."
        )

    if accuracy_vs_training:
        for avt in accuracy_vs_training:
            if avt["assessment"] == "degraded":
                interpretation.append(
                    f"{avt['level']} dropped {abs(avt['delta']):.0%} vs training "
                    f"— possible distribution shift."
                )

    return {
        "routing_analysis": routing_analysis,
        "accuracy_vs_training": accuracy_vs_training,
        "interpretation": interpretation,
    }


# ─── Model health ───


def build_model_health(manifest: dict) -> dict:
    """Build model health assessment from a saved manifest.

    Args:
        manifest: The model's manifest.json contents
    """
    per_level_metrics = manifest.get("per_level_metrics", {})
    taxonomy_levels = manifest.get("taxonomy", {}).get("levels", [])

    # Collect val_accs for production readiness
    val_accs = []
    for level_info in taxonomy_levels:
        name = level_info["name"]
        metrics = per_level_metrics.get(name, {})
        va = metrics.get("val_accuracy", metrics.get("val_acc"))
        if va is not None:
            val_accs.append(va)

    # Production readiness
    flags = []
    min_val = min(val_accs) if val_accs else 0.0

    for level_info in taxonomy_levels:
        name = level_info["name"]
        metrics = per_level_metrics.get(name, {})
        va = metrics.get("val_accuracy", metrics.get("val_acc"))

        if va is not None and va < 0.60:
            flags.append(f"{name} val_accuracy below 60% ({va:.0%})")

        train_acc = metrics.get("train_accuracy", metrics.get("train_acc"))
        if train_acc is not None and va is not None and (train_acc - va) > 0.15:
            flags.append(f"{name} overfitting gap > 15pp")

    # Check calibration for flags
    for level_info in taxonomy_levels:
        name = level_info["name"]
        metrics = per_level_metrics.get(name, {})
        calib_nested = metrics.get("calibration", {})
        ece_after = calib_nested.get("ece_after") if calib_nested else None
        if ece_after is None:
            ece_after = metrics.get("ece_after")
        if ece_after is not None and _ece_quality(ece_after) == "poor":
            flags.append(f"{name} ECE > 0.10 (poor calibration)")

    if not flags and min_val >= 0.75:
        overall = "ready"
    elif len(flags) <= 1 and min_val >= 0.60:
        overall = "caution"
    else:
        overall = "not_ready"

    # Find weakest level
    weakest_level = ""
    if val_accs:
        min_val_idx = val_accs.index(min(val_accs))
        if min_val_idx < len(taxonomy_levels):
            weakest_level = taxonomy_levels[min_val_idx]["name"]

    production_readiness = {
        "overall": overall,
        "min_val_accuracy": round(min_val, 4),
        "weakest_level": weakest_level,
        "flags": flags,
    }

    # Interpretation
    interpretation = []

    if overall == "ready":
        interpretation.append(
            "Model is production-ready: all levels above 75% validation accuracy, no flags."
        )
    elif overall == "caution":
        interpretation.append(
            f"Model needs attention: {weakest_level} is the weakest level "
            f"at {min_val:.0%} validation accuracy."
        )
    else:
        interpretation.append(f"Model is not ready for production: {len(flags)} issue(s) found.")

    for flag in flags:
        interpretation.append(flag)

    return {
        "production_readiness": production_readiness,
        "interpretation": interpretation,
    }
