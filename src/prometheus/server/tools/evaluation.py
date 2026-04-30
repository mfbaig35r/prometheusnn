"""Evaluation, threshold tuning, feedback, embedding listing tools."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from prometheus import server
from prometheus.core.embedding import encode_texts
from prometheus.core.mcp_safety import ToolValidationError, mcp_tool_safe
from prometheus.core.responses import success_response
from prometheus.data.loader import load_dataframe, validate_columns
from prometheus.insights import build_evaluation_insights
from prometheus.server import _get_predictor, logger, mcp
from prometheus.storage.model_store import load_model, load_thresholds, save_thresholds


@mcp.tool()
@mcp_tool_safe
def evaluate_model(
    model_name: str,
    file_path: str,
    text_column: str,
    level_columns: list[str],
) -> dict:
    """Evaluate model on held-out data: per-level accuracy, routing stats.

    Args:
        model_name: Name of trained model
        file_path: Path to evaluation CSV
        text_column: Name of text column
        level_columns: Ordered label columns matching model's taxonomy
    """
    logger.info("tool=evaluate_model model=%s file=%s", model_name, file_path)
    predictor = _get_predictor(model_name)
    taxonomy = predictor.taxonomy

    df = load_dataframe(file_path)
    warnings = validate_columns(df, text_column, level_columns)

    clean_df = df.dropna(subset=[text_column] + level_columns).reset_index(drop=True)
    texts = clean_df[text_column].tolist()
    embeddings = encode_texts(texts, taxonomy.embedding_model)

    results = predictor.predict_batch(embeddings, texts=texts)

    level_accuracies = {}
    for depth, col in enumerate(level_columns):
        if depth >= taxonomy.num_levels:
            break
        true_labels = clean_df[col].tolist()
        pred_labels = [r.path[depth] if depth < len(r.path) else "" for r in results]
        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        level_accuracies[taxonomy.get_level(depth).name] = {
            "accuracy": round(correct / len(true_labels), 4),
            "correct": correct,
            "total": len(true_labels),
        }

    routing_counts: dict[str, int] = {}
    for r in results:
        d = r.routing["decision"]
        routing_counts[d] = routing_counts.get(d, 0) + 1

    try:
        model_path = server._config.model_path(model_name)
        with open(model_path / "manifest.json") as f:
            manifest = json.load(f)
    except Exception:
        manifest = None

    try:
        evaluation_insights = build_evaluation_insights(
            level_accuracies,
            routing_counts,
            len(results),
            manifest,
        )
    except Exception:
        evaluation_insights = None

    return success_response(
        {
            "total_items": len(results),
            "per_level_accuracy": level_accuracies,
            "routing_distribution": routing_counts,
            "evaluation_insights": evaluation_insights,
            "warnings": warnings,
        }
    )


@mcp.tool()
@mcp_tool_safe
def get_confusion_matrix(
    model_name: str,
    level: str,
    file_path: str,
    text_column: str,
    level_columns: list[str],
) -> dict:
    """Get per-class precision/recall at a specific level.

    Args:
        model_name: Name of trained model
        level: Level name to get confusion matrix for
        file_path: Path to evaluation data
        text_column: Text column name
        level_columns: Ordered label columns
    """
    from sklearn.metrics import classification_report

    logger.info("tool=get_confusion_matrix model=%s level=%s", model_name, level)
    predictor = _get_predictor(model_name)
    taxonomy = predictor.taxonomy

    target_level = taxonomy.get_level_by_name(level)
    if target_level is None:
        raise ToolValidationError(f"Level '{level}' not found in model")

    df = load_dataframe(file_path)
    clean_df = df.dropna(subset=[text_column] + level_columns).reset_index(drop=True)
    texts = clean_df[text_column].tolist()
    embeddings = encode_texts(texts, taxonomy.embedding_model)

    results = predictor.predict_batch(embeddings, texts=texts)

    depth = target_level.depth
    true_labels = clean_df[level].tolist()
    pred_labels = [r.path[depth] if depth < len(r.path) else "" for r in results]

    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

    return success_response(
        {
            "level": level,
            "depth": depth,
            "classification_report": report,
        }
    )


@mcp.tool()
@mcp_tool_safe
def get_threshold_report(model_name: str) -> dict:
    """Current thresholds and routing distribution info.

    Args:
        model_name: Name of trained model
    """
    logger.info("tool=get_threshold_report model=%s", model_name)
    model_path = server._config.model_path(model_name)
    config = load_thresholds(model_path)

    _, manifest = load_model(model_path)
    taxonomy = manifest.taxonomy

    per_level = {}
    for depth in range(taxonomy.num_levels):
        level = taxonomy.get_level(depth)
        per_level[level.name] = {
            "depth": depth,
            "threshold": config.threshold_for_depth(depth),
        }

    return success_response(
        {
            "model_name": model_name,
            "per_level_thresholds": per_level,
            "margin": config.margin,
            "z_score": config.z_score,
            "novelty_conf": config.novelty_conf,
            "adaptive_factor": config.adaptive_factor,
        }
    )


@mcp.tool()
@mcp_tool_safe
def set_thresholds(
    model_name: str,
    thresholds: dict[str, float],
) -> dict:
    """Override auto-generated thresholds.

    Args:
        model_name: Name of trained model
        thresholds: Dict of threshold overrides. Keys can be level names, depth indices,
            "margin", "z_score", "novelty_conf", or "adaptive_factor".
    """
    logger.info(
        "tool=set_thresholds model=%s overrides=%s",
        model_name,
        list(thresholds.keys()),
    )
    model_path = server._config.model_path(model_name)
    config = load_thresholds(model_path)

    _, manifest = load_model(model_path)
    taxonomy = manifest.taxonomy

    top_level_fields = {"margin", "z_score", "novelty_conf", "adaptive_factor"}
    for key, value in thresholds.items():
        if key in top_level_fields:
            setattr(config, key, int(value) if key == "adaptive_factor" else float(value))
            continue
        level = taxonomy.get_level_by_name(key)
        if level is not None:
            config.level_thresholds[level.depth] = float(value)
            continue
        try:
            config.level_thresholds[int(key)] = float(value)
        except ValueError:
            raise ToolValidationError(
                f"Unknown threshold key: '{key}'. Expected a level name, a depth index, "
                f"or one of {sorted(top_level_fields)}."
            ) from None

    save_thresholds(model_path, config)
    server._predictor_cache.pop(model_name, None)

    return success_response(
        {"model_name": model_name, "thresholds": config.model_dump()},
        "Thresholds updated",
    )


@mcp.tool()
@mcp_tool_safe
def submit_feedback(
    model_name: str,
    text: str,
    corrected_labels: dict[str, str],
    prediction_id: str | None = None,
) -> dict:
    """Log human corrections for a prediction.

    Args:
        model_name: Name of the model
        text: Original text that was classified
        corrected_labels: Dict of {level_name: correct_label}
        prediction_id: Optional prediction ID to link feedback to
    """
    logger.info("tool=submit_feedback model=%s prediction_id=%s", model_name, prediction_id)
    feedback_dir = server._config.feedback_dir / model_name
    feedback_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "text": text,
        "corrected_labels": corrected_labels,
        "prediction_id": prediction_id,
    }

    feedback_file = feedback_dir / "feedback.jsonl"
    with open(feedback_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return success_response(None, "Feedback recorded")


@mcp.tool()
@mcp_tool_safe
def list_embedding_models() -> dict:
    """List popular sentence-transformer models for embedding."""
    models = [
        {"name": "all-MiniLM-L6-v2", "dim": 384, "speed": "fast", "quality": "good"},
        {"name": "all-mpnet-base-v2", "dim": 768, "speed": "medium", "quality": "best"},
        {
            "name": "paraphrase-MiniLM-L3-v2",
            "dim": 384,
            "speed": "fastest",
            "quality": "fair",
        },
        {
            "name": "all-MiniLM-L12-v2",
            "dim": 384,
            "speed": "medium",
            "quality": "better",
        },
    ]
    return success_response({"models": models})
