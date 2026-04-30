"""Inference tools: predict, predict_batch, classify_with_context, explain_prediction."""

from __future__ import annotations

import pandas as pd
import torch

from prometheus.core.embedding import encode_texts
from prometheus.core.mcp_safety import ToolValidationError, mcp_tool_safe
from prometheus.core.responses import success_response
from prometheus.data.loader import load_dataframe
from prometheus.inference.summary import build_prediction_summary
from prometheus.server import _get_code_mapping, _get_predictor, logger, mcp


def _analyze_forks(result) -> list[str]:
    """Generate insights when forked completions reveal something the best path hides."""
    insights = []
    if not result.forked_completions or len(result.forked_completions) < 2:
        return insights

    best_leaf_conf = result.predictions[-1]["confidence"]

    # 1. Minority branch with higher leaf confidence
    for fc in result.forked_completions[1:]:
        fork_leaf_conf = fc["per_level_probs"][-1]
        fork_leaf_label = fc["suffix"][-1]
        fork_l1 = fc["suffix"][0]
        if fork_leaf_conf > best_leaf_conf + 0.05:
            insights.append(
                f'Alternative path via "{fork_l1}" has higher leaf confidence '
                f'for "{fork_leaf_label}" ({fork_leaf_conf:.0%} vs {best_leaf_conf:.0%})'
            )

    # 2. Multiple branches converge on same leaf label
    leaf_to_branches: dict[str, list[str]] = {}
    for fc in result.forked_completions:
        leaf = fc["suffix"][-1]
        branch = fc["suffix"][0]
        leaf_to_branches.setdefault(leaf, []).append(branch)
    for leaf, branches in leaf_to_branches.items():
        if len(branches) >= 2:
            branch_str = " and ".join(f'"{b}"' for b in branches)
            insights.append(f'Multiple branches ({branch_str}) converge on "{leaf}"')

    # 3. Best path leaf confidence is very low
    if best_leaf_conf < 0.3 and not any("higher leaf" in i for i in insights):
        insights.append(
            f"Best path leaf confidence is low ({best_leaf_conf:.0%}), "
            f"review forked_completions for alternatives"
        )

    return insights


@mcp.tool()
@mcp_tool_safe
def predict(
    model_name: str,
    text: str,
    beam_width: int = 3,
) -> dict:
    """Predict taxonomy path for a single text item.

    Args:
        model_name: Name of trained model
        text: Text description to classify
        beam_width: Beam width for search (1 = greedy)
    """
    logger.info("tool=predict model=%s beam_width=%d", model_name, beam_width)
    predictor = _get_predictor(model_name)
    taxonomy = predictor.taxonomy

    embedding = encode_texts([text], taxonomy.embedding_model, show_progress=False)[0]
    result = predictor.predict(
        embedding,
        text=text,
        use_beam_search=(beam_width > 1),
        beam_width=beam_width if beam_width > 1 else None,
    )

    response = {
        "prediction_id": result.prediction_id,
        "path": result.path,
        "predictions": result.predictions,
        "path_score": result.path_score,
        "novelty": result.novelty,
        "routing": result.routing,
    }
    if result.forked_completions:
        response["forked_completions"] = result.forked_completions
        insights = _analyze_forks(result)
        if insights:
            response["insights"] = insights

    code_mapping = _get_code_mapping(model_name)
    response["summary"] = build_prediction_summary(
        predictions=response["predictions"],
        forked_completions=response.get("forked_completions"),
        routing=response["routing"],
        path_score=response["path_score"],
        level_names=predictor.taxonomy.level_names,
        code_mapping=code_mapping,
    )

    return success_response(response)


@mcp.tool()
@mcp_tool_safe
def predict_batch(
    model_name: str,
    file_path: str,
    text_column: str,
    output_path: str | None = None,
    beam_width: int = 1,
) -> dict:
    """Bulk classify items from a CSV file.

    Args:
        model_name: Name of trained model
        file_path: Path to CSV with items to classify
        text_column: Name of text column
        output_path: Path to write results CSV (optional)
        beam_width: Beam width for search (1 = greedy)
    """
    logger.info(
        "tool=predict_batch model=%s file=%s beam_width=%d",
        model_name,
        file_path,
        beam_width,
    )
    predictor = _get_predictor(model_name)
    taxonomy = predictor.taxonomy

    df = load_dataframe(file_path)
    if text_column not in df.columns:
        raise ToolValidationError(f"Column '{text_column}' not found", field="text_column")

    texts = df[text_column].tolist()
    embeddings = encode_texts(texts, taxonomy.embedding_model)

    results = predictor.predict_batch(
        embeddings,
        texts=texts,
        use_beam_search=(beam_width > 1),
        beam_width=beam_width if beam_width > 1 else None,
    )

    rows = []
    for r in results:
        row = {"text": r.input_text}
        for p in r.predictions:
            row[p["level"]] = p["predicted"]
            row[f"{p['level']}_confidence"] = p["confidence"]
        row["routing"] = r.routing["decision"]
        row["path_score"] = r.path_score
        rows.append(row)

    out_df = pd.DataFrame(rows)

    if output_path:
        out_df.to_csv(output_path, index=False)

    routing_counts = out_df["routing"].value_counts().to_dict()

    return success_response(
        {
            "total_items": len(results),
            "routing_distribution": routing_counts,
            "output_path": output_path,
            "sample_predictions": rows[:5],
        }
    )


@mcp.tool()
@mcp_tool_safe
def classify_with_context(
    model_name: str,
    text: str,
    known_levels: dict[str, str],
) -> dict:
    """Classify with known upper levels locked in.

    Args:
        model_name: Name of trained model
        text: Text to classify
        known_levels: Dict of {level_name: known_value} for upper levels
    """
    logger.info(
        "tool=classify_with_context model=%s known=%s",
        model_name,
        list(known_levels.keys()),
    )
    predictor = _get_predictor(model_name)
    taxonomy = predictor.taxonomy

    embedding = encode_texts([text], taxonomy.embedding_model, show_progress=False)[0]
    emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

    model = predictor.model
    model.eval()

    path_labels = []
    path_confs = []
    context_parts = []

    with torch.no_grad():
        for depth in range(taxonomy.num_levels):
            level = taxonomy.get_level(depth)
            classifier = model.get_level(depth)

            context = torch.cat(context_parts, dim=-1) if context_parts else None

            if level.name in known_levels:
                known_val = known_levels[level.name]
                pred_idx = level.encode(known_val)
                path_labels.append(known_val)
                path_confs.append(1.0)
            else:
                logits = classifier(emb_tensor, context)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                pred_idx = int(probs.argmax())
                path_labels.append(level.classes[pred_idx])
                path_confs.append(float(probs[pred_idx]))

            one_hot = torch.zeros(1, level.num_classes)
            one_hot[0, pred_idx] = 1.0
            context_parts.append(one_hot)

    return success_response(
        {
            "path": path_labels,
            "confidences": [round(c, 4) for c in path_confs],
            "known_levels": known_levels,
        }
    )


@mcp.tool()
@mcp_tool_safe
def explain_prediction(
    model_name: str,
    text: str,
) -> dict:
    """Full beam search tree with all candidates at each level and novelty details.

    Args:
        model_name: Name of trained model
        text: Text to explain
    """
    logger.info("tool=explain_prediction model=%s", model_name)
    predictor = _get_predictor(model_name)
    taxonomy = predictor.taxonomy

    embedding = encode_texts([text], taxonomy.embedding_model, show_progress=False)[0]

    beam_result = predictor.beam.predict(embedding)

    leaf_pred = beam_result.level_predictions[-1]
    min_conf = min(lp.confidence for lp in beam_result.level_predictions)
    novelty = predictor.novelty_detector.score(embedding, leaf_pred.predicted, min_conf)

    return success_response(
        {
            "text": text,
            "best_path": beam_result.path,
            "path_score": round(beam_result.path_score, 4),
            "level_details": [
                {
                    "depth": lp.depth,
                    "level": lp.name,
                    "predicted": lp.predicted,
                    "confidence": round(lp.confidence, 4),
                    "margin": round(lp.margin, 4),
                    "top_candidates": lp.top_candidates,
                }
                for lp in beam_result.level_predictions
            ],
            "alternative_paths": [
                {
                    "path": bp.labels,
                    "score": round(bp.log_score, 4),
                    "probs": [round(p, 4) for p in bp.probs],
                }
                for bp in beam_result.alternative_paths
            ],
            "novelty": {
                "centroid_distance": round(novelty.centroid_distance, 4),
                "centroid_z_score": round(novelty.centroid_z_score, 4),
                "knn_distance": round(novelty.knn_distance, 4),
                "is_novel": novelty.is_novel,
            },
        }
    )
