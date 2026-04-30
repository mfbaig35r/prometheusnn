"""PrometheusNN — Generalized N-Level Cascade Classifier MCP Server."""

from __future__ import annotations

import json
import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from mcp.server.fastmcp import FastMCP

from prometheus.core.config import PrometheusConfig
from prometheus.core.embedding import encode_texts
from prometheus.core.logging import setup_logging
from prometheus.core.responses import internal_error, success_response, validation_error
from prometheus.data.loader import (
    analyze_data,
    load_dataframe,
    prepare_data,
    validate_columns,
)
from prometheus.inference.novelty import NoveltyDetector
from prometheus.inference.predictor import CascadePredictor
from prometheus.inference.router import ThresholdRouter
from prometheus.inference.summary import build_prediction_summary
from prometheus.insights import (
    build_evaluation_insights,
    build_model_health,
    build_training_insights,
)
from prometheus.intake import (
    detect_problem_type,
    estimate_training_time,
    profile_columns,
    recommend_next_steps,
    scope_problem_from_text,
    surface_concerns,
)
from prometheus.storage.model_store import (
    delete_model as delete_model_from_disk,
)
from prometheus.storage.model_store import (
    load_code_mapping,
    load_model,
    load_thresholds,
    save_code_mapping,
    save_model,
)
from prometheus.training.noise import ParentNoiseInjector
from prometheus.training.trainer import TrainingConfig
from prometheus.training.trainer import train_model as run_training

mcp = FastMCP(
    "PrometheusNN",
    instructions="""PrometheusNN — Generalized N-Level Cascade Classifier.

Train, predict, evaluate, and tune hierarchical classifiers on any domain.
The goal is a frictionless experience for users who may have minimal ML background:
guide them from "I have data" or "I have an idea" to a working classifier.

## Where to start (read this first)

**For ANY new conversation about building a classifier, start with one of these two tools — not `analyze_training_data` directly:**

- **`start_here(file_path, goal_description=None)`** — the user has a data file (CSV/JSON).
  Profiles columns, detects whether it's a hierarchical/flat/multi-label problem,
  surfaces concerns BEFORE training, and returns `next_steps` with pre-filled args
  so you can fire the next tool call without guessing column names.

- **`scope_problem(goal_description, data_hint=None)`** — the user describes a goal but
  has no data yet. Returns whether Prometheus fits, what data they'll need, and
  what to watch out for. If the goal isn't a fit (regression, clustering, NER),
  returns a polite redirect.

**Always treat `next_steps` as authoritative** — when `start_here` pre-fills args, use
those exact values. Don't re-ask the user for column names it already detected.

## Input Format
- **CSV/JSON** with one text column and N label columns
- `level_columns` order = hierarchy order (root → leaf)
- Column names become level names automatically (no "L1/L2" renaming)
- Rows with missing values in required columns are dropped with a warning

## Workflow 1: First-Time Training

Use this when the user has training data and wants to build a new classifier.

1. **Start with `start_here(file_path)`** — returns the assessment, concerns, and
   pre-filled next_steps. Show the user the recommended setup AND any concerns
   (especially `severity: critical`) before proceeding.

2. **Run `analyze_training_data`** using the args from `next_steps[0].args`.
   - Check `warnings` for additional issues that needed embedding to surface.
   - Show the user the summary and get confirmation before training.

3. **Train** — call `train_model` with the user's parameters.
   - Default `embedding_model` is `all-MiniLM-L6-v2` (384d, fast, good quality).
   - For higher accuracy, suggest `all-mpnet-base-v2` (768d, slower, best quality).
   - Use `list_embedding_models` if the user asks what's available.
   - Default `noise_rate=0.15` is well-tested. Only increase for very noisy taxonomies.
   - Report per-level val accuracy from the result. If any level is below 0.60, warn the user.

4. **Evaluate** — call `evaluate_model` on held-out data (if available separately).
   - Compare per-level accuracy to training metrics to check for overfitting.
   - Check `routing_distribution` — if >50% are HUMAN_REVIEW, thresholds may be too tight.

5. **Tune thresholds** — if routing distribution is unbalanced:
   - Too many HUMAN_REVIEW → lower thresholds with `set_thresholds`
   - Too many AUTO_ACCEPT on wrong predictions → raise thresholds
   - Use `get_threshold_report` to see current values before changing.

## Workflow 2: Production Prediction

Use this when the user wants to classify new items with a trained model.

1. **Single item** — use `predict` with `beam_width=1` (greedy, fast) or `beam_width=3` (beam search, more robust).
   - **When to use beam search**: items where root-level errors cascade badly, or when the user
     needs confidence in the full path. Beam search is ~3x slower but prevents early mistakes
     from locking in wrong downstream predictions.
   - **When greedy is fine**: high-accuracy models (>90% at all levels), or bulk classification
     where speed matters.

2. **Batch** — use `predict_batch` for CSV files. Always use `beam_width=1` for batch unless
   the user explicitly wants beam search (it's much slower on large files).

3. **Interpret routing decisions**:
   - **AUTO_ACCEPT**: All levels confident above thresholds. Present the full path.
   - **HUMAN_REVIEW**: Some levels uncertain. Show the `confident_path` (levels the model is sure about)
     and highlight which level triggered review and why (`review_reason`).
   - **OUT_OF_TAXONOMY**: Item doesn't match known classes. The novelty z-score indicates how far
     out-of-distribution it is. This suggests a taxonomy gap — the item may need a new category.

4. **Partial classification** — use `classify_with_context` when upper levels are already known
   (e.g., user knows the Category but wants SubCategory/Type predicted). This locks in known
   levels and only predicts the unknown ones.

## Workflow 3: Model Improvement

Use this when a model exists but accuracy needs to improve.

### Diagnose first
1. Call `evaluate_model` to get per-level accuracy.
2. Identify the weakest level — this is where to focus.
3. Call `get_confusion_matrix` for that level to see which classes are confused.
4. Use `explain_prediction` on misclassified examples to understand the beam search decisions.

### Decision tree for improvement

**If root level accuracy is low (<80%)**:
- The problem is fundamental — text descriptions may not distinguish top-level categories.
- Suggest: better text preprocessing, more training data, or a larger embedding model.
- Try `resume_training` with more epochs and a lower learning rate.

**If accuracy drops sharply at deeper levels**:
- This is expected — deeper levels are harder. Check if:
  - Classes at that level have very few samples → need more data.
  - Classes are semantically similar → may need to merge classes or improve descriptions.
- Try increasing `noise_rate` (e.g., 0.20-0.25) to make downstream classifiers more robust.

**If routing is too aggressive (too many HUMAN_REVIEW)**:
- Call `get_threshold_report` to see current thresholds.
- Lower thresholds with `set_thresholds` (e.g., reduce leaf threshold from 0.75 to 0.65).
- This trades off precision for throughput — more items auto-accepted but slightly less reliable.

**If routing misses errors (AUTO_ACCEPT on wrong predictions)**:
- Raise thresholds or increase the margin threshold.
- Call `set_thresholds` with higher values.

### Retrain vs fine-tune
- **Fine-tune** (`resume_training`): Use when adding more data to an existing model, or when
  you want a few more epochs. Uses a lower default learning rate (0.0005).
- **Full retrain** (`train_model`): Use when changing the taxonomy structure, switching embedding
  models, or when fine-tuning isn't improving things.

## Workflow 4: Feedback Loop

Use this for continuous improvement after deployment.

1. When the user corrects a prediction, call `submit_feedback` with the corrected labels.
2. Feedback is logged to `~/.prometheus/feedback/{model_name}/feedback.jsonl`.
3. Periodically, the user can retrain with corrected data:
   - Export feedback, merge corrections into the training CSV.
   - Call `resume_training` or `train_model` with the updated dataset.

## Workflow 5: Model Comparison

Use this when deciding between models (e.g., different embedding models or hyperparameters).

1. Train two models with different configs (e.g., `model_v1` and `model_v2`).
2. Call `evaluate_model` on the same held-out dataset for both.
3. Compare per-level accuracy and routing distributions side by side.
4. Use `explain_prediction` on the same tricky examples to compare beam search behavior.
5. Call `describe_model` for both to compare training configs and calibration temperatures.
6. Delete the losing model with `delete_model`.

## Tool Reference

### Entry Points (USE THESE FIRST)
- `start_here` — assess a CSV/JSON file: detect problem type, surface concerns, return pre-filled next_steps
- `scope_problem` — assess a goal description (no data yet): is it a fit, what data is needed, what to watch for

### Training & Data
- `analyze_training_data` — validate data before training (call after `start_here` confirms setup)
- `train_model` — full pipeline: embed → split → train → calibrate → novelty → save
- `resume_training` — fine-tune existing model with more data/epochs

### Inference
- `predict` — single item classification with routing
- `predict_batch` — bulk CSV classification
- `classify_with_context` — predict with known upper levels locked in

### Model Management
- `list_models` — all trained models
- `describe_model` — full manifest (taxonomy, metrics, config, thresholds)
- `delete_model` — remove from disk
- `export_model` — copy model directory

### Evaluation & Diagnostics
- `evaluate_model` — per-level accuracy on held-out data
- `explain_prediction` — full beam search tree + novelty details
- `get_confusion_matrix` — per-class precision/recall at a specific level
- `get_threshold_report` — current thresholds and their values
- `set_thresholds` — override auto-generated thresholds

### Feedback
- `submit_feedback` — log human corrections

### Embedding
- `list_embedding_models` — available sentence-transformer models

## Notes
- Thresholds are auto-generated as linear decay from 0.90 (root) to 0.75 (leaf).
  For N levels: threshold[i] = 0.90 - i * 0.15 / (N - 1). Override with `set_thresholds`.
- All models saved to `~/.prometheus/models/` as JSON manifest + .pt state dicts. No pickle.
- Calibration (temperature scaling) is applied automatically during training. Higher temperatures
  mean the model was overconfident — this is normal and corrected by calibration.
- Novelty detection uses dual signals: centroid z-score (class shape) + kNN distance (local density).
  Both must be triggered to flag OUT_OF_TAXONOMY (conservative, reduces false positives).
""",
)

_config = PrometheusConfig()
_config.ensure_dirs()
setup_logging(_config.logs_dir)
logger = logging.getLogger(__name__)

# Cache loaded predictors: {model_name: CascadePredictor}
_predictor_cache: dict[str, CascadePredictor] = {}

# Cache loaded code mappings: {model_name: dict | None}
_code_mapping_cache: dict[str, dict | None] = {}


def _get_code_mapping(model_name: str) -> dict | None:
    """Get or load code mapping for a model."""
    if model_name in _code_mapping_cache:
        return _code_mapping_cache[model_name]
    model_path = _config.model_path(model_name)
    mapping = load_code_mapping(model_path)
    _code_mapping_cache[model_name] = mapping
    return mapping


def _get_predictor(model_name: str) -> CascadePredictor:
    """Get or load a CascadePredictor."""
    if model_name in _predictor_cache:
        return _predictor_cache[model_name]

    model_path = _config.model_path(model_name)
    if not (model_path / "manifest.json").exists():
        raise FileNotFoundError(f"Model '{model_name}' not found")

    model, manifest = load_model(model_path)
    taxonomy = manifest.taxonomy
    thresholds = load_thresholds(model_path)

    # Load novelty detector
    novelty = NoveltyDetector(
        z_score_threshold=thresholds.get("z_score", 2.5),
        knn_k=10,
        novelty_conf_threshold=thresholds.get("novelty_conf", 0.70),
    )
    if (model_path / "novelty_centroids.npz").exists():
        novelty.load(model_path)

    adaptive_factor = int(thresholds.pop("adaptive_factor", 2))

    router = ThresholdRouter(thresholds, taxonomy.level_names)
    predictor = CascadePredictor(
        model=model,
        taxonomy=taxonomy,
        novelty_detector=novelty,
        router=router,
        parent_child_map=taxonomy.parent_child_map,
        adaptive_factor=adaptive_factor,
    )
    _predictor_cache[model_name] = predictor
    return predictor


# ─── Entry Points ───


@mcp.tool()
def start_here(file_path: str, goal_description: str | None = None) -> dict:
    """Entry point for users with a data file. Profiles columns, detects problem type, returns pre-filled next_steps.

    Always call this FIRST when a user wants to build a classifier and has a CSV/JSON file.
    Do not call analyze_training_data directly — use the args from this tool's
    `next_steps[0].args` instead.

    Args:
        file_path: Path to a CSV or JSON file
        goal_description: Optional free text describing what the user wants to build
            (e.g., "classify support tickets by urgency"). Helps disambiguate when
            the data shape is unclear.
    """
    try:
        logger.info("tool=start_here file=%s", file_path)
        df = load_dataframe(file_path)
        profiles = profile_columns(df)
        assessment = detect_problem_type(df, profiles, goal_description=goal_description)
        next_steps = recommend_next_steps(file_path, assessment, profiles)
        concerns = surface_concerns(df, assessment, profiles)

        time_low, time_high = (0, 0)
        if assessment.suggested_level_columns:
            time_low, time_high = estimate_training_time(
                rows=len(df),
                num_levels=len(assessment.suggested_level_columns),
            )

        relevant_names = set(assessment.suggested_level_columns)
        if assessment.suggested_text_column:
            relevant_names.add(assessment.suggested_text_column)
        relevant_profiles = [p for p in profiles if p.name in relevant_names]
        unused_columns = [p.name for p in profiles if p.name not in relevant_names]

        return success_response(
            {
                "data_summary": {
                    "rows": len(df),
                    "columns": [p.name for p in profiles],
                    "relevant_profiles": [p.to_dict() for p in relevant_profiles],
                    "unused_columns": unused_columns,
                },
                "assessment": assessment.to_dict(),
                "confidence_guide": (
                    "0.9+ = strong signal, recommendation can be trusted; "
                    "0.7-0.9 = good signal, worth confirming with the user; "
                    "below 0.7 = weak signal, ask the user to clarify their goal."
                ),
                "concerns": [c.to_dict() for c in concerns],
                "estimated_training_time_minutes": (
                    {"low": time_low, "high": time_high} if time_low else None
                ),
                "next_steps": [s.to_dict() for s in next_steps],
            }
        )
    except FileNotFoundError:
        return validation_error(f"File not found: {file_path}", field="file_path")
    except Exception as e:
        logger.error("start_here failed: %s", e, exc_info=True)
        return internal_error(str(e))


@mcp.tool()
def scope_problem(goal_description: str, data_hint: str | None = None) -> dict:
    """Entry point for users without data yet. Returns whether Prometheus fits the goal.

    Use this when the user describes what they want to build but doesn't have a
    file ready. Returns fits/doesn't-fit, what data they'll need, and watch-outs.
    For non-fits, returns a polite redirect to the right tool.

    Args:
        goal_description: Free text describing the goal
            (e.g., "classify customer emails into urgency levels")
        data_hint: Optional hint about available data
            (e.g., "I have ~3000 examples", "no labels yet")
    """
    try:
        logger.info("tool=scope_problem goal=%s", goal_description[:80])
        result = scope_problem_from_text(goal_description, data_hint)
        return success_response(result.to_dict())
    except Exception as e:
        logger.error("scope_problem failed: %s", e, exc_info=True)
        return internal_error(str(e))


# ─── Training & Data ───


@mcp.tool()
def analyze_training_data(
    file_path: str,
    text_column: str,
    level_columns: list[str],
) -> dict:
    """Validate training data: class counts, hierarchy consistency, missing values.

    Args:
        file_path: Path to CSV or JSON file
        text_column: Name of the text/description column
        level_columns: Ordered label columns root → leaf (e.g., ["Category", "SubCategory", "Type"])
    """
    try:
        logger.info("tool=analyze_training_data file=%s levels=%s", file_path, level_columns)
        return success_response(analyze_data(file_path, text_column, level_columns))
    except Exception as e:
        logger.error("analyze_training_data failed: %s", e, exc_info=True)
        return internal_error(str(e))


@mcp.tool()
def train_model(
    file_path: str,
    text_column: str,
    level_columns: list[str],
    model_name: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    epochs: int = 20,
    batch_size: int = 64,
    noise_rate: float = 0.15,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.3,
    learning_rate: float = 0.001,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> dict:
    """Full training pipeline: embed → split → train N levels → calibrate → novelty → save.

    Args:
        file_path: Path to CSV or JSON training data
        text_column: Name of the text/description column
        level_columns: Ordered label columns root → leaf
        model_name: Name for the saved model
        embedding_model: Sentence-transformer model name
        epochs: Max training epochs per level
        batch_size: Training batch size
        noise_rate: Parent noise injection rate (0-1)
        hidden_dims: Hidden layer dimensions (default [256, 64])
        dropout: Dropout rate
        learning_rate: Adam learning rate
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
    try:
        logger.info(
            "tool=train_model model=%s file=%s levels=%s epochs=%d",
            model_name,
            file_path,
            level_columns,
            epochs,
        )
        hidden_dims = hidden_dims or [256, 64]
        progress_log = []

        def log(msg: str):
            progress_log.append(msg)

        # Prepare data
        log("Loading and embedding data...")
        taxonomy, train_data, val_data, test_data, warnings = prepare_data(
            file_path=file_path,
            text_column=text_column,
            level_columns=level_columns,
            model_name=model_name,
            embedding_model=embedding_model,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        embedding_dim = train_data.embeddings.shape[1]
        log(
            f"Data: {taxonomy.total_samples} samples, {taxonomy.num_levels} levels, {embedding_dim}d embeddings"
        )

        # Train
        config = TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            noise_rate=noise_rate,
            dropout=dropout,
            hidden_dims=hidden_dims,
        )

        model, result, noise_injector = run_training(
            taxonomy=taxonomy,
            train_data=train_data,
            val_data=val_data,
            embedding_dim=embedding_dim,
            config=config,
            progress_callback=log,
        )

        # Novelty detector
        log("Fitting novelty detector...")
        leaf = taxonomy.leaf_level
        novelty = NoveltyDetector()
        novelty.fit(
            train_data.embeddings,
            train_data.leaf_labels,
            leaf.classes,
        )

        # Save
        model_path = _config.model_path(model_name)
        log(f"Saving model to {model_path}...")
        save_model(
            path=model_path,
            model=model,
            taxonomy=taxonomy,
            result=result,
            config=config,
            embedding_dim=embedding_dim,
            training_embeddings=train_data.embeddings,
            confusion_profiles=result.confusion_profiles,
            centroids_data=novelty.get_centroids_data(),
        )

        # Also save novelty files directly
        novelty.save(model_path)

        # Invalidate cache
        _predictor_cache.pop(model_name, None)

        # Build learning insights
        try:
            learning_insights = build_training_insights(result.level_metrics, taxonomy, train_data)
        except Exception:
            learning_insights = None

        return success_response(
            {
                "model_name": model_name,
                "taxonomy": {
                    "num_levels": taxonomy.num_levels,
                    "levels": [
                        {"name": lvl.name, "num_classes": lvl.num_classes}
                        for lvl in taxonomy.levels
                    ],
                    "total_samples": taxonomy.total_samples,
                },
                "metrics": result.summary(),
                "splits": {
                    "train": len(train_data),
                    "val": len(val_data),
                    "test": len(test_data),
                },
                "learning_insights": learning_insights,
                "warnings": warnings,
                "progress": progress_log,
            }
        )
    except Exception as e:
        logger.error("train_model failed: %s", e, exc_info=True)
        return internal_error(str(e), traceback.format_exc())


@mcp.tool()
def resume_training(
    model_name: str,
    file_path: str,
    additional_epochs: int = 10,
    learning_rate: float = 0.0005,
    text_column: str | None = None,
    level_columns: list[str] | None = None,
) -> dict:
    """Fine-tune an existing model with more data or epochs.

    Args:
        model_name: Name of existing model to resume
        file_path: Path to training data (can be new or same)
        additional_epochs: Extra epochs to train
        learning_rate: Learning rate for fine-tuning (lower than initial)
        text_column: Override text column (uses original if None)
        level_columns: Override level columns (uses original if None)
    """
    try:
        logger.info(
            "tool=resume_training model=%s epochs=%d lr=%f",
            model_name,
            additional_epochs,
            learning_rate,
        )
        model_path = _config.model_path(model_name)
        model, manifest = load_model(model_path)
        taxonomy = manifest.taxonomy

        text_col = text_column or taxonomy.text_column
        level_cols = level_columns or taxonomy.level_names

        _, train_data, val_data, _, warnings = prepare_data(
            file_path=file_path,
            text_column=text_col,
            level_columns=level_cols,
            model_name=model_name,
            embedding_model=taxonomy.embedding_model,
        )

        config = TrainingConfig(
            epochs=additional_epochs,
            learning_rate=learning_rate,
            batch_size=manifest.training_config.get("batch_size", 64),
            noise_rate=manifest.training_config.get("noise_rate", 0.15),
            hidden_dims=manifest.training_config.get("hidden_dims", [256, 64]),
            dropout=manifest.training_config.get("dropout", 0.3),
        )

        # Re-train using existing model weights (already loaded)
        noise_injector = ParentNoiseInjector(noise_rate=config.noise_rate)
        if (model_path / "confusion_profiles.json").exists():
            with open(model_path / "confusion_profiles.json") as f:
                noise_injector.load_profiles(json.load(f))

        from prometheus.training.calibration import TemperatureScaler
        from prometheus.training.trainer import TrainingResult, _train_level

        # Build HierarchicalClassifier wrapper around loaded model
        embedding_dim = manifest.embedding_dim
        level_metrics = []
        num_classes = taxonomy.num_classes_per_level

        for depth in range(taxonomy.num_levels):
            level = taxonomy.get_level(depth)
            metrics = _train_level(
                model=model,
                depth=depth,
                level_name=level.name,
                train_data=train_data,
                val_data=val_data,
                parent_depths=list(range(depth)),
                num_classes=num_classes,
                noise_injector=noise_injector,
                config=config,
                device=torch.device("cpu"),
            )
            noise_injector.build_confusion_profile(depth, metrics.confusion_matrix)
            level_metrics.append(metrics)

        # Re-calibrate
        scaler = TemperatureScaler(model)
        calib_results = scaler.calibrate_all(val_data, num_classes)
        for depth, cr in calib_results.items():
            level_metrics[depth].calibration = cr

        result = TrainingResult(
            level_metrics=level_metrics,
            confusion_profiles=noise_injector.get_serializable_profiles(),
        )

        save_model(
            path=model_path,
            model=model,
            taxonomy=taxonomy,
            result=result,
            config=config,
            embedding_dim=embedding_dim,
            training_embeddings=train_data.embeddings,
            confusion_profiles=result.confusion_profiles,
        )

        _predictor_cache.pop(model_name, None)

        return success_response(
            {
                "model_name": model_name,
                "additional_epochs": additional_epochs,
                "metrics": result.summary(),
                "warnings": warnings,
            }
        )
    except Exception as e:
        logger.error("resume_training failed: %s", e, exc_info=True)
        return internal_error(str(e), traceback.format_exc())


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


# ─── Inference ───


@mcp.tool()
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
    try:
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
    except Exception as e:
        logger.error("predict failed: %s", e, exc_info=True)
        return internal_error(str(e))


@mcp.tool()
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
    try:
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
            return validation_error(f"Column '{text_column}' not found", field="text_column")

        texts = df[text_column].tolist()
        embeddings = encode_texts(texts, taxonomy.embedding_model)

        results = predictor.predict_batch(
            embeddings,
            texts=texts,
            use_beam_search=(beam_width > 1),
            beam_width=beam_width if beam_width > 1 else None,
        )

        # Build output
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
    except Exception as e:
        logger.error("predict_batch failed: %s", e, exc_info=True)
        return internal_error(str(e), traceback.format_exc())


@mcp.tool()
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
    try:
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
                    # Use known value
                    known_val = known_levels[level.name]
                    pred_idx = level.encode(known_val)
                    path_labels.append(known_val)
                    path_confs.append(1.0)
                else:
                    # Predict
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
    except Exception as e:
        logger.error("classify_with_context failed: %s", e, exc_info=True)
        return internal_error(str(e))


# ─── Model Management ───


@mcp.tool()
def list_models() -> dict:
    """List all trained models with summary stats."""
    try:
        logger.info("tool=list_models")
        models = []
        for name in _config.list_models():
            try:
                with open(_config.model_path(name) / "manifest.json") as f:
                    manifest = json.load(f)
                models.append(
                    {
                        "name": name,
                        "num_levels": len(manifest["taxonomy"]["levels"]),
                        "level_names": [lvl["name"] for lvl in manifest["taxonomy"]["levels"]],
                        "total_samples": manifest["taxonomy"].get("total_samples", 0),
                        "embedding_model": manifest["taxonomy"]["embedding_model"],
                        "created_at": manifest.get("created_at", "unknown"),
                    }
                )
            except Exception:
                models.append({"name": name, "error": "Failed to read manifest"})

        return success_response({"models": models, "count": len(models)})
    except Exception as e:
        logger.error("list_models failed: %s", e, exc_info=True)
        return internal_error(str(e))


@mcp.tool()
def describe_model(model_name: str) -> dict:
    """Get full model details: taxonomy, metrics, thresholds, config.

    Args:
        model_name: Name of the model to describe
    """
    try:
        logger.info("tool=describe_model model=%s", model_name)
        model_path = _config.model_path(model_name)
        if not (model_path / "manifest.json").exists():
            return validation_error(f"Model '{model_name}' not found")

        with open(model_path / "manifest.json") as f:
            manifest = json.load(f)

        try:
            model_health = build_model_health(manifest)
        except Exception:
            model_health = None

        return success_response(
            {
                "model_name": model_name,
                "manifest": manifest,
                "model_health": model_health,
            }
        )
    except Exception as e:
        logger.error("describe_model failed: %s", e, exc_info=True)
        return internal_error(str(e))


@mcp.tool()
def delete_model(model_name: str) -> dict:
    """Delete a model from disk.

    Args:
        model_name: Name of the model to delete
    """
    try:
        logger.info("tool=delete_model model=%s", model_name)
        model_path = _config.model_path(model_name)
        if not model_path.exists():
            return validation_error(f"Model '{model_name}' not found")

        delete_model_from_disk(model_path)
        _predictor_cache.pop(model_name, None)

        return success_response(None, f"Model '{model_name}' deleted")
    except Exception as e:
        logger.error("delete_model failed: %s", e, exc_info=True)
        return internal_error(str(e))


@mcp.tool()
def export_model(
    model_name: str,
    output_path: str,
) -> dict:
    """Copy model directory to a new location.

    Args:
        model_name: Name of the model to export
        output_path: Destination directory
    """
    try:
        logger.info("tool=export_model model=%s dest=%s", model_name, output_path)
        import shutil

        model_path = _config.model_path(model_name)
        if not model_path.exists():
            return validation_error(f"Model '{model_name}' not found")

        dest = Path(output_path)
        if dest.exists():
            return validation_error(f"Destination already exists: {output_path}")

        shutil.copytree(model_path, dest)
        return success_response({"exported_to": str(dest)})
    except Exception as e:
        logger.error("export_model failed: %s", e, exc_info=True)
        return internal_error(str(e))


# ─── Code Mapping ───


@mcp.tool()
def build_code_mapping(
    model_name: str,
    file_path: str,
    code_column: str,
    level_columns: list[str] | None = None,
) -> dict:
    """Build a code mapping from training data so predictions include source codes.

    Reads the CSV, groups unique taxonomy paths, and maps each path to its
    associated code(s). Saves to the model directory for automatic use by predict.

    Args:
        model_name: Name of existing model
        file_path: Path to CSV with code column and level columns
        code_column: Column containing codes (e.g., "hts8", "naics_code")
        level_columns: Ordered label columns (defaults to model's taxonomy level names)
    """
    try:
        logger.info("tool=build_code_mapping model=%s code_col=%s", model_name, code_column)
        model_path = _config.model_path(model_name)
        if not (model_path / "manifest.json").exists():
            return validation_error(f"Model '{model_name}' not found")

        with open(model_path / "manifest.json") as f:
            manifest = json.load(f)

        taxonomy_levels = [lvl["name"] for lvl in manifest["taxonomy"]["levels"]]
        cols = level_columns or taxonomy_levels

        df = load_dataframe(file_path)

        # Validate columns exist
        missing = [c for c in cols + [code_column] if c not in df.columns]
        if missing:
            return validation_error(f"Columns not found: {missing}")

        # Drop rows with missing values in required columns
        df = df.dropna(subset=cols + [code_column]).reset_index(drop=True)

        # Build mapping: pipe-delimited path → sorted deduplicated code list
        mapping: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            path_key = "|".join(str(row[c]) for c in cols)
            code = str(row[code_column]).strip()
            if path_key not in mapping:
                mapping[path_key] = set()
            mapping[path_key].add(code)

        # Convert sets to sorted lists
        mapping = {k: sorted(v) for k, v in mapping.items()}

        save_code_mapping(model_path, code_column, cols, mapping)
        _code_mapping_cache.pop(model_name, None)

        multi_code_paths = sum(1 for v in mapping.values() if len(v) > 1)
        all_codes = set()
        for codes in mapping.values():
            all_codes.update(codes)

        return success_response(
            {
                "model_name": model_name,
                "code_column": code_column,
                "level_columns": cols,
                "unique_paths": len(mapping),
                "unique_codes": len(all_codes),
                "multi_code_paths": multi_code_paths,
            },
            "Code mapping saved",
        )
    except Exception as e:
        logger.error("build_code_mapping failed: %s", e, exc_info=True)
        return internal_error(str(e), traceback.format_exc())


# ─── Evaluation & Diagnostics ───


@mcp.tool()
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
    try:
        logger.info("tool=evaluate_model model=%s file=%s", model_name, file_path)
        predictor = _get_predictor(model_name)
        taxonomy = predictor.taxonomy

        df = load_dataframe(file_path)
        warnings = validate_columns(df, text_column, level_columns)

        clean_df = df.dropna(subset=[text_column] + level_columns).reset_index(drop=True)
        texts = clean_df[text_column].tolist()
        embeddings = encode_texts(texts, taxonomy.embedding_model)

        results = predictor.predict_batch(embeddings, texts=texts)

        # Per-level accuracy
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

        # Routing stats
        routing_counts = {}
        for r in results:
            d = r.routing["decision"]
            routing_counts[d] = routing_counts.get(d, 0) + 1

        # Build evaluation insights
        try:
            model_path = _config.model_path(model_name)
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
    except Exception as e:
        logger.error("evaluate_model failed: %s", e, exc_info=True)
        return internal_error(str(e), traceback.format_exc())


@mcp.tool()
def explain_prediction(
    model_name: str,
    text: str,
) -> dict:
    """Full beam search tree with all candidates at each level and novelty details.

    Args:
        model_name: Name of trained model
        text: Text to explain
    """
    try:
        logger.info("tool=explain_prediction model=%s", model_name)
        predictor = _get_predictor(model_name)
        taxonomy = predictor.taxonomy

        embedding = encode_texts([text], taxonomy.embedding_model, show_progress=False)[0]

        # Use beam search for explanation
        beam_result = predictor.beam.predict(embedding)

        # Novelty details
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
    except Exception as e:
        logger.error("explain_prediction failed: %s", e, exc_info=True)
        return internal_error(str(e))


@mcp.tool()
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
    try:
        logger.info("tool=get_confusion_matrix model=%s level=%s", model_name, level)
        from sklearn.metrics import classification_report

        predictor = _get_predictor(model_name)
        taxonomy = predictor.taxonomy

        target_level = taxonomy.get_level_by_name(level)
        if target_level is None:
            return validation_error(f"Level '{level}' not found in model")

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
    except Exception as e:
        logger.error("get_confusion_matrix failed: %s", e, exc_info=True)
        return internal_error(str(e), traceback.format_exc())


@mcp.tool()
def get_threshold_report(model_name: str) -> dict:
    """Current thresholds and routing distribution info.

    Args:
        model_name: Name of trained model
    """
    try:
        logger.info("tool=get_threshold_report model=%s", model_name)
        model_path = _config.model_path(model_name)
        thresholds = load_thresholds(model_path)

        _, manifest = load_model(model_path)
        taxonomy = manifest.taxonomy

        per_level = {}
        for depth in range(taxonomy.num_levels):
            level = taxonomy.get_level(depth)
            per_level[level.name] = {
                "depth": depth,
                "threshold": thresholds.get(depth, 0.75),
            }

        return success_response(
            {
                "model_name": model_name,
                "per_level_thresholds": per_level,
                "margin": thresholds.get("margin", 0.15),
                "z_score": thresholds.get("z_score", 2.5),
                "novelty_conf": thresholds.get("novelty_conf", 0.70),
            }
        )
    except Exception as e:
        logger.error("get_threshold_report failed: %s", e, exc_info=True)
        return internal_error(str(e))


@mcp.tool()
def set_thresholds(
    model_name: str,
    thresholds: dict[str, float],
) -> dict:
    """Override auto-generated thresholds.

    Args:
        model_name: Name of trained model
        thresholds: Dict of threshold overrides. Keys can be level names, depth indices, "margin", "z_score", "novelty_conf", "adaptive_factor" (int, beam widening multiplier, default 2).
    """
    try:
        logger.info(
            "tool=set_thresholds model=%s overrides=%s",
            model_name,
            list(thresholds.keys()),
        )
        model_path = _config.model_path(model_name)
        current = load_thresholds(model_path)

        _, manifest = load_model(model_path)
        taxonomy = manifest.taxonomy

        # Apply overrides
        for key, value in thresholds.items():
            # Try as level name
            level = taxonomy.get_level_by_name(key)
            if level is not None:
                current[level.depth] = value
            else:
                try:
                    current[int(key)] = value
                except ValueError:
                    current[key] = value

        # Save
        with open(model_path / "thresholds.json", "w") as f:
            json.dump({str(k): v for k, v in current.items()}, f, indent=2)

        _predictor_cache.pop(model_name, None)

        return success_response(
            {
                "model_name": model_name,
                "thresholds": {str(k): v for k, v in current.items()},
            },
            "Thresholds updated",
        )
    except Exception as e:
        logger.error("set_thresholds failed: %s", e, exc_info=True)
        return internal_error(str(e))


# ─── Feedback ───


@mcp.tool()
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
    try:
        logger.info("tool=submit_feedback model=%s prediction_id=%s", model_name, prediction_id)
        feedback_dir = _config.feedback_dir / model_name
        feedback_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model_name": model_name,
            "text": text,
            "corrected_labels": corrected_labels,
            "prediction_id": prediction_id,
        }

        # Append to JSONL file
        feedback_file = feedback_dir / "feedback.jsonl"
        with open(feedback_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return success_response(None, "Feedback recorded")
    except Exception as e:
        logger.error("submit_feedback failed: %s", e, exc_info=True)
        return internal_error(str(e))


# ─── Embedding ───


@mcp.tool()
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


def main():
    mcp.run()


if __name__ == "__main__":
    main()
