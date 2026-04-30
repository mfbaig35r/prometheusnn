"""Training tools: analyze_training_data, train_model, resume_training."""

from __future__ import annotations

import json

import torch

from prometheus import server
from prometheus.core.mcp_safety import mcp_tool_safe
from prometheus.core.responses import success_response
from prometheus.data.loader import analyze_data, prepare_data
from prometheus.inference.novelty import NoveltyDetector
from prometheus.insights import build_training_insights
from prometheus.server import logger, mcp
from prometheus.storage.model_store import load_model, save_model
from prometheus.training.noise import ParentNoiseInjector
from prometheus.training.trainer import TrainingConfig
from prometheus.training.trainer import train_model as run_training


@mcp.tool()
@mcp_tool_safe
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
    logger.info("tool=analyze_training_data file=%s levels=%s", file_path, level_columns)
    return success_response(analyze_data(file_path, text_column, level_columns))


@mcp.tool()
@mcp_tool_safe
def train_model(
    file_path: str,
    text_column: str,
    level_columns: list[str],
    model_name: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    config: TrainingConfig | None = None,
) -> dict:
    """Full training pipeline: embed → split → train N levels → calibrate → novelty → save.

    Args:
        file_path: Path to CSV or JSON training data
        text_column: Name of the text/description column
        level_columns: Ordered label columns root → leaf
        model_name: Name for the saved model
        embedding_model: Sentence-transformer model name
        config: Optional training hyperparameters (epochs, batch_size, noise_rate,
            hidden_dims, dropout, learning_rate, train_ratio, val_ratio, test_ratio).
            All fields have sensible defaults — pass only what you want to override.
    """
    config = config or TrainingConfig()
    logger.info(
        "tool=train_model model=%s file=%s levels=%s epochs=%d",
        model_name,
        file_path,
        level_columns,
        config.epochs,
    )
    progress_log = []

    def log(msg: str):
        progress_log.append(msg)

    log("Loading and embedding data...")
    taxonomy, train_data, val_data, test_data, warnings = prepare_data(
        file_path=file_path,
        text_column=text_column,
        level_columns=level_columns,
        model_name=model_name,
        embedding_model=embedding_model,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    embedding_dim = train_data.embeddings.shape[1]
    log(
        f"Data: {taxonomy.total_samples} samples, {taxonomy.num_levels} levels, "
        f"{embedding_dim}d embeddings"
    )

    model, result, _ = run_training(
        taxonomy=taxonomy,
        train_data=train_data,
        val_data=val_data,
        embedding_dim=embedding_dim,
        config=config,
        progress_callback=log,
    )

    log("Fitting novelty detector...")
    leaf = taxonomy.leaf_level
    novelty = NoveltyDetector()
    novelty.fit(train_data.embeddings, train_data.leaf_labels, leaf.classes)

    model_path = server._config.model_path(model_name)
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
    novelty.save(model_path)

    server._predictor_cache.pop(model_name, None)

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
                    {"name": lvl.name, "num_classes": lvl.num_classes} for lvl in taxonomy.levels
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


@mcp.tool()
@mcp_tool_safe
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
    logger.info(
        "tool=resume_training model=%s epochs=%d lr=%f",
        model_name,
        additional_epochs,
        learning_rate,
    )
    model_path = server._config.model_path(model_name)
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

    noise_injector = ParentNoiseInjector(noise_rate=config.noise_rate)
    if (model_path / "confusion_profiles.json").exists():
        with open(model_path / "confusion_profiles.json") as f:
            noise_injector.load_profiles(json.load(f))

    from prometheus.training.calibration import TemperatureScaler
    from prometheus.training.trainer import TrainingResult, _train_level

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

    server._predictor_cache.pop(model_name, None)

    return success_response(
        {
            "model_name": model_name,
            "additional_epochs": additional_epochs,
            "metrics": result.summary(),
            "warnings": warnings,
        }
    )
