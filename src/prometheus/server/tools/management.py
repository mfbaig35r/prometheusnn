"""Model management tools: list, describe, delete, export, build_code_mapping."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from prometheus import server
from prometheus.core.mcp_safety import ToolValidationError, mcp_tool_safe
from prometheus.core.responses import success_response
from prometheus.data.loader import load_dataframe
from prometheus.insights import build_model_health
from prometheus.server import logger, mcp
from prometheus.storage.model_store import (
    delete_model as delete_model_from_disk,
)
from prometheus.storage.model_store import save_code_mapping


@mcp.tool()
@mcp_tool_safe
def list_models() -> dict:
    """List all trained models with summary stats."""
    logger.info("tool=list_models")
    models = []
    for name in server._config.list_models():
        try:
            with open(server._config.model_path(name) / "manifest.json") as f:
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


@mcp.tool()
@mcp_tool_safe
def describe_model(model_name: str) -> dict:
    """Get full model details: taxonomy, metrics, thresholds, config.

    Args:
        model_name: Name of the model to describe
    """
    logger.info("tool=describe_model model=%s", model_name)
    model_path = server._config.model_path(model_name)
    if not (model_path / "manifest.json").exists():
        raise ToolValidationError(f"Model '{model_name}' not found")

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


@mcp.tool()
@mcp_tool_safe
def delete_model(model_name: str) -> dict:
    """Delete a model from disk.

    Args:
        model_name: Name of the model to delete
    """
    logger.info("tool=delete_model model=%s", model_name)
    model_path = server._config.model_path(model_name)
    if not model_path.exists():
        raise ToolValidationError(f"Model '{model_name}' not found")

    delete_model_from_disk(model_path)
    server._predictor_cache.pop(model_name, None)

    return success_response(None, f"Model '{model_name}' deleted")


@mcp.tool()
@mcp_tool_safe
def export_model(
    model_name: str,
    output_path: str,
) -> dict:
    """Copy model directory to a new location.

    Args:
        model_name: Name of the model to export
        output_path: Destination directory
    """
    logger.info("tool=export_model model=%s dest=%s", model_name, output_path)
    model_path = server._config.model_path(model_name)
    if not model_path.exists():
        raise ToolValidationError(f"Model '{model_name}' not found")

    dest = Path(output_path)
    if dest.exists():
        raise ToolValidationError(f"Destination already exists: {output_path}")

    shutil.copytree(model_path, dest)
    return success_response({"exported_to": str(dest)})


@mcp.tool()
@mcp_tool_safe
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
    logger.info("tool=build_code_mapping model=%s code_col=%s", model_name, code_column)
    model_path = server._config.model_path(model_name)
    if not (model_path / "manifest.json").exists():
        raise ToolValidationError(f"Model '{model_name}' not found")

    with open(model_path / "manifest.json") as f:
        manifest = json.load(f)

    taxonomy_levels = [lvl["name"] for lvl in manifest["taxonomy"]["levels"]]
    cols = level_columns or taxonomy_levels

    df = load_dataframe(file_path)

    missing = [c for c in cols + [code_column] if c not in df.columns]
    if missing:
        raise ToolValidationError(f"Columns not found: {missing}")

    df = df.dropna(subset=cols + [code_column]).reset_index(drop=True)

    mapping: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        path_key = "|".join(str(row[c]) for c in cols)
        code = str(row[code_column]).strip()
        mapping.setdefault(path_key, set()).add(code)

    sorted_mapping = {k: sorted(v) for k, v in mapping.items()}

    save_code_mapping(model_path, code_column, cols, sorted_mapping)
    server._code_mapping_cache.pop(model_name, None)

    multi_code_paths = sum(1 for v in sorted_mapping.values() if len(v) > 1)
    all_codes: set[str] = set()
    for codes in sorted_mapping.values():
        all_codes.update(codes)

    return success_response(
        {
            "model_name": model_name,
            "code_column": code_column,
            "level_columns": cols,
            "unique_paths": len(sorted_mapping),
            "unique_codes": len(all_codes),
            "multi_code_paths": multi_code_paths,
        },
        "Code mapping saved",
    )
