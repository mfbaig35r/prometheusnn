"""PrometheusNN — Generalized N-Level Cascade Classifier MCP Server.

This module exposes the shared FastMCP instance, configuration, caches, and
helpers used by the per-domain tool modules under ``prometheus.server.tools``.

Tool functions are registered as a side effect of importing
``prometheus.server.tools`` at the bottom of this file.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from prometheus.core.config import PrometheusConfig
from prometheus.core.logging import setup_logging
from prometheus.core.lru import LRUCache
from prometheus.inference.novelty import NoveltyDetector
from prometheus.inference.predictor import CascadePredictor
from prometheus.inference.router import ThresholdRouter
from prometheus.storage.model_store import load_code_mapping, load_model, load_thresholds

_INSTRUCTIONS = (Path(__file__).parent / "INSTRUCTIONS.md").read_text()

mcp = FastMCP("PrometheusNN", instructions=_INSTRUCTIONS)

_config = PrometheusConfig()
_config.ensure_dirs()
setup_logging(_config.logs_dir)
logger = logging.getLogger("prometheus.server")

_PREDICTOR_CACHE_SIZE = 8
_CODE_MAPPING_CACHE_SIZE = 16

# Cache loaded predictors with the manifest mtime at load time so we can
# invalidate stale entries when a model is overwritten on disk.
_predictor_cache: LRUCache[str, tuple[CascadePredictor, float]] = LRUCache(
    maxsize=_PREDICTOR_CACHE_SIZE
)

_code_mapping_cache: LRUCache[str, dict | None] = LRUCache(maxsize=_CODE_MAPPING_CACHE_SIZE)


def _get_code_mapping(model_name: str) -> dict | None:
    """Get or load code mapping for a model."""
    cached = _code_mapping_cache.get(model_name)
    if cached is not None or model_name in _code_mapping_cache:
        return cached
    model_path = _config.model_path(model_name)
    mapping = load_code_mapping(model_path)
    _code_mapping_cache.put(model_name, mapping)
    return mapping


def _get_predictor(model_name: str) -> CascadePredictor:
    """Get or load a CascadePredictor. Invalidates the cache if manifest.json
    has been modified since the cached predictor was loaded.
    """
    model_path = _config.model_path(model_name)
    manifest_path = model_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found")

    current_mtime = manifest_path.stat().st_mtime
    cached = _predictor_cache.get(model_name)
    if cached is not None and cached[1] >= current_mtime:
        return cached[0]

    model, manifest = load_model(model_path)
    taxonomy = manifest.taxonomy
    config = load_thresholds(model_path)

    novelty = NoveltyDetector(
        z_score_threshold=config.z_score,
        knn_k=10,
        novelty_conf_threshold=config.novelty_conf,
    )
    if (model_path / "novelty_centroids.npz").exists():
        novelty.load(model_path)

    router = ThresholdRouter(config, taxonomy.level_names)
    predictor = CascadePredictor(
        model=model,
        taxonomy=taxonomy,
        novelty_detector=novelty,
        router=router,
        parent_child_map=taxonomy.parent_child_map,
        adaptive_factor=config.adaptive_factor,
    )
    _predictor_cache.put(model_name, (predictor, current_mtime))
    return predictor


def main() -> None:
    """CLI entry point — runs the MCP server over stdio."""
    mcp.run()


# Importing the tools subpackage registers all tool functions as a side effect.
from prometheus.server import tools as _tools  # noqa: E402, F401

# Re-export tool functions at the package level so callers (and tests) can
# `from prometheus.server import train_model` without knowing the internal layout.
from prometheus.server.tools.evaluation import (  # noqa: E402, F401
    evaluate_model,
    get_confusion_matrix,
    get_threshold_report,
    list_embedding_models,
    set_thresholds,
    submit_feedback,
)
from prometheus.server.tools.inference import (  # noqa: E402, F401
    classify_with_context,
    explain_prediction,
    predict,
    predict_batch,
)
from prometheus.server.tools.intake import (  # noqa: E402, F401
    scope_problem,
    start_here,
)
from prometheus.server.tools.management import (  # noqa: E402, F401
    build_code_mapping,
    delete_model,
    describe_model,
    export_model,
    list_models,
)
from prometheus.server.tools.training import (  # noqa: E402, F401
    analyze_training_data,
    resume_training,
    train_model,
)

if __name__ == "__main__":
    main()
