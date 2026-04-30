# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-30

### Added
- **`prometheusnn-ui` CLI command** — WebSocket server (FastAPI + uvicorn) that
  exposes every registered MCP tool to a browser UI. Run after installing the
  `[ui]` extras and point the hosted demo at it for "connected mode".
- **`[ui]` optional dependency group** — `fastapi>=0.110`,
  `uvicorn[standard]>=0.27`, `websockets>=12.0`. Install with
  `pip install prometheusnn[ui]`.

### Notes
- The websocket route handler is defined at module level; defining it inside
  a factory caused FastAPI to reject handshakes with HTTP 403 in this version.

## [0.1.0] - 2026-04-30

First public release on PyPI. PrometheusNN is an MCP server that guides users
from a CSV file to a trained hierarchical text classifier through a conversational
interface — the LLM mediates, the user never writes Python.

### Added
- **Entry-point tools** for LLM-mediated onboarding:
  - `start_here(file_path)` — profiles a CSV/JSON, detects problem type,
    surfaces concerns *before* training, returns pre-filled `next_steps`
    so the LLM can fire the right tool without guessing column names.
  - `scope_problem(goal_description)` — assesses whether Prometheus fits
    a goal that doesn't have data yet; politely redirects non-fits
    (regression, clustering, NER) to appropriate alternatives.
- **20 MCP tools** covering the full workflow: intake, training, inference,
  model management, evaluation, threshold tuning, feedback.
- **Hierarchical cascade classifier** with parent-noise injection, beam search
  with adaptive widening, fork detection, temperature-calibrated confidence,
  and dual-signal novelty detection.
- **Three-tier routing** (AUTO_ACCEPT / HUMAN_REVIEW / OUT_OF_TAXONOMY)
  with per-level threshold tuning.
- Typed configuration objects via Pydantic v2 throughout — `RouterConfig`,
  `TrainingConfig`, `TaxonomyInfo`, `ColumnProfile`, `ProblemAssessment`, etc.
- LRU-bounded predictor and embedding caches with mtime-based invalidation
  so stale models don't return stale predictions.

### Notes
- Requires Python 3.11+.
- Heavy ML dependencies (`torch`, `sentence-transformers`) are required —
  the package only does meaningful work with them.
- Models are saved to `~/.prometheus/models/` as JSON manifests + `.pt`
  state dicts. No pickle.
