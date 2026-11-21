# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.4] - 2026-11-20

### Security
- **Dependency security update.** The lockfile carried 17 packages with published
  advisories; it now has none (pip-audit, all extras). Minimum versions are
  raised where the vulnerable code is reachable from PrometheusNN, so upgrading
  an existing environment pulls the fixes:
  - `transformers>=5.10.0` and `sentence-transformers>=5.6.0`: model-loading
    flaws that could run code from a crafted model repository. Embedding models
    are loaded by name, so a `train_model` call naming a malicious model reached
    them.
  - `mcp[cli]>=1.30.0,<2`: eleven advisories in the MCP SDK. The `<2` cap is
    because mcp 2.x renamed `FastMCP`; migrating is separate work.
  - `starlette>=1.3.1` in the `[ui]` extra: request-handling flaws in the
    network-facing UI server's framework.
  - `torch` is not given a higher minimum: its two advisories concern
    TorchScript and `.pt2` loading, which PrometheusNN does not use. The
    lockfile moves to 2.15 regardless.
- Checked against the previous lockfile: MiniLM embeddings are unchanged
  (minimum cosine 0.9999999 over 300 texts) and the HTS model gives the same
  path for all 300.

## [0.2.3] - 2026-11-20

### Fixed
- **OUT_OF_TAXONOMY no longer fires on ordinary in-taxonomy items.** Novelty was
  judged by the distance to the *predicted leaf's* centroid, in units of that
  leaf's spread. Leaves with a few near-identical training rows had their spread
  clamped to 0.1, so an input belonging to a sibling leaf scored as extreme, and
  the leaf prediction itself was usually the wrong one. On the HTS sample, 29% of
  held-out tariff items and 35% of plain-English rewrites were routed
  OUT_OF_TAXONOMY. Novelty now uses the distance to the k-th nearest training
  embedding, against the 99th percentile of the training set's own leave-one-out
  kNN distances: 1% of held-out items and 0% of the rewrites are flagged, while
  40 of 40 non-product sentences still are. The centroid z-score is still
  reported, and decides only for models saved without training embeddings.
- **AUTO_ACCEPT is fit to a target precision instead of fixed thresholds.**
  Training now raises each level's threshold, top-down, until auto-accepted
  predictions on the validation split reach `auto_accept_precision` (new
  `TrainingConfig` field, default 0.95). Thresholds never go below the old
  defaults; a level that cannot reach the target is never auto-accepted (its
  threshold is 1.01) and those items go to HUMAN_REVIEW with the confident levels
  above it. On the HTS sample the old thresholds auto-accepted 7.5% of held-out
  items at 62.8% precision; the fitted ones auto-accept none, because leaf
  predictions cannot reach 95% there, while the confident chapter and heading
  are right 98.6% and 95.3% of the time. `train_model` and `resume_training`
  return a `routing` report. With fewer than 40 validation items the defaults are
  kept.
- `resume_training` now refits novelty data for the new weights and training data.
  It previously saved new training embeddings next to stale centroids.

### Changed
- Predictions report `knn_threshold` in their novelty details, and the
  OUT_OF_TAXONOMY reason names the kNN distance and threshold.

### Upgrade notes
- Models saved by 0.2.2 or earlier get the new novelty rule automatically: the
  kNN threshold is fit from their stored training embeddings when they load. Their
  thresholds are unchanged; retrain to fit thresholds to a target precision.
- Expect far fewer AUTO_ACCEPT decisions on large or sparse taxonomies. That is
  the fix: those auto-accepts were often wrong.

## [0.2.2] - 2026-10-30

### Security
- **Novelty data is loaded without pickle.** `novelty_centroids.npz` was loaded
  with `allow_pickle=True`, so a crafted model directory could run code in the
  server when any tool loaded that model. In 0.2.0 and earlier this was
  reachable through the `model_name` path traversal fixed in 0.2.1; in 0.2.1 it
  still applied to a malicious model placed in the models directory. Both
  novelty files now load with `allow_pickle=False`. Models saved by
  PrometheusNN contain only numeric and string arrays and load unchanged; a
  file holding pickled objects is refused with an error asking for a retrain.

## [0.2.1] - 2026-10-30

### Security
- **Model-name path traversal.** `delete_model`, `describe_model`,
  `submit_feedback` and every other tool that takes a `model_name` now accept
  only a single path segment (`[A-Za-z0-9._-]`, no `..`), and every resolved
  path must stay under its base directory. In 0.2.0, `delete_model("..")`
  deleted all of `PROMETHEUS_HOME`, an absolute name could delete any
  directory the process could write, and `submit_feedback` could append to a
  file outside the feedback directory. Upgrade if the MCP server or the UI
  server is reachable by anything you do not fully trust.
- **Unauthenticated UI WebSocket.** `prometheusnn-ui` now requires a token
  (`?token=` on the WebSocket URL; set with `--token` or
  `PROMETHEUS_UI_TOKEN`, otherwise generated and printed at startup) and
  rejects browser handshakes from origins outside an allowlist (the hosted
  demo and `localhost:3000` by default; add more with `--allowed-origin`).
  In 0.2.0 any web page open in the same browser could drive every tool.
- `delete_model` and `export_model` are withheld from the UI unless the
  server is started with `--allow-destructive`.

### Changed
- `delete_model` refuses a directory that has no `manifest.json`.
- `export_model` writes only under the exports directory
  (`PROMETHEUS_EXPORT_DIR`, default `<PROMETHEUS_HOME>/exports`). A relative
  `output_path` is placed there; an absolute path outside it is refused.
  **Breaking** for callers that exported to arbitrary absolute paths.

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
