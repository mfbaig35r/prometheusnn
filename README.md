# PrometheusNN

A hierarchical text classifier that learns any taxonomy. Feed it descriptions and labels at any depth — 2-level product categories, 6-level industry codes, whatever your domain needs — and it builds a cascade of specialized classifiers that know when they're confident, when they're uncertain, and when they're looking at something entirely new.

## Why PrometheusNN

Most classifiers flatten your hierarchy into a single prediction. PrometheusNN preserves it. Each level gets its own classifier trained with noise injection from the level above, so deeper predictions stay robust even when parent levels are uncertain. Beam search explores alternative paths through the taxonomy. Calibrated confidence scores drive automatic routing decisions. Dual-signal novelty detection catches items that don't belong.

The result: a system that doesn't just classify — it knows what it knows.

## Key Capabilities

- **N-level cascade classification** — trains independent classifiers per taxonomy level with parent context
- **Noise-injected training** — injects realistic parent errors during training so deeper levels learn to cope with upstream mistakes
- **Beam search with adaptive widening** — explores alternative classification paths when confidence drops, with fork detection to surface disagreements
- **Temperature-calibrated confidence** — post-hoc calibration produces probabilities you can trust for routing
- **Dual-signal novelty detection** — centroid z-score + kNN distance flags out-of-taxonomy items without false positives
- **Automatic routing** — AUTO_ACCEPT / HUMAN_REVIEW / OUT_OF_TAXONOMY with tunable per-level thresholds
- **Feedback loop** — log corrections, retrain, improve

## Architecture

```
Text descriptions + taxonomy labels
        │
        ▼
  ┌─────────────┐
  │  Embedding  │  sentence-transformers (384d or 768d)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   Cascade   │  N classifiers, one per level
  │  Classifier │  parent noise injection for robustness
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Beam Search │  adaptive widening when confidence drops
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   Router    │  calibrated thresholds → accept / review / reject
  │  + Novelty  │  centroid z-score + kNN distance
  └─────────────┘
```

## Install

```bash
uv pip install -e .
```

## Interface

PrometheusNN exposes 23 tools via the [Model Context Protocol](https://modelcontextprotocol.io), making it accessible to any MCP-compatible client.

Add to your MCP client config:

```json
{
  "mcpServers": {
    "prometheus": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/prometheusnn", "prometheusnn"]
    }
  }
}
```

### Core Workflow

```
1. analyze_training_data("data.csv", "text", ["Category", "Sub", "Type"])
2. train_model("data.csv", "text", ["Category", "Sub", "Type"], "my_model")
3. predict("my_model", "some item description")
4. evaluate_model("my_model", "holdout.csv", "text", ["Category", "Sub", "Type"])
5. set_thresholds("my_model", {"Category": 0.85, "Type": 0.70})
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMETHEUS_HOME` | `~/.prometheus` | Base directory for models, logs, feedback |
| `PROMETHEUS_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Default sentence-transformer model |

## Development

```bash
uv sync --dev
uv run python -m pytest tests/ -v --tb=short    # 151 tests
uv run ruff check src/ tests/                   # lint
uv run ruff format --check src/ tests/          # format
```

## License

MIT
