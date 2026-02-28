# PrometheusNN

Generalized N-level hierarchical cascade classifier, exposed as an MCP server.

Train, predict, evaluate, and tune classifiers on any taxonomy — from 2-level categories to deep 6+ level hierarchies. Uses per-level noise injection, temperature-calibrated confidence, beam search with adaptive widening, and dual-signal novelty detection.

## Install

```bash
uv pip install -e .
```

## MCP Configuration

Add to your `claude_desktop_config.json`:

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

## Quick Start

```
1. analyze_training_data("data.csv", "text", ["Category", "Sub", "Type"])
2. train_model("data.csv", "text", ["Category", "Sub", "Type"], "my_model")
3. predict("my_model", "some item description")
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMETHEUS_HOME` | `~/.prometheus` | Base directory for models, logs, feedback |
| `PROMETHEUS_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Default sentence-transformer model |

## Running Tests

```bash
uv run python -m pytest tests/ -v --tb=short
```

## Documentation

See `docs/` for detailed guides on beam search strategy, threshold tuning, and novelty detection.

## License

MIT
