# PrometheusNN

Build a hierarchical text classifier without writing code. Point Claude (or any MCP client) at your data and it walks you from "I have a CSV" to a trained model — profiling your columns, detecting whether the problem fits, surfacing concerns before training, and recommending the right setup.

PrometheusNN is the ML expert sitting next to you. You bring the data and the goal; it handles the rest.

## What this looks like

```
You:    I have a CSV of product descriptions and I want to categorize them.
        Can you help?

Claude: [calls start_here on the file]

        I looked at your data — 11,400 product descriptions across three
        nested category levels (Chapter → Heading → Subheading). One thing
        worth flagging before we train: the deepest level has ~2 examples
        per class on average, which usually means poor accuracy at that
        depth. Want to do all three levels, or simplify to the top two?

You:    Top two.

Claude: [fires analyze_training_data → train_model with the simpler setup]

        Done. Chapter accuracy 92%, Heading accuracy 78%. Want to try
        classifying something?
```

No column-name guessing, no parameter cargo-culting, no premature training runs.

## Install

```bash
pip install prometheusnn
```

Then add to your MCP client config (Claude Desktop, Cursor, Zed, etc.):

```json
{
  "mcpServers": {
    "prometheus": {
      "command": "prometheusnn"
    }
  }
}
```

That's it. Open your MCP client and start a conversation about classifying something.

## How it works under the hood

Two entry-point tools handle every new conversation:

- **`start_here(file_path)`** — profiles your CSV, detects whether it's a hierarchical, flat, or multi-label classification problem, surfaces concerns before training, and returns pre-filled next steps so the LLM never has to guess column names.

- **`scope_problem(goal_description)`** — for when you don't have data yet. Tells you whether Prometheus fits your goal, what data you'll need, and what to watch for. Politely redirects non-fits (regression, clustering, NER) to the right tools.

From there, the LLM routes through the existing tool surface (training, evaluation, threshold tuning, prediction) based on what `start_here` recommended.

## What's actually being built

Under the conversational surface, PrometheusNN trains a cascade of neural network classifiers — one per level of your taxonomy — with parent-noise injection so deeper levels stay robust when upper levels are uncertain. Beam search explores alternative paths when confidence drops. Temperature calibration produces probabilities you can trust for routing. Dual-signal novelty detection catches items that don't belong.

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

You don't need to know any of this to use it — but it's there if you want to dig in.

## Tool reference

20 MCP tools total. The two you'll be told about; the rest the LLM picks for you.

**Entry points** (start here)
- `start_here` — assess a file, detect problem type, return pre-filled next steps
- `scope_problem` — assess a goal description (no data yet), check fit and data requirements

**Training & data**
- `analyze_training_data`, `train_model`, `resume_training`

**Inference**
- `predict`, `predict_batch`, `classify_with_context`

**Model management**
- `list_models`, `describe_model`, `delete_model`, `export_model`

**Evaluation & tuning**
- `evaluate_model`, `explain_prediction`, `get_confusion_matrix`
- `get_threshold_report`, `set_thresholds`

**Other**
- `submit_feedback`, `list_embedding_models`, `build_code_mapping`

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMETHEUS_HOME` | `~/.prometheus` | Base directory for models, logs, feedback |
| `PROMETHEUS_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Default sentence-transformer model |

## Development

```bash
git clone https://github.com/mfbaig35r/prometheusnn
cd prometheusnn
uv sync --extra dev
uv run python -m pytest tests/ -v --tb=short    # 179 tests
uv run ruff check src/ tests/                   # lint
uv run ruff format --check src/ tests/          # format
```

## License

MIT
