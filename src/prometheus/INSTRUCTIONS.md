PrometheusNN — Generalized N-Level Cascade Classifier.

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
