# Roadmap: Self-Aware Server

Next steps for making Prometheus self-interpreting for cold users and LLMs.

> **Status update (2026-04-29):** The "analyze_training_data Recommendations" and "Post-Training Diagnostics" sections below are partially addressed by the new `intake/` module — `start_here` profiles data, detects problem types, and `surface_concerns()` flags class-imbalance and sparsity issues *before* training. Domain Presets and Prediction Response Enhancements remain open.

## Done (2026-02-26)

1. **Diversity-aware beam pruning** — `beam_search.py`: minority branches at uncertain levels guaranteed to survive pruning
2. **Fork detection for depth-0 uncertainty** — `router.py`: when no level is confident, group forks by L1 class instead of bailing out
3. **Self-interpreting insights** — `server.py` `_analyze_forks()`: flags minority branches with higher leaf confidence, branch convergence, low best-path confidence
4. **`beam_width` per-call** — was silently ignored, now threaded from server → predictor → beam search
5. **`adaptive_factor` via `set_thresholds`** — configurable beam widening multiplier per model
6. **MeSH disease classifier** — 10,984 rows, 2,459 leaf classes, validated all of the above on real polyhierarchical data
7. **Composite text training** — "Name: description" format improved L4 accuracy +2.3 points
8. **Docs** — `docs/beam-search-strategy.md` covers full pipeline

## Next: `analyze_training_data` Recommendations

When `analyze_training_data` detects polyhierarchy (multi-parent warnings), it should append actionable recommendations to the response:

- **"Detected N polyhierarchical items. Recommend composite text column ('name: description') for better leaf disambiguation."**
- **"L4 has X singleton classes (1 sample each). Expect weak leaf accuracy. Consider merging rare classes or cutting hierarchy at L3."**
- **"Class imbalance at L1: 'Occupational Diseases' has 2 samples vs median 415. This class will train poorly."**

Implementation: add a `recommendations` list to the `analyze_training_data` response in `server.py`. Pure heuristics on the stats already computed — no new analysis needed.

## Next: Post-Training Diagnostics

After `train_model` completes, analyze the per-level metrics and suggest threshold adjustments:

- **L1 val_acc < 60%**: "L1 accuracy is low. If your taxonomy is polyhierarchical (items belong to multiple L1 categories), this is expected. Consider lowering L1 confidence threshold (currently 0.90) to 0.70 and increasing adaptive_factor to 3."
- **L4 val_acc < 50%**: "Leaf accuracy is below 50%. Common causes: too many singleton classes, or text descriptions don't distinguish leaf-level categories. Consider composite text training or cutting hierarchy depth."
- **Train/val gap > 15%**: "Overfitting detected at L3 (train 74%, val 58%). Consider increasing dropout or noise_rate."

Implementation: add `_post_training_insights(metrics, taxonomy)` helper, append to `train_model` response as `recommendations` field.

## Next: Prediction Response Enhancements

- **When routing is HUMAN_REVIEW and convergence insight fires**: add `"suggested_label"` field with the converged leaf — saves the reviewer from reading all forks
- **When routing is OUT_OF_TAXONOMY but forks exist**: add `"closest_match"` field with the highest-confidence leaf across all forks — "not in taxonomy, but closest known class is X"

## Future: Domain Presets

Different taxonomy types need different defaults:

| Domain | beam_width | adaptive_factor | Threshold strategy |
|--------|-----------|----------------|-------------------|
| Clean hierarchy (NAICS) | 3 | 2 | Auto (0.90→0.75) |
| Polyhierarchical (MeSH, HTS) | 5 | 3 | Lower L1 (0.70), relax leaf (0.60) |
| Flat/wide (product catalogs) | 3 | 2 | Uniform 0.80 |

Could be exposed as `set_thresholds(model, preset="polyhierarchical")` or auto-detected from `analyze_training_data` warnings.
