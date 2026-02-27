# Beam Search Strategy

This document describes the beam search inference pipeline in Prometheus, including adaptive beam widening, diversity-aware pruning, fork detection, and self-interpreting insights.

## File Reference

- Beam search: `src/prometheus/inference/beam_search.py`
- Router & forks: `src/prometheus/inference/router.py`
- Predictor: `src/prometheus/inference/predictor.py`
- Insights: `src/prometheus/server.py` (`_analyze_forks`)
- Tests: `tests/test_beam_search.py`

## Overview

Prometheus uses an N-level hierarchical cascade classifier. At inference time, beam search explores multiple classification paths through the taxonomy tree and returns the highest-scoring complete path along with alternatives.

Two classifiers are available:
- **GreedyClassifier** — picks the single best class at each level. Fast, no alternatives.
- **BeamSearchClassifier** — maintains `beam_width` candidate paths. Slower, but surfaces competing hypotheses.

## Standard Beam Search

At each taxonomy level (depth 0 through N-1):

1. For each existing beam, run the level classifier to get class probabilities.
2. Expand each beam with the top-k children (k = `beam_width`).
3. Score candidates by cumulative log-probability: `log_score = sum(log(p_i))` across all levels.
4. Prune to `beam_width` beams by global score.

This works well when the model is confident at every level. The problem arises when it isn't.

## Problem: Score-Based Pruning Kills Minority Branches

Consider classifying "plumbing and heating contractor" in a 4-level NAICS taxonomy:

| Level | Top candidate | Confidence | Runner-up | Confidence |
|-------|--------------|------------|-----------|------------|
| Sector | Construction | 0.99 | — | — |
| Subsector | Specialty Trade | 0.99 | — | — |
| IndustryGroup | Building Finishing | 0.56 | Building Equipment | 0.19 |
| NAICSIndustry | (varies) | ~0.17 | — | — |

At IndustryGroup (depth 2), the model is genuinely uncertain — 0.56 vs 0.19. Building Equipment Contractors is the correct answer for plumbing/heating, but its log-score deficit from depth 2 compounds at depth 3. With pure score-based pruning, all final beams collapse into the Building Finishing branch. The correct path never appears in the output.

## Solution: Adaptive Widening + Diversity-Aware Pruning

Two mechanisms work together to fix this.

### 1. Adaptive Beam Widening

When `confidence_thresholds` are configured, the beam width dynamically increases at uncertain levels.

```
effective_width = beam_width
if max_prob < confidence_thresholds[depth]:
    effective_width = beam_width * adaptive_factor  # default: 2x
```

This happens in both `_init_beams()` (depth 0) and `_expand_beams()` (depth 1+). Widening creates more candidates at the uncertain level, giving minority branches a chance to enter the beam. But widening alone isn't enough — at the next level, the expanded candidates get pruned back down, and pure score ranking still kills the minority branches.

### 2. Diversity-Aware Pruning

When widening triggers, the system records the **first uncertain level** as `_diversity_depth`. This depth becomes a diversity anchor: during pruning at all subsequent levels, at least one representative from each unique class at the diversity depth is guaranteed to survive.

#### Tracking

```python
# Reset per prediction
self._diversity_depth = None

# Set on first widening (in _init_beams or _expand_beams)
if self._diversity_depth is None:
    self._diversity_depth = depth
```

The `is None` guard ensures only the first uncertain level becomes the anchor. If multiple levels are uncertain, deeper diversity is handled implicitly — diverse branches at the anchor level naturally produce diverse candidates downstream.

#### Pruning Algorithm (`_prune_with_diversity`)

Input: candidate beams (pre-sorted by log_score descending), target width.

**Phase 1 — Guarantee representation:**
- Group beams by their class index at `_diversity_depth`.
- Take the highest-scored beam from each group (one representative per branch).

**Phase 2 — Fill by score:**
- If representatives don't fill the width, add remaining beams in score order.
- Re-sort the final list by log_score.

```
Beams at depth 3 (6 candidates, width=6):
  BF→OtherFinishing    score=-2.32  ← best of BF group
  BF→FinishCarpentry   score=-2.50
  BF→Flooring          score=-2.51
  BE→Electrical        score=-2.72  ← best of BE group (guaranteed slot)
  BE→Plumbing          score=-3.10
  BF→Painting          score=-3.20

After diversity pruning (width=6): all 6 survive (no pruning needed)
After diversity pruning (width=3): BF→OtherFinishing, BF→FinishCarpentry, BE→Electrical
```

Without diversity pruning at width=3, the result would be the top 3 BF paths. With it, the BE branch survives.

#### Fallback

When `_diversity_depth` is `None` (no thresholds configured, or all levels confident), `_prune_with_diversity` returns `beams[:width]` — identical to the previous pure-score behavior. This means:
- Models without `confidence_thresholds` are completely unaffected.
- Predictions where all levels are confident use plain score ranking.

## Configuration

Both mechanisms are controlled via `BeamSearchClassifier` constructor parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beam_width` | 3 | Base number of paths to maintain |
| `confidence_thresholds` | None | Dict mapping depth → minimum confidence. Levels below threshold trigger widening. |
| `adaptive_factor` | 2 | Multiplier for beam width when widening triggers |

Thresholds are auto-generated during training as linear decay from 0.90 (root) to 0.75 (leaf). They can be overridden via `set_thresholds`.

## Edge Cases

**Single-level taxonomies:** No beam expansion occurs. Diversity pruning is never triggered.

**Out-of-taxonomy items:** Novelty detection runs independently. Diversity pruning may activate if thresholds are configured, but the routing decision is governed by the novelty detector, not the beam structure.

**Greedy mode (beam_width=1):** Uses `GreedyClassifier`, which has no beam expansion, widening, or diversity logic.

**Many competing classes at diversity depth:** If more unique classes exist at the anchor depth than the effective beam width, representatives are capped at `width` (taken in score order). This means the lowest-scored branches may still be dropped. In practice this is rare — uncertain levels typically have 2-4 competing classes, well within a width of 3-6.

## Fork Detection

After beam search and diversity pruning produce the final set of beams, the router (`ThresholdRouter.route_with_forks`) groups them into `forked_completions` — alternative paths presented to the caller.

### How forks are detected

The router identifies a **fork depth** — the level where the beams diverge:

- **`confident_depth` exists** (e.g., NAICS: L1 and L2 are confident): `fork_depth = confident_depth + 1`. Forks are alternative completions below the confident prefix. All forks share the same upper levels (e.g., Construction → Specialty Trade) and diverge at the uncertain level (IndustryGroup).

- **`confident_depth` is None** (e.g., MeSH: L1 itself is uncertain): `fork_depth = 0`. The entire path is uncertain, so forks are grouped by L1 class — one representative per top-level branch, each showing the best full path through that branch.

### Why both modes matter

NAICS "plumbing and heating contractor": L1 (Construction, 0.99) and L2 (Specialty Trade, 0.99) are confident. The question is which IndustryGroup, so forks show competing L3→L4 suffixes below the shared prefix.

MeSH "rheumatoid arthritis": L1 itself is split (Musculoskeletal 0.43 vs Connective Tissue 0.14). There's no confident prefix. Forks show competing full paths — one through Musculoskeletal, one through Connective Tissue — because both are legitimate MeSH placements.

### Format

Each `ForkedCompletion` contains:
- `suffix` — labels from `fork_depth` onward
- `suffix_indices` — class indices from `fork_depth` onward
- `suffix_score` — cumulative log-prob of the suffix portion
- `per_level_probs` — probability at each suffix level

Forks are sorted by `suffix_score` descending (best first).

## Self-Interpreting Insights

The `insights` field in the predict response is generated by `_analyze_forks()` in `server.py`. It runs three checks against the already-computed `forked_completions` — no additional model calls, no ML, just pattern matching on beam search output.

### Insight 1: Minority branch with higher leaf confidence

```
"Alternative path via 'Skin and Connective Tissue Diseases' has higher leaf confidence
 for 'Arthritis, Rheumatoid' (95% vs 61%)"
```

**What it detects:** A forked completion whose leaf-level probability exceeds the best path's leaf probability by more than 5 percentage points.

**Why it matters:** In polyhierarchical taxonomies, the best path (highest cumulative score) often has a weak leaf prediction because its strength comes from a confident upper level. A minority branch with low L1 confidence can produce a much more confident leaf — this insight flags exactly that.

**What a caller should do:** Present both the best path and the flagged alternative. The alternative's leaf prediction may be more reliable even though its overall path score is lower.

### Insight 2: Branch convergence

```
"Multiple branches ('Musculoskeletal Diseases' and 'Skin and Connective Tissue Diseases')
 converge on 'Arthritis, Rheumatoid'"
```

**What it detects:** Two or more forked completions that arrive at the same leaf label through different upper-level branches.

**Why it matters:** Independent paths reaching the same conclusion is strong corroboration. If the Musculoskeletal branch and the Connective Tissue branch both end at "Arthritis, Rheumatoid", the model is confident in the leaf even though it's uncertain about which branch to use. This is the hallmark of polyhierarchical data — the item genuinely belongs in multiple places.

**What a caller should do:** Treat the converged leaf as high-confidence. Optionally present both branches as valid placements.

### Insight 3: Low best-path leaf confidence

```
"Best path leaf confidence is low (22%), review forked_completions for alternatives"
```

**What it detects:** The best path's leaf probability is below 30%, and neither of the above insights fired.

**Why it matters:** Safety net. The model picked a path but isn't confident about where it ends. The forked completions may contain a better answer or the item may need human review.

**What a caller should do:** Don't present the best path as authoritative. Show the top 2-3 forked completions and let the user decide.

### When insights are absent

No `insights` field means: the best path is the clear winner and no fork has a stronger leaf prediction. The caller can trust the best path.

## Design Philosophy

Each layer does one simple thing:

| Layer | What it does | Complexity |
|-------|-------------|-----------|
| Beam search | Explore multiple paths | Expansion + log-score |
| Diversity pruning | Don't let score kill minority branches | One dict lookup per beam |
| Fork detection | Group paths by where they diverge | Prefix matching |
| Insights | Compare leaf confidences across groups | Arithmetic |

No layer "understands" polyhierarchy. Stacked together, they surface it — because they don't throw away information prematurely.

## Tested On

**NAICS** (4 levels, 20 sectors, clean hierarchy, 1,012 samples):
- Diversity pruning keeps Building Equipment Contractors alive for "plumbing and heating contractor"
- Forks show competing IndustryGroup completions below the confident Sector/Subsector prefix

**MeSH Diseases** (4 levels, 22 L1 categories, 2,459 leaf classes, polyhierarchical, 10,984 samples):
- Diversity pruning keeps minority L1 branches alive (Connective Tissue at 14% L1 confidence finds RA at 95% leaf confidence)
- Fork detection at depth 0 surfaces all competing top-level branches
- Convergence insight detects when two branches agree on the same leaf
- Composite text ("Name: scope note") improved L4 accuracy by 2.3 points
