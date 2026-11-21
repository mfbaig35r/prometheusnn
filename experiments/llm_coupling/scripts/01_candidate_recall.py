"""Step 1: can an LLM's candidate list even contain the right subheading?

On the held-out test split: routing mix, leaf recall of the beam at widths 3/5/10,
how many items belong to subheadings with no training rows, and three candidate-set
strategies for review items:

  A  the beam's own leaves
  B  every subheading under the confident prefix
  C  every subheading under the headings in the surviving beams  (the one that works)

Writes data/candidates.json (gitignored; input to 02_sample_items.py).

  uv run python experiments/llm_coupling/scripts/01_candidate_recall.py
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from common import DATA, beam_paths, leaves_under_headings, load

predictor, train, test = load()
tax = predictor.taxonomy
leaf = tax.get_level(2).classes
pcm = tax.parent_child_map
seen = Counter(train.leaf_labels.tolist())
print(f"test split: {len(test)} items")

rows = []
for i in range(len(test)):
    emb = test.embeddings[i]
    r = predictor.predict(emb, use_beam_search=True, beam_width=3)
    beams = {w: beam_paths(predictor, emb, w) for w in (3, 5, 10)}
    d = r.routing["confident_depth"]
    best = beams[3][0]
    if d is None:
        set_b = set(range(len(leaf)))
    elif d == 0:
        set_b = leaves_under_headings(tax, pcm.get((1, best[0]), []))
    else:
        set_b = leaves_under_headings(tax, [best[1]])
    rows.append(
        {
            "i": i,
            "text": test.texts[i],
            "true": [int(test.labels_by_depth[k][i]) for k in range(3)],
            "decision": r.routing["decision"],
            "confident_depth": d,
            "pred_leaf": leaf.index(r.path[-1]),
            "train_rows_for_true_leaf": seen[int(test.leaf_labels[i])],
            **{f"beams{w}": [p[-1] for p in beams[w]] for w in beams},
            "setB_size": len(set_b),
            "setB_hit": int(test.leaf_labels[i]) in set_b,
            "setC": sorted(leaves_under_headings(tax, {p[1] for p in beams[3]})),
        }
    )
predictor.beam.beam_width = 3


def recall(subset, key, k=None):
    return sum(r["true"][2] in (r[key][:k] if k else r[key]) for r in subset) / len(subset)


print("routing:", dict(Counter(r["decision"] for r in rows)))
print(f"{'subset':<18}{'n':>6}{'top1':>8}{'beam3':>8}{'beam5':>8}{'beam10':>8}")
for name in ["ALL", "AUTO_ACCEPT", "HUMAN_REVIEW", "OUT_OF_TAXONOMY"]:
    sub = rows if name == "ALL" else [r for r in rows if r["decision"] == name]
    print(
        f"{name:<18}{len(sub):>6}{recall(sub, 'beams3', 1):>8.1%}{recall(sub, 'beams3'):>8.1%}"
        f"{recall(sub, 'beams5'):>8.1%}{recall(sub, 'beams10'):>8.1%}"
    )

unseen = sum(r["train_rows_for_true_leaf"] == 0 for r in rows)
print(f"\ntrue subheading has no training rows: {unseen}/{len(rows)} ({unseen / len(rows):.1%})")

review = [r for r in rows if r["decision"] == "HUMAN_REVIEW"]
for label, hits, sizes in [
    (
        "A beam leaves",
        [r["true"][2] in r["beams3"] for r in review],
        [len(r["beams3"]) for r in review],
    ),
    (
        "B all under confident prefix",
        [r["setB_hit"] for r in review],
        [r["setB_size"] for r in review],
    ),
    (
        "C all under beam headings",
        [r["true"][2] in r["setC"] for r in review],
        [len(r["setC"]) for r in review],
    ),
]:
    s = np.array(sizes)
    print(
        f"{label:<30} recall {np.mean(hits):.1%}   size median {np.median(s):.0f}, "
        f"p90 {np.percentile(s, 90):.0f}, max {s.max()}"
    )

DATA.mkdir(exist_ok=True)
(DATA / "candidates.json").write_text(json.dumps(rows))
