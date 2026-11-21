"""Step 2: sample 100 HUMAN_REVIEW items and write the judge's item set.

Each item gets candidate set C in shuffled order (no ordering hint from the model).
Fields starting with "_" are the answer key; to_text.py strips them before a judge
sees anything.

  uv run python experiments/llm_coupling/scripts/02_sample_items.py
"""

from __future__ import annotations

import json
import random

from common import DATA, MODEL

from prometheus.server import _get_predictor

leaf = _get_predictor(MODEL).taxonomy.get_level(2).classes
rows = [
    r for r in json.loads((DATA / "candidates.json").read_text()) if r["decision"] == "HUMAN_REVIEW"
]
random.seed(7)
items = []
for k, r in enumerate(random.sample(rows, 100)):
    cands = list(r["setC"])
    random.shuffle(cands)
    items.append(
        {
            "id": k + 1,
            "text": r["text"],
            "candidates": [{"n": j + 1, "label": leaf[c]} for j, c in enumerate(cands)],
            "_true": leaf[r["true"][2]],
            "_true_n": next((j + 1 for j, c in enumerate(cands) if c == r["true"][2]), 0),
            "_cascade": leaf[r["pred_leaf"]],
        }
    )
(DATA / "tariff_items.json").write_text(json.dumps(items, indent=1))
print(
    f"100 review items | answer in candidate set: {sum(it['_true_n'] > 0 for it in items)} | "
    f"cascade top-1 correct: {sum(it['_cascade'] == it['_true'] for it in items)}"
)
