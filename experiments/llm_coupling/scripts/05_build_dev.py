"""Step 5a: a dev set for prompt tuning, disjoint from the 100 scored items.

200 HUMAN_REVIEW items from the held-out test split, excluding the 100 in
tariff_items.json, with the same candidate set C. Also writes data/taxonomy_info.json:
each subheading's heading, chapter number and 6-digit HTS code, which the prompt
variants use. Needs data/candidates.json (01_candidate_recall.py).

  uv run python experiments/llm_coupling/scripts/05_build_dev.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

import pandas as pd
from common import CSV, DATA, MODEL

from prometheus.server import _get_predictor

tax = _get_predictor(MODEL).taxonomy
leaf = tax.get_level(2).classes
heads = tax.get_level(1).classes
parent = {c: p for (d, p), kids in tax.parent_child_map.items() if d == 2 for c in kids}

df = pd.read_csv(CSV, dtype={"hts8": str})
codes = defaultdict(set)
for name, hts8 in zip(df["Subheading"], df["hts8"]):
    codes[name].add(f"{hts8[:4]}.{hts8[4:6]}")
info = {
    name: {
        "heading": heads[parent[i]],
        "chapter": int(sorted(codes[name])[0][:2]),
        "code": sorted(codes[name])[0],
        "codes": len(codes[name]),
    }
    for i, name in enumerate(leaf)
}
(DATA / "taxonomy_info.json").write_text(json.dumps(info))

rows = [
    r for r in json.loads((DATA / "candidates.json").read_text()) if r["decision"] == "HUMAN_REVIEW"
]
random.seed(7)
scored = {r["i"] for r in random.sample(rows, 100)}  # the same draw as 02_sample_items.py
pool = [r for r in rows if r["i"] not in scored]
rng = random.Random(99)
items = []
for k, r in enumerate(rng.sample(pool, 200)):
    cands = list(r["setC"])
    rng.shuffle(cands)
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
(DATA / "dev_items.json").write_text(json.dumps(items, indent=1))

chapters = sorted({info[c["label"]]["chapter"] for it in items for c in it["candidates"]})
scored_items = json.loads((DATA / "tariff_items.json").read_text())
chapters_all = sorted(
    set(chapters) | {info[c["label"]]["chapter"] for it in scored_items for c in it["candidates"]}
)
print(
    f"dev: {len(items)} items, answer offered in {sum(i['_true_n'] > 0 for i in items)}, "
    f"cascade right {sum(i['_cascade'] == i['_true'] for i in items)}"
)
print(f"subheadings with more than one 6-digit code: {sum(v['codes'] > 1 for v in info.values())}")
print(f"chapters needed for notes: dev {len(chapters)}, dev + scored {len(chapters_all)}")
print(json.dumps(chapters_all))
