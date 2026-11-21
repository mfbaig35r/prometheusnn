"""Step 3: run the plain-English rewrites through the front half of the pipeline.

data/plain_texts.json holds the same 100 items rewritten in commercial language by an
LLM that saw only the original text (prompt in prompts/paraphrase.md). Each rewrite goes
MiniLM -> cascade (beam 3) -> routing -> candidate set C. Every item goes to the judge,
whatever its routing, so the run also shows what the LLM does with items the cascade
would have auto-accepted or flagged as novel.

  uv run python experiments/llm_coupling/scripts/03_plain_pipeline.py
"""

from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
from common import DATA, MODEL, beam_paths, leaves_under_headings

from prometheus.core.embedding import encode_texts
from prometheus.server import _get_predictor

tariff = {it["id"]: it for it in json.loads((DATA / "tariff_items.json").read_text())}
plain = {p["id"]: p["plain"] for p in json.loads((DATA / "plain_texts.json").read_text())}

predictor = _get_predictor(MODEL)
tax = predictor.taxonomy
leaf = tax.get_level(2).classes

ids = sorted(plain)
embs = encode_texts([plain[i] for i in ids], tax.embedding_model, show_progress=False)
random.seed(11)
items = []
for i, emb in zip(ids, embs):
    true_label = tariff[i]["_true"]
    true_idx = leaf.index(true_label)
    r = predictor.predict(emb, use_beam_search=True, beam_width=3)
    cands = sorted(leaves_under_headings(tax, {p[1] for p in beam_paths(predictor, emb, 3)}))
    random.shuffle(cands)
    items.append(
        {
            "id": i,
            "text": plain[i],
            "candidates": [{"n": j + 1, "label": leaf[c]} for j, c in enumerate(cands)],
            "_true": true_label,
            "_true_n": next((j + 1 for j, c in enumerate(cands) if c == true_idx), 0),
            "_cascade": r.path[-1],
            "_decision": r.routing["decision"],
        }
    )

(DATA / "plain_items.json").write_text(json.dumps(items, indent=1))
sizes = np.array([len(it["candidates"]) for it in items])
print("routing:", dict(Counter(it["_decision"] for it in items)))
print(
    f"cascade top-1 correct: {sum(it['_cascade'] == it['_true'] for it in items)}/100   "
    f"answer in candidate set: {sum(it['_true_n'] > 0 for it in items)}/100   "
    f"size median {np.median(sizes):.0f}, max {sizes.max()}"
)
