"""Step 4: TypeSafe JEV as the judge, on the same 100-item sets as the LLM judges.

Per item and wording (tariff / plain), against the cascade's candidate set C:
  pick_none  Choice over the candidates plus a "none of these" option (comparable to the
             LLM judges, who could answer 0)
  pick       Choice over the candidates only, in three orders (order sensitivity)
  any        Noul: does any candidate fit? (TypeSafe's suggested alternative to a "none" option)
And JEV alone, with no cascade: a beam search (K=3) down the taxonomy, chapter -> heading ->
subheading, one Choice per node, following TypeSafe's hierarchical classification cookbook.

Raw responses go to runs/jev-1.14/<set>/raw.jsonl, so --analyze-only re-scores without calls.
answers_1.json (from pick_none) has the same shape as the other judges, for score.py.

  TYPESAFE_API_KEY=... uv run python experiments/llm_coupling/scripts/04_jev_benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time

import httpx
import numpy as np
from common import DATA, MODEL, RUNS

from prometheus.server import _get_predictor

JEV_MODEL = "jev-1.14.0"
URL = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_M_INPUT = 0.042  # USD, docs.typesafe.ai/models; output tokens are free
CONCURRENCY = 6
BEAM = 3
NONE = "0"
ASK = {
    0: "Which U.S. HTS chapter covers this product?",
    1: "Which U.S. HTS heading covers this product?",
    2: "Which U.S. HTS subheading is the correct classification for this product?",
}


async def call(client: httpx.AsyncClient, sem: asyncio.Semaphore, state: str, questions: dict):
    body = {"model": JEV_MODEL, "state": state, "questions": questions}
    async with sem:
        for attempt in range(6):
            t0 = time.perf_counter()
            r = await client.post(URL, json=body)
            if r.status_code == 200:
                data = r.json()
                data["_seconds"] = time.perf_counter() - t0
                return data
            if r.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(2**attempt)
                continue
            raise RuntimeError(f"JEV HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError("JEV: retries exhausted")


def choice(labels: list[str], ask: str, none: bool = False) -> dict:
    """Options named 1..n in presentation order; NONE adds a 'none of these' option."""
    criteria = {str(j + 1): label for j, label in enumerate(labels)}
    if none:
        criteria[NONE] = "None of these subheadings correctly classifies this product"
    return {"type": "choice", "instructions": ask, "criteria": criteria}


async def judge_candidates(client, sem, item: dict, rng: random.Random) -> dict:
    labels = [c["label"] for c in item["candidates"]]
    n = len(labels)
    orders = [list(range(n))] + [rng.sample(range(n), n) for _ in range(2)]
    first = await call(
        client,
        sem,
        item["text"],
        {
            "pick_none": choice(labels, ASK[2], none=True),
            "pick": choice(labels, ASK[2]),
            "any": {
                "type": "noul",
                "instructions": "Does any one of these U.S. HTS subheadings correctly classify "
                "this product? " + " | ".join(labels),
            },
        },
    )
    shuffled = await asyncio.gather(
        *(
            call(client, sem, item["text"], {"pick": choice([labels[j] for j in o], ASK[2])})
            for o in orders[1:]
        )
    )
    return {"id": item["id"], "orders": orders, "responses": [first, *shuffled]}


async def walk_taxonomy(client, sem, item: dict, tax) -> dict:
    """JEV alone: beam search over Choice probabilities, root to leaf."""
    pcm = tax.parent_child_map
    levels = [tax.get_level(d).classes for d in range(3)]
    beams = [((), 1.0)]
    calls = []
    for depth in range(3):
        expanded = []
        children_of = [
            list(range(len(levels[0]))) if depth == 0 else pcm.get((depth, path[-1]), [])
            for path, _ in beams
        ]
        todo = [(b, kids) for b, kids in zip(beams, children_of) if len(kids) > 1]
        responses = await asyncio.gather(
            *(
                call(
                    client,
                    sem,
                    item["text"],
                    {"q": choice([levels[depth][k] for k in kids], ASK[depth])},
                )
                for _, kids in todo
            )
        )
        calls += responses
        for ((path, score), kids), resp in zip(todo, responses):
            probs = resp["answers"]["q"]["probabilities"]
            for j, k in enumerate(kids):
                expanded.append((path + (k,), score * probs.get(str(j + 1), 0.0)))
        for (path, score), kids in zip(beams, children_of):
            if len(kids) == 1:
                expanded.append((path + (kids[0],), score))
        expanded.sort(key=lambda x: -x[1])
        beams = expanded[:BEAM]
    best = beams[0]
    return {
        "id": item["id"],
        "path": list(best[0]),
        "path_prob": best[1],
        "beams": [[list(p), s] for p, s in beams],
        "calls": [
            {
                "usage": c["usage"],
                "seconds": c["_seconds"],
                "confidence": c["answers"]["q"]["confidence"],
            }
            for c in calls
        ],
    }


async def run(set_name: str, items: list[dict], tax, out_dir):
    key = os.environ["TYPESAFE_API_KEY"]
    sem = asyncio.Semaphore(CONCURRENCY)
    rng = random.Random(23)
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=120) as client:
        cand = await asyncio.gather(*(judge_candidates(client, sem, it, rng) for it in items))
        walk = await asyncio.gather(*(walk_taxonomy(client, sem, it, tax) for it in items))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "raw.jsonl").open("w") as f:
        for c, w in zip(cand, walk):
            f.write(json.dumps({"candidates": c, "walk": w}) + "\n")
    print(f"{set_name}: {len(items)} items judged")


def analyze(set_name: str, items: list[dict], tax, out_dir):
    leaf = tax.get_level(2).classes
    rows = [json.loads(line) for line in (out_dir / "raw.jsonl").read_text().splitlines()]
    key = {it["id"]: it for it in items}
    answers, stats = [], []
    tokens = seconds = requests = 0
    for row in rows:
        c, w = row["candidates"], row["walk"]
        it = key[c["id"]]
        true_n = it["_true_n"]
        first = c["responses"][0]["answers"]
        pn = first["pick_none"]
        pick_none = int(pn["choice"]) if pn["choice"] != NONE else 0
        answers.append(
            {
                "id": it["id"],
                "pick": pick_none,
                "confidence": pn["confidence"],
                "need_info": False,
                "reason": "",
            }
        )
        # Map every ordering's pick back to the original candidate number.
        picks = []
        for order, resp in zip(c["orders"], c["responses"]):
            pos = int(resp["answers"]["pick"]["choice"]) - 1
            picks.append(order[pos] + 1)
        avg = np.zeros(len(it["candidates"]))
        for order, resp in zip(c["orders"], c["responses"]):
            for name, p in resp["answers"]["pick"]["probabilities"].items():
                avg[order[int(name) - 1]] += p / len(c["orders"])
        for resp in c["responses"]:
            tokens += resp["usage"]["input_tokens"]
            seconds += resp["_seconds"]
            requests += 1
        for call in w["calls"]:
            tokens += call["usage"]["input_tokens"]
            seconds += call["seconds"]
            requests += 1
        walk_leaf = leaf[w["path"][-1]] if len(w["path"]) == 3 else None
        stats.append(
            {
                "true_n": true_n,
                "pick_none": pick_none,
                "conf_none": pn["confidence"],
                "picks": picks,
                "conf": first["pick"]["confidence"],
                "avg_pick": int(avg.argmax()) + 1,
                "any": first["any"]["noul"],
                "walk_right": walk_leaf == it["_true"],
                "walk_prob": w["path_prob"],
                "walk_calls": len(w["calls"]),
                "cascade_right": it["_cascade"] == it["_true"],
            }
        )
    (out_dir / "answers_1.json").write_text(json.dumps(answers, indent=1))

    s = stats
    n = len(s)
    offered = [x for x in s if x["true_n"] > 0]
    absent = [x for x in s if x["true_n"] == 0]
    right_none = [x["pick_none"] == x["true_n"] for x in s]
    print(f"\n=== {set_name} ({n} items; answer offered in {len(offered)}) ===")
    print(f"cascade alone                       {sum(x['cascade_right'] for x in s)}/{n}")
    print(
        f"JEV pick, 'none' option             {sum(right_none)}/{n}   "
        f"(offered {sum(x['pick_none'] == x['true_n'] for x in offered)}/{len(offered)}, "
        f"'none' said {sum(x['pick_none'] == 0 for x in s)}, right {sum(x['pick_none'] == 0 == x['true_n'] for x in s)})"
    )
    for k in range(3):
        print(
            f"JEV pick, no 'none', order {k + 1}        "
            f"{sum(x['picks'][k] == x['true_n'] for x in offered)}/{len(offered)} offered"
        )
    print(
        f"JEV pick, probabilities averaged     {sum(x['avg_pick'] == x['true_n'] for x in offered)}/{len(offered)} offered"
    )
    same = sum(len(set(x["picks"])) == 1 for x in s)
    fail_sets = [
        {i for i, x in enumerate(offered) if x["picks"][k] != x["true_n"]} for k in range(3)
    ]
    union = set().union(*fail_sets)
    print(
        f"order stability: same pick in all 3 orders {same}/{n}; failures in order 1 "
        f"{len(fail_sets[0])}, in any order {len(union)} (+{len(union) - len(fail_sets[0])})"
    )
    for t in (0.5, 0.7, 0.9):
        abst = [x for x in s if x["any"] < t]
        print(
            f"Noul 'any fits' < {t}: abstains on {len(abst)} "
            f"(truly absent among them {sum(x['true_n'] == 0 for x in abst)} of {len(absent)} absent)"
        )
    print("selective accuracy (pick_none, by confidence):")
    for t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
        sel = [ok for ok, x in zip(right_none, s) if x["conf_none"] >= t]
        if sel:
            print(f"   conf >= {t:<4}  covers {len(sel):>3}/{n}  right {sum(sel) / len(sel):.1%}")
    print(
        f"JEV alone, taxonomy walk (beam {BEAM})  {sum(x['walk_right'] for x in s)}/{n}   "
        f"(avg {np.mean([x['walk_calls'] for x in s]):.1f} calls/item)"
    )
    print(
        f"usage: {requests} requests, {tokens:,} input tokens, ${tokens / 1e6 * PRICE_PER_M_INPUT:.4f}, "
        f"mean {seconds / requests * 1000:.0f} ms/request"
    )


def main():
    tax = _get_predictor(MODEL).taxonomy
    analyze_only = "--analyze-only" in sys.argv
    for set_name in ("tariff", "plain"):
        items = json.loads((DATA / f"{set_name}_items.json").read_text())
        out_dir = RUNS / "jev-1.14" / set_name
        if not analyze_only:
            asyncio.run(run(set_name, items, tax, out_dir))
        analyze(set_name, items, tax, out_dir)


if __name__ == "__main__":
    main()
