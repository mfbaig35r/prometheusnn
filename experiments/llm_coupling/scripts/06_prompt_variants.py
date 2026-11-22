"""Step 6: prompt variants for JEV, tuned on the dev set, then scored once on the 100 items.

Variants (all pick from the cascade's candidate set C, with a way to say "none"):
  base        one flat Choice: "Heading: subheading" labels, options named 1..n (step 4's setup)
  twostep     Choice of heading first, then Choice of subheading among that heading's
              candidates (top two headings, scored p(heading) x p(subheading))
  structured  one flat Choice, options named by HTS code, each described as
              {"heading": ..., "subheading": ...}
  *_notes     the same, with the classification rule (GRI 1) and the legal chapter notes
              (from hts-api, data/hts_chapter_notes.json) for the candidates' chapters,
              in the question instructions
  structured_notes_state  structured, with the notes in the state ({product, legal_notes})
              instead of the instructions

Error analysis on step 4 motivated these: 32 of 38 JEV misses picked a subheading under
the wrong heading, which the flat "Heading: subheading" list blurs.

  TYPESAFE_API_KEY=... uv run python experiments/llm_coupling/scripts/06_prompt_variants.py \
      --sets dev --variants base,twostep,structured
  ... --analyze-only to re-score saved runs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import httpx
import numpy as np
from common import DATA, RUNS

JEV_MODEL = "jev-1.14.0"
URL = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_M_INPUT = 0.042  # USD, docs.typesafe.ai/models
CONCURRENCY = 6
OUT = RUNS / "jev-1.14-prompts"
ASK_SUB = "Which U.S. HTS subheading is the correct classification for this product?"
ASK_HEAD = "Which U.S. HTS heading covers this product?"
GRI = (
    "Classify under the General Rules of Interpretation: the terms of the headings and the "
    "legal section and chapter notes decide the classification; they take precedence over "
    "descriptions. Relevant legal notes:\n"
)
NOTE_CAP = 6000  # characters per chapter, to stay well inside the 32k-token question limit

INFO = json.loads((DATA / "taxonomy_info.json").read_text())
NOTES_PATH = DATA / "hts_chapter_notes.json"
NOTES = json.loads(NOTES_PATH.read_text()) if NOTES_PATH.exists() else {}


def tail(label: str) -> str:
    """The subheading's own text, without the heading prefix every sibling repeats."""
    head = INFO[label]["heading"]
    return label[len(head) + 2 :] if label.startswith(head + ": ") else label


def notes_for(labels: list[str]) -> str:
    chapters = sorted({INFO[label]["chapter"] for label in labels})
    blocks = []
    for ch in chapters:
        text = NOTES.get(f"{ch:02d}", "")
        if text:
            blocks.append(f"Chapter {ch:02d}:\n{text[:NOTE_CAP]}")
    return "\n\n".join(blocks)


def ask(base: str, labels: list[str], notes: bool) -> str:
    return f"{base}\n\n{GRI}{notes_for(labels)}" if notes else base


async def call(client, sem, state, questions):
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


def usage(*responses) -> dict:
    return {
        "tokens": sum(r["usage"]["input_tokens"] for r in responses),
        "seconds": sum(r["_seconds"] for r in responses),
        "requests": len(responses),
    }


async def flat(client, sem, item, notes: bool, structured: bool, notes_in_state=False) -> dict:
    labels = [c["label"] for c in item["candidates"]]
    if structured:
        names, seen = [], set()
        for label in labels:
            name = INFO[label]["code"]
            while name in seen:  # some subheading labels share a 6-digit code
                name += "'"
            seen.add(name)
            names.append(name)
        criteria = {
            name: {"heading": INFO[label]["heading"], "subheading": tail(label)}
            for name, label in zip(names, labels)
        }
        none = "none"
    else:
        names = [str(c["n"]) for c in item["candidates"]]
        criteria = dict(zip(names, labels))
        none = "0"
    criteria[none] = "None of these subheadings correctly classifies this product"
    if notes_in_state:
        state = {"product": item["text"], "legal_notes": notes_for(labels)}
        q = {"type": "choice", "instructions": f"{ASK_SUB}\n\n{GRI.split(' Relevant')[0]}"}
    else:
        state = item["text"]
        q = {"type": "choice", "instructions": ask(ASK_SUB, labels, notes)}
    q["criteria"] = criteria
    resp = await call(client, sem, state, {"q": q})
    a = resp["answers"]["q"]
    pick = 0 if a["choice"] == none else item["candidates"][names.index(a["choice"])]["n"]
    return {"id": item["id"], "pick": pick, "confidence": a["confidence"], **usage(resp)}


async def twostep(client, sem, item, notes: bool) -> dict:
    by_head: dict[str, list[dict]] = {}
    for c in item["candidates"]:
        by_head.setdefault(INFO[c["label"]]["heading"], []).append(c)
    heads = list(by_head)
    all_labels = [c["label"] for c in item["candidates"]]
    criteria = {str(j + 1): h for j, h in enumerate(heads)}
    criteria["0"] = "None of these headings covers this product"
    q = {"type": "choice", "instructions": ask(ASK_HEAD, all_labels, notes), "criteria": criteria}
    first = await call(client, sem, item["text"], {"q": q})
    probs = first["answers"]["q"]["probabilities"]
    ranked = sorted(((p, k) for k, p in probs.items() if k != "0"), reverse=True)
    if first["answers"]["q"]["choice"] == "0":
        return {"id": item["id"], "pick": 0, "confidence": probs["0"], **usage(first)}
    top = [(p, k) for p, k in ranked[:2] if p >= 0.05] or ranked[:1]

    async def subheading(k):
        kids = by_head[heads[int(k) - 1]]
        if len(kids) == 1:
            return kids, None, {"1": 1.0}
        crit = {str(j + 1): tail(c["label"]) for j, c in enumerate(kids)}
        labels = [c["label"] for c in kids]
        qs = {"type": "choice", "instructions": ask(ASK_SUB, labels, notes), "criteria": crit}
        resp = await call(client, sem, item["text"], {"q": qs})
        return kids, resp, resp["answers"]["q"]["probabilities"]

    seconds = await asyncio.gather(*(subheading(k) for _, k in top))
    best = (-1.0, None)
    for (p_head, _), (kids, _, sub_probs) in zip(top, seconds):
        for name, p in sub_probs.items():
            score = p_head * p
            if score > best[0]:
                best = (score, kids[int(name) - 1]["n"])
    responses = [first] + [r for _, r, _ in seconds if r is not None]
    return {"id": item["id"], "pick": best[1], "confidence": best[0], **usage(*responses)}


VARIANTS = {
    "base": lambda c, s, it: flat(c, s, it, notes=False, structured=False),
    "structured": lambda c, s, it: flat(c, s, it, notes=False, structured=True),
    "twostep": lambda c, s, it: twostep(c, s, it, notes=False),
    "base_notes": lambda c, s, it: flat(c, s, it, notes=True, structured=False),
    "structured_notes": lambda c, s, it: flat(c, s, it, notes=True, structured=True),
    "twostep_notes": lambda c, s, it: twostep(c, s, it, notes=True),
    "structured_notes_state": lambda c, s, it: flat(
        c, s, it, notes=False, structured=True, notes_in_state=True
    ),
}


async def run(variant: str, set_name: str, items: list[dict]):
    sem = asyncio.Semaphore(CONCURRENCY)
    headers = {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"}
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        results = await asyncio.gather(*(VARIANTS[variant](client, sem, it) for it in items))
    out = OUT / variant / set_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers_1.json").write_text(json.dumps(results, indent=1))


def score(variant: str, set_name: str, items: list[dict]) -> dict | None:
    path = OUT / variant / set_name / "answers_1.json"
    if not path.exists():
        return None
    key = {it["id"]: it for it in items}
    res = json.loads(path.read_text())
    right = np.array([r["pick"] == key[r["id"]]["_true_n"] for r in res])
    conf = np.array([r["confidence"] for r in res])
    offered = np.array([key[r["id"]]["_true_n"] > 0 for r in res])
    said_none = np.array([r["pick"] == 0 for r in res])
    # Largest share of items auto-acceptable at >= 95% precision (with at least 20 accepted).
    best_cov = 0.0
    for t in sorted(set(conf)):
        sel = conf >= t
        if sel.sum() >= 20 and right[sel].mean() >= 0.95:
            best_cov = max(best_cov, sel.mean())
    tokens = sum(r["tokens"] for r in res)
    return {
        "variant": variant,
        "right": int(right.sum()),
        "n": len(res),
        "offered_right": f"{int(right[offered].sum())}/{int(offered.sum())}",
        "none": f"{int((said_none & ~offered).sum())}/{int(said_none.sum())}",
        "prec@0.7": f"{right[conf >= 0.7].mean():.0%} of {int((conf >= 0.7).sum())}",
        "cov@95%": f"{best_cov:.0%}",
        "usd": round(tokens / 1e6 * PRICE_PER_M_INPUT, 4),
        "ms_per_item": round(1000 * sum(r["seconds"] for r in res) / len(res)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", default="dev")
    p.add_argument("--variants", default="base,twostep,structured")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    for set_name in args.sets.split(","):
        items = json.loads((DATA / f"{set_name}_items.json").read_text())
        rows = []
        for v in args.variants.split(","):
            if "_notes" in v and not NOTES:
                raise SystemExit("notes variants need data/hts_chapter_notes.json")
            if not args.analyze_only:
                asyncio.run(run(v, set_name, items))
            rows.append(score(v, set_name, items))
        print(f"\n=== {set_name} ({len(items)} items) ===")
        cols = [
            "variant",
            "right",
            "offered_right",
            "none",
            "prec@0.7",
            "cov@95%",
            "usd",
            "ms_per_item",
        ]
        print("  ".join(f"{c:<16}" for c in cols))
        for r in filter(None, rows):
            print("  ".join(f"{str(r[c]):<16}" for c in cols))


if __name__ == "__main__":
    main()
