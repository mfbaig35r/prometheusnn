"""Score one judge run against its answer key.

  python3 experiments/llm_coupling/scripts/score.py runs/sonnet-5/plain data/plain_items.json

Paths are relative to experiments/llm_coupling/. A pick of 0 means "none of these" and
counts as correct when the right subheading was not among the candidates.
"""

import json
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
run_dir, key_file = EXP / sys.argv[1], EXP / sys.argv[2]
key = {it["id"]: it for it in json.loads(key_file.read_text())}
picks = {
    a["id"]: a for f in sorted(run_dir.glob("answers_*.json")) for a in json.loads(f.read_text())
}


def right(i: int) -> bool:
    return picks[i]["pick"] == key[i]["_true_n"]


def cascade_right(i: int) -> bool:
    return key[i]["_cascade"] == key[i]["_true"]


offered = [i for i in picks if key[i]["_true_n"] > 0]
none_said = [i for i in picks if picks[i]["pick"] == 0]

print(f"items scored:               {len(picks)}/{len(key)}")
print(f"cascade top-1 alone:        {sum(map(cascade_right, picks))}")
print(f"cascade + LLM pick:         {sum(map(right, picks))}")
print(f"  when answer was offered:  {sum(map(right, offered))}/{len(offered)}")
print(f"  'none of these':          {len(none_said)} (correct: {sum(map(right, none_said))})")
conf = Counter(picks[i].get("confidence") for i in picks)
conf_right = Counter(picks[i].get("confidence") for i in picks if right(i))
print(
    "  by stated confidence:     "
    + ", ".join(f"{c} {conf_right[c]}/{conf[c]}" for c in ("high", "medium", "low") if conf[c])
)
print(
    f"  fixed a wrong cascade answer: {sum(right(i) and not cascade_right(i) for i in picks)}; "
    f"overturned a right one: {sum(cascade_right(i) and not right(i) for i in picks)}"
)
if "_decision" in next(iter(key.values())):
    for dec in ("AUTO_ACCEPT", "HUMAN_REVIEW", "OUT_OF_TAXONOMY"):
        ids = [i for i in picks if key[i]["_decision"] == dec]
        if ids:
            print(
                f"  routed {dec:<16} {sum(map(right, ids))}/{len(ids)} (cascade {sum(map(cascade_right, ids))})"
            )
