"""Write blind judge batches: answer-key fields removed, one candidate per line.

  python3 experiments/llm_coupling/scripts/to_text.py data/tariff_items.json <out_dir> [per_batch]

Paths are relative to experiments/llm_coupling/. Keep the output directory away from
data/ so a judge pointed at it never sits next to an answer key.
"""

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
src, outdir = EXP / sys.argv[1], Path(sys.argv[2])
per = int(sys.argv[3]) if len(sys.argv) > 3 else 50

items = json.loads(src.read_text())
outdir.mkdir(parents=True, exist_ok=True)
for k in range(0, len(items), per):
    lines = []
    for it in items[k : k + per]:
        lines += [f"### ITEM {it['id']}", f"DESCRIPTION: {it['text']}", "CANDIDATES:"]
        lines += [f"{c['n']}. {c['label']}" for c in it["candidates"]]
        lines.append("")
    (outdir / f"batch_{k // per + 1}.txt").write_text("\n".join(lines))
print(f"wrote {len(range(0, len(items), per))} batches to {outdir}")
