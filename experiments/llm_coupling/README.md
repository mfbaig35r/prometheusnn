# Experiment: coupling the cascade with an LLM judge

**Question.** PrometheusNN's cascade is fast, calibrated and cheap, but at the leaf level of a
large taxonomy it is often wrong. LLMs know a great deal but are slow and expensive per call.
Can the cascade do the cheap structured work and hand an LLM only the hard part, the way DLSS
renders cheaply and lets a network resolve only what the renderer cannot?

**Setup.** The `hts_viz` model: the U.S. Harmonized Tariff Schedule sample in
`data/hts_training.csv` (11,414 rows; 96 chapters, 1,228 headings, 4,724 subheadings), trained
with default settings. Everything below runs on its held-out test split (1,141 rows, rebuilt
exactly with `prepare_data(random_state=42)`).

## Results

### 1. Picking from the beam's own candidates does not work

| HUMAN_REVIEW items (720) | |
|---|---|
| Cascade top-1 leaf correct | 25.6% |
| True leaf among the beam's surviving paths (width 3 / 5 / 10) | 44.0% / 48.8% / 51.0% |
| Held-out items whose true subheading has **no training rows** (all 1,141) | 27.6% |

Over a quarter of held-out items belong to subheadings the model never saw, so no beam can
offer them.

### 2. Let the taxonomy supply the candidates

| Candidate set for review items | Contains the answer | Size (median / p90 / max) |
|---|---|---|
| A: the beam's leaves | 44.0% | 6 / 6 / 6 |
| B: every subheading under the confident prefix | 98.2% | 169 / 4,724 / 4,724 |
| **C: every subheading under the beam's headings** | **83.2%** | **20 / 48 / 136** |

Set C uses the cascade to narrow to a few headings and the taxonomy to list every subheading
under them, including ones with no training data. That is the design tested below.

### 3. LLM judges on 100 review items

100 HUMAN_REVIEW items sampled from the test split (`random.seed(7)`), candidates shuffled, the
judge blind to the answer and to the cascade's pick. "Plain English" is the same 100 items
rewritten by an LLM into invoice or catalog language and run through the whole pipeline again.
A pick of "none of these" counts as correct when the answer was not offered.

| Correct out of 100 | Tariff wording | Plain English |
|---|---|---|
| Cascade top-1 alone | 25 | 23 |
| Answer is in candidate set C (ceiling) | 89 | 85 |
| + Claude Haiku 4.5 | 72 | 68 |
| + Claude Sonnet 5 | 86 | pending |
| + Claude Opus 5.5 | 98 | 95 |

- **Model tier dominates; wording barely matters.** Rewriting costs 4 points of candidate
  recall and 3 to 4 points per judge.
- **Opus 5.5:** never wrong when it answered "none of these" (9/9, 13/13), overturned a correct
  cascade answer once in 200 items, and its "high" confidence was right 128 of 129 times.
- **Haiku 4.5:** overturned 7 correct cascade answers per run; "high" confidence was right only
  75 to 83% of the time, and agreeing with the cascade did not make it more reliable (72%).
- **Sonnet 5:** in between on accuracy (its plain-English run is incomplete: only items 51 to
  100 are judged so far, in `runs/sonnet-5/plain/answers_2.json`); "high" confidence was right 56 of 59 times on the
  tariff wording.
- **The novelty gate discards classifiable items.** On plain English the cascade routed 35 items
  to OUT_OF_TAXONOMY; Opus classified 32 of them correctly.
- **AUTO_ACCEPT is miscalibrated at the leaf:** 62.8% correct on the 86 held-out items it
  auto-accepted.

### Caveats

- 100 items, one run per setting: read the numbers as direction, not as precise rates.
- The "plain English" rewrites are only moderately plain; technical goods stay technical and 11
  of 100 are near-copies of the original.
- The judges likely know the public tariff text, which flatters them. No real invoice data yet.
- Judges ran as Claude Code subagents, not through the API; cost was not measured. The first
  Opus tariff run used 25-item JSON batches, every other run the 50-item text format in
  `scripts/to_text.py`.

## Layout

```
scripts/common.py               paths, the held-out split, candidate-set helpers
scripts/01_candidate_recall.py  sections 1 and 2; writes data/candidates.json (gitignored)
scripts/02_sample_items.py      data/tariff_items.json: the 100 items + answer key
scripts/03_plain_pipeline.py    data/plain_items.json from data/plain_texts.json
scripts/to_text.py              blind judge batches (answer-key fields stripped)
scripts/score.py                score a run against its key
prompts/judge.md                judge instructions
prompts/paraphrase.md           instructions that produced data/plain_texts.json
runs/<model>/<tariff|plain>/    each judge's raw answers
```

Fields starting with `_` in `data/*_items.json` are the answer key. Never point a judge at
`data/`; generate batches into a separate folder with `to_text.py`.

## Reproduce

```sh
# Train the model (about 30 s). Scripts 01 to 03 assume the name hts_viz.
uv run python -c "
from prometheus.server.tools.training import train_model
train_model('data/hts_training.csv', 'text', ['Chapter', 'Heading', 'Subheading'], 'hts_viz')"

uv run python experiments/llm_coupling/scripts/01_candidate_recall.py
uv run python experiments/llm_coupling/scripts/02_sample_items.py
uv run python experiments/llm_coupling/scripts/03_plain_pipeline.py

# Judge: build blind batches outside data/, run prompts/judge.md on each, save answers_N.json
python3 experiments/llm_coupling/scripts/to_text.py data/plain_items.json /tmp/judge
python3 experiments/llm_coupling/scripts/score.py runs/opus-5.5/plain data/plain_items.json
```

Training is not seeded, so a retrained model will differ slightly from the committed results.
Steps 1 to 3 reproduced this run's numbers and item sets exactly against the original model.
