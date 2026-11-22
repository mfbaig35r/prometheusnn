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
| + Claude Sonnet 5 | 86 | 87 |
| + Claude Opus 5.5 | 98 | 95 |

- **Model tier dominates; wording barely matters.** Rewriting costs 4 points of candidate
  recall; Haiku loses 4 points, Opus 3, Sonnet none (86 to 87).
- **Opus 5.5:** never wrong when it answered "none of these" (9/9, 13/13), overturned a correct
  cascade answer once in 200 items, and its "high" confidence was right 128 of 129 times.
- **Haiku 4.5:** overturned 7 correct cascade answers per run; "high" confidence was right only
  75 to 83% of the time, and agreeing with the cascade did not make it more reliable (72%).
- **Sonnet 5:** in between, 86 / 87. "None of these" was right 19 of 20 times, it overturned
  8 correct cascade answers over 200 items, and "high" confidence was right 114 of 121 times
  (94%). On plain English it classified 28 of the 35 items routed to OUT_OF_TAXONOMY.
- **The novelty gate discards classifiable items.** On plain English the cascade routed 35 items
  to OUT_OF_TAXONOMY; Opus classified 32 of them correctly.
- **AUTO_ACCEPT is miscalibrated at the leaf:** 62.8% correct on the 86 held-out items it
  auto-accepted.

Both were fixed in PrometheusNN 0.2.3 (kNN novelty; thresholds fit to a target precision).

### 4. TypeSafe JEV as the judge

[JEV](https://docs.typesafe.ai/models) (`jev-1.14.0`) is a non-generative model that returns a
probability for each option of a declared set. Same 100 items and candidate sets as section 3,
through the API (`POST /v1/systemone`, one Choice question with a "none" option).

| Correct out of 100 | Tariff wording | Plain English |
|---|---|---|
| + JEV, plain labels (`04_jev_benchmark.py`) | 78 | 76 |
| + JEV, structured options (`06_prompt_variants.py`) | 81 | 82 |
| **JEV structured if confidence >= 0.7, else Sonnet 5** | **88** (36 items to Sonnet) | **91** (35 to Sonnet) |

- **Between Haiku and Sonnet alone; better than Sonnet as a first pass.** Sending only JEV's
  low-confidence items to Sonnet beat Sonnet alone (86 / 87) while Sonnet saw about a third of
  the items. The 0.7 cutoff was not tuned on these items; 0.8 gives the same totals.
- **The prompt fix: structured options.** About 70% of candidate labels are bare subheading
  text ("Other", "Of synthetic fibers") with no heading. Naming each option by its HTS code and
  describing it as `{"heading", "subheading"}` took the 200-item dev set from 150 to 163 right,
  and the share auto-acceptable at 95% precision from 14% to 70%.
- **JEV's misses are mostly sibling subheadings** (16 of 20 tariff, 13 of 18 plain), not wrong
  headings. Two-step prompting (heading, then subheading) did not help.
- **Legal chapter notes** (from hts-api, in the instructions or in the state) gave no
  consistent gain over structured options: +4 tariff, -2 plain, at twice the tokens.
- **JEV alone, no cascade** (beam search down the taxonomy, about 7 calls per item) scored 79 /
  73, about the same as picking from the cascade's candidates.
- **Order sensitivity:** the same pick under three option orders on 84-85% of items; averaging
  probabilities across orders did not help.
- **Cost and speed:** about 100 ms per request and about $0.0001 per item (structured);
  everything in this section used about 8.3M input tokens (about $0.35).

Prompt variants were chosen on a separate dev set (200 review items from the same held-out
split, none of them among the 100), then scored once on the 100; see
`06_prompt_variants.py` for all seven variants and their dev results.

### Caveats

- 100 items, one run per setting: read the numbers as direction, not as precise rates.
- The "plain English" rewrites are only moderately plain; technical goods stay technical and 11
  of 100 are near-copies of the original.
- The judges likely know the public tariff text, which flatters them. No real invoice data yet.
- Claude judges ran as Claude Code subagents, not through the API; their cost was not
  measured. The first Opus tariff run used 25-item JSON batches, every other run the 50-item
  text format in `scripts/to_text.py`.
- JEV's numbers are one run per setting; TypeSafe does not publish the architecture, and
  JEV is hosted only (product text leaves the machine).

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
prompts/jev.md                  the JEV request formats (plain labels, structured)
scripts/04_jev_benchmark.py     JEV on the 100 items: plain labels, option orders, "any fits"
                                question, and JEV alone down the taxonomy
scripts/05_build_dev.py         data/dev_items.json (200 dev items) and data/taxonomy_info.json
scripts/06_prompt_variants.py   seven JEV prompt variants (dev), and the chosen ones on the 100
data/hts_chapter_notes.json     legal chapter notes (no statistical notes) for 91 chapters,
                                from hts-api get_chapter_notes
runs/<model>/<tariff|plain>/    each judge's raw answers
runs/jev-1.14/, runs/jev-1.14-prompts/<variant>/<set>/   JEV answers (and raw responses)
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

# JEV (needs a TypeSafe key)
TYPESAFE_API_KEY=... uv run python experiments/llm_coupling/scripts/04_jev_benchmark.py
uv run python experiments/llm_coupling/scripts/05_build_dev.py
TYPESAFE_API_KEY=... uv run python experiments/llm_coupling/scripts/06_prompt_variants.py \
  --sets dev --variants base,structured,twostep
```

Training is not seeded, so a retrained model will differ slightly from the committed results.
Steps 1 to 3 reproduced this run's numbers and item sets exactly against the original model.
They depend on its 0.2.2 routing: `hts_viz` was later refit to 0.2.3 routing for the
Observatory, which changes which items are HUMAN_REVIEW, so rerunning 01 and 02 against the
refit model selects different items. The committed item sets are the ones that were judged.
