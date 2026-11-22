# JEV request formats

Both send one Choice question to `POST https://api.typesafe.ai/v1/systemone` with
`"model": "jev-1.14.0"`, the product description as the `state`, and the candidate set C
(every subheading under the cascade's beam headings), plus a "none" option.

## Plain labels (04_jev_benchmark.py, `pick_none`)

```json
{
  "type": "choice",
  "instructions": "Which U.S. HTS subheading is the correct classification for this product?",
  "criteria": {
    "1": "<subheading label>",
    "2": "<subheading label>",
    "0": "None of these subheadings correctly classifies this product"
  }
}
```

## Structured options (06_prompt_variants.py, `structured`): the chosen format

```json
{
  "type": "choice",
  "instructions": "Which U.S. HTS subheading is the correct classification for this product?",
  "criteria": {
    "6104.63": {"heading": "<heading text>", "subheading": "<subheading text>"},
    "6104.69": {"heading": "<heading text>", "subheading": "<subheading text>"},
    "none": "None of these subheadings correctly classifies this product"
  }
}
```

Option names are 6-digit HTS codes (a `'` is appended when two candidates share one). The
subheading text drops the heading prefix where the label repeats it.
