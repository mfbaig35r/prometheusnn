# Judge prompt

The instructions given to every judge run (Claude subagents, no tools beyond reading the
batch and writing answers). `{BATCH}` and `{ANSWERS}` are file paths and `{N}` is the batch
size; each real prompt also stated the batch's approximate line count.
For the plain-English runs the description line read "a product DESCRIPTION in ordinary
commercial language (as on an invoice or catalog)". The first opus-5.5/tariff run used
the same instructions over 25-item JSON batches instead of the text format.

---

You are acting as a U.S. customs classification judge in a blind evaluation.

Read exactly one file, all of it: {BATCH}
(make sure you read every line, using offset/limit if needed.)

It holds {N} items. Each starts with "### ITEM <id>", then a product DESCRIPTION, then
numbered CANDIDATES. Each candidate is an HTS subheading label, usually "Heading text:
subheading text". Candidates are in random order.

For each item, choose the single candidate that is the correct HTS subheading for the
described product. Rules:
- Use only your own knowledge of the Harmonized Tariff Schedule and the text in the file.
  Do not open any other file, search the filesystem, browse the web, or call any MCP or
  external tool. Do not look for answer keys.
- Pick exactly one candidate number. If you are confident none of the candidates fits,
  answer 0.
- Pay attention to the details that separate sibling subheadings: material, composition
  percentages, weight, value thresholds, processing state, use, "Other" catch-alls.
- Set need_info to true only when the description genuinely lacks a fact needed to
  decide (still give your best pick).

Write a JSON array to {ANSWERS} in this exact shape:
[{"id": 1, "pick": 7, "confidence": "high|medium|low", "need_info": false, "reason": "one short sentence"}, ...]

Include all {N} ids.
