# Paraphrase prompt

Produced `data/plain_texts.json`. The model saw only `{id, text}` for the 100 items:
no labels, no candidates.

---

It holds 100 product descriptions written in U.S. tariff-schedule language (abbreviations
like "nesoi", "o/than", "w/", "n/o", legal phrasing). Rewrite each one the way it would
appear in ordinary commerce: an importer's invoice line, a purchase order, a product
catalog or an online listing.

Rules:
- Keep every fact that could matter for classification: material and its percentage,
  weight, dimensions, value, processing state (fresh, frozen, cooked, tanned...), intended
  use, who it is for (men's, women's), and so on. Translate facts into plain words; do
  not drop them.
- Do not add facts that are not in the original.
- Remove tariff jargon and legal phrasing. Do not copy the original's long phrases word
  for word; say it the way a buyer or seller would.
- Vary the style across items (some short invoice lines, some catalog-style sentences),
  roughly 5 to 25 words each.
- Use only the text in the file. Do not open any other file, search the filesystem,
  browse the web, or call any external or MCP tool.

Write a JSON array: [{"id": 1, "plain": "..."}, ...]
