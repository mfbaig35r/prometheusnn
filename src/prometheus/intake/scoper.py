"""scope_problem logic — guidance for users without data yet."""

from __future__ import annotations

import re

from prometheus.intake.models import ProblemType, ScopeResult

_CLASSIFICATION_KEYWORDS = {
    "classify", "classification", "categorize", "categorise", "category",
    "label", "tag", "bucket", "group into", "assign to", "sort into",
    "type of", "kind of", "which", "map ", "mapping",
    "industry code", "product code", "tariff code",
}

_HIERARCHICAL_KEYWORDS = {
    "hierarchy", "hierarchical", "nested", "taxonomy",
    "subcategory", "sub-category", "subcategories",
    "nested levels", "levels of categories", "category levels",
    "tree of categories", "parent category", "child category",
}

_HIERARCHICAL_PATTERN = re.compile(r"\b\d+\s+(?:nested\s+)?(?:levels?\s+of|tiers?\s+of)")

_REGRESSION_KEYWORDS = {
    "regression", "forecast", "how much", "how many", "what price",
}

_REGRESSION_PATTERNS = [
    re.compile(r"predict\s+(?:the\s+)?(?:price|value|amount|number|cost|revenue|sales)"),
    re.compile(r"estimate\s+(?:the\s+)?(?:price|value|amount|number|cost|revenue)"),
]

_CLUSTERING_KEYWORDS = {
    "cluster", "group similar", "find patterns", "segment",
    "discover", "unsupervised", "without labels",
    "find natural groups", "find groups", "natural groups",
}

_EXTRACTION_KEYWORDS = {
    "extract", "pull out", "find entities", "named entity", "ner",
    "highlight", "annotate", "parse",
}

_NO_LABELS_KEYWORDS = {
    "no labels", "unlabeled", "unlabelled", "don't have labels",
    "haven't labeled", "haven't labelled", "no training data",
}

_COUNT_PATTERN = re.compile(r"(\d[\d,]*)\s*(?:categories|classes|labels|buckets|types|groups)")
_SAMPLE_COUNT_PATTERN = re.compile(r"(\d[\d,]*)\s*(?:examples|rows|samples|items|records)")


def scope_problem_from_text(
    goal_description: str,
    data_hint: str | None = None,
) -> ScopeResult:
    """Assess whether Prometheus fits a described goal."""
    text = goal_description.lower()
    hint = (data_hint or "").lower()
    combined = f"{text} {hint}".strip()

    if _matches_any(combined, _NO_LABELS_KEYWORDS):
        return ScopeResult(
            fits_prometheus=False,
            reasoning=(
                "Prometheus needs labeled training examples — text descriptions "
                "paired with the correct category. Without labels, it can't learn."
            ),
            redirect_suggestion=(
                "If you want to discover natural groupings without labels, try "
                "clustering tools (scikit-learn KMeans, HDBSCAN) instead. "
                "If you have a small set of examples you could hand-label, "
                "even a few hundred would be enough to come back here."
            ),
            next_step=(
                "Label 200-500 examples by hand, then return with a CSV and "
                "call start_here(file_path)."
            ),
        )

    if _matches_any(text, _REGRESSION_KEYWORDS) or any(p.search(text) for p in _REGRESSION_PATTERNS):
        return ScopeResult(
            fits_prometheus=False,
            reasoning=(
                "Sounds like a regression problem — predicting a numeric value "
                "rather than a category. Prometheus only handles classification "
                "(picking from a fixed set of categories)."
            ),
            redirect_suggestion=(
                "For numeric prediction, look at scikit-learn (LinearRegression, "
                "RandomForestRegressor) or XGBoost."
            ),
            next_step="If you can re-frame as 'low/medium/high' buckets, Prometheus would fit.",
        )

    if _matches_any(text, _CLUSTERING_KEYWORDS) and not _matches_any(text, _CLASSIFICATION_KEYWORDS):
        return ScopeResult(
            fits_prometheus=False,
            reasoning=(
                "Sounds like clustering — finding natural groups without "
                "predefined categories. Prometheus learns to predict known "
                "categories from labeled examples."
            ),
            redirect_suggestion=(
                "For clustering, try scikit-learn (KMeans, DBSCAN, HDBSCAN) "
                "or sentence-transformers + UMAP."
            ),
            next_step=(
                "If you have categories in mind and want to assign items to them, "
                "Prometheus does fit — come back with example data."
            ),
        )

    if _matches_any(text, _EXTRACTION_KEYWORDS):
        return ScopeResult(
            fits_prometheus=False,
            reasoning=(
                "Sounds like information extraction (pulling out specific "
                "entities or fields from text), not classification. Prometheus "
                "assigns a whole document to a category — it doesn't extract spans."
            ),
            redirect_suggestion=(
                "Look at spaCy's NER or fine-tuned LLMs with structured output."
            ),
            next_step="",
        )

    if _matches_any(text, _CLASSIFICATION_KEYWORDS) or _looks_like_classification(text):
        return _build_fit_result(text, hint)

    return ScopeResult(
        fits_prometheus=False,
        reasoning=(
            "I couldn't tell from the description whether this is a classification "
            "problem. Prometheus is for assigning text to predefined categories "
            "(e.g., 'classify these emails into urgency levels')."
        ),
        next_step=(
            "Re-describe the goal as 'I want to classify [items] into [N categories]' "
            "and call scope_problem again, or just call start_here with your data."
        ),
    )


def _build_fit_result(text: str, hint: str) -> ScopeResult:
    is_hierarchical = _matches_any(text, _HIERARCHICAL_KEYWORDS) or bool(
        _HIERARCHICAL_PATTERN.search(text)
    )
    detected = (
        ProblemType.HIERARCHICAL_CLASSIFICATION
        if is_hierarchical
        else ProblemType.FLAT_CLASSIFICATION
    )

    label_count = _extract_count(text, _COUNT_PATTERN)
    sample_count = _extract_count(hint, _SAMPLE_COUNT_PATTERN) or _extract_count(
        text, _SAMPLE_COUNT_PATTERN
    )

    min_examples, ideal_examples = _required_examples(label_count, is_hierarchical)

    what_youll_need: dict = {
        "min_examples": min_examples,
        "ideal_examples": ideal_examples,
        "data_format": (
            "CSV with one text column and one label column per hierarchy level"
            if is_hierarchical
            else "CSV with one text column and one label column"
        ),
        "example_row": (
            "text='Live purebred breeding horses', "
            "Chapter='Live Animals', Heading='Live horses'"
            if is_hierarchical
            else "text='My order is late', category='shipping_issue'"
        ),
    }

    watch_out: list[str] = []
    if label_count and label_count > 100:
        watch_out.append(
            f"With {label_count} categories you'll need significantly more data — "
            f"plan for at least {label_count * 20} examples for usable accuracy."
        )
    watch_out.append(
        "Class imbalance hurts — if 80% of your examples fall in one category, "
        "the model will lazily predict that for everything."
    )
    if is_hierarchical:
        watch_out.append(
            "Deeper hierarchy levels are exponentially harder. Plan to get the top "
            "level right first; deeper levels often need 10x more data per class."
        )

    if sample_count and sample_count < min_examples:
        watch_out.insert(
            0,
            f"You mentioned ~{sample_count} examples but the minimum is around "
            f"{min_examples}. The model will likely underfit — try to gather more.",
        )

    reasoning = (
        f"Sounds like {'hierarchical' if is_hierarchical else 'flat single-label'} "
        f"classification — assigning text to "
        f"{label_count or 'a fixed set of'} "
        f"categor{'ies' if (label_count or 2) > 1 else 'y'}. Prometheus handles this well."
    )

    return ScopeResult(
        fits_prometheus=True,
        reasoning=reasoning,
        detected_problem_type=detected,
        what_youll_need=what_youll_need,
        watch_out_for=watch_out,
        next_step="When you have data, call start_here(file_path) — it will profile your CSV and recommend the setup.",
    )


def _matches_any(text: str, keywords: set[str]) -> bool:
    return any(kw in text for kw in keywords)


def _looks_like_classification(text: str) -> bool:
    return any(p in text for p in ("into ", " as ", " to a ", " to the "))


def _extract_count(text: str, pattern: re.Pattern) -> int | None:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _required_examples(label_count: int | None, is_hierarchical: bool) -> tuple[int, int]:
    """Rule of thumb: 50-100 examples per class for usable accuracy."""
    if label_count is None:
        base_min, base_ideal = 400, 1500
    else:
        base_min = max(200, label_count * 20)
        base_ideal = max(800, label_count * 80)
    if is_hierarchical:
        base_min *= 2
        base_ideal *= 2
    return base_min, base_ideal
