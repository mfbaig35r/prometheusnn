"""Problem-type detection from column profiles."""

from __future__ import annotations

import pandas as pd

from prometheus.intake.models import ColumnKind, ColumnProfile, ProblemAssessment, ProblemType


def detect_problem_type(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    goal_description: str | None = None,
) -> ProblemAssessment:
    """Pick the best-fit problem type from profiled columns."""
    text_cols = [p for p in profiles if p.kind == ColumnKind.TEXT]
    cat_cols = [p for p in profiles if p.kind == ColumnKind.CATEGORICAL]
    numeric_cols = [p for p in profiles if p.kind == ColumnKind.NUMERIC]

    if not text_cols and not _categorical_could_be_text(cat_cols):
        return _not_a_fit(
            "No text-like column found. Prometheus needs at least one column "
            "of free-text descriptions to classify.",
            profiles,
            numeric_cols,
        )

    text_col = _pick_text_column(text_cols, cat_cols)

    label_candidates = [c for c in cat_cols if c.name != text_col]
    label_candidates = [c for c in label_candidates if c.cardinality >= 2]

    if not label_candidates:
        return _not_a_fit(
            f"Found a text column ('{text_col}') but no categorical label columns. "
            "Prometheus needs at least one column of labels to learn from.",
            profiles,
            numeric_cols,
        )

    hierarchies = detect_hierarchical_relationships(df, [c.name for c in label_candidates])

    if hierarchies:
        ordered = _order_by_hierarchy(hierarchies, [c.name for c in label_candidates])
        return ProblemAssessment(
            problem_type=ProblemType.HIERARCHICAL_CLASSIFICATION,
            confidence=0.9,
            reasoning=_hierarchical_reasoning(text_col, ordered, label_candidates, hierarchies),
            suggested_text_column=text_col,
            suggested_level_columns=ordered,
            detected_hierarchies=hierarchies,
        )

    best = max(label_candidates, key=lambda c: _label_quality(c, len(df)))
    return ProblemAssessment(
        problem_type=ProblemType.FLAT_CLASSIFICATION,
        confidence=0.85,
        reasoning=(
            f"Found text column '{text_col}' and categorical column '{best.name}' "
            f"with {best.cardinality} classes. No hierarchical nesting detected — "
            f"this looks like flat single-label classification."
        ),
        suggested_text_column=text_col,
        suggested_level_columns=[best.name],
    )


def detect_hierarchical_relationships(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> list[tuple[str, str]]:
    """Find (parent, child) pairs where each child value maps to exactly one parent."""
    relationships = []
    for parent in categorical_columns:
        for child in categorical_columns:
            if parent == child:
                continue
            if _nests_within(df, parent_col=parent, child_col=child):
                relationships.append((parent, child))
    return _dedupe_transitive(relationships)


def _nests_within(df: pd.DataFrame, parent_col: str, child_col: str) -> bool:
    """True if every value of child_col maps to exactly one value of parent_col."""
    pair = df[[parent_col, child_col]].dropna()
    if len(pair) == 0:
        return False
    parent_card = pair[parent_col].nunique()
    child_card = pair[child_col].nunique()
    if child_card <= parent_card:
        return False
    parents_per_child = pair.groupby(child_col)[parent_col].nunique()
    return bool((parents_per_child == 1).all())


def _dedupe_transitive(rels: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop transitive edges: if A→B and B→C and A→C, drop A→C."""
    direct = set(rels)
    to_drop = set()
    for a, b in rels:
        for b2, c in rels:
            if b == b2 and (a, c) in direct:
                to_drop.add((a, c))
    return [r for r in rels if r not in to_drop]


def _order_by_hierarchy(
    hierarchies: list[tuple[str, str]],
    columns: list[str],
) -> list[str]:
    """Topological sort: roots first, leaves last."""
    parents = {p for p, _ in hierarchies}
    children = {c for _, c in hierarchies}
    roots = [c for c in columns if c in parents and c not in children]
    if not roots:
        roots = [c for c in columns if c in parents]

    ordered: list[str] = []
    seen: set[str] = set()

    def visit(node: str) -> None:
        if node in seen:
            return
        seen.add(node)
        ordered.append(node)
        for p, c in hierarchies:
            if p == node:
                visit(c)

    for root in roots:
        visit(root)

    for c in columns:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    return ordered


def _categorical_could_be_text(cat_cols: list[ColumnProfile]) -> bool:
    """A categorical column with high cardinality and long values may actually be text."""
    return any(
        c.cardinality > 100 and (c.avg_text_length or 0) > 30 for c in cat_cols
    )


def _pick_text_column(
    text_cols: list[ColumnProfile],
    cat_cols: list[ColumnProfile],
) -> str:
    """Prefer the longest-average-length text column."""
    candidates = list(text_cols)
    if not candidates:
        candidates = [c for c in cat_cols if c.cardinality > 100]
    return max(candidates, key=lambda c: c.avg_text_length or 0).name


def _label_quality(col: ColumnProfile, total_rows: int) -> float:
    """Score a candidate label column. Prefer moderate cardinality."""
    if col.cardinality < 2:
        return -1.0
    avg_per_class = total_rows / col.cardinality
    if avg_per_class < 5:
        return -0.5
    if col.cardinality > 500:
        return 0.5
    return 1.0


def _hierarchical_reasoning(
    text_col: str,
    ordered: list[str],
    label_candidates: list[ColumnProfile],
    hierarchies: list[tuple[str, str]],
) -> str:
    by_name = {c.name: c for c in label_candidates}
    parts = [f"Found text column '{text_col}' and {len(ordered)} categorical columns "]
    parts.append(f"with nested relationships: {' → '.join(ordered)}. ")
    counts = ", ".join(
        f"'{name}' ({by_name[name].cardinality} classes)" if name in by_name else f"'{name}'"
        for name in ordered
    )
    parts.append(f"Class counts: {counts}. ")
    parts.append("Each child value maps to exactly one parent value, confirming the hierarchy.")
    return "".join(parts)


def _not_a_fit(
    reason: str,
    profiles: list[ColumnProfile],
    numeric_cols: list[ColumnProfile],
) -> ProblemAssessment:
    redirect = ""
    if numeric_cols and len(numeric_cols) >= 2:
        redirect = (
            " The data looks more numeric — consider regression or clustering tools "
            "(scikit-learn, XGBoost) instead."
        )
    return ProblemAssessment(
        problem_type=ProblemType.NOT_A_FIT,
        confidence=0.9,
        reasoning=reason + redirect,
    )
