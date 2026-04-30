"""Build pre-filled next_steps, surface concerns, estimate training time."""

from __future__ import annotations

import pandas as pd

from prometheus.intake.models import (
    ColumnProfile,
    Concern,
    NextStep,
    ProblemAssessment,
    ProblemType,
)

_MIN_SAMPLES_PER_CLASS = 5
_NULL_RATE_WARNING = 0.10
_NULL_RATE_CRITICAL = 0.30
_HIGH_CARDINALITY_RATIO = 0.5  # classes / rows above this is problematic


def recommend_next_steps(
    file_path: str,
    assessment: ProblemAssessment,
    profiles: list[ColumnProfile],
) -> list[NextStep]:
    """Build pre-filled next-tool-call recommendations."""
    if assessment.problem_type == ProblemType.NOT_A_FIT:
        return []

    text_col = assessment.suggested_text_column
    levels = assessment.suggested_level_columns
    if not text_col or not levels:
        return []

    steps = [
        NextStep(
            label=f"Run the recommended {len(levels)}-level setup",
            tool="analyze_training_data",
            args={
                "file_path": file_path,
                "text_column": text_col,
                "level_columns": levels,
            },
            rationale=(
                "Validates the data and surfaces any per-class issues before training. "
                "Always call this before train_model."
            ),
        ),
    ]

    if assessment.problem_type == ProblemType.HIERARCHICAL_CLASSIFICATION and len(levels) > 2:
        simpler = levels[:-1]
        dropped = levels[-1]
        steps.append(
            NextStep(
                label=f"Try a simpler {len(simpler)}-level version (drop '{dropped}')",
                tool="analyze_training_data",
                args={
                    "file_path": file_path,
                    "text_column": text_col,
                    "level_columns": simpler,
                },
                rationale=(
                    f"Deeper levels often have too few samples per class. Dropping the "
                    f"leaf '{dropped}' usually yields a more accurate model — worth comparing."
                ),
            )
        )

    return steps


def surface_concerns(
    df: pd.DataFrame,
    assessment: ProblemAssessment,
    profiles: list[ColumnProfile],
) -> list[Concern]:
    """Flag issues the user should know about before training."""
    concerns: list[Concern] = []

    if assessment.problem_type == ProblemType.NOT_A_FIT:
        return concerns

    profile_by_name = {p.name: p for p in profiles}

    for col in assessment.suggested_level_columns:
        prof = profile_by_name.get(col)
        if not prof:
            continue
        avg_per_class = len(df) / prof.cardinality if prof.cardinality else 0
        if avg_per_class < _MIN_SAMPLES_PER_CLASS:
            concerns.append(
                Concern(
                    severity="critical",
                    column=col,
                    message=(
                        f"'{col}' has {prof.cardinality} classes across {len(df)} rows "
                        f"(~{avg_per_class:.1f} samples per class). "
                        f"Below {_MIN_SAMPLES_PER_CLASS} samples per class, "
                        f"accuracy at this level will be poor — consider dropping it or "
                        f"merging rare classes."
                    ),
                )
            )
        elif prof.cardinality / len(df) > _HIGH_CARDINALITY_RATIO:
            concerns.append(
                Concern(
                    severity="warning",
                    column=col,
                    message=(
                        f"'{col}' has {prof.cardinality} classes but only {len(df)} rows. "
                        f"Expect ~{avg_per_class:.0f} samples per class — accuracy will be limited."
                    ),
                )
            )

    for col in [assessment.suggested_text_column] + list(assessment.suggested_level_columns):
        if not col:
            continue
        prof = profile_by_name.get(col)
        if not prof:
            continue
        if prof.null_rate >= _NULL_RATE_CRITICAL:
            concerns.append(
                Concern(
                    severity="critical",
                    column=col,
                    message=(
                        f"'{col}' is {prof.null_rate:.0%} null. "
                        f"Most rows will be dropped during training."
                    ),
                )
            )
        elif prof.null_rate >= _NULL_RATE_WARNING:
            concerns.append(
                Concern(
                    severity="warning",
                    column=col,
                    message=f"'{col}' is {prof.null_rate:.0%} null — affected rows will be dropped.",
                )
            )

    text_prof = profile_by_name.get(assessment.suggested_text_column or "")
    if text_prof and text_prof.avg_text_length is not None and text_prof.avg_text_length < 10:
        concerns.append(
            Concern(
                severity="warning",
                column=text_prof.name,
                message=(
                    f"Text column '{text_prof.name}' averages only "
                    f"{text_prof.avg_text_length:.0f} characters. "
                    f"Very short text limits how well the model can distinguish classes."
                ),
            )
        )

    return concerns


def estimate_training_time(
    rows: int,
    num_levels: int,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> tuple[int, int]:
    """Estimate training time in minutes (low, high) on CPU.

    Rough heuristic: embedding pass dominates for small models, training
    dominates for large taxonomies.
    """
    is_large = "mpnet" in embedding_model.lower() or "large" in embedding_model.lower()
    embed_per_1k = 1.5 if is_large else 0.5

    embed_minutes = (rows / 1000) * embed_per_1k
    train_minutes = (rows / 1000) * 0.4 * num_levels

    total = embed_minutes + train_minutes
    low = max(1, int(total * 0.7))
    high = max(low + 1, int(total * 1.3) + 1)
    return low, high
