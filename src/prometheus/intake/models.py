"""Pydantic models for intake assessment results."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_serializer


class ColumnKind(StrEnum):
    """Heuristic classification of a column's role."""

    TEXT = "text"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    IDENTIFIER = "identifier"
    DATETIME = "datetime"
    UNKNOWN = "unknown"


class ProblemType(StrEnum):
    """Detected problem type from data shape."""

    HIERARCHICAL_CLASSIFICATION = "hierarchical_classification"
    FLAT_CLASSIFICATION = "flat_classification"
    MULTI_LABEL = "multi_label"
    NOT_A_FIT = "not_a_fit"


class ColumnProfile(BaseModel):
    """Profile of a single column."""

    name: str
    kind: ColumnKind
    dtype: str
    cardinality: int
    null_rate: float
    sample_values: list[str]
    avg_text_length: float | None = None
    median_text_length: float | None = None

    @field_serializer("null_rate")
    def _round_null_rate(self, v: float) -> float:
        return round(v, 4)


class Concern(BaseModel):
    """A flagged issue the user should know about before training."""

    severity: str
    message: str
    column: str | None = None


class NextStep(BaseModel):
    """A recommended next tool call with pre-filled args."""

    label: str
    tool: str
    args: dict
    rationale: str


class ProblemAssessment(BaseModel):
    """Result of analyzing a dataset for classifier suitability."""

    problem_type: ProblemType
    confidence: float
    reasoning: str
    suggested_text_column: str | None = None
    suggested_level_columns: list[str] = Field(default_factory=list)
    detected_hierarchies: list[tuple[str, str]] = Field(default_factory=list)

    @field_serializer("confidence")
    def _round_confidence(self, v: float) -> float:
        return round(v, 3)

    @field_serializer("detected_hierarchies")
    def _reshape_hierarchies(self, v: list[tuple[str, str]]) -> list[dict]:
        return [{"parent": p, "child": c} for p, c in v]


class ScopeResult(BaseModel):
    """Result of `scope_problem` for users without data yet."""

    fits_prometheus: bool
    reasoning: str
    detected_problem_type: ProblemType | None = None
    what_youll_need: dict | None = None
    watch_out_for: list[str] = Field(default_factory=list)
    next_step: str = ""
    redirect_suggestion: str | None = None
