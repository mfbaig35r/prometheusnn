"""Dataclasses for intake assessment results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


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


@dataclass
class ColumnProfile:
    """Profile of a single column."""

    name: str
    kind: ColumnKind
    dtype: str
    cardinality: int
    null_rate: float
    sample_values: list[str]
    avg_text_length: float | None = None
    median_text_length: float | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "dtype": self.dtype,
            "cardinality": self.cardinality,
            "null_rate": round(self.null_rate, 4),
            "sample_values": self.sample_values,
            "avg_text_length": self.avg_text_length,
            "median_text_length": self.median_text_length,
        }


@dataclass
class Concern:
    """A flagged issue the user should know about before training."""

    severity: str  # "info" | "warning" | "critical"
    message: str
    column: str | None = None

    def to_dict(self) -> dict:
        d = {"severity": self.severity, "message": self.message}
        if self.column:
            d["column"] = self.column
        return d


@dataclass
class NextStep:
    """A recommended next tool call with pre-filled args."""

    label: str
    tool: str
    args: dict
    rationale: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "tool": self.tool,
            "args": self.args,
            "rationale": self.rationale,
        }


@dataclass
class ProblemAssessment:
    """Result of analyzing a dataset for classifier suitability."""

    problem_type: ProblemType
    confidence: float
    reasoning: str
    suggested_text_column: str | None = None
    suggested_level_columns: list[str] = field(default_factory=list)
    detected_hierarchies: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "problem_type": self.problem_type.value,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "suggested_text_column": self.suggested_text_column,
            "suggested_level_columns": self.suggested_level_columns,
            "detected_hierarchies": [
                {"parent": p, "child": c} for p, c in self.detected_hierarchies
            ],
        }


@dataclass
class ScopeResult:
    """Result of `scope_problem` for users without data yet."""

    fits_prometheus: bool
    reasoning: str
    detected_problem_type: ProblemType | None = None
    what_youll_need: dict | None = None
    watch_out_for: list[str] = field(default_factory=list)
    next_step: str = ""
    redirect_suggestion: str | None = None

    def to_dict(self) -> dict:
        d = {
            "fits_prometheus": self.fits_prometheus,
            "reasoning": self.reasoning,
            "watch_out_for": self.watch_out_for,
            "next_step": self.next_step,
        }
        if self.detected_problem_type:
            d["detected_problem_type"] = self.detected_problem_type.value
        if self.what_youll_need:
            d["what_youll_need"] = self.what_youll_need
        if self.redirect_suggestion:
            d["redirect_suggestion"] = self.redirect_suggestion
        return d
