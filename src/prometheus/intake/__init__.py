"""Intake module — entry-point assessment tools for new users.

Exposes the public surface used by the `start_here` and `scope_problem` MCP tools.
"""

from prometheus.intake.detector import detect_problem_type
from prometheus.intake.models import (
    ColumnProfile,
    Concern,
    NextStep,
    ProblemAssessment,
    ProblemType,
    ScopeResult,
)
from prometheus.intake.profiler import profile_columns
from prometheus.intake.recommender import (
    estimate_training_time,
    recommend_next_steps,
    surface_concerns,
)
from prometheus.intake.scoper import scope_problem_from_text

__all__ = [
    "ColumnProfile",
    "Concern",
    "NextStep",
    "ProblemAssessment",
    "ProblemType",
    "ScopeResult",
    "detect_problem_type",
    "estimate_training_time",
    "profile_columns",
    "recommend_next_steps",
    "scope_problem_from_text",
    "surface_concerns",
]
