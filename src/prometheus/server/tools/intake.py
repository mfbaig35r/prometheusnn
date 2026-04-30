"""Entry-point tools: start_here, scope_problem."""

from __future__ import annotations

from pathlib import Path

from prometheus.core.mcp_safety import ToolValidationError, mcp_tool_safe
from prometheus.core.responses import success_response
from prometheus.data.loader import load_dataframe
from prometheus.intake import (
    detect_problem_type,
    estimate_training_time,
    profile_columns,
    recommend_next_steps,
    scope_problem_from_text,
    surface_concerns,
)
from prometheus.server import logger, mcp


@mcp.tool()
@mcp_tool_safe
def start_here(file_path: str, goal_description: str | None = None) -> dict:
    """Entry point for users with a data file. Profiles columns, detects problem type, returns pre-filled next_steps.

    Always call this FIRST when a user wants to build a classifier and has a CSV/JSON file.
    Do not call analyze_training_data directly — use the args from this tool's
    `next_steps[0].args` instead.

    Args:
        file_path: Path to a CSV or JSON file
        goal_description: Optional free text describing what the user wants to build
            (e.g., "classify support tickets by urgency"). Helps disambiguate when
            the data shape is unclear.
    """
    logger.info("tool=start_here file=%s", file_path)
    if not Path(file_path).exists():
        raise ToolValidationError(f"File not found: {file_path}", field="file_path")
    df = load_dataframe(file_path)
    profiles = profile_columns(df)
    assessment = detect_problem_type(df, profiles, goal_description=goal_description)
    next_steps = recommend_next_steps(file_path, assessment, profiles)
    concerns = surface_concerns(df, assessment, profiles)

    time_low, time_high = (0, 0)
    if assessment.suggested_level_columns:
        time_low, time_high = estimate_training_time(
            rows=len(df),
            num_levels=len(assessment.suggested_level_columns),
        )

    relevant_names = set(assessment.suggested_level_columns)
    if assessment.suggested_text_column:
        relevant_names.add(assessment.suggested_text_column)
    relevant_profiles = [p for p in profiles if p.name in relevant_names]
    unused_columns = [p.name for p in profiles if p.name not in relevant_names]

    return success_response(
        {
            "data_summary": {
                "rows": len(df),
                "columns": [p.name for p in profiles],
                "relevant_profiles": [p.model_dump(exclude_none=True) for p in relevant_profiles],
                "unused_columns": unused_columns,
            },
            "assessment": assessment.model_dump(exclude_none=True),
            "confidence_guide": (
                "0.9+ = strong signal, recommendation can be trusted; "
                "0.7-0.9 = good signal, worth confirming with the user; "
                "below 0.7 = weak signal, ask the user to clarify their goal."
            ),
            "concerns": [c.model_dump(exclude_none=True) for c in concerns],
            "estimated_training_time_minutes": (
                {"low": time_low, "high": time_high} if time_low else None
            ),
            "next_steps": [s.model_dump(exclude_none=True) for s in next_steps],
        }
    )


@mcp.tool()
@mcp_tool_safe
def scope_problem(goal_description: str, data_hint: str | None = None) -> dict:
    """Entry point for users without data yet. Returns whether Prometheus fits the goal.

    Use this when the user describes what they want to build but doesn't have a
    file ready. Returns fits/doesn't-fit, what data they'll need, and watch-outs.
    For non-fits, returns a polite redirect to the right tool.

    Args:
        goal_description: Free text describing the goal
            (e.g., "classify customer emails into urgency levels")
        data_hint: Optional hint about available data
            (e.g., "I have ~3000 examples", "no labels yet")
    """
    logger.info("tool=scope_problem goal=%s", goal_description[:80])
    result = scope_problem_from_text(goal_description, data_hint)
    return success_response(result)
