"""Typed MCP response envelopes.

Three discriminated variants — ``status`` field tells the client which one:

- ``SuccessResponse`` (status="success") — wraps tool output payload
- ``ErrorResponse`` (status="error") — internal/unexpected error
- ``ValidationErrorResponse`` (status="validation_error") — bad input
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class SuccessResponse(BaseModel):
    """Successful tool response. Extra fields from the tool's payload are merged in."""

    model_config = ConfigDict(extra="allow")

    status: Literal["success"] = "success"
    message: str | None = None


class ErrorResponse(BaseModel):
    """Internal error response — used when a tool raises unexpectedly."""

    status: Literal["error"] = "error"
    error: str


class ValidationErrorResponse(BaseModel):
    """Validation error response — used when tool inputs fail to validate."""

    status: Literal["validation_error"] = "validation_error"
    error: str
    field: str | None = None


def success_response(
    data: dict[str, Any] | BaseModel | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Build a success response. Pass a dict or Pydantic model for payload, or None for message-only ack."""
    payload: dict[str, Any] = {}
    if isinstance(data, BaseModel):
        payload = data.model_dump(exclude_none=True)
    elif isinstance(data, dict):
        payload = data
    return SuccessResponse(message=message, **payload).model_dump(exclude_none=True)


def internal_error(message: str) -> dict[str, Any]:
    """Build an internal error response. Stack traces are logged server-side, not returned."""
    return ErrorResponse(error=message).model_dump(exclude_none=True)


def validation_error(message: str, field: str | None = None) -> dict[str, Any]:
    """Build a validation error response."""
    return ValidationErrorResponse(error=message, field=field).model_dump(exclude_none=True)
