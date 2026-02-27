"""Standardized MCP response helpers."""

from typing import Any


def success_response(data: Any, message: str | None = None) -> dict:
    """Return a successful MCP response."""
    result = {"status": "success"}
    if message:
        result["message"] = message
    if isinstance(data, dict):
        result.update(data)
    else:
        result["data"] = data
    return result


def internal_error(message: str, details: Any = None) -> dict:
    """Return an internal error MCP response."""
    result = {"status": "error", "error": message}
    if details is not None:
        result["details"] = details
    return result


def validation_error(message: str, field: str | None = None) -> dict:
    """Return a validation error MCP response."""
    result = {"status": "validation_error", "error": message}
    if field:
        result["field"] = field
    return result
