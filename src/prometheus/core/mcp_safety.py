"""Decorator that wraps MCP tools with uniform error handling."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from prometheus.core.responses import internal_error, validation_error

logger = logging.getLogger(__name__)


class ToolValidationError(Exception):
    """Raise from inside a tool to return a validation_error response.

    The mcp_tool_safe decorator catches this and converts it to the proper
    response shape. Use this for bad-input failures (missing file, bad column
    name, etc.) instead of returning validation_error() directly.
    """

    def __init__(self, message: str, field: str | None = None):
        self.message = message
        self.field = field
        super().__init__(message)


def mcp_tool_safe(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an MCP tool with logging + uniform exception handling.

    Catches:
    - ToolValidationError → validation_error response (with field)
    - Any other Exception → internal_error response, logged with traceback

    Apply *inside* @mcp.tool() so the registered name and signature stay correct.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return func(*args, **kwargs)
        except ToolValidationError as e:
            return validation_error(e.message, field=e.field)
        except Exception as e:
            logger.error("%s failed: %s", func.__name__, e, exc_info=True)
            return internal_error(str(e))

    return wrapper
