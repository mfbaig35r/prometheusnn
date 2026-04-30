"""Per-domain MCP tool modules.

Each submodule registers its tools against the shared ``mcp`` instance from
``prometheus.server`` as a side effect of being imported.
"""

from prometheus.server.tools import (  # noqa: F401
    evaluation,
    inference,
    intake,
    management,
    training,
)
