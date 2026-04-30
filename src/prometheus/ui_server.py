"""WebSocket server that exposes the prometheusnn MCP tools to a browser UI.

Run with the ``prometheusnn-ui`` CLI command after ``pip install
prometheusnn[ui]``. The companion frontend is at
https://prometheusnn.cognitionkernel.agency — toggle to "connected mode"
and point it at this server (default: ws://localhost:8000/ws/prometheus)
to drive the same UI against your own data, on your own hardware.

Protocol — JSON over WebSocket. Atomic request/response per tool call.

Client → Server:
    {"type": "ping"}
    {"type": "tool_call", "id": "<uuid>", "tool": "start_here", "args": {...}}
    {"type": "list_tools"}

Server → Client:
    {"type": "pong"}
    {"type": "tool_started", "id": "<uuid>", "tool": "start_here"}
    {"type": "tool_result", "id": "<uuid>", "tool": "start_here", "result": {...}}
    {"type": "tool_error", "id": "<uuid>", "tool": "start_here", "error": "..."}
    {"type": "tool_list", "tools": [{"name": "...", "description": "..."}]}
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("prometheus.ui_server")

# ─── Tool registry ────────────────────────────────────────────────────────────


def _build_tool_registry() -> dict[str, Callable[..., dict]]:
    """Map tool name → tool function for every @mcp.tool registered in
    prometheus.server. Reads from the FastMCP instance so this stays in sync
    automatically when new tools are added.
    """
    from prometheus.server import mcp

    tool_manager = getattr(mcp, "_tool_manager", None)
    if tool_manager is None:
        raise RuntimeError("Could not access FastMCP tool registry — incompatible mcp version?")
    return {name: tool.fn for name, tool in tool_manager._tools.items()}


_TOOLS_CACHE: dict[str, Callable[..., dict]] | None = None


def _get_tools() -> dict[str, Callable[..., dict]]:
    """Lazy-init + cache the tool registry."""
    global _TOOLS_CACHE
    if _TOOLS_CACHE is None:
        _TOOLS_CACHE = _build_tool_registry()
    return _TOOLS_CACHE


def _serialize_tool_metadata(tool_fn: Callable[..., dict]) -> dict[str, Any]:
    """Inspect a tool function and produce JSON-serializable metadata."""
    sig = inspect.signature(tool_fn)
    params = []
    for pname, param in sig.parameters.items():
        annotation = param.annotation
        ann_str = (
            str(annotation).replace("typing.", "").replace("<class '", "").replace("'>", "")
            if annotation is not inspect.Parameter.empty
            else "Any"
        )
        params.append(
            {
                "name": pname,
                "type": ann_str,
                "required": param.default is inspect.Parameter.empty,
            }
        )
    return {
        "name": tool_fn.__name__,
        "description": (tool_fn.__doc__ or "").strip().split("\n")[0],
        "parameters": params,
    }


# ─── Message handler ─────────────────────────────────────────────────────────


async def _handle_message(ws, msg: dict[str, Any]) -> None:
    msg_type = msg.get("type")
    tools = _get_tools()

    if msg_type == "ping":
        await ws.send_json({"type": "pong"})
        return

    if msg_type == "list_tools":
        await ws.send_json(
            {
                "type": "tool_list",
                "tools": [_serialize_tool_metadata(fn) for fn in tools.values()],
            }
        )
        return

    if msg_type == "tool_call":
        call_id = msg.get("id", "no-id")
        tool_name = msg.get("tool")
        args = msg.get("args", {}) or {}

        if tool_name not in tools:
            await ws.send_json(
                {
                    "type": "tool_error",
                    "id": call_id,
                    "tool": tool_name,
                    "error": f"unknown tool: {tool_name}",
                }
            )
            return

        await ws.send_json({"type": "tool_started", "id": call_id, "tool": tool_name})
        try:
            # Tools are synchronous; run in a thread so they don't block the
            # event loop (training can take minutes).
            result = await asyncio.to_thread(tools[tool_name], **args)
            await ws.send_json(
                {
                    "type": "tool_result",
                    "id": call_id,
                    "tool": tool_name,
                    "result": result,
                }
            )
        except Exception as e:
            logger.error("tool %s failed: %s", tool_name, e, exc_info=True)
            await ws.send_json(
                {
                    "type": "tool_error",
                    "id": call_id,
                    "tool": tool_name,
                    "error": str(e),
                }
            )
        return

    await ws.send_json({"type": "tool_error", "error": f"unknown message type: {msg_type}"})


# ─── App + module-level routes ───────────────────────────────────────────────
#
# fastapi is gated behind the [ui] extra. We attempt the import at module
# level; if it fails the module still imports (so ``import prometheus.ui_server``
# doesn't blow up without the extra) but ``app`` is None and ``main()`` will
# error with an install hint.
#
# IMPORTANT: route handlers MUST be defined at module level — defining them
# inside a function body (e.g. ``create_app()``) causes FastAPI to reject
# WebSocket handshakes with HTTP 403 in this version. Discovered the hard way.

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    app: FastAPI | None = FastAPI(title="prometheusnn-ui", version="0.2.0")

    @app.get("/health")
    def _health() -> dict:
        return {"status": "ok", "tool_count": len(_get_tools())}

    @app.websocket("/ws/prometheus")
    async def _ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
        logger.info("client connected: %s", peer)
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "tool_error", "error": "invalid JSON"})
                    continue
                await _handle_message(websocket, msg)
        except WebSocketDisconnect:
            logger.info("client disconnected: %s", peer)

except ImportError:
    app = None


def create_app():
    """Factory used by uvicorn --factory. Returns the module-level app."""
    if app is None:
        raise SystemExit(
            "prometheusnn-ui requires the [ui] extras. Install with:\n"
            "    pip install prometheusnn[ui]"
        )
    return app


# ─── CLI entry point ─────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point — runs the WebSocket server via uvicorn."""
    parser = argparse.ArgumentParser(
        prog="prometheusnn-ui",
        description="Run the prometheusnn WebSocket server for the browser UI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            "prometheusnn-ui requires the [ui] extras. Install with:\n"
            "    pip install prometheusnn[ui]"
        ) from e

    logging.basicConfig(level=args.log_level.upper())
    # Warm the tool registry so the count is logged before serving
    tools = _get_tools()
    logger.info("Registered %d tools", len(tools))
    print(
        f"\n  prometheusnn-ui listening on ws://{args.host}:{args.port}/ws/prometheus\n"
        f"  Open https://prometheusnn.cognitionkernel.agency and toggle 'Connected mode'\n"
    )
    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
