"""WebSocket server that exposes the prometheusnn MCP tools to a browser UI.

Run with the ``prometheusnn-ui`` CLI command after ``pip install
prometheusnn[ui]``. The companion frontend is at
https://prometheusnn.cognitionkernel.agency — toggle to "connected mode"
and point it at this server (default: ws://localhost:8000/ws/prometheus)
to drive the same UI against your own data, on your own hardware.

Security. Browsers let any web page open a WebSocket to localhost, so the server:

- requires a per-launch token, passed as ``?token=...`` in the URL (printed at startup, or set
  with ``--token`` / ``PROMETHEUS_UI_TOKEN``); paste the printed URL into connected mode;
- accepts browser connections only from allowed origins (the companion site and local dev by
  default; ``--allowed-origin`` to change);
- does not offer ``delete_model`` or ``export_model`` unless started with ``--allow-destructive``.

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
import hmac
import inspect
import json
import logging
import os
import secrets
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


# ─── Connection security ─────────────────────────────────────────────────────

DEFAULT_ALLOWED_ORIGINS = (
    "https://prometheusnn.cognitionkernel.agency",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)
# Tools that delete or write outside the model store; off over the UI unless --allow-destructive.
DESTRUCTIVE_TOOLS = frozenset({"delete_model", "export_model"})

_SECURITY: dict[str, Any] = {
    "token": None,
    "allowed_origins": frozenset(DEFAULT_ALLOWED_ORIGINS),
    "allow_destructive": False,
}


def configure(
    *,
    token: str | None,
    allowed_origins: tuple[str, ...] | list[str] = DEFAULT_ALLOWED_ORIGINS,
    allow_destructive: bool = False,
) -> None:
    """Set connection security. main() calls this; tests call it directly."""
    _SECURITY["token"] = token
    _SECURITY["allowed_origins"] = frozenset(o.rstrip("/") for o in allowed_origins)
    _SECURITY["allow_destructive"] = allow_destructive


def check_handshake(origin: str | None, token: str | None) -> str | None:
    """None if the connection may proceed, otherwise the reason it is refused.

    Browsers always send Origin on a WebSocket handshake, so a present Origin must be allowed;
    a missing Origin means a non-browser client, which still needs the token.
    """
    expected = _SECURITY["token"]
    if not expected:
        return "server has no token configured"
    if origin is not None and origin.rstrip("/") not in _SECURITY["allowed_origins"]:
        return f"origin not allowed: {origin}"
    if not token or not hmac.compare_digest(token, expected):
        return "missing or invalid token"
    return None


def _ui_tools() -> dict[str, Callable[..., dict]]:
    tools = _get_tools()
    if _SECURITY["allow_destructive"]:
        return tools
    return {n: f for n, f in tools.items() if n not in DESTRUCTIVE_TOOLS}


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
    tools = _ui_tools()

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
            error = (
                f"{tool_name} is not available over the UI server (start it with --allow-destructive)"
                if tool_name in DESTRUCTIVE_TOOLS
                else f"unknown tool: {tool_name}"
            )
            await ws.send_json(
                {"type": "tool_error", "id": call_id, "tool": tool_name, "error": error}
            )
            return
        if not isinstance(args, dict):
            await ws.send_json(
                {
                    "type": "tool_error",
                    "id": call_id,
                    "tool": tool_name,
                    "error": "args must be an object",
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

    app: FastAPI | None = FastAPI(title="prometheusnn-ui", version="0.2.2")

    @app.get("/health")
    def _health() -> dict:
        return {"status": "ok", "tool_count": len(_get_tools())}

    @app.websocket("/ws/prometheus")
    async def _ws_endpoint(websocket: WebSocket) -> None:
        refused = check_handshake(
            websocket.headers.get("origin"), websocket.query_params.get("token")
        )
        if refused is not None:
            logger.warning("refused WebSocket connection: %s", refused)
            await websocket.close(code=1008)  # before accept(): the handshake fails
            return
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
    """Factory used by uvicorn --factory. Returns the module-level app.

    Outside main() (e.g. ``uvicorn --factory``), the token comes from PROMETHEUS_UI_TOKEN or is
    generated and printed, so the server never runs without one.
    """
    if app is None:
        raise SystemExit(
            "prometheusnn-ui requires the [ui] extras. Install with:\n"
            "    pip install prometheusnn[ui]"
        )
    if not _SECURITY["token"]:
        token = os.environ.get("PROMETHEUS_UI_TOKEN") or secrets.token_urlsafe(24)
        configure(token=token)
        if "PROMETHEUS_UI_TOKEN" not in os.environ:
            print(f"\n  prometheusnn-ui token: {token}  (append ?token=... to the URL)\n")
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
    parser.add_argument(
        "--token",
        default=os.environ.get("PROMETHEUS_UI_TOKEN"),
        help="Connection token (default: PROMETHEUS_UI_TOKEN, or a new random one per launch)",
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        dest="allowed_origins",
        help="Browser origin allowed to connect; repeatable (default: the companion site and "
        "localhost:3000)",
    )
    parser.add_argument(
        "--allow-destructive",
        action="store_true",
        help="Also offer delete_model and export_model over the UI",
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
    token = args.token or secrets.token_urlsafe(24)
    configure(
        token=token,
        allowed_origins=args.allowed_origins or DEFAULT_ALLOWED_ORIGINS,
        allow_destructive=args.allow_destructive,
    )
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning("binding to %s exposes the server beyond this machine", args.host)
    # Warm the tool registry so the count is logged before serving
    tools = _get_tools()
    logger.info("Registered %d tools", len(tools))
    print(
        f"\n  prometheusnn-ui listening on ws://{args.host}:{args.port}/ws/prometheus\n"
        f"  Open https://prometheusnn.cognitionkernel.agency, toggle 'Connected mode', and use:\n"
        f"    ws://{args.host}:{args.port}/ws/prometheus?token={token}\n"
    )
    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
