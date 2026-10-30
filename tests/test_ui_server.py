"""WebSocket UI server: origin allowlist, per-launch token, destructive tools withheld."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from prometheus import ui_server  # noqa: E402

TOKEN = "test-token-123"
GOOD = "https://prometheusnn.cognitionkernel.agency"


@pytest.fixture(autouse=True)
def secured():
    ui_server.configure(token=TOKEN)
    yield
    ui_server.configure(token=None)


@pytest.fixture
def client():
    return TestClient(ui_server.app)


def _connect(client, origin=GOOD, token=TOKEN):
    url = "/ws/prometheus" + (f"?token={token}" if token is not None else "")
    headers = {"origin": origin} if origin is not None else {}
    return client.websocket_connect(url, headers=headers)


def test_allowed_origin_with_token_connects(client):
    with _connect(client) as ws:
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


@pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost:9999", "null"])
def test_foreign_origin_is_refused(client, origin):
    with pytest.raises(WebSocketDisconnect) as exc, _connect(client, origin=origin):
        pass
    assert exc.value.code == 1008


@pytest.mark.parametrize("token", [None, "", "wrong"])
def test_missing_or_wrong_token_is_refused(client, token):
    with pytest.raises(WebSocketDisconnect), _connect(client, token=token):
        pass


def test_non_browser_client_still_needs_the_token(client):
    with _connect(client, origin=None) as ws:
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
    with pytest.raises(WebSocketDisconnect), _connect(client, origin=None, token="nope"):
        pass


def test_no_token_configured_refuses_everything(client):
    ui_server.configure(token=None)
    with pytest.raises(WebSocketDisconnect), _connect(client):
        pass


def test_destructive_tools_are_withheld_by_default(client):
    with _connect(client) as ws:
        ws.send_json({"type": "list_tools"})
        names = {t["name"] for t in ws.receive_json()["tools"]}
        assert not names & ui_server.DESTRUCTIVE_TOOLS
        assert "start_here" in names
        ws.send_json(
            {"type": "tool_call", "id": "1", "tool": "delete_model", "args": {"model_name": "x"}}
        )
        msg = ws.receive_json()
        assert msg["type"] == "tool_error" and "--allow-destructive" in msg["error"]


def test_allow_destructive_opt_in(client):
    ui_server.configure(token=TOKEN, allow_destructive=True)
    with _connect(client) as ws:
        ws.send_json({"type": "list_tools"})
        names = {t["name"] for t in ws.receive_json()["tools"]}
        assert names >= ui_server.DESTRUCTIVE_TOOLS


def test_non_object_args_are_rejected(client):
    with _connect(client) as ws:
        ws.send_json({"type": "tool_call", "id": "2", "tool": "list_models", "args": ["a"]})
        assert ws.receive_json()["error"] == "args must be an object"


def test_custom_origin_list(client):
    ui_server.configure(token=TOKEN, allowed_origins=["http://localhost:5173"])
    with _connect(client, origin="http://localhost:5173") as ws:
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
    with pytest.raises(WebSocketDisconnect), _connect(client, origin=GOOD):
        pass
