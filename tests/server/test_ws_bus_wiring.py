"""The event bridge must listen on the bus the server actually publishes to.

`jarvis serve` builds its own ``EventBus`` and hands it to every publisher,
but the bridge was wired to the module singleton from ``get_event_bus()``.
Two different buses: everything published went one way and the WebSocket
listened the other, so ``/v1/agents/events`` was silent in any real server
run. The live agent view looked merely empty, and approval events would have
vanished the same way.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openjarvis.core.events import EventBus, EventType, get_event_bus  # noqa: E402
from openjarvis.server.api_routes import include_all_routes  # noqa: E402


def _app(bus: EventBus | None):
    app = fastapi.FastAPI()
    app.state.api_key = ""
    app.state.agent_manager = None
    app.state.bus = bus
    include_all_routes(app)
    return app


def test_events_published_on_the_app_bus_reach_the_socket() -> None:
    bus = EventBus(record_history=False)
    assert bus is not get_event_bus(), "fixture must use a non-singleton bus"

    with TestClient(_app(bus)).websocket_connect("/v1/agents/events") as ws:
        bus.publish(
            EventType.TOOL_CALL_BLOCKED,
            {"tool": "shell_exec", "agent_id": "a1", "queued": True},
        )
        frame = ws.receive_json()

    assert frame["type"] == "tool_call_blocked"
    assert frame["data"]["tool"] == "shell_exec"


def test_approval_outcomes_are_forwarded() -> None:
    """Without this the UI can only watch an approved item disappear."""
    bus = EventBus(record_history=False)
    with TestClient(_app(bus)).websocket_connect("/v1/agents/events") as ws:
        bus.publish(
            EventType.APPROVAL_EXECUTED,
            {"action_id": "abc", "success": True, "description": "Run pytest"},
        )
        frame = ws.receive_json()

    assert frame["type"] == "approval_executed"
    assert frame["data"]["success"] is True


def test_an_app_without_a_bus_falls_back_to_the_singleton() -> None:
    """Apps built without one must keep working, not lose their events."""
    with TestClient(_app(None)).websocket_connect("/v1/agents/events") as ws:
        get_event_bus().publish(
            EventType.INFERENCE_START, {"model": "gpt-4o-mini", "agent_id": "a1"}
        )
        frame = ws.receive_json()

    assert frame["type"] == "inference_start"
