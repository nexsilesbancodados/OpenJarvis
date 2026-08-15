"""Approving an action must actually run it.

The loop was open at the far end: the gate queued a call, the UI approved it,
and nothing happened — the action waited for the 5am proactive sweep. From the
user's side the button did nothing, which is worse than not offering it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pytest

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.security.approval_executor import (
    execute_approved_action,
    execute_tool_action,
)
from openjarvis.tools._stubs import BaseTool, ToolSpec
from openjarvis.tools.approval_store import (
    STATUS_APPROVED,
    STATUS_DENIED,
    STATUS_EXECUTED,
    TIER_HIGH,
    ApprovalStore,
)


class _Recorder(BaseTool):
    """Registers real side effects so 'did it run' is observable."""

    ran: List[dict] = []
    tool_id = "recorder"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="recorder",
            description="records its arguments",
            parameters={
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        )

    def execute(self, **params: Any) -> ToolResult:
        _Recorder.ran.append(params)
        return ToolResult(
            tool_name="recorder", content=f"noted {params['note']}", success=True
        )


@pytest.fixture(autouse=True)
def _register() -> Any:
    _Recorder.ran = []
    if not ToolRegistry.contains("recorder"):
        ToolRegistry.register_value("recorder", _Recorder)
    yield


@pytest.fixture()
def store(tmp_path: Path) -> ApprovalStore:
    return ApprovalStore(db_path=tmp_path / "approvals.db")


def _queue(store: ApprovalStore, *, note: str = "hello", tool: str = "recorder"):
    return store.queue_action(
        action_type=f"tool:{tool}",
        description=f"Run {tool}",
        payload={"tool": tool, "agent_id": "a1", "arguments": {"note": note}},
        permission_key=tool,
        tier=TIER_HIGH,
    )


class TestTheLoopCloses:
    def test_approving_runs_the_tool(self, store: ApprovalStore) -> None:
        action = _queue(store)
        store.update_status(action.id, STATUS_APPROVED)

        ok, message = execute_approved_action(action.id, store=store)

        assert ok is True
        assert _Recorder.ran == [{"note": "hello"}], "the tool never ran"
        assert "noted hello" in message

    def test_the_action_is_marked_executed(self, store: ApprovalStore) -> None:
        action = _queue(store)
        store.update_status(action.id, STATUS_APPROVED)
        execute_approved_action(action.id, store=store)
        assert store.get_action(action.id).status == STATUS_EXECUTED

    def test_the_outcome_is_published(self, store: ApprovalStore) -> None:
        bus = EventBus()
        seen: List[dict] = []
        bus.subscribe(EventType.APPROVAL_EXECUTED, lambda e: seen.append(e.data))

        action = _queue(store)
        store.update_status(action.id, STATUS_APPROVED)
        execute_approved_action(action.id, store=store, bus=bus)

        assert seen and seen[0]["success"] is True
        assert seen[0]["action_id"] == action.id


class TestExactlyOnce:
    def test_an_approval_authorises_one_execution(self, store: ApprovalStore) -> None:
        """A duplicated side effect cannot be undone by apologising for it."""
        action = _queue(store)
        store.update_status(action.id, STATUS_APPROVED)

        assert execute_approved_action(action.id, store=store)[0] is True
        ok, message = execute_approved_action(action.id, store=store)

        assert ok is False
        assert len(_Recorder.ran) == 1, "ran twice on one approval"
        assert "executed" in message

    def test_a_pending_action_is_refused(self, store: ApprovalStore) -> None:
        action = _queue(store)
        ok, message = execute_approved_action(action.id, store=store)
        assert ok is False
        assert _Recorder.ran == []
        assert "not approved" in message

    def test_a_denied_action_is_refused(self, store: ApprovalStore) -> None:
        action = _queue(store)
        store.update_status(action.id, STATUS_DENIED)
        assert execute_approved_action(action.id, store=store)[0] is False
        assert _Recorder.ran == []

    def test_an_unknown_id_is_reported_not_raised(self, store: ApprovalStore) -> None:
        ok, message = execute_approved_action("nope", store=store)
        assert ok is False
        assert "no longer in the queue" in message


class TestFailureIsLegible:
    def test_a_vanished_tool_says_so(self, store: ApprovalStore) -> None:
        action = _queue(store, tool="tool_that_was_removed")
        store.update_status(action.id, STATUS_APPROVED)
        ok, message = execute_approved_action(action.id, store=store)
        assert ok is False
        assert "not available" in message
        assert "Traceback" not in message

    def test_bad_arguments_surface_as_validation(self, store: ApprovalStore) -> None:
        action = store.queue_action(
            action_type="tool:recorder",
            description="Run recorder",
            payload={"tool": "recorder", "arguments": {}},  # 'note' is required
            permission_key="recorder",
            tier=TIER_HIGH,
        )
        store.update_status(action.id, STATUS_APPROVED)
        ok, message = execute_approved_action(action.id, store=store)
        assert ok is False
        assert "missing required argument" in message

    def test_a_tool_failure_still_marks_the_action_spent(
        self, store: ApprovalStore
    ) -> None:
        """Otherwise a failing tool would be retried by every later sweep."""
        action = _queue(store, tool="tool_that_was_removed")
        store.update_status(action.id, STATUS_APPROVED)
        execute_approved_action(action.id, store=store)
        assert store.get_action(action.id).status == STATUS_EXECUTED


class TestGuardsStillApply:
    def test_capability_denial_is_honoured(self, store: ApprovalStore) -> None:
        """User consent is not the same as unlimited authority."""

        class _DenyAll:
            def check(self, *_a: Any, **_k: Any) -> bool:
                return False

        class _Guarded(_Recorder):
            tool_id = "guarded"

            @property
            def spec(self) -> ToolSpec:
                base = super().spec
                return ToolSpec(
                    name="guarded",
                    description=base.description,
                    parameters=base.parameters,
                    required_capabilities=["code:execute"],
                )

        ToolRegistry.register_value("guarded", _Guarded)
        action = _queue(store, tool="guarded")

        ok, message = execute_tool_action(action, capability_policy=_DenyAll())
        assert ok is False
        assert "denied" in message.lower()
        assert _Recorder.ran == []


class TestActionShape:
    def test_the_tool_name_comes_from_the_action_type(
        self, store: ApprovalStore
    ) -> None:
        action = store.queue_action(
            action_type="tool:recorder",
            description="",
            payload={"arguments": {"note": "via action_type"}},  # no 'tool' key
            permission_key="recorder",
            tier=TIER_HIGH,
        )
        assert execute_tool_action(action)[0] is True
        assert _Recorder.ran == [{"note": "via action_type"}]

    def test_a_non_tool_action_is_left_alone(self, store: ApprovalStore) -> None:
        """Proactive action types have their own dispatcher."""
        action = store.queue_action(
            action_type="email_delete",
            description="",
            payload={},
            permission_key="email_delete",
            tier=TIER_HIGH,
        )
        ok, message = execute_tool_action(action)
        assert ok is False
        assert "does not name a tool" in message

    def test_payload_arguments_round_trip_as_json(self, store: ApprovalStore) -> None:
        action = _queue(store, note="acentuação e emoji 🙂")
        assert execute_tool_action(action)[0] is True
        assert _Recorder.ran[0]["note"] == "acentuação e emoji 🙂"
        json.dumps(action.payload)  # must stay serialisable for the store


class TestOwnership:
    """This executor owns tool:* actions and nothing else.

    Proactive types (email_delete, sms_send, ...) are dispatched by
    tools/proactive_tools.py on a later sweep. Claiming them here — even only
    by marking them executed — would make that sweep find nothing left and
    silently drop an action the user had approved.
    """

    def test_a_proactive_action_is_left_approved(self, store: ApprovalStore) -> None:
        action = store.queue_action(
            action_type="email_delete",
            description="Delete a newsletter",
            payload={"message_id": "abc"},
            permission_key="email_delete:domain:example.com",
            tier=TIER_HIGH,
        )
        store.update_status(action.id, STATUS_APPROVED)

        ok, message = execute_approved_action(action.id, store=store)

        assert ok is False
        assert "proactive dispatcher" in message
        assert store.get_action(action.id).status == STATUS_APPROVED, (
            "the proactive sweep would no longer find this action"
        )
