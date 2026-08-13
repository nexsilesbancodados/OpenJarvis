"""The audit log must cover what the assistant did, not only what it scanned.

Before this, ``AuditLogger`` subscribed to three content-scan events only, so
a tool running, being refused by the approval gate, or being denied a
capability left no trace — the half of "auditable actions" that matters.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openjarvis.core.events import EventBus, EventType
from openjarvis.security.audit import AuditLogger
from openjarvis.security.types import SecurityEventType


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def logger(tmp_path: Path, bus: EventBus) -> AuditLogger:
    return AuditLogger(db_path=tmp_path / "audit.db", bus=bus)


def _types(logger: AuditLogger) -> list[str]:
    return [row.event_type for row in logger.query()]


class TestToolCoverage:
    def test_a_blocked_call_is_recorded(self, logger: AuditLogger, bus: EventBus):
        bus.publish(
            EventType.TOOL_CALL_BLOCKED,
            {"tool": "shell_exec", "agent_id": "a1", "queued": True},
        )
        rows = logger.query()
        assert len(rows) == 1
        assert rows[0].event_type == SecurityEventType.TOOL_BLOCKED.value
        assert rows[0].action_taken == "queued"
        assert "shell_exec" in rows[0].content_preview

    def test_a_denied_capability_is_recorded(self, logger: AuditLogger, bus: EventBus):
        bus.publish(
            EventType.CAPABILITY_DENIED,
            {"tool": "file_write", "agent_id": "a1", "capability": "file:write"},
        )
        assert _types(logger) == [SecurityEventType.TOOL_BLOCKED.value]

    def test_a_consequential_call_is_recorded(self, logger: AuditLogger, bus: EventBus):
        bus.publish(
            EventType.TOOL_CALL_END,
            {"tool": "shell_exec", "agent_id": "a1", "success": True},
        )
        rows = logger.query()
        assert rows[0].event_type == SecurityEventType.TOOL_INVOKED.value
        assert rows[0].action_taken == "ok"

    def test_failure_is_distinguished_from_success(
        self, logger: AuditLogger, bus: EventBus
    ):
        bus.publish(
            EventType.TOOL_CALL_END,
            {"tool": "git_commit", "success": False},
        )
        assert logger.query()[0].action_taken == "failed"


class TestNoiseControl:
    """An audit trail nobody can read is not an audit trail."""

    @pytest.mark.parametrize("tool", ["think", "calculator", "file_read", "web_search"])
    def test_routine_calls_are_not_recorded(
        self, logger: AuditLogger, bus: EventBus, tool: str
    ):
        bus.publish(EventType.TOOL_CALL_END, {"tool": tool, "success": True})
        assert logger.query() == []

    def test_an_unclassifiable_tool_is_recorded_rather_than_dropped(
        self, logger: AuditLogger, bus: EventBus
    ):
        """A gap in the trail is worse than a redundant row."""
        bus.publish(
            EventType.TOOL_CALL_END,
            {"tool": "some_tool_nobody_declared", "success": True},
        )
        assert len(logger.query()) == 1

    def test_a_call_with_no_tool_name_is_skipped(
        self, logger: AuditLogger, bus: EventBus
    ):
        bus.publish(EventType.TOOL_CALL_END, {"success": True})
        assert logger.query() == []


class TestSecrecy:
    def test_arguments_never_reach_the_log(self, logger: AuditLogger, bus: EventBus):
        """This table is append-only and hash-chained; secrets must stay out."""
        bus.publish(
            EventType.TOOL_CALL_END,
            {
                "tool": "http_request",
                "success": True,
                "arguments": {"url": "https://x/y", "token": "sk-supersecret"},
            },
        )
        blob = " ".join(f"{r.content_preview} {r.action_taken}" for r in logger.query())
        assert "sk-supersecret" not in blob


class TestChainStillVerifies:
    def test_tool_rows_do_not_break_the_merkle_chain(
        self, logger: AuditLogger, bus: EventBus
    ):
        bus.publish(EventType.TOOL_CALL_END, {"tool": "shell_exec", "success": True})
        bus.publish(EventType.TOOL_CALL_BLOCKED, {"tool": "channel_send"})
        bus.publish(EventType.TOOL_CALL_END, {"tool": "git_commit", "success": True})
        ok, problem = logger.verify_chain()
        assert ok is True, problem
