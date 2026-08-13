"""Tests for risk classification and the tool-call approval gate.

The behaviour under test is the fix for a real hole: server-side agents ran
every tool with ``confirm_callback=lambda _prompt: True``, so the risk tiers
and remembered permissions in the approval store never applied to them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.security.approval_gate import ApprovalGate
from openjarvis.security.tool_risk import assess_tool_call
from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolSpec
from openjarvis.tools.approval_store import (
    DECISION_ALWAYS_APPROVE,
    DECISION_ALWAYS_DENY,
    STATUS_APPROVED,
    STATUS_EXECUTED,
    TIER_HIGH,
    TIER_LOW,
    TIER_MEDIUM,
    TIER_TRIVIAL,
    ApprovalStore,
)


@pytest.fixture()
def store(tmp_path: Path) -> ApprovalStore:
    return ApprovalStore(db_path=tmp_path / "approvals.db")


@pytest.fixture()
def gate(store: ApprovalStore) -> ApprovalGate:
    return ApprovalGate(store)


class _Tool(BaseTool):
    """Tool that records whether it ever ran."""

    tool_id = "probe"

    def __init__(self, name: str = "shell_exec", **spec_kwargs: Any) -> None:
        self.name = name
        self.ran = False
        self._spec_kwargs = spec_kwargs

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="probe",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
            **self._spec_kwargs,
        )

    def execute(self, **params: Any) -> ToolResult:
        self.ran = True
        return ToolResult(tool_name=self.name, content="ran", success=True)


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------


class TestRiskTiers:
    @pytest.mark.parametrize(
        ("tool", "expected"),
        [
            ("think", TIER_TRIVIAL),
            ("file_read", TIER_TRIVIAL),
            ("git_diff", TIER_TRIVIAL),
            ("web_search", TIER_LOW),
            ("file_write", TIER_MEDIUM),
            ("git_commit", TIER_MEDIUM),
            ("shell_exec", TIER_HIGH),
            ("channel_send", TIER_HIGH),
        ],
    )
    def test_known_tools(self, tool: str, expected: str) -> None:
        assert assess_tool_call(tool, {}).tier == expected

    @pytest.mark.parametrize(
        "command",
        ["rm -rf /", "git push --force origin main", "DROP TABLE users", "sudo reboot"],
    )
    def test_destructive_commands_escalate_to_high(self, command: str) -> None:
        """The act, not the tool, decides: `ls` and `rm -rf` are not alike."""
        risk = assess_tool_call("code_interpreter", {"command": command})
        assert risk.tier == TIER_HIGH
        assert risk.reason

    def test_benign_command_keeps_its_table_tier(self) -> None:
        assert assess_tool_call("code_interpreter", {"command": "ls -la"}).tier == (
            TIER_MEDIUM
        )

    def test_config_override_wins(self) -> None:
        risk = assess_tool_call("shell_exec", {}, overrides={"shell_exec": TIER_LOW})
        assert risk.tier == TIER_LOW


class TestUnknownTools:
    """A newly registered tool must never default to "safe"."""

    def test_execution_capability_is_high(self) -> None:
        spec = ToolSpec(
            name="brand_new", description="", required_capabilities=["code:execute"]
        )
        assert assess_tool_call("brand_new", {}, spec=spec).tier == TIER_HIGH

    def test_requires_confirmation_is_high(self) -> None:
        spec = ToolSpec(name="brand_new", description="", requires_confirmation=True)
        assert assess_tool_call("brand_new", {}, spec=spec).tier == TIER_HIGH

    def test_write_capability_is_medium(self) -> None:
        spec = ToolSpec(
            name="brand_new", description="", required_capabilities=["file:write"]
        )
        assert assess_tool_call("brand_new", {}, spec=spec).tier == TIER_MEDIUM

    def test_no_declared_side_effects_is_trivial(self) -> None:
        spec = ToolSpec(name="brand_new", description="")
        assert assess_tool_call("brand_new", {}, spec=spec).tier == TIER_TRIVIAL


class TestPermissionKeys:
    """A remembered yes must not be broader than what the user agreed to."""

    def test_channel_send_is_scoped_to_recipient(self) -> None:
        a = assess_tool_call(
            "channel_send", {"channel": "telegram", "conversation_id": "joao"}
        )
        b = assess_tool_call(
            "channel_send", {"channel": "telegram", "conversation_id": "maria"}
        )
        assert a.permission_key != b.permission_key

    def test_http_request_is_scoped_to_host(self) -> None:
        risk = assess_tool_call("http_request", {"url": "https://api.example.com/x"})
        assert risk.permission_key == "http_request:host:api.example.com"

    def test_unscopable_tool_falls_back_to_its_name(self) -> None:
        assert assess_tool_call("think", {}).permission_key == "think"

    def test_description_is_human_readable(self) -> None:
        risk = assess_tool_call("shell_exec", {"command": "pytest -q"})
        assert "pytest -q" in risk.description


# ---------------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------------


class TestGate:
    def test_trivial_runs_without_queueing(
        self, gate: ApprovalGate, store: ApprovalStore
    ) -> None:
        assert gate.review("think", {"thought": "hi"}).allowed is True
        assert store.list_pending() == []

    def test_high_risk_is_queued_and_refused(
        self, gate: ApprovalGate, store: ApprovalStore
    ) -> None:
        decision = gate.review("shell_exec", {"command": "rm -rf /tmp/x"})
        assert decision.allowed is False
        assert decision.action_id
        pending = store.list_pending()
        assert len(pending) == 1
        assert pending[0].tier == TIER_HIGH

    def test_refusal_explains_itself_in_plain_language(
        self, gate: ApprovalGate
    ) -> None:
        decision = gate.review("shell_exec", {"command": "pytest"})
        assert "approval" in decision.message.lower()
        assert "Traceback" not in decision.message

    def test_remembered_approval_lets_it_through(
        self, gate: ApprovalGate, store: ApprovalStore
    ) -> None:
        risk = assess_tool_call("file_write", {"path": "a.txt"})
        store.set_permission(
            risk.permission_key, DECISION_ALWAYS_APPROVE, approved=True
        )
        assert gate.review("file_write", {"path": "a.txt"}).allowed is True

    def test_remembered_denial_blocks_without_queueing(
        self, gate: ApprovalGate, store: ApprovalStore
    ) -> None:
        risk = assess_tool_call("file_write", {"path": "a.txt"})
        store.set_permission(risk.permission_key, DECISION_ALWAYS_DENY)
        decision = gate.review("file_write", {"path": "a.txt"})
        assert decision.allowed is False
        assert store.list_pending() == []

    def test_high_tier_ignores_a_remembered_approval(
        self, gate: ApprovalGate, store: ApprovalStore
    ) -> None:
        """Always-allow must not defeat the tier that exists to always ask."""
        risk = assess_tool_call("shell_exec", {"command": "pytest"})
        store.set_permission(
            risk.permission_key, DECISION_ALWAYS_APPROVE, approved=True
        )
        assert gate.review("shell_exec", {"command": "pytest"}).allowed is False

    def test_an_approved_action_is_spent_once(
        self, gate: ApprovalGate, store: ApprovalStore
    ) -> None:
        first = gate.review("file_write", {"path": "a.txt"})
        store.update_status(first.action_id, STATUS_APPROVED)

        assert gate.review("file_write", {"path": "a.txt"}).allowed is True
        assert store.get_action(first.action_id).status == STATUS_EXECUTED
        # The grant is consumed: asking again queues a fresh request.
        assert gate.review("file_write", {"path": "a.txt"}).allowed is False

    def test_widening_auto_approve_tiers(self, store: ApprovalStore) -> None:
        lenient = ApprovalGate(store, auto_approve_tiers=("trivial", "low"))
        assert lenient.review("web_search", {"query": "x"}).allowed is True
        assert lenient.review("file_write", {"path": "a"}).allowed is False

    def test_a_broken_store_fails_closed(self, store: ApprovalStore) -> None:
        """An unwritable queue must not become permission to run unreviewed."""

        def boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError("disk full")

        store.queue_action = boom  # type: ignore[method-assign]
        decision = ApprovalGate(store).review("shell_exec", {"command": "x"})
        assert decision.allowed is False


# ---------------------------------------------------------------------------
# ToolExecutor wiring — the actual hole being closed
# ---------------------------------------------------------------------------


class TestExecutorWiring:
    def test_without_a_gate_behaviour_is_unchanged(self) -> None:
        """Opting out must be byte-for-byte the old behaviour (CLI path)."""
        tool = _Tool("shell_exec")
        executor = ToolExecutor(
            [tool], interactive=True, confirm_callback=lambda _p: True
        )
        result = executor.execute(
            ToolCall(id="1", name="shell_exec", arguments='{"command":"ls"}')
        )
        assert result.success is True
        assert tool.ran is True

    def test_gate_stops_the_tool_from_running(self) -> None:
        tool = _Tool("shell_exec")
        executor = ToolExecutor(
            [tool],
            interactive=True,
            confirm_callback=lambda _p: True,
            approval_gate=_StubGate(allow=False),
        )
        result = executor.execute(
            ToolCall(id="1", name="shell_exec", arguments='{"command":"rm -rf /"}')
        )
        assert result.success is False
        assert tool.ran is False, "the gate refused but the tool ran anyway"
        assert result.metadata["approval_required"] is True

    def test_gate_allows_a_confirmation_flagged_tool_without_a_tty(self) -> None:
        """The gate replaces the TTY prompt; it must not stack with it.

        requires_confirmation + no interactive callback used to be an
        automatic refusal. With a gate installed the gate decides, otherwise
        an approved shell_exec would still be rejected for want of a TTY.
        """
        tool = _Tool("shell_exec", requires_confirmation=True)
        executor = ToolExecutor(
            [tool], interactive=False, approval_gate=_StubGate(allow=True)
        )
        result = executor.execute(
            ToolCall(id="1", name="shell_exec", arguments='{"command":"ls"}')
        )
        assert result.success is True
        assert tool.ran is True

    def test_validation_runs_before_the_gate(self) -> None:
        """Malformed arguments should not reach a human as an approval."""
        gate = _StubGate(allow=False)
        tool = _Tool("git_commit")
        executor = ToolExecutor([tool], approval_gate=gate)
        result = executor.execute(
            ToolCall(id="1", name="git_commit", arguments="not json")
        )
        assert "Invalid arguments JSON" in result.content
        assert gate.calls == 0


class _StubGate:
    def __init__(self, *, allow: bool) -> None:
        self.allow = allow
        self.calls = 0

    def review(self, tool_name: str, params: Any, **kwargs: Any) -> Any:
        self.calls += 1
        from openjarvis.security.approval_gate import GateDecision

        if self.allow:
            return GateDecision(allowed=True, tier=TIER_HIGH)
        return GateDecision(
            allowed=False, reason="needs approval", action_id="abc", tier=TIER_HIGH
        )


class TestUsabilityBoundary:
    """Tiering has to survive contact with how the assistant actually works.

    A gate that queues routine internal bookkeeping is a gate the user turns
    off, which is worse than no gate. These pin the line between "the
    assistant's own state" and "the world outside it".
    """

    @pytest.mark.parametrize(
        "tool", ["memory_store", "memory_manage", "user_profile_manage"]
    )
    def test_the_assistants_own_notes_are_not_gated(self, tool: str) -> None:
        assert assess_tool_call(tool, {"content": "x"}).tier == TIER_TRIVIAL

    @pytest.mark.parametrize("tool", ["memory_index", "web_search", "schedule_task"])
    def test_touching_the_outside_world_is_gated(self, tool: str) -> None:
        assert assess_tool_call(tool, {}).tier != TIER_TRIVIAL

    def test_a_memory_write_runs_without_queueing(
        self, gate: ApprovalGate, store: ApprovalStore
    ) -> None:
        assert gate.review("memory_store", {"content": "remember me"}).allowed is True
        assert store.list_pending() == []
