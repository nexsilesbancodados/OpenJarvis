"""Gate tool execution on the approval queue instead of auto-approving.

``ToolExecutor`` already had a confirmation hook, but it is synchronous and
returns a bool -- fine for a CLI that can block on a TTY prompt, useless for a
server handling a voice turn. Every server path therefore passed
``confirm_callback=lambda _prompt: True``, so the risk tiers and remembered
permissions in :mod:`openjarvis.tools.approval_store` protected nothing.

This gate resolves that without blocking. A call that needs a human is
**queued and refused**, not awaited: the tool returns a failure the model can
read out loud ("I need your approval to do that"), the action lands in the
queue the UI already polls, and the next attempt after approval proceeds. A
blocking wait would hold an HTTP request or a voice turn open for as long as
the user takes to notice, which is the wrong failure mode for an assistant
you talk to.

Precedence for one call:

1. ``trivial`` tier — run.
2. A remembered ``always_deny`` for this permission key — refuse.
3. A remembered ``always_approve`` — run, provided the tier allows a
   remembered decision at all (``high`` never does).
4. An already-approved queued action matching this key — run once, then mark
   it executed so the grant cannot be replayed.
5. Otherwise — queue and refuse.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from openjarvis.core.events import EventBus, EventType
from openjarvis.security.tool_risk import RiskAssessment, assess_tool_call
from openjarvis.tools.approval_store import (
    DECISION_ALWAYS_APPROVE,
    DECISION_ALWAYS_DENY,
    STATUS_EXECUTED,
    ApprovalStore,
)

logger = logging.getLogger(__name__)

__all__ = ["GateDecision", "ApprovalGate"]


@dataclass(frozen=True)
class GateDecision:
    """Outcome of consulting the gate for one tool call."""

    allowed: bool
    reason: str = ""
    action_id: str = ""
    tier: str = ""

    @property
    def message(self) -> str:
        """Text handed back to the model when the call is refused."""
        return self.reason


class ApprovalGate:
    """Decide whether a tool call may run, queueing it when a human is needed.

    Parameters
    ----------
    store:
        Where pending actions and remembered permissions live. Defaults to the
        shared :class:`ApprovalStore`.
    bus:
        Optional event bus. Queued and denied calls publish
        ``TOOL_CALL_BLOCKED`` so the UI and audit log can see them.
    overrides:
        Per-tool tier overrides from configuration.
    auto_approve_tiers:
        Tiers that run without asking. Defaults to ``("trivial",)``. Widening
        this is how an operator says "stop asking me about low-risk writes".
    ttl_hours:
        How long a queued action stays actionable.
    """

    def __init__(
        self,
        store: Optional[ApprovalStore] = None,
        *,
        bus: Optional[EventBus] = None,
        overrides: Optional[Mapping[str, str]] = None,
        auto_approve_tiers: tuple[str, ...] = ("trivial",),
        ttl_hours: int = 24,
    ) -> None:
        self._store = store if store is not None else ApprovalStore()
        self._bus = bus
        self._overrides = dict(overrides or {})
        self._auto = set(auto_approve_tiers)
        self._ttl_hours = ttl_hours

    # ------------------------------------------------------------------

    def assess(
        self,
        tool_name: str,
        params: Mapping[str, Any],
        *,
        spec: Optional[Any] = None,
    ) -> RiskAssessment:
        return assess_tool_call(
            tool_name,
            params,
            spec=spec,
            overrides=self._overrides,
        )

    def review(
        self,
        tool_name: str,
        params: Mapping[str, Any],
        *,
        spec: Optional[Any] = None,
        agent_id: str = "",
    ) -> GateDecision:
        """Return whether this call may proceed, queueing it if not."""
        risk = self.assess(tool_name, params, spec=spec)

        if risk.tier in self._auto:
            return GateDecision(allowed=True, tier=risk.tier)

        rule = self._store.get_permission(risk.permission_key)
        if rule is not None:
            if rule.decision == DECISION_ALWAYS_DENY:
                return self._refuse(
                    risk,
                    agent_id,
                    f"You previously told me never to allow this "
                    f"({risk.permission_key}).",
                    queued=False,
                )
            if rule.decision == DECISION_ALWAYS_APPROVE and risk.can_be_remembered:
                return GateDecision(allowed=True, reason="remembered", tier=risk.tier)

        consumed = self._consume_approved(risk)
        if consumed is not None:
            return GateDecision(allowed=True, action_id=consumed, tier=risk.tier)

        return self._queue(risk, tool_name, params, agent_id)

    # ------------------------------------------------------------------

    def _consume_approved(self, risk: RiskAssessment) -> Optional[str]:
        """Spend a matching approved action, so a grant is not reusable."""
        try:
            approved = self._store.list_approved()
        except Exception:  # pragma: no cover - store unavailable
            logger.debug("approval store unreadable", exc_info=True)
            return None
        for action in approved:
            if action.permission_key == risk.permission_key:
                self._store.update_status(action.id, STATUS_EXECUTED)
                return action.id
        return None

    def _queue(
        self,
        risk: RiskAssessment,
        tool_name: str,
        params: Mapping[str, Any],
        agent_id: str,
    ) -> GateDecision:
        payload = {
            "tool": tool_name,
            "agent_id": agent_id,
            "arguments": _safe_payload(params),
        }
        if risk.reason:
            payload["reason"] = risk.reason
        try:
            action = self._store.queue_action(
                action_type=f"tool:{tool_name}",
                description=risk.description,
                payload=payload,
                permission_key=risk.permission_key,
                tier=risk.tier,
                ttl_hours=self._ttl_hours,
            )
        except Exception:  # pragma: no cover - store unavailable
            # Fail closed. A queue we cannot write to must not become a reason
            # to run an unreviewed action.
            logger.exception("could not queue %s for approval", tool_name)
            return self._refuse(
                risk,
                agent_id,
                "I could not record an approval request, so I did not run it.",
                queued=False,
            )

        detail = f" ({risk.reason})" if risk.reason else ""
        return self._refuse(
            risk,
            agent_id,
            f"{risk.description}{detail} needs your approval before I run it."
            f" I've added it to your approvals queue.",
            queued=True,
            action_id=action.id,
        )

    def _refuse(
        self,
        risk: RiskAssessment,
        agent_id: str,
        message: str,
        *,
        queued: bool,
        action_id: str = "",
    ) -> GateDecision:
        if self._bus is not None:
            try:
                self._bus.publish(
                    EventType.TOOL_CALL_BLOCKED,
                    {
                        "agent_id": agent_id,
                        "permission_key": risk.permission_key,
                        "tier": risk.tier,
                        "queued": queued,
                        "action_id": action_id,
                        "reason": risk.reason,
                    },
                )
            except Exception:  # pragma: no cover - bus must never break a call
                logger.debug("could not publish TOOL_CALL_BLOCKED", exc_info=True)
        return GateDecision(
            allowed=False,
            reason=message,
            action_id=action_id,
            tier=risk.tier,
        )


def _safe_payload(params: Mapping[str, Any]) -> Any:
    """Keep the payload JSON-serialisable; the store persists it as JSON."""
    try:
        json.dumps(params)
        return dict(params)
    except (TypeError, ValueError):
        return {k: repr(v) for k, v in params.items()}
