"""Run a tool call that a human has approved.

The approval loop was open at the far end. The gate queued a call and refused
it, the UI listed it and could approve it — and then nothing happened. The
action sat at ``approved`` until the 5am proactive cron happened to sweep the
queue, so from the user's side approving a request did nothing at all, which
is worse than not offering the button.

This closes it: approving executes the call it was approving, in the
background, and reports the outcome on the event bus.

The executor deliberately runs **without** an approval gate. Re-gating a call
the user just approved would either loop forever or silently re-queue, and the
decision has already been made by the only party entitled to make it. Every
other guard — capability policy, boundary scanning, taint, timeouts — still
applies, because "the user said yes to this action" is not the same as "this
action may do anything".
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Tuple

from openjarvis.core.events import EventBus, EventType
from openjarvis.tools.approval_store import (
    STATUS_APPROVED,
    STATUS_EXECUTED,
    ApprovalStore,
    PendingAction,
)

logger = logging.getLogger(__name__)

__all__ = ["TOOL_ACTION_PREFIX", "execute_tool_action", "execute_approved_action"]

#: Action types the gate creates, as ``tool:<tool_name>``.
TOOL_ACTION_PREFIX = "tool:"


def _tool_name(action: PendingAction) -> str:
    if action.action_type.startswith(TOOL_ACTION_PREFIX):
        return action.action_type[len(TOOL_ACTION_PREFIX) :]
    payload = action.payload if isinstance(action.payload, dict) else {}
    return str(payload.get("tool") or "")


def execute_tool_action(
    action: PendingAction,
    *,
    bus: Optional[EventBus] = None,
    capability_policy: Any = None,
    boundary_guard: Any = None,
    engine: Any = None,
    model: str = "",
    memory_backend: Any = None,
    channel_backend: Any = None,
) -> Tuple[bool, str]:
    """Run one approved ``tool:*`` action. Returns ``(success, message)``.

    The runtime dependencies are threaded through because a tool rebuilt here
    is a fresh instance: memory tools need their backend and channel tools
    need theirs, or they would run against nothing and report success.
    """
    name = _tool_name(action)
    if not name:
        return False, f"Action {action.id} does not name a tool."

    from openjarvis.agents.tool_resolver import (
        ensure_registries_populated,
        instantiate_registered_tool,
    )
    from openjarvis.core.registry import ToolRegistry
    from openjarvis.core.types import ToolCall
    from openjarvis.tools._stubs import ToolExecutor

    ensure_registries_populated()
    if not ToolRegistry.contains(name):
        return False, f"Tool '{name}' is not available any more."

    try:
        tool = instantiate_registered_tool(
            ToolRegistry.get(name),
            name,
            engine=engine,
            model=model,
            memory_backend=memory_backend,
            channel_backend=channel_backend,
        )
    except Exception as exc:
        logger.exception("could not build tool %s for approved action", name)
        return False, f"Could not prepare '{name}': {exc}"
    if tool is None:
        return False, f"Tool '{name}' could not be constructed."

    payload = action.payload if isinstance(action.payload, dict) else {}
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}

    executor = ToolExecutor(
        [tool],
        bus=bus,
        capability_policy=capability_policy,
        boundary_guard=boundary_guard,
        agent_id=str(payload.get("agent_id") or ""),
        # No approval_gate: this call *is* the approval.
    )
    result = executor.execute(
        ToolCall(id=action.id, name=name, arguments=json.dumps(arguments))
    )
    return bool(result.success), str(result.content or "")


def execute_approved_action(
    action_id: str,
    *,
    store: Optional[ApprovalStore] = None,
    bus: Optional[EventBus] = None,
    capability_policy: Any = None,
    boundary_guard: Any = None,
    engine: Any = None,
    model: str = "",
    memory_backend: Any = None,
    channel_backend: Any = None,
) -> Tuple[bool, str]:
    """Execute an approved action by id, exactly once.

    The status is moved to ``executed`` *before* the call runs. A tool that
    crashes the worker mid-flight must not leave an action that a later sweep
    would run a second time — an approval authorises one execution, and a
    duplicated side effect is not recoverable by apologising for it.
    """
    store = store if store is not None else ApprovalStore()
    action = store.get_action(action_id)
    if action is None:
        return False, "That action is no longer in the queue."
    if action.status != STATUS_APPROVED:
        return False, f"Action {action_id} is {action.status}, not approved."

    # Only ``tool:*`` actions belong to this executor. Proactive types like
    # email_delete have their own dispatcher in tools/proactive_tools.py and
    # are swept later; claiming them here — even just by marking them
    # executed — would make that sweep find nothing left to do and silently
    # drop the action the user approved.
    if not _tool_name(action):
        return False, (
            f"Action {action_id} is not a tool call; leaving it for the "
            "proactive dispatcher."
        )

    store.update_status(action_id, STATUS_EXECUTED)

    try:
        success, message = execute_tool_action(
            action,
            bus=bus,
            capability_policy=capability_policy,
            boundary_guard=boundary_guard,
            engine=engine,
            model=model,
            memory_backend=memory_backend,
            channel_backend=channel_backend,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("approved action %s failed", action_id)
        success, message = False, str(exc)

    if bus is not None:
        try:
            bus.publish(
                EventType.APPROVAL_EXECUTED,
                {
                    "action_id": action_id,
                    "action_type": action.action_type,
                    "description": action.description,
                    "permission_key": action.permission_key,
                    "success": success,
                    # Truncated: this reaches the UI and the audit trail, and
                    # a tool's full output can be very large.
                    "message": message[:500],
                },
            )
        except Exception:  # pragma: no cover - the bus must not break execution
            logger.debug("could not publish APPROVAL_EXECUTED", exc_info=True)

    return success, message
