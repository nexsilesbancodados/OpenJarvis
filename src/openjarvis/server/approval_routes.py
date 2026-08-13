"""REST endpoints for the proactive-agent approval queue."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from openjarvis.tools.approval_store import (
    DECISION_ALWAYS_APPROVE,
    DECISION_ALWAYS_DENY,
    STATUS_APPROVED,
    STATUS_DENIED,
    TIER_HIGH,
    ApprovalStore,
    PendingAction,
)

try:
    from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
except ImportError:
    raise ImportError("fastapi is required for approval routes")

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton that shares the same DB file as ProactiveAgent (WAL mode is safe)
_store: Optional[ApprovalStore] = None


def _get_store() -> ApprovalStore:
    global _store
    if _store is None:
        _store = ApprovalStore()
    return _store


def _serialize(action: PendingAction) -> Dict[str, Any]:
    return {
        "id": action.id,
        "action_type": action.action_type,
        "description": action.description,
        "payload": action.payload,
        "permission_key": action.permission_key,
        "tier": action.tier,
        "status": action.status,
        "created_at": action.created_at,
        "expires_at": action.expires_at,
    }


@router.get("/v1/approvals/pending")
async def list_pending_approvals() -> Dict[str, Any]:
    store = _get_store()
    store.expire_stale()
    actions = store.list_pending()
    return {"actions": [_serialize(a) for a in actions], "count": len(actions)}


def _remember(action: Any, *, approved: bool) -> bool:
    """Persist an always-approve/deny rule for this action's permission key.

    ``high`` tier is excluded by design: the tier exists precisely because the
    consequence is large enough to be worth asking every time, so letting the
    UI store a standing yes for it would defeat the classification.
    """
    if action.tier == TIER_HIGH:
        return False
    _get_store().set_permission(
        action.permission_key,
        DECISION_ALWAYS_APPROVE if approved else DECISION_ALWAYS_DENY,
        approved=approved,
        notes=f"set from the approvals UI for {action.action_type}",
    )
    return True


def _run_approved(action_id: str, app_state: Any) -> None:
    """Execute an approved action off the request path.

    Tool calls can take minutes — shell_exec alone allows 300s — so holding
    the HTTP response open until it finishes would time the UI out on exactly
    the actions worth approving. The outcome is published on the event bus
    instead.
    """
    from openjarvis.security.approval_executor import execute_approved_action

    try:
        execute_approved_action(
            action_id,
            store=_get_store(),
            bus=getattr(app_state, "bus", None),
            capability_policy=getattr(app_state, "capability_policy", None),
            boundary_guard=getattr(app_state, "boundary_guard", None),
            engine=getattr(app_state, "engine", None),
            model=getattr(app_state, "model", "") or "",
            memory_backend=getattr(app_state, "memory_backend", None),
            channel_backend=getattr(app_state, "channel_backend", None),
        )
    except Exception:
        logger.exception("approved action %s could not be executed", action_id)


@router.post("/v1/approvals/{action_id}/approve")
async def approve_action(
    action_id: str,
    request: Request,
    background: BackgroundTasks,
    always: bool = False,
) -> Dict[str, Any]:
    """Approve one queued action, remember the decision, and run it.

    Approving used to only flip a status. The action then waited for the 5am
    proactive sweep, so from the user's side pressing Approve did nothing —
    worse than not offering the button. It now executes in the background and
    reports the outcome on the event bus.

    ``?always=true`` additionally stores an always-approve rule for the
    action's permission key, which is how the learned permission memory the
    store already implemented finally becomes reachable from the UI.
    """
    store = _get_store()
    action = store.get_action(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found")
    store.update_status(action_id, STATUS_APPROVED)
    remembered = _remember(action, approved=True) if always else False
    background.add_task(_run_approved, action_id, request.app.state)
    logger.info(
        "Action %s approved via UI (always=%s, remembered=%s)",
        action_id,
        always,
        remembered,
    )
    return {
        "status": "approved",
        "id": action_id,
        "remembered": remembered,
        "permission_key": action.permission_key,
        "executing": True,
    }


@router.post("/v1/approvals/{action_id}/deny")
async def deny_action(action_id: str, always: bool = False) -> Dict[str, Any]:
    """Deny one queued action, optionally remembering the decision."""
    store = _get_store()
    action = store.get_action(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found")
    store.update_status(action_id, STATUS_DENIED)
    remembered = _remember(action, approved=False) if always else False
    logger.info(
        "Action %s denied via UI (always=%s, remembered=%s)",
        action_id,
        always,
        remembered,
    )
    return {
        "status": "denied",
        "id": action_id,
        "remembered": remembered,
        "permission_key": action.permission_key,
    }


__all__ = ["router"]
