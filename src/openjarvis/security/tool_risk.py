"""Classify a tool call by how much damage it can do.

The approval queue in :mod:`openjarvis.tools.approval_store` already models
risk tiers and remembers decisions, but only the proactive agent ever fed it.
Every other path -- including every agent reachable by voice -- ran tools with
``confirm_callback=lambda _prompt: True``. This module supplies the missing
half: given a tool name and its arguments, what tier is this, and what should
the remembered decision be keyed on.

Tiers follow ``approval_store``:

``trivial``
    Read-only, reversible, no side effects the user would notice. Runs.
``low``
    Writes something small and local. Ask once, then remember.
``medium``
    Writes outside the workspace, spends money, or runs arbitrary code.
``high``
    Destructive, irreversible, or visible to other people. Always ask.

The permission key is what a remembered "always allow" applies to. It is
deliberately narrower than the tool name: approving ``git_commit`` in one
repository should not silently approve it everywhere, and allowing a message
to one contact should not allow messages to everyone. Where a sensible
narrowing exists the key includes it; otherwise it falls back to the tool
name, which errs toward asking again rather than toward assuming.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from openjarvis.tools.approval_store import (
    TIER_HIGH,
    TIER_LOW,
    TIER_MEDIUM,
    TIER_TRIVIAL,
)

__all__ = ["RiskAssessment", "assess_tool_call", "DEFAULT_TIERS"]

# Tier per tool. Anything absent is treated as ``medium`` when the tool
# declares a capability or asks for confirmation, and ``trivial`` otherwise --
# see :func:`assess_tool_call`. Being wrong in the cautious direction costs a
# question; being wrong the other way costs a deleted directory.
DEFAULT_TIERS: Dict[str, str] = {
    # Reading and reasoning.
    "think": TIER_TRIVIAL,
    "calculator": TIER_TRIVIAL,
    "file_read": TIER_TRIVIAL,
    "git_status": TIER_TRIVIAL,
    "git_diff": TIER_TRIVIAL,
    "git_log": TIER_TRIVIAL,
    "memory_retrieve": TIER_TRIVIAL,
    "memory_search": TIER_TRIVIAL,
    "knowledge_search": TIER_TRIVIAL,
    "retrieval": TIER_TRIVIAL,
    "scan_chunks": TIER_TRIVIAL,
    "pdf_extract": TIER_TRIVIAL,
    "audio_transcribe": TIER_TRIVIAL,
    "browser_extract": TIER_TRIVIAL,
    "browser_axtree": TIER_TRIVIAL,
    # Opening a visible browser window is an observable external action.
    "browser_open": TIER_LOW,
    "browser_screenshot": TIER_TRIVIAL,
    # The assistant's own notes. Writing here is not an external act: it is
    # reversible, invisible outside the app, and something a personal
    # assistant does constantly. Gating it would put an approval prompt
    # between the user and every remembered fact, which is how a safety
    # feature turns into a reason to switch the safety feature off.
    "memory_store": TIER_TRIVIAL,
    "memory_manage": TIER_TRIVIAL,
    "user_profile_manage": TIER_TRIVIAL,
    # Reaches the network but only reads.
    "web_search": TIER_LOW,
    "browser_navigate": TIER_LOW,
    # Ingests arbitrary user paths into the store, so unlike memory_store this
    # one does touch things the user did not hand over explicitly.
    "memory_index": TIER_LOW,
    # Creates future autonomous action — worth asking the first time.
    "schedule_task": TIER_LOW,
    # Writes, spends, or executes.
    "file_write": TIER_MEDIUM,
    "apply_patch": TIER_MEDIUM,
    "git_commit": TIER_MEDIUM,
    "http_request": TIER_MEDIUM,
    "code_interpreter": TIER_MEDIUM,
    "code_interpreter_docker": TIER_MEDIUM,
    "repl": TIER_MEDIUM,
    "db_query": TIER_MEDIUM,
    "image_generate": TIER_MEDIUM,
    "text_to_speech": TIER_MEDIUM,
    "browser_click": TIER_MEDIUM,
    "browser_type": TIER_MEDIUM,
    "skill_manage": TIER_MEDIUM,
    # Arbitrary execution, or visible to someone other than the user.
    "shell_exec": TIER_HIGH,
    "docker_shell_exec": TIER_HIGH,
    "channel_send": TIER_HIGH,
    "agent_spawn": TIER_HIGH,
    "agent_kill": TIER_HIGH,
}

# Substrings that push an otherwise-ordinary call to ``high``. Matched against
# the command text of execution tools only -- not against arbitrary arguments,
# where they would fire on prose.
_DESTRUCTIVE_MARKERS = (
    "rm -rf",
    "rm -r",
    "del /f",
    "rmdir /s",
    "format ",
    "mkfs",
    "dd if=",
    "drop table",
    "drop database",
    "truncate table",
    "git push --force",
    "git push -f",
    "git reset --hard",
    "git clean -fd",
    ">/dev/sd",
    "shutdown",
    "reboot",
    "remove-item -recurse",
    "sudo ",
)

_COMMAND_KEYS = ("command", "code", "query", "sql", "script")


@dataclass(frozen=True)
class RiskAssessment:
    """How risky one tool call is, and what a remembered decision covers."""

    tier: str
    permission_key: str
    description: str
    reason: str = ""

    @property
    def auto_executes(self) -> bool:
        """True when this tier runs without asking."""
        return self.tier == TIER_TRIVIAL

    @property
    def can_be_remembered(self) -> bool:
        """True when an "always allow" may be stored for this tier.

        ``high`` deliberately cannot: the whole point of the tier is that the
        consequence is large enough to be worth a question every time.
        """
        return self.tier in (TIER_LOW, TIER_MEDIUM)


def _text_of(params: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _looks_destructive(command: str) -> str:
    lowered = command.lower()
    for marker in _DESTRUCTIVE_MARKERS:
        if marker in lowered:
            return marker.strip()
    return ""


def _repo_of(path: str) -> str:
    """Narrow a filesystem path to its repository or parent directory."""
    try:
        current = os.path.abspath(path)
    except (OSError, ValueError):
        return path
    if os.path.isfile(current):
        current = os.path.dirname(current)
    walk = current
    for _ in range(24):
        if os.path.isdir(os.path.join(walk, ".git")):
            return walk
        parent = os.path.dirname(walk)
        if parent == walk:
            break
        walk = parent
    return current


def _key_for(tool_name: str, params: Mapping[str, Any]) -> str:
    """Build the permission key a remembered decision is scoped to."""
    if tool_name in ("git_commit", "git_push"):
        target = params.get("repo") or params.get("path") or os.getcwd()
        return f"{tool_name}:repo:{_repo_of(str(target))}"
    if tool_name in ("file_write", "apply_patch", "file_read"):
        path = params.get("path") or params.get("file_path") or ""
        if path:
            return f"{tool_name}:dir:{_repo_of(str(path))}"
    if tool_name == "channel_send":
        channel = params.get("channel") or params.get("channel_type") or "unknown"
        recipient = params.get("conversation_id") or params.get("to") or "any"
        return f"channel_send:{channel}:{recipient}"
    if tool_name in ("http_request", "browser_open", "browser_navigate"):
        # Scope to the host. Without this, one "always allow" on a URL the
        # user recognised would grant every future URL — and for browser_open
        # that means an injected page could have the agent open anything in
        # the user's real browser, with their real session.
        url = str(params.get("url") or "")
        host = url.split("//", 1)[-1].split("/", 1)[0] if "//" in url else url
        if host:
            return f"{tool_name}:host:{host}"
    return tool_name


def _describe(tool_name: str, params: Mapping[str, Any]) -> str:
    """One line a person can approve or refuse without reading JSON."""
    command = _text_of(params, _COMMAND_KEYS)
    if command:
        snippet = command.strip().splitlines()[0][:160]
        return f"Run `{snippet}` via {tool_name}"
    for key in ("path", "file_path", "url", "query", "content"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return f"{tool_name} on {value.strip()[:160]}"
    return f"Run {tool_name}"


def assess_tool_call(
    tool_name: str,
    params: Mapping[str, Any],
    *,
    spec: Optional[Any] = None,
    overrides: Optional[Mapping[str, str]] = None,
    tier_lookup: Optional[Callable[[str], Optional[str]]] = None,
) -> RiskAssessment:
    """Classify one tool call.

    ``overrides`` lets configuration retier a tool without editing this table.
    ``spec`` is the tool's :class:`~openjarvis.tools._stubs.ToolSpec`; when a
    tool is not in the table its declared capabilities and
    ``requires_confirmation`` flag decide the fallback, so a newly registered
    tool is never silently trivial.
    """
    params = params or {}
    reason = ""

    tier: Optional[str] = None
    if overrides:
        tier = overrides.get(tool_name)
    if tier is None and tier_lookup is not None:
        tier = tier_lookup(tool_name)
    if tier is None:
        tier = DEFAULT_TIERS.get(tool_name)

    if tier is None:
        # Unknown tool: infer from what it declares about itself.
        capabilities = set(getattr(spec, "required_capabilities", []) or [])
        if getattr(spec, "requires_confirmation", False) or (
            capabilities & {"code:execute", "system:admin"}
        ):
            tier = TIER_HIGH
            reason = "unknown tool declaring execution or admin capability"
        elif capabilities & {"file:write", "memory:write", "channel:send"}:
            tier = TIER_MEDIUM
            reason = "unknown tool declaring a write capability"
        elif capabilities & {"network:fetch"}:
            tier = TIER_LOW
            reason = "unknown tool reaching the network"
        else:
            tier = TIER_TRIVIAL
            reason = "unknown tool declaring no side effects"

    # A destructive-looking command outranks whatever the table said.
    command = _text_of(params, _COMMAND_KEYS)
    if command:
        marker = _looks_destructive(command)
        if marker:
            tier = TIER_HIGH
            reason = f"command contains {marker!r}"

    return RiskAssessment(
        tier=tier,
        permission_key=_key_for(tool_name, params),
        description=_describe(tool_name, params),
        reason=reason,
    )
