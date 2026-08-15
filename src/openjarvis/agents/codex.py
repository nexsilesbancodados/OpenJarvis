"""CodexAgent -- delegates coding work to the OpenAI Codex CLI.

Runs ``codex exec`` non-interactively and reads its JSONL event stream. This
mirrors :mod:`openjarvis.agents.claude_code` and :mod:`openjarvis.agents.opencode`:
an external coding agent owns its own loop, tools and sandbox, and OpenJarvis
treats the whole run as one agent turn.

Unlike the opencode adapter, no local server is started and the OpenJarvis
engine is not wired in as a provider -- Codex authenticates on its own (a
ChatGPT login or ``OPENAI_API_KEY``) and picks its own model. ``-m/--model``
is forwarded when a caller asks for a specific one. The ``engine`` argument is
accepted for :class:`BaseAgent` conformance and is otherwise unused.

Install: ``npm i -g @openai/codex``, then ``codex login``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import ToolResult
from openjarvis.engine._stubs import InferenceEngine

logger = logging.getLogger(__name__)

# `codex exec -s`. read-only is the default here on purpose: this agent is
# reachable by voice through the orchestrator, and the approval queue does not
# yet gate what an external coding agent does inside its own sandbox. Callers
# widen it deliberately.
SANDBOX_MODES = ("read-only", "workspace-write", "danger-full-access")

# JSONL item types that represent work worth surfacing as a ToolResult. Codex
# may emit item types this list does not know about; unknown types are ignored
# rather than guessed at, so a CLI upgrade cannot break the adapter.
_TOOL_ITEM_TYPES = {
    "command_execution": "shell",
    "file_change": "edit",
    "mcp_tool_call": "mcp",
    "web_search": "web_search",
}

_FAILED_STATUSES = {"failed", "error", "denied", "aborted", "cancelled"}


def is_codex_available() -> bool:
    """Return True if the ``codex`` binary is on PATH."""
    return shutil.which("codex") is not None


def _item_text(item: Dict[str, Any]) -> str:
    """Best-effort human-readable body for a Codex item."""
    for key in ("text", "output", "aggregated_output", "message", "summary"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("command", "path", "query"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _parse_events(stdout: str) -> Tuple[str, List[ToolResult], Dict[str, Any]]:
    """Split a ``codex exec --json`` stream into message, tools and usage.

    The stream is JSONL, one event per line::

        {"type": "thread.started", "thread_id": "..."}
        {"type": "turn.started"}
        {"type": "item.completed", "item": {"type": "agent_message", "text": "..."}}
        {"type": "turn.completed", "usage": {"input_tokens": 1, ...}}

    Codex also prints non-JSON lines (progress, warnings) to the same stream,
    so anything that does not parse is skipped rather than treated as failure.
    """
    messages: List[str] = []
    tools: List[ToolResult] = []
    usage: Dict[str, Any] = {}

    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        kind = event.get("type", "")
        if kind == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
            continue
        if kind == "turn.failed":
            detail = event.get("error") or event.get("message") or "turn failed"
            tools.append(
                ToolResult(tool_name="codex", content=str(detail), success=False)
            )
            continue
        if kind not in ("item.completed", "item.updated"):
            continue

        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")

        if item_type == "agent_message":
            text = item.get("text")
            if isinstance(text, str) and text:
                messages.append(text)
            continue

        tool_name = _TOOL_ITEM_TYPES.get(item_type)
        if tool_name is None:
            continue
        status = str(item.get("status", "")).lower()
        tools.append(
            ToolResult(
                tool_name=tool_name,
                content=_item_text(item),
                success=status not in _FAILED_STATUSES,
                metadata={"item_type": item_type, "status": status or "completed"},
            )
        )

    return "\n\n".join(messages).strip(), tools, usage


@AgentRegistry.register("codex")
class CodexAgent(BaseAgent):
    """Agent that delegates a coding task to the ``codex`` CLI.

    Parameters
    ----------
    workspace:
        Directory Codex treats as its working root. Defaults to the process
        cwd.
    sandbox:
        One of :data:`SANDBOX_MODES`. Defaults to ``read-only`` -- Codex can
        read the workspace and reason about it, but cannot write or run
        commands until a caller opts in.
    codex_model:
        Forwarded as ``-m``. Empty means Codex uses its own configured model;
        the OpenJarvis ``model`` argument is *not* forwarded, because it names
        an OpenJarvis-side model that Codex may not know.
    add_dirs:
        Extra directories to make writable alongside the workspace.
    skip_git_repo_check:
        Allow running outside a git repository. Codex refuses by default,
        which is a useful guard when it can write.
    ephemeral:
        Do not persist session files to disk.
    """

    agent_id = "codex"
    accepts_tools = False
    _default_temperature = 0.7
    _default_max_tokens = 1024

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        bus: Optional[EventBus] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        workspace: str = "",
        sandbox: str = "read-only",
        codex_model: str = "",
        add_dirs: Optional[List[str]] = None,
        skip_git_repo_check: bool = False,
        ephemeral: bool = True,
        timeout: int = 600,
        codex_bin: str = "",
    ) -> None:
        super().__init__(
            engine,
            model,
            bus=bus,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if sandbox not in SANDBOX_MODES:
            raise ValueError(f"sandbox must be one of {SANDBOX_MODES}, got {sandbox!r}")
        self._workspace = workspace or os.getcwd()
        self._sandbox = sandbox
        self._codex_model = codex_model
        self._add_dirs = list(add_dirs or [])
        self._skip_git_repo_check = skip_git_repo_check
        self._ephemeral = ephemeral
        self._timeout = timeout
        self._codex_bin = codex_bin or "codex"

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def _build_argv(self, prompt: str, last_message_path: Path) -> List[str]:
        argv = [
            self._codex_bin,
            "exec",
            "--json",
            "--color",
            "never",
            "-s",
            self._sandbox,
            "-C",
            self._workspace,
            "-o",
            str(last_message_path),
        ]
        if self._codex_model:
            argv += ["-m", self._codex_model]
        for extra in self._add_dirs:
            argv += ["--add-dir", extra]
        if self._skip_git_repo_check:
            argv.append("--skip-git-repo-check")
        if self._ephemeral:
            argv.append("--ephemeral")
        # The prompt goes last, positionally. Codex reads stdin when the
        # prompt is absent or "-", so it must never be either.
        argv.append(prompt)
        return argv

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run one ``codex exec`` invocation and map it to an AgentResult."""
        self._emit_turn_start(input)

        prompt = (input or "").strip()
        if not prompt:
            self._emit_turn_end(turns=1, error=True)
            return AgentResult(
                content="Codex needs a task description to work on.",
                turns=1,
                metadata={"error": True, "error_type": "empty_prompt"},
            )

        if not is_codex_available():
            self._emit_turn_end(turns=1, error=True)
            return AgentResult(
                content=(
                    "The Codex CLI is not installed. Install it with"
                    " `npm i -g @openai/codex`, then run `codex login`."
                ),
                turns=1,
                metadata={"error": True, "error_type": "missing_binary"},
            )

        tmp = Path(tempfile.mkdtemp(prefix="openjarvis_codex_"))
        last_message_path = tmp / "last_message.txt"
        try:
            argv = self._build_argv(prompt, last_message_path)
            logger.debug("codex argv: %s", argv)
            try:
                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    # Codex falls back to reading the prompt from stdin. With
                    # an inherited stdin it can block forever waiting on a
                    # terminal that will never type.
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                self._emit_turn_end(turns=1, error=True)
                return AgentResult(
                    content=f"Codex timed out after {self._timeout}s.",
                    turns=1,
                    metadata={"error": True, "error_type": "timeout"},
                )
            except OSError as exc:
                self._emit_turn_end(turns=1, error=True)
                return AgentResult(
                    content=f"Could not start the Codex CLI: {exc}",
                    turns=1,
                    metadata={"error": True, "error_type": "spawn_failed"},
                )

            message, tool_results, usage = _parse_events(proc.stdout or "")

            # `-o` holds the final assistant message verbatim. Prefer it: the
            # event stream can be truncated by a crash mid-write, and this
            # file is written last.
            if last_message_path.is_file():
                final = last_message_path.read_text(encoding="utf-8").strip()
                if final:
                    message = final

            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                detail = message or stderr or "Codex exited without output."
                logger.error("codex exec exited %d: %s", proc.returncode, stderr)
                self._emit_turn_end(turns=1, error=True)
                return AgentResult(
                    content=detail,
                    tool_results=tool_results,
                    turns=1,
                    metadata={
                        "error": True,
                        "error_type": "nonzero_exit",
                        "returncode": proc.returncode,
                        "agent": "codex",
                    },
                )

            self._emit_turn_end(turns=1, error=False)
            return AgentResult(
                content=message or "Codex finished without a closing message.",
                tool_results=tool_results,
                turns=1,
                metadata={
                    "agent": "codex",
                    "sandbox": self._sandbox,
                    "workspace": self._workspace,
                    "usage": usage,
                    "tool_calls": len(tool_results),
                },
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
