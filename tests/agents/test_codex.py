"""Tests for the Codex CLI agent adapter.

The JSONL shapes asserted here were captured from a real
``codex exec --json`` run against codex-cli 0.147.0, not invented.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, List
from unittest.mock import patch

import pytest

from openjarvis.agents.codex import (
    SANDBOX_MODES,
    CodexAgent,
    _parse_events,
    is_codex_available,
)
from openjarvis.core.registry import AgentRegistry


class _FakeEngine:
    engine_id = "fake"

    def generate(self, *a: Any, **k: Any) -> dict:  # pragma: no cover
        raise AssertionError("CodexAgent must not call the OpenJarvis engine")


def _agent(**kwargs: Any) -> CodexAgent:
    return CodexAgent(_FakeEngine(), "some-model", **kwargs)  # type: ignore[arg-type]


def _jsonl(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


REAL_STREAM = _jsonl(
    {"type": "thread.started", "thread_id": "019ff9b9-4180-7f73"},
    {"type": "turn.started"},
    {
        "type": "item.completed",
        "item": {"id": "item_0", "type": "agent_message", "text": "pronto"},
    },
    {
        "type": "turn.completed",
        "usage": {"input_tokens": 14547, "output_tokens": 6},
    },
)


class TestRegistration:
    def test_registered_under_codex(self) -> None:
        import importlib

        import openjarvis.agents.codex as module

        # The suite clears registries between tests, and a module already in
        # sys.modules will not re-run its decorator on plain import. Reload so
        # this asserts the registration rather than import order.
        importlib.reload(module)

        assert AgentRegistry.contains("codex")
        assert AgentRegistry.get("codex").agent_id == "codex"

    def test_does_not_accept_openjarvis_tools(self) -> None:
        """Codex brings its own tool loop; the executor must not feed it ours."""
        assert CodexAgent.accepts_tools is False


class TestParseEvents:
    def test_parses_a_real_stream(self) -> None:
        message, tools, usage = _parse_events(REAL_STREAM)
        assert message == "pronto"
        assert tools == []
        assert usage["input_tokens"] == 14547

    def test_joins_multiple_agent_messages(self) -> None:
        stream = _jsonl(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "um"}},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "dois"},
            },
        )
        assert _parse_events(stream)[0] == "um\n\ndois"

    def test_maps_work_items_to_tool_results(self) -> None:
        stream = _jsonl(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "pytest -q",
                    "aggregated_output": "2 passed",
                    "status": "completed",
                },
            },
            {
                "type": "item.completed",
                "item": {"type": "file_change", "path": "a.py", "status": "completed"},
            },
        )
        _, tools, _ = _parse_events(stream)
        assert [t.tool_name for t in tools] == ["shell", "edit"]
        assert tools[0].content == "2 passed"
        assert all(t.success for t in tools)

    @pytest.mark.parametrize("status", ["failed", "error", "denied", "aborted"])
    def test_failed_items_are_marked_unsuccessful(self, status: str) -> None:
        stream = _jsonl(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "false",
                    "status": status,
                },
            }
        )
        _, tools, _ = _parse_events(stream)
        assert tools[0].success is False

    def test_turn_failed_surfaces_as_a_failed_result(self) -> None:
        stream = _jsonl({"type": "turn.failed", "error": "rate limited"})
        _, tools, _ = _parse_events(stream)
        assert tools[0].success is False
        assert "rate limited" in tools[0].content

    def test_ignores_noise_and_unknown_items(self) -> None:
        """A CLI upgrade adding item types must not break the adapter."""
        stream = "Reading additional input from stdin...\n\nnot json at all\n" + _jsonl(
            {"type": "item.completed", "item": {"type": "brand_new_thing"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
        )
        message, tools, _ = _parse_events(stream)
        assert message == "ok"
        assert tools == []

    def test_empty_stream_is_not_an_error(self) -> None:
        assert _parse_events("") == ("", [], {})


class TestArgv:
    def test_defaults_are_read_only_and_never_read_stdin(self, tmp_path: Path) -> None:
        argv = _agent(workspace=str(tmp_path))._build_argv("fix it", tmp_path / "o.txt")
        assert argv[:3] == ["codex", "exec", "--json"]
        assert argv[argv.index("-s") + 1] == "read-only"
        assert argv[argv.index("-C") + 1] == str(tmp_path)
        # The prompt must be positional and last: codex reads stdin when the
        # prompt is missing or "-".
        assert argv[-1] == "fix it"
        assert "-" not in argv[-1:]

    def test_forwards_optional_flags(self, tmp_path: Path) -> None:
        argv = _agent(
            workspace=str(tmp_path),
            sandbox="workspace-write",
            codex_model="gpt-5-codex",
            add_dirs=["/extra"],
            skip_git_repo_check=True,
        )._build_argv("go", tmp_path / "o.txt")
        assert argv[argv.index("-m") + 1] == "gpt-5-codex"
        assert argv[argv.index("-s") + 1] == "workspace-write"
        assert argv[argv.index("--add-dir") + 1] == "/extra"
        assert "--skip-git-repo-check" in argv

    def test_openjarvis_model_is_not_forwarded(self, tmp_path: Path) -> None:
        """`some-model` names an OpenJarvis model Codex would not recognise."""
        argv = _agent(workspace=str(tmp_path))._build_argv("go", tmp_path / "o.txt")
        assert "some-model" not in argv
        assert "-m" not in argv

    def test_rejects_an_unknown_sandbox(self) -> None:
        with pytest.raises(ValueError, match="sandbox must be one of"):
            _agent(sandbox="yolo")

    def test_every_declared_mode_is_accepted(self) -> None:
        for mode in SANDBOX_MODES:
            assert _agent(sandbox=mode)._sandbox == mode


class TestRun:
    def test_reports_a_missing_binary_in_plain_language(self) -> None:
        with patch("openjarvis.agents.codex.is_codex_available", return_value=False):
            result = _agent().run("do something")
        assert result.metadata["error_type"] == "missing_binary"
        assert "npm i -g @openai/codex" in result.content
        assert "Traceback" not in result.content

    def test_rejects_an_empty_prompt_without_spawning(self) -> None:
        with patch("subprocess.run") as run:
            result = _agent().run("   ")
        run.assert_not_called()
        assert result.metadata["error_type"] == "empty_prompt"

    def test_happy_path(self, tmp_path: Path) -> None:
        def fake_run(argv: List[str], **kwargs: Any) -> Any:
            # Codex writes the final message to the -o path.
            Path(argv[argv.index("-o") + 1]).write_text("pronto", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, REAL_STREAM, "")

        with (
            patch("openjarvis.agents.codex.is_codex_available", return_value=True),
            patch("subprocess.run", side_effect=fake_run),
        ):
            result = _agent(workspace=str(tmp_path)).run("diga pronto")

        assert result.content == "pronto"
        assert result.metadata["agent"] == "codex"
        assert result.metadata["sandbox"] == "read-only"
        assert result.metadata["usage"]["output_tokens"] == 6
        assert result.metadata.get("error") is None

    def test_stdin_is_closed_so_codex_cannot_block(self, tmp_path: Path) -> None:
        seen: dict = {}

        def fake_run(argv: List[str], **kwargs: Any) -> Any:
            seen.update(kwargs)
            return subprocess.CompletedProcess(argv, 0, REAL_STREAM, "")

        with (
            patch("openjarvis.agents.codex.is_codex_available", return_value=True),
            patch("subprocess.run", side_effect=fake_run),
        ):
            _agent(workspace=str(tmp_path)).run("go")

        assert seen["stdin"] is subprocess.DEVNULL

    def test_output_file_wins_over_the_event_stream(self, tmp_path: Path) -> None:
        """The -o file is written last, so it survives a truncated stream."""

        def fake_run(argv: List[str], **kwargs: Any) -> Any:
            Path(argv[argv.index("-o") + 1]).write_text("final", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, REAL_STREAM, "")

        with (
            patch("openjarvis.agents.codex.is_codex_available", return_value=True),
            patch("subprocess.run", side_effect=fake_run),
        ):
            result = _agent(workspace=str(tmp_path)).run("go")
        assert result.content == "final"

    def test_timeout_is_reported_not_raised(self, tmp_path: Path) -> None:
        with (
            patch("openjarvis.agents.codex.is_codex_available", return_value=True),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("codex", 1)),
        ):
            result = _agent(workspace=str(tmp_path), timeout=1).run("go")
        assert result.metadata["error_type"] == "timeout"
        assert "timed out" in result.content

    def test_nonzero_exit_keeps_partial_tool_results(self, tmp_path: Path) -> None:
        stream = _jsonl(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "pytest",
                    "status": "failed",
                },
            }
        )
        with (
            patch("openjarvis.agents.codex.is_codex_available", return_value=True),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess([], 1, stream, "boom"),
            ),
        ):
            result = _agent(workspace=str(tmp_path)).run("go")
        assert result.metadata["error"] is True
        assert result.metadata["returncode"] == 1
        assert len(result.tool_results) == 1
        assert "boom" in result.content

    def test_spawn_failure_is_reported(self, tmp_path: Path) -> None:
        with (
            patch("openjarvis.agents.codex.is_codex_available", return_value=True),
            patch("subprocess.run", side_effect=OSError("denied")),
        ):
            result = _agent(workspace=str(tmp_path)).run("go")
        assert result.metadata["error_type"] == "spawn_failed"
        assert "denied" in result.content


class TestAvailability:
    def test_probe_uses_path_lookup(self) -> None:
        with patch("shutil.which", return_value=None):
            assert is_codex_available() is False
        with patch("shutil.which", return_value="/usr/bin/codex"):
            assert is_codex_available() is True
