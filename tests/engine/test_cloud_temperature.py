"""Models that reject a non-default temperature must not fail the turn.

`gpt-5-nano` and `gpt-5` are in the model picker and answered every request
with:

    400 - Unsupported value: 'temperature' does not support 0.7 with this
    model. Only the default (1) value is supported.

The name list caught `gpt-5-mini` only, and the streaming path had no retry,
so choosing either model broke chat outright.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from openjarvis.engine.cloud import (
    _is_openai_reasoning_model,
    _is_unsupported_temperature_error,
)


class TestModelDetection:
    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-nano-2025-08-07",
            "gpt-5-codex",
            "o1",
            "o3-mini",
            "o4-mini",
            "GPT-5-NANO",
        ],
    )
    def test_restricted_models_omit_temperature(self, model: str) -> None:
        assert _is_openai_reasoning_model(model) is True

    @pytest.mark.parametrize(
        "model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "claude-sonnet-4-6", "llama3"]
    )
    def test_ordinary_models_keep_temperature(self, model: str) -> None:
        assert _is_openai_reasoning_model(model) is False


class TestErrorDetection:
    def test_recognises_the_real_api_message(self) -> None:
        real = (
            "Error code: 400 - {'error': {'message': \"Unsupported value: "
            "'temperature' does not support 0.7 with this model. Only the "
            "default (1) value is supported.\", 'type': "
            "'invalid_request_error', 'param': 'temperature', 'code': "
            "'unsupported_value'}}"
        )
        assert _is_unsupported_temperature_error(Exception(real)) is True

    def test_ignores_unrelated_errors(self) -> None:
        assert _is_unsupported_temperature_error(Exception("rate limited")) is False
        assert _is_unsupported_temperature_error(Exception("bad api key")) is False


class _Chunk:
    def __init__(self, text: str) -> None:
        self.choices = [type("C", (), {"delta": type("D", (), {"content": text})()})()]


class _Completions:
    """Rejects temperature once, like the real API, then succeeds."""

    def __init__(self, *, reject: bool) -> None:
        self.reject = reject
        self.calls: List[dict] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.reject and "temperature" in kwargs:
            raise Exception(
                "Error code: 400 - Unsupported value: 'temperature' does not "
                "support 0.7 with this model. Only the default (1) value is "
                "supported. code: unsupported_value"
            )
        return [_Chunk("ok")]


class _Client:
    def __init__(self, completions: _Completions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


@pytest.mark.asyncio
class TestStreamingRetry:
    async def _run(self, engine: Any, model: str) -> str:
        from openjarvis.core.types import Message, Role

        out = ""
        async for token in engine._stream_openai(
            [Message(role=Role.USER, content="oi")],
            model=model,
            temperature=0.7,
            max_tokens=64,
        ):
            out += token
        return out

    def _engine(self, completions: _Completions) -> Any:
        from openjarvis.engine.cloud import CloudEngine

        engine = CloudEngine.__new__(CloudEngine)
        engine._openai_client = _Client(completions)
        return engine

    async def test_a_restricted_model_never_sends_temperature(self) -> None:
        completions = _Completions(reject=True)
        assert await self._run(self._engine(completions), "gpt-5-nano") == "ok"
        assert len(completions.calls) == 1, "should not have needed a retry"
        assert "temperature" not in completions.calls[0]

    async def test_an_unknown_restricted_model_retries_once(self) -> None:
        """The name list cannot keep up with every model OpenAI ships."""
        completions = _Completions(reject=True)
        assert await self._run(self._engine(completions), "gpt-6-future") == "ok"
        assert len(completions.calls) == 2
        assert "temperature" in completions.calls[0]
        assert "temperature" not in completions.calls[1]

    async def test_ordinary_models_still_get_temperature(self) -> None:
        completions = _Completions(reject=False)
        assert await self._run(self._engine(completions), "gpt-4o-mini") == "ok"
        assert completions.calls[0]["temperature"] == 0.7

    async def test_unrelated_errors_are_not_swallowed(self) -> None:
        class Boom(_Completions):
            def create(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                raise Exception("rate limited")

        completions = Boom(reject=False)
        with pytest.raises(Exception, match="rate limited"):
            await self._run(self._engine(completions), "gpt-4o-mini")
        assert len(completions.calls) == 1, "must not retry a non-temperature error"
