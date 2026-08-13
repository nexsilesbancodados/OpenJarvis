"""OpenAI TTS backend — cloud-based voice synthesis via OpenAI API."""

from __future__ import annotations

import os
from typing import List

import httpx

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSBackend, TTSResult

_OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"


def _openai_tts_request(
    api_key: str,
    text: str,
    voice: str,
    model: str = "tts-1",
    speed: float = 1.0,
    response_format: str = "mp3",
) -> bytes:
    """Call the OpenAI TTS API and return raw audio bytes."""
    resp = httpx.post(
        _OPENAI_TTS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": response_format,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.content


@TTSRegistry.register("openai_tts")
class OpenAITTSBackend(TTSBackend):
    """OpenAI TTS backend — cloud synthesis."""

    backend_id = "openai_tts"

    #: Time-to-first-byte dominates a spoken turn, and it is almost entirely
    #: fixed cost: measured against this API, a 3-character phrase and a
    #: 112-character one both take about two seconds on ``tts-1``. Switching
    #: model is the single cheapest latency win available —
    #:
    #:     tts-1            first byte 2142 ms, complete 2661 ms
    #:     gpt-4o-mini-tts  first byte  774 ms, complete 1440 ms
    #:
    #: so this build defaults to the faster one. Set ``OPENAI_TTS_MODEL`` to
    #: pin ``tts-1``, ``tts-1-hd``, or anything newer.
    DEFAULT_MODEL = "gpt-4o-mini-tts"

    def __init__(self, *, api_key: str = "", model: str = "") -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = (
            model or os.environ.get("OPENAI_TTS_MODEL", "") or self.DEFAULT_MODEL
        )

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = "nova",
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> TTSResult:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        audio = _openai_tts_request(
            self._api_key,
            text,
            voice=voice_id,
            model=self._model,
            speed=speed,
            response_format=output_format,
        )

        return TTSResult(
            audio=audio,
            format=output_format,
            voice_id=voice_id,
            metadata={"backend": "openai_tts", "model": self._model},
        )

    def available_voices(self) -> List[str]:
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def health(self) -> bool:
        return bool(self._api_key)
