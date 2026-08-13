"""WebSocket transport for the continuous voice session.

A thin adapter over :class:`openjarvis.speech.session.VoiceSession`. All the
interesting behaviour — interruption, what history keeps, synthesis running
ahead of playback — lives there and is unit-tested without a socket. This file
is only wiring: frames in, frames out, and the three collaborators the session
needs (answer, synthesize, transcribe).

Protocol
--------
Client → server

===========================  ====================================================
``{"type":"speech_start"}``  The client's VAD heard the user. Barge-in trigger.
``{"type":"speech_end"}``    Utterance closed; transcribe what was buffered.
``{"type":"text", "text"}``  Skip STT (typed input, or client-side recognition).
``{"type":"cancel"}``        Stop the current turn without it counting as
                             an interruption.
``{"type":"mute", "value"}`` Stop and stay quiet.
binary                       PCM/encoded audio for the current utterance.
===========================  ====================================================

Server → client mirrors :class:`VoiceSession`'s emissions — ``state``,
``delta``, ``speaking``, ``interrupted``, ``turn_complete``, ``error`` — plus
``transcript`` when STT resolves, and binary frames carrying audio to play.

Audio buffering is bounded: a client that never sends ``speech_end`` must not
be able to grow the server's memory without limit.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, List, Optional

from openjarvis.speech.session import VoiceSession

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
except ImportError:  # pragma: no cover - server extra not installed
    APIRouter = None  # type: ignore[assignment]

__all__ = ["create_voice_router"]

#: Roughly 60s of 16 kHz mono PCM16. An utterance longer than this is a stuck
#: VAD, not speech, and buffering it unbounded is how a socket becomes an OOM.
MAX_UTTERANCE_BYTES = 2 * 16_000 * 60


# Spoken answers have different constraints from written ones, and the
# difference is measurable rather than stylistic. Synthesis cannot start until
# a sentence closes, so one 140-character sentence means waiting for the whole
# generation *and then* the whole TTS call before any sound — measured at
# ~4.2s here. The same content as three short sentences starts speaking as
# soon as the first one lands. Markdown matters too: read aloud, "**" and "-"
# become noise.
VOICE_SYSTEM_PROMPT = (
    "You are Jarvis, replying out loud. Your answer is spoken, never read.\n"
    "- Use short sentences. Break every thought into its own sentence and end "
    "it with a full stop, so speech can begin before you have finished "
    "thinking.\n"
    "- Lead with the answer, then the detail. If the reply is one line, say "
    "one line.\n"
    "- No markdown, no bullet points, no headings, no emoji, no code blocks — "
    "they are read aloud as noise.\n"
    "- Write numbers, dates and units the way a person would say them.\n"
    "- Answer in the language the user spoke."
)


def _make_answer(app_state: Any, model: str):
    """Build the reply generator, streaming tokens from the configured engine."""

    async def answer(text: str) -> AsyncIterator[str]:
        engine = getattr(app_state, "engine", None)
        if engine is None:
            yield ("No inference engine is configured, so I cannot answer yet.")
            return

        from openjarvis.core.types import Message, Role

        messages = [
            Message(role=Role.SYSTEM, content=VOICE_SYSTEM_PROMPT),
            Message(role=Role.USER, content=text),
        ]
        chosen = model or getattr(app_state, "model", "") or ""
        async for token in engine.stream(messages, model=chosen):
            yield token

    return answer


def _make_synthesizer(app_state: Any, voice_id: str, speed: float):
    """Build the TTS callable, or one that yields silence when TTS is absent.

    Returning empty bytes rather than raising is deliberate: a missing TTS
    backend should degrade to a text-only conversation, not break the session.
    """
    import asyncio

    async def synthesize(text: str) -> bytes:
        backend = getattr(app_state, "tts_backend", None)
        if backend is None:
            return b""
        try:
            result = await asyncio.to_thread(
                backend.synthesize, text, voice_id=voice_id, speed=speed
            )
            return result.audio
        except Exception:
            logger.exception("voice synthesis failed")
            return b""

    return synthesize


async def _transcribe(app_state: Any, audio: bytes, fmt: str) -> str:
    import asyncio

    from openjarvis.speech.wav import ensure_container

    backend = getattr(app_state, "speech_backend", None)
    if backend is None or not audio:
        return ""
    # The client streams headerless PCM16 straight from the AudioWorklet.
    # Every backend wants a container: OpenAI infers the codec from the
    # filename and rejects unknown ones, faster-whisper hands the bytes to a
    # demuxer. Without this the turn fails with an opaque API error instead of
    # producing a transcript.
    audio, fmt = ensure_container(audio, fmt)
    try:
        result = await asyncio.to_thread(backend.transcribe, audio, format=fmt)
        return (result.text or "").strip()
    except Exception:
        logger.exception("voice transcription failed")
        return ""


async def _warm(app_state: Any) -> None:
    """Pay connection setup before the user's first sentence, not during it.

    Both calls are deliberately tiny and entirely best-effort: a warmup that
    fails, costs real time, or raises into the session would be worse than no
    warmup at all.
    """
    engine = getattr(app_state, "engine", None)
    tts = getattr(app_state, "tts_backend", None)

    async def warm_engine() -> None:
        if engine is None:
            return
        from openjarvis.core.types import Message, Role

        model = getattr(app_state, "model", "") or ""
        async for _token in engine.stream(
            [Message(role=Role.USER, content="hi")], model=model, max_tokens=1
        ):
            break

    async def warm_tts() -> None:
        if tts is None:
            return
        await asyncio.to_thread(tts.synthesize, "ok")

    for coro in (warm_engine(), warm_tts()):
        try:
            await asyncio.wait_for(coro, timeout=20)
        except Exception:
            logger.debug("voice warmup skipped", exc_info=True)


def create_voice_router() -> Any:
    """Create the router exposing ``WS /v1/voice/session``."""
    if APIRouter is None:  # pragma: no cover
        return None

    router = APIRouter()

    @router.websocket("/v1/voice/session")
    async def voice_session(websocket: WebSocket) -> None:
        from openjarvis.server.auth_middleware import websocket_authorized

        app_state = websocket.app.state
        expected_key = getattr(app_state, "api_key", "")
        if not websocket_authorized(websocket, expected_key):
            await websocket.close(code=1008)
            return
        await websocket.accept()

        params = websocket.query_params
        audio_format = params.get("format", "webm")
        session = VoiceSession(
            answer=_make_answer(app_state, params.get("model", "")),
            synthesize=_make_synthesizer(
                app_state,
                params.get("voice", "nova"),
                _float(params.get("speed"), 1.0),
            ),
            send_json=websocket.send_json,
            send_audio=websocket.send_bytes,
        )

        # The first turn of a session pays TLS handshakes and client
        # construction on top of everything else — measured at ~1.2s more than
        # a warm turn, on exactly the interaction that forms the user's
        # impression. Spend it now, in the background, while they are still
        # deciding what to say.
        warmup = asyncio.create_task(_warm(app_state))

        buffer: List[bytes] = []
        buffered = 0
        overflowed = False

        def reset() -> None:
            nonlocal buffered, overflowed
            buffer.clear()
            buffered = 0
            overflowed = False

        try:
            await session.start()
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                chunk: Optional[bytes] = message.get("bytes")
                if chunk:
                    if buffered + len(chunk) > MAX_UTTERANCE_BYTES:
                        overflowed = True
                        continue
                    buffer.append(chunk)
                    buffered += len(chunk)
                    continue

                raw = message.get("text")
                if not raw:
                    continue
                payload = _parse(raw)
                kind = payload.get("type")

                if kind == "speech_start":
                    reset()
                    await session.on_user_speech_start()
                elif kind == "speech_end":
                    if overflowed:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "That was too long for me to hear "
                                "in one go — try again in a shorter sentence.",
                            }
                        )
                        reset()
                        await session.on_user_speech_end("")
                        continue
                    audio = b"".join(buffer)
                    reset()
                    text = await _transcribe(app_state, audio, audio_format)
                    await websocket.send_json({"type": "transcript", "text": text})
                    await session.on_user_speech_end(text)
                elif kind == "text":
                    reset()
                    await session.on_user_speech_end(str(payload.get("text", "")))
                elif kind == "cancel":
                    await session.on_cancel()
                elif kind == "mute":
                    await session.set_muted(bool(payload.get("value", True)))
                elif kind == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("voice session failed")
        finally:
            warmup.cancel()
            # Always tear the turn down: an abandoned socket must not leave a
            # generation task and a synthesis task running behind it.
            await session.close()

    return router


def _parse(raw: str) -> dict:
    import json

    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def _float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
