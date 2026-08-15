"""Tests for the WS /v1/voice/session transport.

The session's behaviour is covered in tests/speech/test_session.py. What is
asserted here is the wiring: auth, frame handling, buffering limits, and that
the socket cannot leave a turn running behind it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openjarvis.server.voice_routes import (  # noqa: E402
    MAX_UTTERANCE_BYTES,
    create_voice_router,
)


class _Result:
    def __init__(self, text: str) -> None:
        self.text = text


class _STT:
    def __init__(self, text: str = "abre o chrome") -> None:
        self.text = text
        self.calls: List[bytes] = []

    def transcribe(self, audio: bytes, *, format: str = "wav") -> _Result:
        self.calls.append(audio)
        return _Result(self.text)


class _TTS:
    backend_id = "fake"

    def __init__(self) -> None:
        self.spoken: List[str] = []

    def synthesize(self, text: str, **kwargs: Any) -> Any:
        self.spoken.append(text)
        return SimpleNamespace(audio=b"AUDIO", format="mp3")


class _Engine:
    engine_id = "fake"

    def __init__(self, reply: str = "Claro. Abrindo agora para voce.") -> None:
        self.reply = reply

    async def stream(self, messages, *, model="", **kwargs):
        for word in self.reply.split(" "):
            yield word + " "


def _app(*, stt: Any = None, tts: Any = None, engine: Any = None, api_key: str = ""):
    app = fastapi.FastAPI()
    router = create_voice_router()
    assert router is not None
    app.include_router(router)
    app.state.api_key = api_key
    app.state.speech_backend = stt
    app.state.tts_backend = tts
    app.state.engine = engine
    app.state.model = "test-model"
    return app


def _collect(ws, limit: int = 60) -> tuple[list[dict], list[bytes]]:
    """Drain frames until the turn completes or the budget runs out."""
    json_frames: list[dict] = []
    audio_frames: list[bytes] = []
    for _ in range(limit):
        message = ws.receive()
        if message.get("bytes") is not None:
            audio_frames.append(message["bytes"])
            continue
        text = message.get("text")
        if text is None:
            break
        import json as _json

        payload = _json.loads(text)
        json_frames.append(payload)
        if payload.get("type") in ("turn_complete", "error"):
            break
    return json_frames, audio_frames


class TestAuth:
    def test_rejects_an_unauthenticated_socket(self) -> None:
        app = _app(api_key="oj_sk_secret")
        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/v1/voice/session"):
                pass

    def test_accepts_the_configured_token(self) -> None:
        app = _app(api_key="oj_sk_secret", engine=_Engine(), tts=_TTS())
        client = TestClient(app)
        with client.websocket_connect("/v1/voice/session?token=oj_sk_secret") as ws:
            assert ws.receive_json()["type"] == "state"

    def test_keyless_server_stays_open(self) -> None:
        app = _app(engine=_Engine(), tts=_TTS())
        with TestClient(app).websocket_connect("/v1/voice/session") as ws:
            assert ws.receive_json() == {"type": "state", "value": "listening"}


class TestTurn:
    def test_text_input_produces_speech(self) -> None:
        tts = _TTS()
        app = _app(engine=_Engine(), tts=tts)
        with TestClient(app).websocket_connect("/v1/voice/session") as ws:
            ws.receive_json()  # listening
            ws.send_json({"type": "text", "text": "abre o chrome"})
            frames, audio = _collect(ws)

        assert audio, "no audio was sent to the client"
        assert tts.spoken, "TTS was never called"
        assert any(f["type"] == "turn_complete" for f in frames)

    def test_audio_is_transcribed_on_speech_end(self) -> None:
        stt = _STT("qual o clima hoje")
        app = _app(stt=stt, engine=_Engine(), tts=_TTS())
        with TestClient(app).websocket_connect("/v1/voice/session") as ws:
            ws.receive_json()
            ws.send_json({"type": "speech_start"})
            ws.send_bytes(b"\x00\x01" * 100)
            ws.send_json({"type": "speech_end"})
            frames, _ = _collect(ws)

        assert stt.calls, "buffered audio never reached the STT backend"
        transcripts = [f for f in frames if f["type"] == "transcript"]
        assert transcripts and transcripts[0]["text"] == "qual o clima hoje"

    def test_silence_does_not_start_a_turn(self) -> None:
        stt = _STT("")
        tts = _TTS()
        app = _app(stt=stt, engine=_Engine(), tts=tts)
        with TestClient(app).websocket_connect("/v1/voice/session") as ws:
            ws.receive_json()
            ws.send_json({"type": "speech_start"})
            ws.send_bytes(b"\x00" * 50)
            ws.send_json({"type": "speech_end"})
            ws.receive_json()  # user_speaking
            ws.send_json({"type": "ping"})
            for _ in range(20):
                frame = ws.receive_json()
                if frame.get("type") == "pong":
                    break
        # Assert on the reply specifically rather than on the call count: the
        # session also warms the TTS backend when the socket opens, and a bare
        # count would make this test fail for the wrong reason.
        assert not any(word in " ".join(tts.spoken) for word in ("Claro", "Abrindo")), (
            "answered an empty transcription"
        )


class TestDegradation:
    def test_missing_tts_still_completes_the_turn(self) -> None:
        """Text-only is a worse experience, not a broken one."""
        app = _app(engine=_Engine(), tts=None)
        with TestClient(app).websocket_connect("/v1/voice/session") as ws:
            ws.receive_json()
            ws.send_json({"type": "text", "text": "oi"})
            frames, _ = _collect(ws)
        assert any(f["type"] == "turn_complete" for f in frames)

    def test_missing_engine_says_so_in_a_sentence(self) -> None:
        app = _app(engine=None, tts=_TTS())
        with TestClient(app).websocket_connect("/v1/voice/session") as ws:
            ws.receive_json()
            ws.send_json({"type": "text", "text": "oi"})
            frames, _ = _collect(ws)
        said = " ".join(f.get("text", "") for f in frames if f["type"] == "delta")
        assert "engine" in said.lower()
        assert "Traceback" not in said

    def test_malformed_frames_are_ignored(self) -> None:
        app = _app(engine=_Engine(), tts=_TTS())
        with TestClient(app).websocket_connect("/v1/voice/session") as ws:
            ws.receive_json()
            ws.send_text("not json")
            ws.send_json({"type": "nonsense"})
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"


class TestBufferBounds:
    def test_an_overlong_utterance_is_refused_not_buffered(self) -> None:
        """A stuck VAD must not be able to grow server memory without limit."""
        app = _app(stt=_STT(), engine=_Engine(), tts=_TTS())
        with TestClient(app).websocket_connect("/v1/voice/session") as ws:
            ws.receive_json()
            ws.send_json({"type": "speech_start"})
            block = b"\x00" * 200_000
            for _ in range(MAX_UTTERANCE_BYTES // len(block) + 2):
                ws.send_bytes(block)
            ws.send_json({"type": "speech_end"})
            frames, _ = _collect(ws)

        errors = [f for f in frames if f["type"] == "error"]
        assert errors
        assert "shorter" in errors[0]["message"]


class TestMounting:
    def test_the_route_is_registered_on_the_real_app(self) -> None:
        from openjarvis.server.api_routes import include_all_routes

        app = fastapi.FastAPI()
        app.state.agent_manager = None
        include_all_routes(app)
        paths = {getattr(r, "path", "") for r in app.routes}
        assert "/v1/voice/session" in paths
