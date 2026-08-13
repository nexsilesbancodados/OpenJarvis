"""Tests for the continuous voice session — mostly about interruption."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, List

import pytest

from openjarvis.speech.session import VoiceSession, VoiceState

pytestmark = pytest.mark.asyncio

REPLY = "Encontrei tres opcoes. A primeira e a mais barata. A segunda e rapida. "


class Harness:
    """Collects everything the session sends, and lets tests pace generation."""

    def __init__(self, reply: str = REPLY, *, chunk_size: int = 8) -> None:
        self.reply = reply
        self.chunk_size = chunk_size
        self.json: List[dict] = []
        self.audio: List[bytes] = []
        self.synthesized: List[str] = []
        self.tokens_emitted = 0
        self.generation_finished = False
        self.gate: asyncio.Event | None = None

    async def answer(self, text: str) -> AsyncIterator[str]:
        for i in range(0, len(self.reply), self.chunk_size):
            if self.gate is not None:
                await self.gate.wait()
            self.tokens_emitted += 1
            yield self.reply[i : i + self.chunk_size]
            await asyncio.sleep(0)
        self.generation_finished = True

    async def synthesize(self, text: str) -> bytes:
        self.synthesized.append(text)
        await asyncio.sleep(0)
        return f"audio:{text}".encode()

    async def send_json(self, payload: dict) -> None:
        self.json.append(payload)

    async def send_audio(self, data: bytes) -> None:
        self.audio.append(data)

    # -- helpers --------------------------------------------------------

    def session(self) -> VoiceSession:
        return VoiceSession(
            answer=self.answer,
            synthesize=self.synthesize,
            send_json=self.send_json,
            send_audio=self.send_audio,
        )

    def states(self) -> List[str]:
        return [m["value"] for m in self.json if m["type"] == "state"]

    def spoken_chunks(self) -> List[str]:
        return [m["text"] for m in self.json if m["type"] == "speaking"]


async def _settle() -> None:
    for _ in range(60):
        await asyncio.sleep(0)


class TestHappyPath:
    async def test_a_full_turn_speaks_and_returns_to_listening(self) -> None:
        h = Harness()
        s = h.session()
        await s.start()
        await s.on_user_speech_end("quais as opcoes")
        await _settle()

        assert h.audio, "nothing was spoken"
        assert s.state is VoiceState.LISTENING
        assert s.history[-1].interrupted is False
        assert "Encontrei tres opcoes." in s.history[-1].spoken

    async def test_audio_starts_before_generation_finishes(self) -> None:
        """The whole point of sentence chunking."""
        h = Harness()
        s = h.session()
        await s.start()
        task = asyncio.create_task(s.on_user_speech_end("oi"))
        while not h.audio:
            await asyncio.sleep(0)
        assert h.generation_finished is False
        await task
        await _settle()

    async def test_states_follow_the_conversational_loop(self) -> None:
        h = Harness()
        s = h.session()
        await s.start()
        await s.on_user_speech_end("oi")
        await _settle()
        seq = h.states()
        assert seq[0] == "listening"
        assert "thinking" in seq
        assert "speaking" in seq
        assert seq[-1] == "listening"


class TestBargeIn:
    async def test_speaking_stops_when_the_user_starts_talking(self) -> None:
        h = Harness()
        s = h.session()
        await s.start()
        asyncio.create_task(s.on_user_speech_end("quais as opcoes"))
        while not h.audio:
            await asyncio.sleep(0)

        spoken_before = len(h.audio)
        await s.on_user_speech_start()
        await _settle()

        assert len(h.audio) == spoken_before, "kept talking after the interrupt"
        assert s.state is VoiceState.USER_SPEAKING

    async def test_generation_is_actually_cancelled(self) -> None:
        """Not "run to completion and discard" — the tokens must stop."""
        h = Harness()
        s = h.session()
        await s.start()
        asyncio.create_task(s.on_user_speech_end("quais as opcoes"))
        while not h.audio:
            await asyncio.sleep(0)

        await s.on_user_speech_start()
        emitted_at_interrupt = h.tokens_emitted
        await _settle()

        assert h.tokens_emitted == emitted_at_interrupt
        assert h.generation_finished is False

    async def test_history_records_only_what_was_heard(self) -> None:
        h = Harness()
        s = h.session()
        await s.start()
        asyncio.create_task(s.on_user_speech_end("quais as opcoes"))
        while not h.audio:
            await asyncio.sleep(0)
        await s.on_user_speech_start()
        await _settle()

        record = s.history[-1]
        assert record.interrupted is True
        assert record.spoken, "the user did hear something"
        assert "A segunda" not in record.spoken
        # Whatever was synthesised but never reached the speaker is kept
        # rather than silently lost, and never counted as spoken. It may be
        # empty — generation is cancelled at the interrupt, so the later
        # sentences are often never produced at all.
        if record.unspoken:
            assert record.unspoken not in record.spoken

    async def test_synthesised_but_unplayed_text_is_kept(self) -> None:
        """When chunks do queue up behind the interrupt, they are retrievable.

        This is the "como voce ia dizer?" case — distinct from text that was
        never generated because the model was stopped in time.
        """

        class SlowHarness(Harness):
            async def synthesize(self, text: str) -> bytes:
                self.synthesized.append(text)
                await asyncio.sleep(0.05)
                return f"audio:{text}".encode()

        h = SlowHarness()
        s = h.session()
        await s.start()
        asyncio.create_task(s.on_user_speech_end("quais as opcoes"))
        # Wait until the first sentence has played and the second is still
        # being synthesised — that second one is the queued-but-unheard text.
        while not (h.audio and len(h.synthesized) >= 2):
            await asyncio.sleep(0.01)
        await s.on_user_speech_start()
        await asyncio.sleep(0.15)

        record = s.history[-1]
        assert record.interrupted is True
        assert record.spoken, "the first sentence was heard"
        assert record.unspoken, "the queued sentence should be retrievable"
        assert record.unspoken not in record.spoken

    async def test_an_interrupt_is_announced_to_the_client(self) -> None:
        h = Harness()
        s = h.session()
        await s.start()
        asyncio.create_task(s.on_user_speech_end("oi"))
        while not h.audio:
            await asyncio.sleep(0)
        await s.on_user_speech_start()
        await _settle()
        assert any(m["type"] == "interrupted" for m in h.json)

    async def test_the_next_turn_runs_normally_after_an_interrupt(self) -> None:
        h = Harness()
        s = h.session()
        await s.start()
        asyncio.create_task(s.on_user_speech_end("primeira"))
        while not h.audio:
            await asyncio.sleep(0)
        await s.on_user_speech_start()
        await s.on_user_speech_end("na verdade, a segunda")
        await _settle()

        assert len(s.history) == 2
        assert s.history[-1].interrupted is False
        assert s.state is VoiceState.LISTENING

    async def test_interrupting_twice_does_not_wedge_the_session(self) -> None:
        h = Harness()
        s = h.session()
        await s.start()
        asyncio.create_task(s.on_user_speech_end("um"))
        while not h.audio:
            await asyncio.sleep(0)
        await s.on_user_speech_start()
        await s.on_user_speech_start()
        await s.on_user_speech_end("dois")
        await _settle()
        assert s.state is VoiceState.LISTENING

    async def test_interrupt_while_idle_is_harmless(self) -> None:
        h = Harness()
        s = h.session()
        await s.start()
        await s.on_user_speech_start()
        assert s.state is VoiceState.USER_SPEAKING
        assert s.history == []


class TestLateWork:
    async def test_audio_synthesized_after_an_interrupt_is_not_played(self) -> None:
        """Synthesis in flight when the interrupt lands must be dropped."""

        class SlowHarness(Harness):
            async def synthesize(self, text: str) -> bytes:
                self.synthesized.append(text)
                await asyncio.sleep(0.05)
                return f"audio:{text}".encode()

        h = SlowHarness()
        s = h.session()
        await s.start()
        asyncio.create_task(s.on_user_speech_end("oi"))
        while not h.synthesized:
            await asyncio.sleep(0)

        await s.on_user_speech_start()
        await asyncio.sleep(0.15)

        assert h.audio == [], "spoke over the user with late audio"


class TestControls:
    async def test_cancel_is_not_recorded_as_an_interruption(self) -> None:
        h = Harness()
        s = h.session()
        await s.start()
        asyncio.create_task(s.on_user_speech_end("oi"))
        while not h.audio:
            await asyncio.sleep(0)
        await s.on_cancel()
        await _settle()
        assert s.history[-1].interrupted is False
        assert s.state is VoiceState.LISTENING

    async def test_muting_stops_the_current_turn(self) -> None:
        h = Harness()
        s = h.session()
        await s.start()
        asyncio.create_task(s.on_user_speech_end("oi"))
        while not h.audio:
            await asyncio.sleep(0)
        await s.set_muted(True)
        await _settle()
        count = len(h.audio)

        await s.on_user_speech_end("ainda esta ai?")
        await _settle()
        assert len(h.audio) == count, "answered while muted"

    async def test_empty_transcription_does_not_start_a_turn(self) -> None:
        h = Harness()
        s = h.session()
        await s.start()
        await s.on_user_speech_end("   ")
        await _settle()
        assert s.history == []
        assert s.state is VoiceState.LISTENING


class TestFailure:
    async def test_a_generator_error_is_spoken_not_traced(self) -> None:
        class Broken(Harness):
            async def answer(self, text: str):
                raise RuntimeError("engine exploded")
                yield ""  # pragma: no cover

        h = Broken()
        s = h.session()
        await s.start()
        await s.on_user_speech_end("oi")
        await _settle()

        errors = [m for m in h.json if m["type"] == "error"]
        assert errors
        assert "Traceback" not in errors[0]["message"]
        assert s.state is VoiceState.LISTENING, "must recover, not stay stuck"

    async def test_a_dead_transport_does_not_wedge_the_turn(self) -> None:
        h = Harness()
        s = h.session()

        async def broken(_payload: dict) -> None:
            raise ConnectionError("socket closed")

        s.send_json = broken
        await s.start()
        await s.on_user_speech_end("oi")
        await _settle()
        assert s.state is VoiceState.LISTENING


class TestFluidity:
    """ "Instantanea e fluida" is a latency property, so it gets tests."""

    async def test_synthesis_overlaps_playback(self) -> None:
        """Sentence N+1 must be synthesising while sentence N is spoken.

        Awaiting each chunk before starting the next leaves a gap between
        sentences exactly as long as the TTS call — the difference between
        fluid speech and stilted speech.
        """
        overlapped = False

        class TrackingHarness(Harness):
            def __init__(self) -> None:
                super().__init__()
                self.in_flight = 0

            async def synthesize(self, text: str) -> bytes:
                nonlocal overlapped
                self.in_flight += 1
                self.synthesized.append(text)
                await asyncio.sleep(0.02)
                if self.in_flight > 1:
                    overlapped = True
                self.in_flight -= 1
                return f"audio:{text}".encode()

        h = TrackingHarness()
        s = h.session()
        await s.start()
        await s.on_user_speech_end("quais as opcoes")
        await asyncio.sleep(0.3)
        assert overlapped, "synthesis was serial; sentences will have gaps"

    async def test_speaking_state_only_once_audio_is_going_out(self) -> None:
        """The orb must not claim "speaking" while TTS is still working."""

        class SlowHarness(Harness):
            async def synthesize(self, text: str) -> bytes:
                self.synthesized.append(text)
                await asyncio.sleep(0.05)
                return b"audio"

        h = SlowHarness()
        s = h.session()
        await s.start()
        asyncio.create_task(s.on_user_speech_end("oi"))
        while not h.synthesized:
            await asyncio.sleep(0.005)
        assert "speaking" not in h.states()
        await asyncio.sleep(0.2)
        assert "speaking" in h.states()

    async def test_speaking_over_a_thinking_assistant_also_interrupts(self) -> None:
        """The user replacing the request before any audio is still barge-in."""

        class SlowHarness(Harness):
            async def synthesize(self, text: str) -> bytes:
                self.synthesized.append(text)
                await asyncio.sleep(0.05)
                return b"audio"

        h = SlowHarness()
        s = h.session()
        await s.start()
        asyncio.create_task(s.on_user_speech_end("pergunta antiga"))
        while not h.synthesized:
            await asyncio.sleep(0.005)

        await s.on_user_speech_start()
        await asyncio.sleep(0.2)
        assert h.audio == [], "answered a question the user had replaced"
