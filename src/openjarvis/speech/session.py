"""Continuous voice-turn orchestration, independent of any transport.

The conversational loop this drives:

    listening → speech detected → understanding → answering → speaking →
    listening

with the property the user actually asked for: **speaking is interruptible**.
When the client reports that the user started talking, the in-flight answer is
cancelled, playback stops, and only the audio that really reached the speaker
is written to history (see :mod:`openjarvis.speech.turn`).

Everything here is transport-agnostic on purpose. The WebSocket route is a
thin adapter over this class, which means the interesting behaviour — what
happens when someone interrupts mid-sentence, what happens when they interrupt
twice, what history is left behind — is testable without audio hardware, a
browser, or a socket.

Two rules the implementation exists to enforce:

* **Cancellation is cooperative and immediate.** A barge-in must stop the
  generator at the next token, not run it to completion and discard the
  output — otherwise the model keeps paying for tokens nobody hears and the
  next turn queues behind the old one.
* **Late chunks from a cancelled turn are dropped, not spoken.** Synthesis
  already in flight when the interrupt lands will still come back; playing it
  would mean the assistant talks over the user immediately after being told
  to stop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, List, Optional, Protocol

from openjarvis.speech.chunking import SentenceAccumulator
from openjarvis.speech.turn import SpokenTurn

logger = logging.getLogger(__name__)

__all__ = ["VoiceState", "VoiceSession", "TurnRecord", "Synthesizer"]


class VoiceState(str, Enum):
    """What the assistant is doing. Drives the orb, one-to-one."""

    IDLE = "idle"
    LISTENING = "listening"
    #: The client's VAD says the user is mid-utterance.
    USER_SPEAKING = "user_speaking"
    THINKING = "thinking"
    TOOL = "tool"
    SPEAKING = "speaking"
    ERROR = "error"


class Synthesizer(Protocol):
    """Turns one chunk of text into audio bytes."""

    async def __call__(self, text: str) -> bytes: ...


@dataclass
class TurnRecord:
    """What a finished turn left behind, for history and diagnostics."""

    user_text: str
    spoken: str
    unspoken: str
    interrupted: bool
    first_audio_ms: Optional[float] = None
    total_ms: Optional[float] = None


@dataclass
class VoiceSession:
    """One continuous voice conversation.

    Parameters
    ----------
    answer:
        Async generator factory: given the user's text, yields reply deltas.
        Cancelling the task running it must stop generation.
    synthesize:
        Turns a sentence into audio bytes.
    send_json / send_audio:
        Transport callbacks. Failures are logged, never raised into the loop —
        a dead socket must not leave the session wedged mid-turn.
    """

    answer: Callable[[str], "AsyncIterator[str]"]  # noqa: F821
    synthesize: Synthesizer
    send_json: Callable[[dict], Awaitable[None]]
    send_audio: Callable[[bytes], Awaitable[None]]

    state: VoiceState = VoiceState.IDLE
    history: List[TurnRecord] = field(default_factory=list)
    muted: bool = False

    _turn: Optional[SpokenTurn] = None
    _task: Optional[asyncio.Task] = None
    _generation: int = 0
    _started_at: float = 0.0
    _first_audio_at: Optional[float] = None
    _user_text: str = ""

    # -- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        await self._set_state(VoiceState.LISTENING)

    async def close(self) -> None:
        await self._abort_turn()
        self.state = VoiceState.IDLE

    # -- inbound events -------------------------------------------------

    async def on_user_speech_start(self) -> None:
        """The client's VAD detected speech. This is the barge-in trigger.

        Any turn in flight is abandoned — not only one that has reached the
        speaker. Talking over the assistant while it is still thinking is the
        same act as talking over it mid-sentence: the user is replacing the
        request, and finishing the old answer would speak into a question
        nobody asked any more.
        """
        if self._turn_in_flight:
            await self._interrupt()
        if not self.muted:
            await self._set_state(VoiceState.USER_SPEAKING)

    @property
    def _turn_in_flight(self) -> bool:
        if self._task is not None and not self._task.done():
            return True
        return self._turn is not None and not self._turn.finished

    async def on_user_speech_end(self, text: str) -> None:
        """The utterance closed and was transcribed."""
        text = (text or "").strip()
        if not text or self.muted:
            await self._set_state(VoiceState.LISTENING)
            return
        await self._abort_turn()
        self._task = asyncio.create_task(self._run_turn(text))

    async def on_cancel(self) -> None:
        """Stopped deliberately — escape, a button, a new request."""
        if self._turn is not None and not self._turn.finished:
            spoken = self._turn.cancel(played_ratio=0.0)
            self._record(spoken, interrupted=False)
        await self._abort_turn()
        await self._set_state(VoiceState.LISTENING)

    async def set_muted(self, muted: bool) -> None:
        self.muted = muted
        if muted:
            await self.on_cancel()
        await self._emit({"type": "muted", "value": muted})

    # -- the turn -------------------------------------------------------

    async def _run_turn(self, text: str) -> None:
        generation = self._generation
        self._user_text = text
        self._turn = SpokenTurn()
        self._started_at = time.monotonic()
        self._first_audio_at = None
        accumulator = SentenceAccumulator()

        # Producer and consumer run concurrently. An earlier version drained
        # inline from the producer, gated on `len(pending) > 1` — which meant
        # the first sentence could not be spoken until the *second* one had
        # been dispatched, so time-to-first-audio silently included generating
        # a sentence the user did not need to hear yet. Playback must depend
        # on chunk N being ready, never on chunk N+1 existing.
        pending: asyncio.Queue = asyncio.Queue()
        # Bounds how far synthesis may run ahead of the speaker. More than
        # this buys nothing once playback is the bottleneck, and every extra
        # chunk is TTS spend a barge-in may throw away.
        lookahead = asyncio.Semaphore(2)

        async def consume() -> None:
            while True:
                item = await pending.get()
                if item is None:
                    return
                try:
                    await self._play_next(item, generation)
                finally:
                    lookahead.release()

        consumer = asyncio.create_task(consume())

        async def dispatch(chunk: str) -> None:
            assert self._turn is not None
            await lookahead.acquire()
            index = self._turn.enqueue(chunk)
            task = asyncio.create_task(self._synthesize(chunk, index))
            await pending.put((index, task))

        try:
            await self._set_state(VoiceState.THINKING)
            async for delta in self.answer(text):
                if self._stale(generation):
                    return
                if delta:
                    await self._emit({"type": "delta", "text": delta})
                for chunk in accumulator.feed(delta):
                    if self._stale(generation):
                        return
                    await dispatch(chunk)

            for chunk in accumulator.flush():
                if self._stale(generation):
                    return
                await dispatch(chunk)

            await pending.put(None)
            await consumer

            if self._stale(generation):
                consumer.cancel()
                return
            spoken = self._turn.complete()
            self._record(spoken, interrupted=False)
            await self._emit({"type": "turn_complete", "text": spoken})
            await self._set_state(VoiceState.LISTENING)

        except asyncio.CancelledError:
            consumer.cancel()
            raise
        except Exception as exc:
            consumer.cancel()
            logger.exception("voice turn failed")
            await self._emit(
                {
                    "type": "error",
                    # Spoken aloud, so it has to be a sentence, not a trace.
                    "message": "Something went wrong while I was answering.",
                    "detail": str(exc),
                }
            )
            await self._set_state(VoiceState.ERROR)
            await self._set_state(VoiceState.LISTENING)

    async def _synthesize(self, chunk: str, index: int) -> tuple[str, bytes]:
        return chunk, await self.synthesize(chunk)

    async def _play_next(
        self,
        item: tuple[int, asyncio.Task],
        generation: int,
    ) -> None:
        """Await one synthesis and play it, unless the turn moved on."""
        assert self._turn is not None
        index, task = item
        try:
            chunk, audio = await task
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("synthesis failed for chunk %d", index)
            return

        # Synthesis is slow enough that an interrupt very often lands while it
        # is in flight. Playing this now would mean talking over the user
        # immediately after being told to stop.
        if self._stale(generation) or self._turn.finished:
            return

        await self._set_state(VoiceState.SPEAKING)
        self._turn.start_playing(index)
        await self._emit({"type": "speaking", "text": chunk, "chunk": index})
        await self._send_audio(audio)
        if self._first_audio_at is None:
            self._first_audio_at = time.monotonic()
        self._turn.finish_playing(index)

    async def _interrupt(self, played_ratio: float = 0.0) -> None:
        if self._turn is not None and not self._turn.finished:
            spoken = self._turn.interrupt(played_ratio=played_ratio)
            self._record(spoken, interrupted=True)
            await self._emit({"type": "interrupted", "spoken": spoken})
        await self._abort_turn()

    async def _abort_turn(self) -> None:
        """Bump the generation and stop the task, so late work is ignored.

        The generation bump matters as much as the cancel: synthesis tasks
        already awaiting a network round-trip will still resolve, and the
        stale check is what stops their audio from being played.
        """
        self._generation += 1
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def _stale(self, generation: int) -> bool:
        return generation != self._generation

    def _record(self, spoken: str, *, interrupted: bool) -> None:
        assert self._turn is not None
        now = time.monotonic()
        self.history.append(
            TurnRecord(
                user_text=self._user_text,
                spoken=spoken,
                unspoken=self._turn.unspoken_text(),
                interrupted=interrupted,
                first_audio_ms=(
                    (self._first_audio_at - self._started_at) * 1000
                    if self._first_audio_at
                    else None
                ),
                total_ms=(now - self._started_at) * 1000,
            )
        )

    # -- transport ------------------------------------------------------

    async def _set_state(self, state: VoiceState) -> None:
        if state is self.state:
            return
        self.state = state
        await self._emit({"type": "state", "value": state.value})

    async def _emit(self, payload: dict) -> None:
        try:
            await self.send_json(payload)
        except Exception:
            logger.debug("voice send_json failed", exc_info=True)

    async def _send_audio(self, audio: bytes) -> None:
        try:
            await self.send_audio(audio)
        except Exception:
            logger.debug("voice send_audio failed", exc_info=True)
