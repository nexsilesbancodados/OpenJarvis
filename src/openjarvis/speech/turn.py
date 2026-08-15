"""Track a spoken turn so an interruption leaves honest history behind.

Stopping the audio is the easy half of barge-in. The hard half is what the
conversation remembers afterwards. If the model generated

    "Encontrei tres opcoes. A primeira e a mais barata. A segunda e a mais
     rapida. A terceira e a mais completa."

and the user cut in after the second sentence, storing the whole paragraph as
the assistant's turn makes the next reply incoherent: the user says "pega a
segunda" meaning the second of the *two* they heard, while the model believes
it already offered three. Storing nothing is just as wrong — the user did hear
something and is referring to it.

What has to be recorded is the prefix that actually reached the speaker. This
module tracks that, chunk by chunk, and is deliberately free of I/O so the
rule can be tested without audio hardware:

    turn = SpokenTurn()
    turn.enqueue("Encontrei tres opcoes.")      # queued for synthesis
    turn.start_playing(0); turn.finish_playing(0)
    turn.enqueue("A primeira e a mais barata.")
    turn.start_playing(1)
    turn.interrupt(played_ratio=0.4)            # user spoke mid-sentence
    turn.spoken_text()  # -> "Encontrei tres opcoes. A primeira e a"

A partially played chunk is truncated on a word boundary. Being slightly
conservative there is deliberate: claiming the assistant said a word it did
not is worse than dropping one it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

__all__ = ["ChunkState", "SpokenChunk", "SpokenTurn", "TurnOutcome"]


class ChunkState(str, Enum):
    """Where one synthesis chunk got to before the turn ended."""

    QUEUED = "queued"
    PLAYING = "playing"
    PLAYED = "played"
    #: Cut short by the user speaking, or by an explicit cancel.
    CUT = "cut"
    #: Never reached the speaker — synthesis or playback was abandoned.
    DROPPED = "dropped"


class TurnOutcome(str, Enum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


@dataclass
class SpokenChunk:
    index: int
    text: str
    state: ChunkState = ChunkState.QUEUED
    #: Fraction of this chunk that reached the speaker, 0.0–1.0.
    played_ratio: float = 0.0

    @property
    def spoken(self) -> str:
        """The part of this chunk the user actually heard."""
        if self.state is ChunkState.PLAYED:
            return self.text
        if self.state is not ChunkState.CUT:
            return ""
        return _truncate_on_word(self.text, self.played_ratio)


def _truncate_on_word(text: str, ratio: float) -> str:
    """Cut ``text`` at ``ratio`` of its length, rounding down to a word."""
    if ratio <= 0.0:
        return ""
    if ratio >= 1.0:
        return text
    cut = int(len(text) * ratio)
    head = text[:cut]
    if cut < len(text) and not text[cut].isspace():
        # Mid-word: drop the partial word rather than invent a whole one.
        space = head.rfind(" ")
        head = head[:space] if space > 0 else ""
    return head.strip()


@dataclass
class SpokenTurn:
    """One assistant speaking turn, tracked chunk by chunk."""

    chunks: List[SpokenChunk] = field(default_factory=list)
    outcome: Optional[TurnOutcome] = None

    # -- building -------------------------------------------------------

    def enqueue(self, text: str) -> int:
        """Add a chunk about to be synthesised. Returns its index."""
        if self.finished:
            raise RuntimeError("cannot enqueue onto a finished turn")
        index = len(self.chunks)
        self.chunks.append(SpokenChunk(index=index, text=text))
        return index

    def start_playing(self, index: int) -> None:
        self.chunks[index].state = ChunkState.PLAYING

    def finish_playing(self, index: int) -> None:
        chunk = self.chunks[index]
        chunk.state = ChunkState.PLAYED
        chunk.played_ratio = 1.0

    # -- ending ---------------------------------------------------------

    def interrupt(self, *, played_ratio: float = 0.0) -> str:
        """The user started speaking. Stop, and return what was really said.

        ``played_ratio`` describes the chunk that was mid-flight. Anything
        still queued never reached the speaker and is dropped, not recorded.
        """
        return self._end(TurnOutcome.INTERRUPTED, played_ratio)

    def cancel(self, *, played_ratio: float = 0.0) -> str:
        """Stopped deliberately (mute, escape, new request) rather than by voice."""
        return self._end(TurnOutcome.CANCELLED, played_ratio)

    def complete(self) -> str:
        """The turn finished on its own."""
        for chunk in self.chunks:
            if chunk.state in (ChunkState.QUEUED, ChunkState.PLAYING):
                chunk.state = ChunkState.PLAYED
                chunk.played_ratio = 1.0
        self.outcome = TurnOutcome.COMPLETED
        return self.spoken_text()

    def _end(self, outcome: TurnOutcome, played_ratio: float) -> str:
        ratio = min(1.0, max(0.0, played_ratio))
        for chunk in self.chunks:
            if chunk.state is ChunkState.PLAYING:
                chunk.state = ChunkState.CUT
                chunk.played_ratio = ratio
            elif chunk.state is ChunkState.QUEUED:
                chunk.state = ChunkState.DROPPED
        self.outcome = outcome
        return self.spoken_text()

    # -- reading --------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.outcome is not None

    @property
    def was_interrupted(self) -> bool:
        return self.outcome is TurnOutcome.INTERRUPTED

    def spoken_text(self) -> str:
        """Everything the user actually heard, in order."""
        return " ".join(c.spoken for c in self.chunks if c.spoken).strip()

    def unspoken_text(self) -> str:
        """What was generated but never heard.

        Worth keeping separate rather than discarding: "como voce ia dizer?"
        is a reasonable thing to ask after interrupting.
        """
        parts: List[str] = []
        for chunk in self.chunks:
            if chunk.state is ChunkState.DROPPED:
                parts.append(chunk.text)
            elif chunk.state is ChunkState.CUT:
                spoken = chunk.spoken
                rest = chunk.text[len(spoken) :].strip() if spoken else chunk.text
                if rest:
                    parts.append(rest)
        return " ".join(parts).strip()

    def history_entry(self) -> str:
        """The assistant message to store in conversation history.

        An interrupted turn is marked, so the model can see it was cut off
        instead of assuming the user ignored a complete answer.
        """
        spoken = self.spoken_text()
        if self.was_interrupted and spoken:
            return f"{spoken} —"
        return spoken
