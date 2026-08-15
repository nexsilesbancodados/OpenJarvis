"""Split spoken text into synthesis-sized pieces.

A voice turn is serial: transcribe, generate, synthesize, play. Synthesising
the whole reply before the first sound means time-to-first-audio is the sum of
every stage, which is what makes an assistant feel like software rather than a
conversation. Splitting the reply into sentences and synthesising them in
order collapses that to "however long the first sentence takes".

The rules here are deliberately conservative, because a wrong split is
audible:

* Never break inside an abbreviation, a decimal, or an ellipsis. "Dr. Silva"
  and "R$ 1.500,00" must stay whole — a pause mid-number is worse than a
  slightly long first chunk.
* Never emit a chunk so short that the synthesiser produces a clipped
  fragment; merge it forward instead.
* Fall back to clause boundaries, then to whitespace, when a "sentence" runs
  far past what a listener would wait for.

Text arriving token-by-token is handled by :class:`SentenceAccumulator`, which
only releases a sentence once it is certain no further token can change the
decision.
"""

from __future__ import annotations

import re
from typing import Iterator, List

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MIN_CHARS",
    "SentenceAccumulator",
    "split_for_speech",
]

# Below this, a chunk is merged with a neighbour: synthesisers give clipped,
# oddly-prosodied output for one- and two-word fragments like "Ok." Kept low
# on purpose — "Deixa eu pensar..." is a perfectly good thing to say on its
# own, and a high threshold would glue real sentences together and delay the
# first audio, which is the problem this module exists to solve.
DEFAULT_MIN_CHARS = 12
# Above this we split at the best available boundary even mid-sentence. A
# listener will not wait for a 400-character sentence to finish synthesising.
DEFAULT_MAX_CHARS = 240

_TERMINATORS = ".!?…。！？"

# Abbreviations whose trailing period does not end a sentence. Portuguese and
# English, since the persona speaks both.
_ABBREVIATIONS = frozenset(
    {
        "dr",
        "dra",
        "sr",
        "sra",
        "srta",
        "prof",
        "profa",
        "eng",
        "adv",
        "av",
        "r",
        "ltda",
        "cia",
        "etc",
        "ex",
        "obs",
        "pág",
        "pag",
        "fl",
        "mr",
        "mrs",
        "ms",
        "st",
        "jr",
        "vs",
        "approx",
        "dept",
        "est",
        "fig",
        "inc",
        "no",
        "vol",
        "e.g",
        "i.e",
    }
)

_CLAUSE_BREAKS = ";:,—–"

# A terminator ends a sentence when whitespace follows it. The whitespace is
# what distinguishes "certa. Achei" from "3.14" and "openjarvis.io", so no
# lookahead at the next character is needed — and requiring one would be
# actively wrong for streaming, where the buffer often ends right after the
# space and the sentence would never be released.
_BOUNDARY = re.compile(r"([" + re.escape(_TERMINATORS) + r"]+[\"'”’\)\]]*)(\s+)")


def _ends_with_abbreviation(text: str) -> bool:
    tail = text.rstrip()
    if not tail.endswith("."):
        return False
    word = re.split(r"[\s(\[]", tail[:-1])[-1].lower()
    if not word:
        return False
    if word in _ABBREVIATIONS:
        return True
    # Single letter followed by a period: an initial, as in "J. Silva".
    if len(word) == 1 and word.isalpha():
        return True
    # Digit before the period: a decimal or a numbered item.
    return word[-1].isdigit()


def _split_sentences(text: str) -> List[str]:
    """Split on sentence terminators, respecting abbreviations."""
    pieces: List[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end(1)
        candidate = text[start:end]
        if _ends_with_abbreviation(candidate):
            continue
        pieces.append(candidate.strip())
        start = match.end(2)
    remainder = text[start:].strip()
    if remainder:
        pieces.append(remainder)
    return [p for p in pieces if p]


def _hard_wrap(piece: str, max_chars: int) -> List[str]:
    """Break an over-long piece at the best boundary available."""
    if len(piece) <= max_chars:
        return [piece]

    out: List[str] = []
    rest = piece
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max(window.rfind(c) for c in _CLAUSE_BREAKS)
        if cut < max_chars // 3:
            cut = window.rfind(" ")
        if cut < max_chars // 3:
            cut = max_chars - 1
        out.append(rest[: cut + 1].strip())
        rest = rest[cut + 1 :].lstrip()
    if rest:
        out.append(rest)
    return [p for p in out if p]


def split_for_speech(
    text: str,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[str]:
    """Split ``text`` into chunks suitable for sequential synthesis.

    Chunks are returned in reading order and, joined with a space, contain the
    same words as the input. An empty or whitespace-only input yields ``[]``.
    """
    text = (text or "").strip()
    if not text:
        return []

    pieces: List[str] = []
    for sentence in _split_sentences(text):
        pieces.extend(_hard_wrap(sentence, max_chars))
    return _merge_runts(pieces, min_chars=min_chars, max_chars=max_chars)


def _merge_runts(pieces: List[str], *, min_chars: int, max_chars: int) -> List[str]:
    """Absorb fragments too short to synthesise cleanly into a neighbour.

    Forward first: "Ok." belongs at the head of the sentence that follows it,
    which is how it would be spoken. Only a trailing runt with nothing after
    it merges backward. Nothing is ever dropped, and a lone short chunk is
    returned as-is — there is no neighbour to merge with, and saying "Sim."
    is better than saying nothing.
    """
    out: List[str] = []
    carry = ""
    for piece in pieces:
        if carry:
            piece = f"{carry} {piece}"
            carry = ""
        if len(piece) < min_chars:
            carry = piece
            continue
        out.append(piece)

    if carry:
        if out and len(out[-1]) + len(carry) + 1 <= max_chars:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


class SentenceAccumulator:
    """Release complete sentences from a token stream, as soon as they are safe.

    Feed it deltas from a streaming LLM; it yields chunks that will not change
    if more tokens arrive. A sentence is held back until the terminator is
    followed by whitespace, because ``"1."`` may still become ``"1.5"``.

    Call :meth:`flush` when the stream ends to release whatever remains.
    """

    def __init__(
        self,
        *,
        min_chars: int = DEFAULT_MIN_CHARS,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self._buffer = ""
        self._min = min_chars
        self._max = max_chars

    def feed(self, delta: str) -> Iterator[str]:
        """Absorb a token and yield any chunk that is now final."""
        if not delta:
            return
        self._buffer += delta

        while True:
            release = self._next_release()
            if release is None:
                return
            head, self._buffer = release
            if head:
                yield head

    def _next_release(self) -> tuple[str, str] | None:
        # Sentence boundary: needs trailing whitespace to be certain.
        last: int | None = None
        for match in _BOUNDARY.finditer(self._buffer):
            candidate = self._buffer[: match.end(1)]
            if not _ends_with_abbreviation(candidate):
                last = match.end(2)
        if last is not None:
            head = self._buffer[:last].strip()
            if len(head) >= self._min:
                return head, self._buffer[last:]

        # No sentence yet, but the buffer is long enough that waiting costs
        # more than an imperfect break.
        if len(self._buffer) > self._max:
            pieces = _hard_wrap(self._buffer, self._max)
            if len(pieces) > 1:
                head = pieces[0]
                return head, self._buffer[len(head) :].lstrip()
        return None

    def flush(self) -> List[str]:
        """Return the remaining chunks and reset the buffer."""
        remaining, self._buffer = self._buffer.strip(), ""
        if not remaining:
            return []
        return split_for_speech(remaining, min_chars=self._min, max_chars=self._max)
