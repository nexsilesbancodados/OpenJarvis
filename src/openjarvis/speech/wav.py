"""Wrap raw PCM in a WAV container.

The voice client streams headerless 16 kHz mono PCM16 — the natural output of
an AudioWorklet, and the cheapest thing to send over a socket. Every STT
backend, though, wants a recognisable container: the OpenAI API infers the
codec from the filename and rejects anything outside its allowlist, and
faster-whisper hands the bytes to a demuxer. Sending raw samples produces a
confusing API error rather than a transcript, so the header is added here,
once, at the boundary between transport and recognition.

44 bytes of header, no re-encoding, no dependency.
"""

from __future__ import annotations

import struct

__all__ = ["RAW_PCM_FORMATS", "is_raw_pcm", "pcm_to_wav", "ensure_container"]

#: Format names the client may use for headerless little-endian PCM16.
RAW_PCM_FORMATS = frozenset({"pcm", "pcm16", "raw", "s16le", "pcm_s16le"})

_RIFF = b"RIFF"
_WAVE = b"WAVE"


def is_raw_pcm(fmt: str) -> bool:
    return (fmt or "").strip().lower().lstrip(".") in RAW_PCM_FORMATS


def pcm_to_wav(
    pcm: bytes,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Prepend a canonical 44-byte WAV header to ``pcm``."""
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    header = b"".join(
        (
            _RIFF,
            struct.pack("<I", 36 + len(pcm)),
            _WAVE,
            b"fmt ",
            struct.pack("<I", 16),  # PCM header size
            struct.pack("<H", 1),  # format 1 = uncompressed PCM
            struct.pack("<H", channels),
            struct.pack("<I", sample_rate),
            struct.pack("<I", byte_rate),
            struct.pack("<H", block_align),
            struct.pack("<H", sample_width * 8),
            b"data",
            struct.pack("<I", len(pcm)),
        )
    )
    return header + pcm


def ensure_container(audio: bytes, fmt: str, *, sample_rate: int = 16_000):
    """Return ``(audio, format)`` in a form a speech backend will accept.

    Raw PCM becomes WAV; anything already containerised is passed through
    untouched. Audio that merely *claims* to be raw but already carries a RIFF
    header is left alone too — double-wrapping would corrupt it, and a client
    sending WAV under the wrong label is a mistake worth surviving.
    """
    if not audio or not is_raw_pcm(fmt):
        return audio, fmt
    if audio[:4] == _RIFF:
        return audio, "wav"
    return pcm_to_wav(audio, sample_rate=sample_rate), "wav"
