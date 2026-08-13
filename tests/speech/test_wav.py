"""Tests for wrapping raw PCM in a WAV container.

This exists because of a real failure: the voice client streams headerless
PCM16 from the AudioWorklet, and every STT backend rejected it. OpenAI infers
the codec from the filename and ``audio.pcm16`` is not in its allowlist, so
the turn died with an API error instead of producing a transcript.
"""

from __future__ import annotations

import struct
import wave
from io import BytesIO

import pytest

from openjarvis.speech.wav import ensure_container, is_raw_pcm, pcm_to_wav

PCM = b"\x00\x01" * 8_000  # 1s of 16 kHz mono PCM16


class TestHeader:
    def test_the_standard_library_can_read_it_back(self) -> None:
        """The strongest available check: a real WAV parser accepts it."""
        with wave.open(BytesIO(pcm_to_wav(PCM)), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 16_000
            assert handle.readframes(handle.getnframes()) == PCM

    def test_header_is_exactly_44_bytes(self) -> None:
        assert len(pcm_to_wav(PCM)) == len(PCM) + 44

    def test_riff_sizes_match_the_payload(self) -> None:
        blob = pcm_to_wav(PCM)
        assert blob[:4] == b"RIFF"
        assert blob[8:12] == b"WAVE"
        assert struct.unpack("<I", blob[4:8])[0] == 36 + len(PCM)
        assert struct.unpack("<I", blob[40:44])[0] == len(PCM)

    def test_respects_a_different_sample_rate(self) -> None:
        with wave.open(BytesIO(pcm_to_wav(PCM, sample_rate=48_000)), "rb") as handle:
            assert handle.getframerate() == 48_000

    def test_empty_pcm_still_produces_a_valid_file(self) -> None:
        with wave.open(BytesIO(pcm_to_wav(b"")), "rb") as handle:
            assert handle.getnframes() == 0


class TestFormatDetection:
    @pytest.mark.parametrize(
        "fmt", ["pcm", "pcm16", "raw", "s16le", "PCM16", ".pcm16", " pcm "]
    )
    def test_raw_names_are_recognised(self, fmt: str) -> None:
        assert is_raw_pcm(fmt) is True

    @pytest.mark.parametrize("fmt", ["wav", "mp3", "webm", "m4a", "", "ogg"])
    def test_container_names_are_left_alone(self, fmt: str) -> None:
        assert is_raw_pcm(fmt) is False


class TestEnsureContainer:
    def test_raw_pcm_becomes_wav(self) -> None:
        audio, fmt = ensure_container(PCM, "pcm16")
        assert fmt == "wav"
        assert audio[:4] == b"RIFF"

    @pytest.mark.parametrize("fmt", ["webm", "wav", "mp3"])
    def test_containerised_audio_is_untouched(self, fmt: str) -> None:
        audio, out = ensure_container(b"\x1aE\xdf\xa3fake", fmt)
        assert (audio, out) == (b"\x1aE\xdf\xa3fake", fmt)

    def test_does_not_double_wrap_mislabelled_wav(self) -> None:
        """A client sending WAV under the wrong label is worth surviving."""
        already = pcm_to_wav(PCM)
        audio, fmt = ensure_container(already, "pcm16")
        assert audio == already
        assert fmt == "wav"

    def test_empty_audio_is_passed_through(self) -> None:
        assert ensure_container(b"", "pcm16") == (b"", "pcm16")


class TestAgainstTheBackendContract:
    def test_the_result_is_a_format_openai_accepts(self) -> None:
        """`audio.pcm16` was rejected; `audio.wav` is on the allowlist."""
        from openjarvis.speech.openai_whisper import OpenAIWhisperBackend

        _, fmt = ensure_container(PCM, "pcm16")
        assert fmt in OpenAIWhisperBackend.supported_formats(
            OpenAIWhisperBackend.__new__(OpenAIWhisperBackend)
        )
