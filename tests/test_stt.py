"""Tests for stt.py — Whisper.cpp STT engine.

These tests cover:
- Public API surface
- transcribe() with empty / silent audio returns TranscribeResult(text="", no_speech_prob=1.0)
- transcribe() returns a TranscribeResult for normal audio
- TranscribeResult.text is a str; TranscribeResult.no_speech_prob is a float in [0, 1]
- Artifact stripping (BLANK_AUDIO tokens, leading/trailing whitespace)
- Model configurability (model name parameter)
- All output goes to stderr (stdout stays clean)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile

import numpy as np
import pytest

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _silence(seconds: float = 1.0) -> np.ndarray:
    """Return a silent 16 kHz float32 array."""
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def _noise(seconds: float = 0.5) -> np.ndarray:
    """Return low-amplitude noise at 16 kHz float32."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(SAMPLE_RATE * seconds)) * 0.01).astype(np.float32)


# ---------------------------------------------------------------------------
# Empty / silent audio
# ---------------------------------------------------------------------------


class TestTranscribeSilentAudio:
    """transcribe() must return TranscribeResult with empty text for empty/silent input."""

    def test_empty_array_returns_empty_text(self):
        from lazy_claude.stt import transcribe
        audio = np.array([], dtype=np.float32)
        result = transcribe(audio)
        assert result.text == "", f"Expected text='', got: {result.text!r}"

    def test_empty_array_returns_high_no_speech_prob(self):
        from lazy_claude.stt import transcribe
        audio = np.array([], dtype=np.float32)
        result = transcribe(audio)
        assert result.no_speech_prob == 1.0

    def test_zero_samples_returns_empty_text(self):
        from lazy_claude.stt import transcribe
        audio = np.zeros(0, dtype=np.float32)
        result = transcribe(audio)
        assert result.text == ""

    def test_silence_returns_transcribe_result(self):
        """Pure silence should return a TranscribeResult (text possibly empty)."""
        from lazy_claude.stt import transcribe, TranscribeResult
        audio = _silence(1.0)
        result = transcribe(audio)
        assert isinstance(result, TranscribeResult)
        assert isinstance(result.text, str)

    def test_transcribe_returns_transcribe_result_type(self):
        from lazy_claude.stt import transcribe, TranscribeResult
        result = transcribe(_silence(0.5))
        assert isinstance(result, TranscribeResult)


# ---------------------------------------------------------------------------
# Artifact stripping
# ---------------------------------------------------------------------------


class TestArtifactStripping:
    """Verify that common whisper hallucination tokens are stripped."""

    def test_strip_blank_audio_token(self):
        from lazy_claude.stt import _strip_artifacts
        assert _strip_artifacts("[BLANK_AUDIO]") == ""

    def test_strip_blank_audio_token_with_whitespace(self):
        from lazy_claude.stt import _strip_artifacts
        assert _strip_artifacts("  [BLANK_AUDIO]  ") == ""

    def test_strip_leading_trailing_whitespace(self):
        from lazy_claude.stt import _strip_artifacts
        assert _strip_artifacts("  hello world  ") == "hello world"

    def test_strip_multiple_blank_audio_tokens(self):
        from lazy_claude.stt import _strip_artifacts
        assert _strip_artifacts("[BLANK_AUDIO] [BLANK_AUDIO]") == ""

    def test_strip_music_token(self):
        from lazy_claude.stt import _strip_artifacts
        # Common whisper hallucination for background music
        assert _strip_artifacts("[MUSIC]") == ""

    def test_preserve_normal_text(self):
        from lazy_claude.stt import _strip_artifacts
        assert _strip_artifacts("hello world") == "hello world"

    def test_strip_repeated_phrases(self):
        from lazy_claude.stt import _strip_artifacts
        # Whisper sometimes hallucinates repeated phrases
        result = _strip_artifacts("hello hello hello hello hello")
        # The result should not contain 5 consecutive repetitions of the same word
        assert result != "hello hello hello hello hello" or result == ""

    def test_strip_thank_you_for_watching_artifact(self):
        from lazy_claude.stt import _strip_artifacts
        # Common hallucination at end of audio
        assert _strip_artifacts("Thank you for watching.") == ""

    def test_strip_subscribe_artifact(self):
        from lazy_claude.stt import _strip_artifacts
        assert _strip_artifacts("Subscribe to our channel.") == ""

    def test_normal_speech_not_stripped(self):
        from lazy_claude.stt import _strip_artifacts
        text = "open the file in vim"
        assert _strip_artifacts(text) == text


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Test load_model() returns a usable model object."""

    def test_load_default_model(self):
        from lazy_claude.stt import load_model
        model = load_model()
        assert model is not None

    def test_load_model_returns_same_type(self):
        from lazy_claude.stt import load_model
        m1 = load_model()
        m2 = load_model()
        assert type(m1) == type(m2)

    def test_load_model_custom_name(self):
        """load_model('base.en') should work the same as default."""
        from lazy_claude.stt import load_model
        model = load_model("base.en")
        assert model is not None


# ---------------------------------------------------------------------------
# Transcribe with a loaded model
# ---------------------------------------------------------------------------


class TestTranscribeWithModel:
    """Pass a pre-loaded model to avoid repeated downloads."""

    def setup_method(self):
        from lazy_claude.stt import load_model
        self.model = load_model()

    def test_transcribe_silence_with_model(self):
        from lazy_claude.stt import transcribe, TranscribeResult
        result = transcribe(_silence(1.0), model=self.model)
        assert isinstance(result, TranscribeResult)
        assert isinstance(result.text, str)
        assert isinstance(result.no_speech_prob, float)

    def test_transcribe_noise_with_model(self):
        from lazy_claude.stt import transcribe, TranscribeResult
        result = transcribe(_noise(1.0), model=self.model)
        assert isinstance(result, TranscribeResult)
        assert isinstance(result.text, str)

    def test_transcribe_empty_with_model(self):
        from lazy_claude.stt import transcribe
        result = transcribe(np.array([], dtype=np.float32), model=self.model)
        assert result.text == ""
        assert result.no_speech_prob == 1.0

    def test_transcribe_no_speech_prob_in_range(self):
        from lazy_claude.stt import transcribe
        result = transcribe(_silence(1.0), model=self.model)
        assert 0.0 <= result.no_speech_prob <= 1.0

    def test_transcribe_does_not_write_to_stdout(self):
        """transcribe() must not emit anything on stdout."""
        code = """
import numpy as np
from lazy_claude.stt import transcribe, load_model
model = load_model()
audio = np.zeros(16000, dtype=np.float32)
result = transcribe(audio, model=model)
"""
        with tempfile.TemporaryFile() as cap:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                stdout=cap,
                stderr=subprocess.DEVNULL,
                cwd="/Users/sonwork/Workspace/lazy-claude",
            )
            cap.seek(0)
            out = cap.read()
        assert proc.returncode == 0, "transcribe() must not crash"
        assert out == b"", f"Expected no stdout, got: {out!r}"


# ---------------------------------------------------------------------------
# stdout cleanliness
# ---------------------------------------------------------------------------


class TestStdoutClean:
    """Importing and using stt module must not pollute stdout."""

    def test_import_stt_produces_no_stdout(self):
        with tempfile.TemporaryFile() as cap:
            proc = subprocess.run(
                [sys.executable, "-c", "from lazy_claude import stt"],
                stdout=cap,
                stderr=subprocess.DEVNULL,
                cwd="/Users/sonwork/Workspace/lazy-claude",
            )
            cap.seek(0)
            out = cap.read()
        assert proc.returncode == 0
        assert out == b"", f"Expected no stdout on import, got: {out!r}"
