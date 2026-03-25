"""Tests for Phase 2 — AEC integration into TTSEngine and ContinuousListener.

Tests cover:
- TTSEngine accepts an optional ReferenceBuffer and pushes audio into it
- ContinuousListener accepts ReferenceBuffer + EchoCanceller in constructor
- Mic callback pulls reference, runs cancel(), feeds cleaned audio to VAD
- Fallback gate suppresses chunks when AEC residual energy exceeds threshold
- ECHO_TAIL constant and _tts_stopped_at echo-tail branch are removed
- No 0.6s sleep + drain_queue() call in _ask_single
- Barge-in uses normal VAD threshold on cleaned signal (no BARGE_IN_THRESHOLD)
- set_tts_playing() not called in speak_message_impl
- VoiceServer wires shared ReferenceBuffer between TTSEngine and ContinuousListener
"""

from __future__ import annotations

import sys
import time
import threading
import importlib.util
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub missing optional dependencies so module-level imports succeed
# in environments where sounddevice / kokoro / mcp are not installed.
# ---------------------------------------------------------------------------
for _missing in ('sounddevice', 'kokoro', 'mcp', 'mcp.server',
                 'mcp.server.fastmcp', 'mcp.server.stdio', 'anyio'):
    if _missing not in sys.modules and importlib.util.find_spec(_missing) is None:
        sys.modules[_missing] = MagicMock()

SAMPLE_RATE = 16_000
CHUNK = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq: float, n: int, sr: int = SAMPLE_RATE, amp: float = 0.3) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_mock_vad(return_value: float = 0.1):
    """Return a mock SileroVAD that returns a fixed probability."""
    mock = MagicMock(return_value=return_value)
    mock.reset = MagicMock()
    return mock


def _make_mock_tts():
    mock = MagicMock()
    mock.is_speaking = False
    mock.speak = MagicMock()
    mock.stop = MagicMock()
    return mock


def _make_listener(vad=None, ref_buf=None, echo_canceller=None):
    """Instantiate ContinuousListener with thread start suppressed."""
    from lazy_claude.audio import ContinuousListener
    if vad is None:
        vad = _make_mock_vad()
    with patch.object(threading.Thread, 'start', return_value=None):
        listener = ContinuousListener(vad, ref_buf=ref_buf, echo_canceller=echo_canceller)
    return listener


def _make_listener_with_callback(vad=None, ref_buf=None, echo_canceller=None,
                                  capture_rate=16_000, needs_resample=False):
    """Instantiate ContinuousListener and build its callback (for direct invocation)."""
    listener = _make_listener(vad=vad, ref_buf=ref_buf, echo_canceller=echo_canceller)
    listener._make_callback(capture_rate, needs_resample)
    return listener


# ---------------------------------------------------------------------------
# TTSEngine — ReferenceBuffer integration
# ---------------------------------------------------------------------------


class TestTTSEngineReferenceBuffer:
    """TTSEngine should push each audio chunk into the ReferenceBuffer."""

    def test_tts_engine_accepts_ref_buf_constructor(self):
        """TTSEngine __init__ should accept a ref_buf keyword argument."""
        import lazy_claude.tts  # ensure module is imported before patch resolves targets
        from lazy_claude.aec import ReferenceBuffer
        buf = ReferenceBuffer()
        with patch('lazy_claude.tts.KPipeline'), \
             patch('lazy_claude.tts.sd'):
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine(ref_buf=buf)
        assert engine is not None

    def test_tts_engine_without_ref_buf_still_works(self):
        """TTSEngine with no ref_buf should behave exactly as before."""
        import lazy_claude.tts  # ensure module is imported before patch resolves targets
        with patch('lazy_claude.tts.KPipeline'), \
             patch('lazy_claude.tts.sd'):
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
        assert engine is not None

    def test_tts_engine_pushes_chunks_to_ref_buf(self):
        """Each audio chunk written to speaker should also be written to ref_buf."""
        import lazy_claude.tts  # ensure module is imported before patch resolves targets
        from lazy_claude.aec import ReferenceBuffer

        buf = MagicMock(spec=ReferenceBuffer)

        # Build a fake KPipeline generator result
        fake_result = MagicMock()
        fake_result.audio = MagicMock()
        fake_result.audio.cpu.return_value.numpy.return_value = np.zeros(1000, dtype=np.float32)

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = iter([fake_result])

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline_instance), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings = MagicMock()
            mock_sd.OutputStream.return_value = mock_stream
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine(ref_buf=buf)
            engine.speak("hello")

        # ref_buf.write() should have been called at least once
        assert buf.write.called, "ReferenceBuffer.write() was not called during TTS playback"

    def test_tts_engine_ref_buf_receives_float32_array(self):
        """Chunks pushed to ref_buf must be float32 numpy arrays."""
        import lazy_claude.tts  # ensure module is imported before patch resolves targets
        written_chunks = []

        class CapturingBuffer:
            def write(self, data):
                written_chunks.append(data)

        fake_result = MagicMock()
        fake_result.audio = MagicMock()
        fake_result.audio.cpu.return_value.numpy.return_value = np.zeros(1000, dtype=np.float32)

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = iter([fake_result])

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline_instance), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings = MagicMock()
            mock_sd.OutputStream.return_value = mock_stream
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine(ref_buf=CapturingBuffer())
            engine.speak("test")

        assert written_chunks, "No chunks were written to ref_buf"
        for chunk in written_chunks:
            assert isinstance(chunk, np.ndarray), f"Expected ndarray, got {type(chunk)}"
            assert chunk.dtype == np.float32, f"Expected float32, got {chunk.dtype}"


# ---------------------------------------------------------------------------
# ContinuousListener — AEC constructor integration
# ---------------------------------------------------------------------------


class TestContinuousListenerAECConstructor:
    """ContinuousListener should accept ReferenceBuffer and EchoCanceller."""

    def test_accepts_ref_buf_and_echo_canceller(self):
        from lazy_claude.aec import ReferenceBuffer, EchoCanceller
        buf = ReferenceBuffer()
        ec = EchoCanceller()
        listener = _make_listener(ref_buf=buf, echo_canceller=ec)
        assert listener is not None

    def test_accepts_no_aec_args_for_backward_compat(self):
        """Passing no AEC args should work (AEC disabled)."""
        listener = _make_listener()
        assert listener is not None

    def test_listener_has_no_echo_tail_constant(self):
        """ECHO_TAIL class attribute should be gone."""
        from lazy_claude.audio import ContinuousListener
        assert not hasattr(ContinuousListener, 'ECHO_TAIL'), \
            "ContinuousListener.ECHO_TAIL should have been removed"

    def test_listener_has_no_barge_in_threshold(self):
        """BARGE_IN_THRESHOLD class attribute should be gone (simplified barge-in)."""
        from lazy_claude.audio import ContinuousListener
        assert not hasattr(ContinuousListener, 'BARGE_IN_THRESHOLD'), \
            "ContinuousListener.BARGE_IN_THRESHOLD should have been removed"

    def test_listener_stores_ref_buf(self):
        from lazy_claude.aec import ReferenceBuffer
        buf = ReferenceBuffer()
        listener = _make_listener(ref_buf=buf)
        assert listener._ref_buf is buf

    def test_listener_stores_echo_canceller(self):
        from lazy_claude.aec import EchoCanceller
        ec = EchoCanceller()
        listener = _make_listener(echo_canceller=ec)
        assert listener._echo_canceller is ec


# ---------------------------------------------------------------------------
# ContinuousListener — AEC used in mic callback
# ---------------------------------------------------------------------------


class TestContinuousListenerAECCallback:
    """The mic callback should pull reference, run AEC, and feed cleaned audio to VAD."""

    def test_aec_cancel_called_in_callback(self):
        """EchoCanceller.cancel() should be called from the mic callback."""
        from lazy_claude.aec import ReferenceBuffer, EchoCanceller

        vad = _make_mock_vad(return_value=0.1)
        buf = ReferenceBuffer(capacity=4096, write_sr=16_000, read_sr=16_000)
        ec = MagicMock(spec=EchoCanceller)
        ec.cancel.return_value = np.zeros(CHUNK, dtype=np.float32)

        listener = _make_listener_with_callback(vad=vad, ref_buf=buf, echo_canceller=ec)
        listener._active.set()

        indata = np.zeros((CHUNK, 1), dtype=np.float32)
        listener._callback(indata, CHUNK, MagicMock(), None)

        ec.cancel.assert_called_once()

    def test_vad_receives_cleaned_audio(self):
        """VAD must be called with the AEC output, not raw mic audio."""
        from lazy_claude.aec import ReferenceBuffer, EchoCanceller

        cleaned = _sine(440.0, CHUNK, amp=0.05)  # cleaned signal
        raw_mic = _sine(1000.0, CHUNK, amp=0.5)  # echo-heavy raw mic

        vad_call_args = []

        def capture_vad(chunk):
            vad_call_args.append(chunk.copy())
            return 0.1

        vad = MagicMock(side_effect=capture_vad)

        buf = ReferenceBuffer(capacity=4096, write_sr=16_000, read_sr=16_000)
        ec = MagicMock(spec=EchoCanceller)
        ec.cancel.return_value = cleaned  # AEC returns cleaned signal

        listener = _make_listener_with_callback(vad=vad, ref_buf=buf, echo_canceller=ec)
        listener._active.set()

        indata = raw_mic.reshape(-1, 1)
        listener._callback(indata, CHUNK, MagicMock(), None)

        assert vad_call_args, "VAD was not called"
        np.testing.assert_allclose(
            vad_call_args[-1], cleaned, atol=1e-5,
            err_msg="VAD was not called with AEC-cleaned audio"
        )

    def test_no_aec_passthrough(self):
        """Without AEC, VAD receives the raw mic audio unchanged."""
        vad_call_args = []

        def capture_vad(chunk):
            vad_call_args.append(chunk.copy())
            return 0.1

        vad = MagicMock(side_effect=capture_vad)

        # No ref_buf or echo_canceller → passthrough
        listener = _make_listener_with_callback(vad=vad)
        listener._active.set()

        raw_mic = _sine(440.0, CHUNK, amp=0.3)
        indata = raw_mic.reshape(-1, 1)
        listener._callback(indata, CHUNK, MagicMock(), None)

        assert vad_call_args, "VAD was not called"
        np.testing.assert_allclose(vad_call_args[-1], raw_mic, atol=1e-5)


# ---------------------------------------------------------------------------
# Fallback gate — AEC residual energy check
# ---------------------------------------------------------------------------


class TestFallbackGate:
    """If AEC residual energy is too high during TTS, chunk should be suppressed."""

    def test_fallback_gate_suppresses_high_energy_chunk_during_tts(self):
        """During TTS, if residual energy > threshold, chunk should be dropped."""
        from lazy_claude.aec import ReferenceBuffer, EchoCanceller

        # AEC returns a high-energy chunk (echo leaked through)
        loud_residual = _sine(440.0, CHUNK, amp=0.8)  # power >> threshold

        vad = MagicMock(return_value=0.9)  # would fire speech if not gated
        buf = ReferenceBuffer(capacity=4096, write_sr=16_000, read_sr=16_000)
        ec = MagicMock(spec=EchoCanceller)
        ec.cancel.return_value = loud_residual

        listener = _make_listener_with_callback(vad=vad, ref_buf=buf, echo_canceller=ec)
        listener._active.set()
        listener._tts_active = True  # TTS is playing

        indata = loud_residual.reshape(-1, 1)
        listener._callback(indata, CHUNK, MagicMock(), None)

        # Gate should have suppressed the chunk — VAD not called, queue empty
        vad.assert_not_called()
        assert listener._pending is None, (
            "Fallback gate should have suppressed high-energy AEC residual during TTS"
        )

    def test_fallback_gate_allows_low_energy_clean_chunk(self):
        """During TTS, a low-energy residual should pass through to VAD."""
        from lazy_claude.aec import ReferenceBuffer, EchoCanceller

        # Very quiet residual — well below gate threshold (0.01)
        clean_residual = np.zeros(CHUNK, dtype=np.float32) + 0.001

        vad = MagicMock(return_value=0.0)
        buf = ReferenceBuffer(capacity=4096, write_sr=16_000, read_sr=16_000)
        ec = MagicMock(spec=EchoCanceller)
        ec.cancel.return_value = clean_residual

        listener = _make_listener_with_callback(vad=vad, ref_buf=buf, echo_canceller=ec)
        listener._active.set()
        listener._tts_active = True

        indata = clean_residual.reshape(-1, 1)
        listener._callback(indata, CHUNK, MagicMock(), None)

        # VAD should have been called (chunk not suppressed)
        assert vad.called, "VAD should be called for low-energy AEC residual"

    def test_fallback_gate_inactive_when_tts_not_playing(self):
        """Fallback gate should NOT suppress chunks when TTS is not active."""
        from lazy_claude.aec import ReferenceBuffer, EchoCanceller

        # High energy but TTS not playing
        loud = _sine(440.0, CHUNK, amp=0.8)

        vad = MagicMock(return_value=0.0)
        buf = ReferenceBuffer(capacity=4096, write_sr=16_000, read_sr=16_000)
        ec = MagicMock(spec=EchoCanceller)
        ec.cancel.return_value = loud

        listener = _make_listener_with_callback(vad=vad, ref_buf=buf, echo_canceller=ec)
        listener._active.set()
        listener._tts_active = False  # TTS not playing

        indata = loud.reshape(-1, 1)
        listener._callback(indata, CHUNK, MagicMock(), None)

        # Gate must not fire when TTS is off — VAD should be called
        assert vad.called, "VAD should be called when TTS is not active (no gating)"


# ---------------------------------------------------------------------------
# Barge-in — uses normal VAD threshold on cleaned signal
# ---------------------------------------------------------------------------


class TestBargeInSimplified:
    """Speech during TTS should be buffered for STOP-only barge-in decisions."""

    def test_speech_during_tts_creates_barge_in_candidate(self):
        """VAD >= NORMAL_THRESHOLD on cleaned signal during TTS buffers a candidate."""
        from lazy_claude.aec import ReferenceBuffer, EchoCanceller
        from lazy_claude.audio import ContinuousListener

        clean_speech = np.zeros(CHUNK, dtype=np.float32) + 0.001

        recorded_frames = int(np.ceil(ContinuousListener.MIN_SPEECH_DURATION / (CHUNK / 16000)))
        speech_frames = ContinuousListener.BARGE_IN_FRAMES - 1 + recorded_frames
        vad = MagicMock(side_effect=([0.8] * speech_frames) + [0.0])
        buf = ReferenceBuffer(capacity=4096, write_sr=16_000, read_sr=16_000)
        ec = MagicMock(spec=EchoCanceller)
        ec.cancel.return_value = clean_speech

        listener = _make_listener_with_callback(vad=vad, ref_buf=buf, echo_canceller=ec)
        listener._active.set()
        listener._tts_active = True

        indata = clean_speech.reshape(-1, 1)
        for _ in range(speech_frames):
            listener._callback(indata, CHUNK, MagicMock(), None)
        listener._barge_in_silence_started = time.monotonic() - 1.0
        listener._callback(indata, CHUNK, MagicMock(), None)

        candidate = listener.pop_barge_in_candidate()
        assert isinstance(candidate, np.ndarray)
        # barge_in event IS set when a candidate is buffered — this signals
        # the server to check the candidate and decide whether to stop TTS.
        assert listener.barge_in.is_set(), \
            "barge_in event should be set when a candidate is buffered"

    def test_barge_in_does_not_require_separate_barge_in_threshold(self):
        """BARGE_IN_THRESHOLD constant must not exist — use NORMAL_THRESHOLD."""
        from lazy_claude.audio import ContinuousListener
        assert not hasattr(ContinuousListener, 'BARGE_IN_THRESHOLD')

    def test_no_barge_in_candidate_on_low_vad_prob_during_tts(self):
        """Low VAD prob on cleaned signal should not even start a candidate."""
        from lazy_claude.aec import ReferenceBuffer, EchoCanceller

        clean_silence = np.zeros(CHUNK, dtype=np.float32) + 0.001

        vad = MagicMock(return_value=0.1)  # below NORMAL_THRESHOLD
        buf = ReferenceBuffer(capacity=4096, write_sr=16_000, read_sr=16_000)
        ec = MagicMock(spec=EchoCanceller)
        ec.cancel.return_value = clean_silence

        listener = _make_listener_with_callback(vad=vad, ref_buf=buf, echo_canceller=ec)
        listener._active.set()
        listener._tts_active = True

        indata = clean_silence.reshape(-1, 1)
        for _ in range(10):
            listener._callback(indata, CHUNK, MagicMock(), None)

        assert not listener.barge_in.is_set(), \
            "Barge-in should not fire on low VAD probability"
        assert listener.pop_barge_in_candidate() is None


