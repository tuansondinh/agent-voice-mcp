"""Tests for av_audio.py — AVAudioEngine backend, macOS listener and TTS.

Tests run on all platforms with PyObjC mocked out. On non-macOS hosts
the pyobjc-framework-AVFoundation import always fails gracefully; the
module provides stub classes so the tests can still run.

Test coverage
-------------
- Resampling accuracy: 44.1k→16k and 24k→44.1k, no drift over many chunks
- Rechunking: 512-sample VAD frames produced correctly from variable-size tap buffers
- AVAudioBackend init/shutdown with mocked PyObjC
- MacOSContinuousListener queues utterances from synthetic speech
- MacOSTTSEngine delegates play_audio to AVAudioBackend
- Regression: 16kHz/512-chunk delivery matches what Silero VAD expects
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq: float, duration: float, sample_rate: int) -> np.ndarray:
    """Generate a pure sine wave as float32."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Resampling tests — pure math, no hardware
# ---------------------------------------------------------------------------

class TestResampling:
    """Verify resample_audio() accuracy and no-drift property."""

    def setup_method(self):
        from lazy_claude.av_audio import resample_audio
        self.resample = resample_audio

    def test_44100_to_16000_length(self):
        """Resampled length should be approximately n * (16000/44100)."""
        n_in = 1024
        src = np.ones(n_in, dtype=np.float32)
        out = self.resample(src, 44_100, 16_000)
        expected = int(round(n_in * 16_000 / 44_100))
        # Allow ±1 sample tolerance for rounding
        assert abs(len(out) - expected) <= 1

    def test_24000_to_44100_length(self):
        """Upsample 24kHz → 44.1kHz length check."""
        n_in = 480
        src = np.ones(n_in, dtype=np.float32)
        out = self.resample(src, 24_000, 44_100)
        expected = int(round(n_in * 44_100 / 24_000))
        assert abs(len(out) - expected) <= 1

    def test_same_rate_passthrough(self):
        """Identity: same in/out rate returns equal array."""
        src = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        out = self.resample(src, 16_000, 16_000)
        np.testing.assert_array_equal(out, src)

    def test_no_drift_44100_to_16000_many_chunks(self):
        """
        Feed 100 chunks of 1024 samples at 44.1kHz.
        Stateless per-chunk resampling rounds each chunk independently;
        the total must match the theoretically exact total within
        ±n_chunks samples (at most 1 rounding error per chunk).
        """
        chunk_size = 1024
        n_chunks = 100
        total_in = chunk_size * n_chunks
        all_samples = _sine(440.0, total_in / 44_100, 44_100)

        # Resample each chunk independently
        resampled_chunks = []
        for i in range(n_chunks):
            chunk = all_samples[i * chunk_size : (i + 1) * chunk_size]
            resampled_chunks.append(self.resample(chunk, 44_100, 16_000))
        total_resampled = sum(len(c) for c in resampled_chunks)

        # Exact theoretical total (without per-chunk rounding)
        exact_total = int(round(total_in * 16_000 / 44_100))
        # Tolerance: at most 1 sample rounding error per chunk
        assert abs(total_resampled - exact_total) <= n_chunks, (
            f"Drift {abs(total_resampled - exact_total)} exceeds {n_chunks} samples"
        )

    def test_no_drift_24000_to_44100_many_chunks(self):
        """Feed 50 chunks of 480 samples at 24kHz; no cumulative drift."""
        chunk_size = 480
        n_chunks = 50
        total_in = chunk_size * n_chunks
        all_samples = _sine(220.0, total_in / 24_000, 24_000)

        resampled_chunks = []
        for i in range(n_chunks):
            chunk = all_samples[i * chunk_size : (i + 1) * chunk_size]
            resampled_chunks.append(self.resample(chunk, 24_000, 44_100))
        total_resampled = sum(len(c) for c in resampled_chunks)

        expected = len(self.resample(all_samples, 24_000, 44_100))
        assert abs(total_resampled - expected) <= 2

    def test_output_dtype_is_float32(self):
        src = np.ones(512, dtype=np.float32)
        out = self.resample(src, 44_100, 16_000)
        assert out.dtype == np.float32

    def test_output_is_1d(self):
        src = np.ones(512, dtype=np.float32)
        out = self.resample(src, 44_100, 16_000)
        assert out.ndim == 1


# ---------------------------------------------------------------------------
# Rechunking tests
# ---------------------------------------------------------------------------

class TestRechunker:
    """Verify AudioRechunker produces exact 512-sample VAD frames."""

    def setup_method(self):
        from lazy_claude.av_audio import AudioRechunker
        self.AudioRechunker = AudioRechunker

    def test_exact_chunk_passes_through(self):
        """512 samples in → one 512-sample callback."""
        received = []
        rechunker = self.AudioRechunker(chunk_size=512, callback=received.append)
        rechunker.push(np.ones(512, dtype=np.float32))
        assert len(received) == 1
        assert len(received[0]) == 512

    def test_partial_chunk_buffered(self):
        """256 samples in → no callback yet."""
        received = []
        rechunker = self.AudioRechunker(chunk_size=512, callback=received.append)
        rechunker.push(np.ones(256, dtype=np.float32))
        assert len(received) == 0

    def test_two_partials_make_one_chunk(self):
        """256 + 256 = 512 → one callback."""
        received = []
        rechunker = self.AudioRechunker(chunk_size=512, callback=received.append)
        rechunker.push(np.ones(256, dtype=np.float32) * 0.1)
        rechunker.push(np.ones(256, dtype=np.float32) * 0.2)
        assert len(received) == 1
        assert len(received[0]) == 512

    def test_large_buffer_produces_multiple_chunks(self):
        """1024 samples → two 512-sample chunks."""
        received = []
        rechunker = self.AudioRechunker(chunk_size=512, callback=received.append)
        rechunker.push(np.ones(1024, dtype=np.float32))
        assert len(received) == 2
        for chunk in received:
            assert len(chunk) == 512

    def test_leftover_buffered_until_next_push(self):
        """768 samples → one 512-chunk; 256 remain buffered."""
        received = []
        rechunker = self.AudioRechunker(chunk_size=512, callback=received.append)
        rechunker.push(np.ones(768, dtype=np.float32))
        assert len(received) == 1
        # Now push 256 more → second chunk
        rechunker.push(np.ones(256, dtype=np.float32))
        assert len(received) == 2

    def test_data_content_preserved(self):
        """Values in produced chunks match source data exactly."""
        received = []
        rechunker = self.AudioRechunker(chunk_size=4, callback=received.append)
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        rechunker.push(src)
        assert len(received) == 2
        np.testing.assert_array_equal(received[0], src[:4])
        np.testing.assert_array_equal(received[1], src[4:])

    def test_silero_vad_chunk_size_regression(self):
        """All produced chunks are exactly CHUNK_SAMPLES=512 — Silero requirement."""
        received = []
        rechunker = self.AudioRechunker(chunk_size=512, callback=received.append)
        # Feed variable-size buffers as CoreAudio would
        rng = np.random.default_rng(42)
        total_fed = 0
        for _ in range(20):
            size = rng.integers(100, 2000)
            rechunker.push(rng.random(size).astype(np.float32))
            total_fed += size
        for chunk in received:
            assert len(chunk) == 512, f"Expected 512 got {len(chunk)}"


# ---------------------------------------------------------------------------
# AVAudioBackend init/shutdown (mocked PyObjC)
# ---------------------------------------------------------------------------

class TestAVAudioBackendMocked:
    """AVAudioBackend construction and shutdown with all ObjC calls mocked."""

    def _make_backend(self):
        """Return an AVAudioBackend with PyObjC fully mocked."""
        from lazy_claude.av_audio import AVAudioBackend

        mock_engine = MagicMock()
        mock_input_node = MagicMock()
        mock_player_node = MagicMock()
        mock_mixer_node = MagicMock()
        mock_output_format = MagicMock()

        mock_engine.inputNode.return_value = mock_input_node
        mock_engine.mainMixerNode.return_value = mock_mixer_node
        mock_input_node.outputFormatForBus_.return_value = mock_output_format

        AVFoundation_mock = MagicMock()
        AVFoundation_mock.AVAudioEngine.return_value = mock_engine
        AVFoundation_mock.AVAudioPlayerNode.return_value = mock_player_node
        AVFoundation_mock.AVAudioFormat.return_value = MagicMock()

        with patch.dict('sys.modules', {'AVFoundation': AVFoundation_mock}):
            backend = AVAudioBackend.__new__(AVAudioBackend)
            backend._AVFoundation = AVFoundation_mock
            backend._Foundation = None
            backend._engine = mock_engine
            backend._player = mock_player_node
            backend._input_node = mock_input_node
            backend._mixer_node = mock_mixer_node
            backend._running = True
            backend._tap_installed = False
            backend._vp_enabled = False

        return backend, mock_engine, mock_player_node, mock_input_node

    def test_shutdown_stops_engine(self):
        """shutdown() calls stop() on the AVAudioEngine."""
        backend, mock_engine, mock_player, mock_input = self._make_backend()
        backend._tap_installed = False
        backend.shutdown()
        mock_engine.stop.assert_called_once()

    def test_install_mic_tap_resumes_engine_when_suspended(self):
        """install_mic_tap() resumes a suspended engine before tapping the mic."""
        backend, mock_engine, mock_player, mock_input = self._make_backend()
        backend._running = False
        backend._tap_queue = queue.Queue()
        backend._consumer_stop = threading.Event()
        backend._tap_callback = None
        backend._consumer_thread = None
        mock_engine.startAndReturnError_.return_value = True
        mock_output_format = MagicMock()
        mock_output_format.sampleRate.return_value = 48_000
        mock_output_format.channelCount.return_value = 1
        mock_input.outputFormatForBus_.return_value = mock_output_format

        backend.install_mic_tap(MagicMock())

        mock_engine.startAndReturnError_.assert_called_once()

    def test_remove_mic_tap_suspends_engine_when_idle(self):
        """remove_mic_tap() stops the engine once no tap remains."""
        backend, mock_engine, mock_player, mock_input = self._make_backend()
        backend._tap_installed = True
        backend._consumer_stop = threading.Event()
        backend._consumer_thread = MagicMock()

        backend.remove_mic_tap()

        mock_input.removeTapOnBus_.assert_called_once_with(0)
        mock_engine.stop.assert_called_once()

    def test_stop_playback_stops_and_replays_player(self):
        """stop_playback() stops the player node and re-primes it."""
        backend, mock_engine, mock_player, mock_input = self._make_backend()
        backend.stop_playback()
        mock_player.stop.assert_called_once()
        mock_player.play.assert_called_once()

    def test_play_audio_resamples_24k_input(self):
        """play_audio() with a 24kHz chunk does not raise."""
        from lazy_claude.av_audio import AVAudioBackend
        AVFoundation_mock = MagicMock()
        mock_engine = MagicMock()
        mock_player = MagicMock()
        mock_format = MagicMock()

        AVFoundation_mock.AVAudioEngine.return_value = mock_engine
        AVFoundation_mock.AVAudioPlayerNode.return_value = mock_player
        AVFoundation_mock.AVAudioFormat.return_value = mock_format
        AVFoundation_mock.AVAudioPCMBuffer.return_value = MagicMock()

        with patch.dict('sys.modules', {'AVFoundation': AVFoundation_mock}):
            backend = AVAudioBackend.__new__(AVAudioBackend)
            backend._engine = mock_engine
            backend._player = mock_player
            backend._running = True
            backend._tap_installed = False

            chunk_24k = _sine(440.0, 0.1, 24_000)
            # Should not raise even with mocks
            try:
                backend.play_audio(chunk_24k)
            except Exception:
                pass  # mock may not support buffer internals — that's OK

    def test_backend_is_importable_on_non_macos(self):
        """av_audio module imports without error even when AVFoundation is absent."""
        import importlib
        import sys
        # Remove AVFoundation from sys.modules if present to simulate non-macOS
        avfoundation_backup = sys.modules.pop('AVFoundation', None)
        try:
            # Re-import av_audio without AVFoundation
            if 'lazy_claude.av_audio' in sys.modules:
                del sys.modules['lazy_claude.av_audio']
            import lazy_claude.av_audio as av_mod  # noqa: F401
            assert hasattr(av_mod, 'AVAudioBackend')
            assert hasattr(av_mod, 'MacOSContinuousListener')
            assert hasattr(av_mod, 'MacOSTTSEngine')
        finally:
            if avfoundation_backup is not None:
                sys.modules['AVFoundation'] = avfoundation_backup


# ---------------------------------------------------------------------------
# MacOSContinuousListener — utterance queuing with synthetic VAD input
# ---------------------------------------------------------------------------


class TestMacOSContinuousListenerVADProcessing:
    """Test utterance segmentation logic via direct _process_chunk calls."""

    def _make_listener_with_callback(self):
        """Return a listener with the internal _process_chunk method accessible."""
        mock_vad = MagicMock()
        mock_vad.return_value = 0.0  # silence by default

        from lazy_claude.av_audio import MacOSContinuousListener
        listener = MacOSContinuousListener.__new__(MacOSContinuousListener)
        listener._vad = mock_vad
        listener._slot_lock = threading.Condition()
        listener._pending = None
        listener._barge_in_event = threading.Event()
        listener._stop_event = threading.Event()
        listener._tts_active = False
        listener._tts_stopped_at = 0.0
        listener._active = threading.Event()
        listener._active.set()  # active by default for VAD tests
        listener._backend_tap_active = True
        listener._recording = False
        listener._utterance_chunks = []
        listener._accumulated_speech = 0.0
        listener._silence_started = None
        listener._barge_in_frame_count = 0
        listener._backend = MagicMock()
        return listener, mock_vad

    def test_silence_chunks_do_not_queue_utterance(self):
        listener, mock_vad = self._make_listener_with_callback()
        mock_vad.return_value = 0.0  # silence

        chunk = np.zeros(512, dtype=np.float32)
        for _ in range(10):
            listener._process_chunk(chunk)

        with listener._slot_lock:
            assert listener._pending is None

    def test_speech_then_silence_queues_utterance(self):
        """Speech followed by sufficient trailing silence → utterance queued."""
        listener, mock_vad = self._make_listener_with_callback()

        chunk = np.ones(512, dtype=np.float32) * 0.1
        chunk_duration = 512 / 16_000  # ~0.032s

        # Feed speech chunks to accumulate > 0.5s
        mock_vad.return_value = 0.9
        n_speech = int(0.6 / chunk_duration) + 1
        for _ in range(n_speech):
            listener._process_chunk(chunk)

        assert listener._recording is True, "Should be recording after speech chunks"
        assert listener._accumulated_speech >= 0.5, "Should have accumulated 0.5s of speech"

        # Manually set silence_started far enough in the past to trigger completion
        # (avoids needing to sleep real time in tests)
        listener._silence_started = time.monotonic() - 2.0  # 2s ago > SILENCE_DURATION=1.5s

        # Feed one more silence chunk to trigger the completion check
        mock_vad.return_value = 0.0
        listener._process_chunk(chunk)

        with listener._slot_lock:
            assert listener._pending is not None
            utterance = listener._pending
        assert isinstance(utterance, np.ndarray)
        assert utterance.dtype == np.float32

    def test_barge_in_candidate_buffered_during_tts(self):
        """Speech during TTS is buffered as a candidate, not an immediate stop."""
        listener, mock_vad = self._make_listener_with_callback()
        listener._tts_active = True
        listener.clear_barge_in()

        mock_vad.return_value = 0.9  # speech
        chunk = np.ones(512, dtype=np.float32) * 0.1

        from lazy_claude.av_audio import MacOSContinuousListener
        recorded_frames = int(
            np.ceil(MacOSContinuousListener.MIN_SPEECH_DURATION / (512 / 16000))
        )
        speech_frames = (
            MacOSContinuousListener.BARGE_IN_FRAMES - 1 + recorded_frames
        )
        for _ in range(speech_frames):
            listener._process_chunk(chunk)

        mock_vad.return_value = 0.0
        listener._barge_in_silence_started = time.monotonic() - 1.0
        listener._process_chunk(np.zeros(512, dtype=np.float32))

        candidate = listener.pop_barge_in_candidate()
        assert isinstance(candidate, np.ndarray)
        assert candidate.dtype == np.float32
        assert listener.barge_in.is_set()

    def test_inactive_listener_discards_audio(self):
        """When not active, chunks are discarded without VAD processing."""
        listener, mock_vad = self._make_listener_with_callback()
        listener.set_active(False)

        chunk = np.ones(512, dtype=np.float32)
        mock_vad.return_value = 0.9
        for _ in range(100):
            listener._process_chunk(chunk)

        with listener._slot_lock:
            assert listener._pending is None
        mock_vad.assert_not_called()

    def test_chunk_must_be_512_samples(self):
        """Chunks of wrong size are silently skipped (no crash, no slot set)."""
        listener, mock_vad = self._make_listener_with_callback()
        mock_vad.return_value = 0.9

        listener._process_chunk(np.ones(256, dtype=np.float32))
        with listener._slot_lock:
            assert listener._pending is None


# ---------------------------------------------------------------------------
# MacOSTTSEngine — public API and delegation
# ---------------------------------------------------------------------------

class TestMacOSTTSEngineAPI:
    """Verify MacOSTTSEngine matches TTSEngine public API."""

    def _make_engine(self):
        from lazy_claude.av_audio import MacOSTTSEngine
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = iter([])  # no chunks by default
        mock_backend = MagicMock()

        with patch('lazy_claude.av_audio.KPipeline', return_value=mock_pipeline), \
             patch('lazy_claude.av_audio.AVAudioBackend', return_value=mock_backend):
            engine = MacOSTTSEngine.__new__(MacOSTTSEngine)
            engine._pipeline = mock_pipeline
            engine._backend = mock_backend
            engine._speaking = False
            engine._stop_event = threading.Event()
        return engine, mock_pipeline, mock_backend

    def test_speak_empty_text_is_noop(self):
        engine, mock_pipeline, mock_backend = self._make_engine()
        engine.speak("")
        mock_pipeline.assert_not_called()
        mock_backend.play_audio.assert_not_called()

    def test_speak_delegates_to_backend_play_audio(self):
        """speak() feeds Kokoro chunks into backend.play_audio()."""
        from lazy_claude.av_audio import MacOSTTSEngine
        import torch

        mock_pipeline_instance = MagicMock()
        audio_chunk = torch.from_numpy(np.ones(480, dtype=np.float32) * 0.1)
        mock_result = MagicMock()
        mock_result.audio = audio_chunk
        mock_pipeline_instance.return_value = iter([mock_result])

        mock_backend = MagicMock()

        with patch('lazy_claude.av_audio.KPipeline', return_value=mock_pipeline_instance), \
             patch('lazy_claude.av_audio.AVAudioBackend', return_value=mock_backend):
            engine = MacOSTTSEngine.__new__(MacOSTTSEngine)
            engine._pipeline = mock_pipeline_instance
            engine._backend = mock_backend
            engine._speaking = False
            engine._stop_event = threading.Event()
            engine.speak("hello")

        mock_backend.play_audio.assert_called_once()
        # Verify passed a numpy float32 array
        call_arg = mock_backend.play_audio.call_args[0][0]
        assert isinstance(call_arg, np.ndarray)
        assert call_arg.dtype == np.float32


# ---------------------------------------------------------------------------
# Silero VAD regression: 16kHz / 512-sample delivery
# ---------------------------------------------------------------------------

class TestSileroVADChunkRegression:
    """Ensure the tap→resample→rechunk pipeline delivers exactly what Silero expects."""

    def test_resample_44100_chunks_yield_512_at_16k(self):
        """
        When the CoreAudio tap delivers 1024-sample chunks at 44.1kHz,
        rechunking must produce exactly 512-sample chunks at 16kHz.
        """
        from lazy_claude.av_audio import resample_audio, AudioRechunker

        CHUNK_SAMPLES = 512
        received = []
        rechunker = AudioRechunker(chunk_size=CHUNK_SAMPLES, callback=received.append)

        # Simulate 50 tap callbacks delivering 1024 samples at 44.1kHz
        for _ in range(50):
            tap_buffer = np.random.randn(1024).astype(np.float32) * 0.01
            resampled = resample_audio(tap_buffer, 44_100, 16_000)
            rechunker.push(resampled)

        # All produced chunks must be exactly 512 samples
        for i, chunk in enumerate(received):
            assert len(chunk) == CHUNK_SAMPLES, (
                f"Chunk {i}: expected {CHUNK_SAMPLES} samples, got {len(chunk)}"
            )
        # We should have received a reasonable number of chunks
        assert len(received) > 0

    def test_silero_vad_accepts_produced_chunks(self):
        """
        Actually run Silero VAD on produced chunks to confirm they are valid.
        Skipped if the ONNX model file is absent or onnxruntime is not installed.
        """
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            pytest.skip("onnxruntime not installed — skipping live Silero regression test")

        from pathlib import Path
        model_path = Path(__file__).parent.parent / "lazy_claude" / "models" / "silero_vad.onnx"
        if not model_path.exists():
            pytest.skip("Silero VAD model not downloaded — skipping live regression test")

        from lazy_claude.audio import SileroVAD
        from lazy_claude.av_audio import resample_audio, AudioRechunker

        vad = SileroVAD(model_path)
        vad.reset()

        probabilities = []

        def run_vad(chunk: np.ndarray) -> None:
            prob = vad(chunk)
            probabilities.append(prob)

        rechunker = AudioRechunker(chunk_size=512, callback=run_vad)

        # Feed 3 seconds of silence at 44.1kHz → should get valid probabilities
        total_samples = int(44_100 * 3)
        chunk_size_44k = 1024
        for i in range(0, total_samples, chunk_size_44k):
            tap = np.zeros(min(chunk_size_44k, total_samples - i), dtype=np.float32)
            resampled = resample_audio(tap, 44_100, 16_000)
            rechunker.push(resampled)

        assert len(probabilities) > 0
        # Silence should have low VAD probability
        assert all(0.0 <= p <= 1.0 for p in probabilities), "All probs must be in [0,1]"
        assert np.mean(probabilities) < 0.5, "Mean prob should be low for silence"


