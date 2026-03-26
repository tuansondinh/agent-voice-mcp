"""Tests for audio.py — VAD state machine and audio capture utilities.

These tests do NOT require a microphone.  They test:
- VAD state machine transitions
- Silence / speech detection thresholds
- Timeout behaviour
- Return type and sample rate of recorded audio
"""

from __future__ import annotations

import threading
import time
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Unit tests for the VAD state machine (no hardware required)
# ---------------------------------------------------------------------------


class TestVADStateMachine:
    """Test the VadStateMachine class directly."""

    def setup_method(self):
        # Import here so any import-time errors surface as test failures
        from lazy_claude.audio import VadStateMachine
        self.VadStateMachine = VadStateMachine

    def _machine(self, **kwargs):
        defaults = dict(
            silence_duration=1.5,
            min_speech_duration=0.5,
            no_speech_timeout=15.0,
            sample_rate=16000,
            chunk_size=512,
        )
        defaults.update(kwargs)
        return self.VadStateMachine(**defaults)

    def test_initial_state_is_waiting(self):
        m = self._machine()
        assert m.state == "WAITING"

    def test_single_speech_chunk_transitions_to_speaking(self):
        m = self._machine()
        # Feed a speech chunk (prob > threshold)
        done = m.update(speech_prob=0.9, timestamp=0.0)
        assert not done
        assert m.state == "SPEAKING"

    def test_silence_in_waiting_stays_waiting(self):
        m = self._machine()
        done = m.update(speech_prob=0.1, timestamp=0.0)
        assert not done
        assert m.state == "WAITING"

    def test_speech_then_silence_transitions_to_trailing(self):
        m = self._machine(min_speech_duration=0.0)
        m.update(speech_prob=0.9, timestamp=0.0)
        assert m.state == "SPEAKING"
        done = m.update(speech_prob=0.1, timestamp=0.1)
        assert not done
        assert m.state == "TRAILING_SILENCE"

    def test_trailing_silence_long_enough_triggers_done(self):
        m = self._machine(min_speech_duration=0.0, silence_duration=0.5)
        # Start speaking
        m.update(speech_prob=0.9, timestamp=0.0)
        # Start trailing silence
        m.update(speech_prob=0.1, timestamp=0.1)
        assert m.state == "TRAILING_SILENCE"
        # Silence not long enough yet
        done = m.update(speech_prob=0.1, timestamp=0.4)
        assert not done
        # Silence now exceeds 0.5s
        done = m.update(speech_prob=0.1, timestamp=0.7)
        assert done
        assert m.state == "DONE"

    def test_speech_resumes_during_trailing_silence(self):
        m = self._machine(min_speech_duration=0.0, silence_duration=1.5)
        m.update(speech_prob=0.9, timestamp=0.0)
        m.update(speech_prob=0.1, timestamp=0.1)
        assert m.state == "TRAILING_SILENCE"
        # Speech resumes
        done = m.update(speech_prob=0.9, timestamp=0.5)
        assert not done
        assert m.state == "SPEAKING"

    def test_no_speech_timeout_triggers_done(self):
        m = self._machine(no_speech_timeout=1.0)
        done = m.update(speech_prob=0.1, timestamp=0.0)
        assert not done
        done = m.update(speech_prob=0.1, timestamp=1.5)
        assert done
        assert m.state == "DONE"
        assert m.timed_out is True

    def test_no_speech_timeout_does_not_fire_once_speech_started(self):
        m = self._machine(no_speech_timeout=1.0)
        m.update(speech_prob=0.9, timestamp=0.0)
        # timestamp way past no_speech_timeout, but speech already started
        done = m.update(speech_prob=0.1, timestamp=2.0)
        assert not done  # still in TRAILING_SILENCE, not timed out
        assert m.timed_out is False

    def test_min_speech_duration_prevents_early_done(self):
        """With min_speech_duration=0.5s, trailing silence should not finish
        the recording if total speech accumulated is less than 0.5s."""
        chunk_duration = 512 / 16000  # ~0.032 s
        m = self._machine(min_speech_duration=0.5, silence_duration=0.1)
        # Only ~one chunk of speech (~32 ms) — below min_speech_duration
        m.update(speech_prob=0.9, timestamp=0.0)
        # Long silence — would normally trigger done
        done = m.update(speech_prob=0.1, timestamp=0.5)
        assert not done
        # Still TRAILING_SILENCE, not DONE
        assert m.state in ("TRAILING_SILENCE", "WAITING", "SPEAKING")


# ---------------------------------------------------------------------------
# Integration smoke tests (no mic — just verify public API exists)
# ---------------------------------------------------------------------------


class TestVADModelLoading:
    """Test that the Silero VAD ONNX model can be loaded and run on a dummy chunk."""

    def test_load_vad_model_returns_callable(self):
        from lazy_claude.audio import load_vad_model
        model = load_vad_model()
        assert callable(model)

    def test_vad_model_returns_probability_for_silence(self):
        from lazy_claude.audio import load_vad_model
        model = load_vad_model()
        # Pure silence chunk: 512 float32 zeros at 16kHz
        chunk = np.zeros(512, dtype=np.float32)
        prob = model(chunk)
        assert isinstance(prob, float), f"Expected float, got {type(prob)}"
        assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

    def test_vad_model_returns_probability_for_noise(self):
        from lazy_claude.audio import load_vad_model
        model = load_vad_model()
        rng = np.random.default_rng(42)
        chunk = rng.standard_normal(512).astype(np.float32) * 0.1
        prob = model(chunk)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_vad_model_resets_state(self):
        from lazy_claude.audio import load_vad_model
        model = load_vad_model()
        chunk = np.zeros(512, dtype=np.float32)
        prob1 = model(chunk)
        model.reset()
        prob2 = model(chunk)
        # After reset, same silent chunk should give same result
        assert abs(prob1 - prob2) < 1e-5, f"Probabilities differ after reset: {prob1} vs {prob2}"


# ---------------------------------------------------------------------------
# Phase 2: Single-slot Condition-based listener tests (ContinuousListener)
# ---------------------------------------------------------------------------


class TestContinuousListenerSingleSlot:
    """Phase 2: verify Condition-based single-slot design for ContinuousListener."""

    def _make_listener(self):
        """Return a ContinuousListener with mocked dependencies (no audio hardware)."""
        from unittest.mock import MagicMock
        from lazy_claude.audio import ContinuousListener

        mock_vad = MagicMock()
        mock_vad.return_value = 0.0

        listener = ContinuousListener.__new__(ContinuousListener)
        listener._vad = mock_vad
        listener._ref_buf = None
        listener._echo_canceller = None
        listener._slot_lock = threading.Condition()
        listener._pending = None
        listener._barge_in_event = threading.Event()
        listener._stop_event = threading.Event()
        listener._tts_active = False
        listener._tts_stopped_at = 0.0
        listener._active = threading.Event()
        listener._active.set()
        listener._recording = False
        listener._utterance_chunks = []
        listener._accumulated_speech = 0.0
        listener._silence_started = None
        listener._barge_in_frame_count = 0
        listener._device_changed = False
        listener._callback = None
        return listener, mock_vad

    def _build_callback(self, listener):
        """Build and attach the sounddevice callback without starting audio."""
        listener._callback = listener._make_callback(16_000, False)
        return listener._callback

    def _drive_speech(self, listener, callback, n_speech=20):
        """Feed speech then expire silence timer to produce an utterance."""
        from unittest.mock import MagicMock
        listener._vad.return_value = 0.9
        chunk = np.ones(512, dtype=np.float32) * 0.1
        indata = chunk.reshape(-1, 1)
        for _ in range(n_speech):
            callback(indata, 512, MagicMock(), None)
        # Expire silence timer
        listener._silence_started = time.monotonic() - 2.0
        listener._vad.return_value = 0.0
        callback(indata, 512, MagicMock(), None)

    def test_slot_lock_is_condition(self):
        listener, _ = self._make_listener()
        assert isinstance(listener._slot_lock, threading.Condition)

    def test_pending_starts_none(self):
        listener, _ = self._make_listener()
        with listener._slot_lock:
            assert listener._pending is None

    def test_utterance_stored_in_pending_slot(self):
        listener, _ = self._make_listener()
        callback = self._build_callback(listener)
        self._drive_speech(listener, callback)
        with listener._slot_lock:
            assert listener._pending is not None
            assert isinstance(listener._pending, np.ndarray)

    def test_second_utterance_replaces_first(self):
        """Second produced utterance replaces first in slot before consumer reads."""
        listener, _ = self._make_listener()
        callback = self._build_callback(listener)
        self._drive_speech(listener, callback)
        with listener._slot_lock:
            first = listener._pending
        assert first is not None

        # Reset recording state
        listener._recording = False
        listener._accumulated_speech = 0.0
        listener._silence_started = None

        self._drive_speech(listener, callback)
        with listener._slot_lock:
            second = listener._pending

        assert second is not None
        assert second is not first

    def test_drain_queue_clears_pending_slot(self):
        listener, _ = self._make_listener()
        with listener._slot_lock:
            listener._pending = np.zeros(512, dtype=np.float32)
        listener.drain_queue()
        with listener._slot_lock:
            assert listener._pending is None

    def test_get_next_speech_returns_none_on_timeout(self):
        listener, _ = self._make_listener()
        result = listener.get_next_speech(timeout=0.05)
        assert result is None

    def test_get_next_speech_returns_utterance(self):
        listener, _ = self._make_listener()
        audio = np.ones(1024, dtype=np.float32)
        with listener._slot_lock:
            listener._pending = audio
            listener._slot_lock.notify_all()
        result = listener.get_next_speech(timeout=1.0)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_get_next_speech_clears_pending_after_read(self):
        listener, _ = self._make_listener()
        audio = np.ones(1024, dtype=np.float32)
        with listener._slot_lock:
            listener._pending = audio
            listener._slot_lock.notify_all()
        listener.get_next_speech(timeout=1.0)
        with listener._slot_lock:
            assert listener._pending is None

    def test_producer_notifies_consumer(self):
        """Background consumer thread wakes when utterance is placed in slot."""
        listener, _ = self._make_listener()
        result_holder = [None]

        def consumer():
            result_holder[0] = listener.get_next_speech(timeout=2.0)

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        time.sleep(0.05)

        audio = np.ones(512, dtype=np.float32)
        with listener._slot_lock:
            listener._pending = audio
            listener._slot_lock.notify_all()

        t.join(timeout=2.0)
        assert result_holder[0] is not None
        assert isinstance(result_holder[0], np.ndarray)
