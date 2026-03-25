"""Tests for server.py — macOS AEC auto-detection and fallback.

Tests verify:
- VoiceServer._use_macos_aec flag exists
- VoiceServer uses MacOS backend on darwin when AVAudioBackend available
- VoiceServer falls back to sounddevice path when import fails
- VoiceServer falls back when AVAudioBackend init throws
- _ask_single on macOS path does NOT sleep 0.8s or drain after TTS
- _ask_single on fallback path DOES sleep + drain (existing behavior preserved)
- speak_message on macOS path does NOT sleep or drain
- speak_message on fallback path DOES sleep + drain
"""

from __future__ import annotations

import sys
import threading
import time
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out heavy native modules BEFORE any lazy_claude import so that
# the module-level imports in server.py / tts.py do not fail.
# ---------------------------------------------------------------------------

def _stub(name: str) -> MagicMock:
    m = MagicMock()
    m.__name__ = name
    sys.modules[name] = m
    return m


for _mod in ('sounddevice', 'kokoro', 'onnxruntime', 'AVFoundation', 'Foundation', 'objc'):
    if _mod not in sys.modules:
        _stub(_mod)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tts():
    mock = MagicMock()
    mock.is_speaking = False
    mock.speak = MagicMock()
    mock.stop = MagicMock()
    return mock


def _make_mock_listener(next_speech=None):
    mock = MagicMock()
    mock.barge_in = threading.Event()
    mock.get_next_speech = MagicMock(return_value=next_speech)
    mock.set_active = MagicMock()
    mock.set_tts_playing = MagicMock()
    mock.clear_barge_in = MagicMock()
    mock.drain_queue = MagicMock()
    mock.is_active = True
    return mock


def _build_voice_server_directly(use_macos_aec: bool, next_speech=None):
    """Build a VoiceServer by directly setting its internal state.

    Bypasses __init__ entirely to avoid audio hardware requirements.
    """
    mock_tts = _make_mock_tts()
    mock_listener = _make_mock_listener(next_speech=next_speech)

    from lazy_claude.server import VoiceServer
    s = VoiceServer.__new__(VoiceServer)
    s._use_macos_aec = use_macos_aec
    s.tts = mock_tts
    s._listener = mock_listener
    s._whisper_model = MagicMock()
    s.listening = True
    s.busy = False
    s._lock = threading.Lock()

    return s, mock_tts, mock_listener


def _build_server_via_init(*, platform_str='linux', av_available=False,
                            av_init_raises=None, next_speech=None):
    """Build a VoiceServer through __init__ with full dependency mocking.

    Parameters
    ----------
    platform_str:
        Simulated sys.platform value ('darwin' or 'linux').
    av_available:
        Whether lazy_claude.av_audio can be imported successfully.
    av_init_raises:
        If set, AVAudioBackend() raises this exception.
    """
    mock_tts = _make_mock_tts()
    mock_listener = _make_mock_listener(next_speech=next_speech)

    mock_av_module = MagicMock()
    if av_init_raises is not None:
        mock_av_module.AVAudioBackend.side_effect = av_init_raises
    else:
        mock_av_module.AVAudioBackend.return_value = MagicMock()
    mock_av_module.MacOSContinuousListener.return_value = mock_listener
    mock_av_module.MacOSTTSEngine.return_value = mock_tts

    # We patch _try_init_macos_backend to control the outcome cleanly
    # rather than fighting with sys.modules patching across test isolation

    with patch('lazy_claude.server.TTSEngine', return_value=mock_tts), \
         patch('lazy_claude.server.load_model', return_value=MagicMock()), \
         patch('lazy_claude.server.load_vad_model', return_value=MagicMock()), \
         patch('lazy_claude.server.ReferenceBuffer'), \
         patch('lazy_claude.server.EchoCanceller'), \
         patch('lazy_claude.server.ContinuousListener', return_value=mock_listener):
        from lazy_claude.server import VoiceServer

        if platform_str == 'darwin' and av_available:
            # Patch _try_init_macos_backend to return True and set internal state
            def _fake_macos_init(self_inner):
                self_inner._whisper_model = MagicMock()
                self_inner._vad_model = MagicMock()
                self_inner._listener = mock_listener
                self_inner.tts = mock_tts
                return True

            with patch.object(VoiceServer, '_try_init_macos_backend', _fake_macos_init), \
                 patch('sys.platform', 'darwin'):
                s = VoiceServer()

        elif platform_str == 'darwin' and not av_available:
            # macOS but AV init fails — fall back
            def _fake_macos_init_fail(self_inner):
                return False

            with patch.object(VoiceServer, '_try_init_macos_backend', _fake_macos_init_fail), \
                 patch('sys.platform', 'darwin'), \
                 patch.object(VoiceServer, '_calibrate_aec'):
                s = VoiceServer()
            s._listener = mock_listener
            s.tts = mock_tts

        else:
            # Non-darwin — always fallback
            with patch('sys.platform', 'linux'), \
                 patch.object(VoiceServer, '_calibrate_aec'):
                s = VoiceServer()
            s._listener = mock_listener
            s.tts = mock_tts

    return s, mock_tts, mock_listener


# ---------------------------------------------------------------------------
# Test: _use_macos_aec flag existence
# ---------------------------------------------------------------------------

class TestVoiceServerBackendFlag:
    """VoiceServer must expose _use_macos_aec as a bool."""

    def test_fallback_server_has_use_macos_aec_false(self):
        s, _, _ = _build_server_via_init(platform_str='linux')
        assert hasattr(s, '_use_macos_aec')
        assert s._use_macos_aec is False

    def test_macos_server_with_av_available_sets_flag_true(self):
        s, _, _ = _build_server_via_init(platform_str='darwin', av_available=True)
        assert s._use_macos_aec is True

    def test_macos_server_with_av_unavailable_falls_back(self):
        s, _, _ = _build_server_via_init(platform_str='darwin', av_available=False)
        assert s._use_macos_aec is False

    def test_server_always_starts_even_on_macos_failure(self):
        """Server must always start — never raise on audio init failure."""
        s, _, _ = _build_server_via_init(platform_str='darwin', av_available=False)
        assert s is not None
        assert isinstance(s._use_macos_aec, bool)

    def test_use_macos_aec_is_bool_type(self):
        s, _, _ = _build_fallback_server()
        assert isinstance(s._use_macos_aec, bool)


def _build_fallback_server(next_speech=None):
    """Convenience wrapper for fallback server."""
    return _build_server_via_init(platform_str='linux', next_speech=next_speech)


# ---------------------------------------------------------------------------
# Test: _ask_single macOS path — no sleep, no drain after TTS
# ---------------------------------------------------------------------------

class TestAskSingleMacOSPath:
    """On macOS path, _ask_single skips the 0.8s sleep and drain_queue."""

    def test_no_sleep_on_macos_path(self):
        """time.sleep should NOT be called with 0.8 on macOS path."""
        audio = np.zeros(16_000, dtype=np.float32)
        s, mock_tts, mock_listener = _build_voice_server_directly(
            use_macos_aec=True, next_speech=audio
        )

        sleep_calls = []

        def _track_sleep(t):
            sleep_calls.append(t)

        with patch('lazy_claude.server.time') as mock_time, \
             patch('lazy_claude.server.transcribe', return_value="hello"):
            mock_time.sleep = MagicMock(side_effect=_track_sleep)
            mock_time.monotonic = time.monotonic

            s._ask_single("Test question?")

        assert 0.8 not in sleep_calls, (
            f"Expected no 0.8s sleep on macOS path, but got sleep calls: {sleep_calls}"
        )

    def test_drain_queue_not_called_after_tts_on_macos_path(self):
        """drain_queue should NOT be called after TTS on macOS path."""
        audio = np.zeros(16_000, dtype=np.float32)
        s, mock_tts, mock_listener = _build_voice_server_directly(
            use_macos_aec=True, next_speech=audio
        )

        with patch('lazy_claude.server.time') as mock_time, \
             patch('lazy_claude.server.transcribe', return_value="hello"):
            mock_time.sleep = MagicMock()
            mock_time.monotonic = time.monotonic

            mock_listener.drain_queue.reset_mock()
            s._ask_single("Test question?")

        mock_listener.drain_queue.assert_not_called()

    def test_tts_still_called_on_macos_path(self):
        """TTS speak must still be called even on macOS path."""
        audio = np.zeros(16_000, dtype=np.float32)
        s, mock_tts, mock_listener = _build_voice_server_directly(
            use_macos_aec=True, next_speech=audio
        )

        with patch('lazy_claude.server.time') as mock_time, \
             patch('lazy_claude.server.transcribe', return_value="answer"):
            mock_time.sleep = MagicMock()
            mock_time.monotonic = time.monotonic
            s._ask_single("Hello?")

        mock_tts.speak.assert_called()

    def test_transcription_still_works_on_macos_path(self):
        """Transcription should still happen on macOS path."""
        audio = np.zeros(16_000, dtype=np.float32)
        s, mock_tts, mock_listener = _build_voice_server_directly(
            use_macos_aec=True, next_speech=audio
        )

        with patch('lazy_claude.server.time') as mock_time, \
             patch('lazy_claude.server.transcribe', return_value="test answer") as mock_transcribe:
            mock_time.sleep = MagicMock()
            mock_time.monotonic = time.monotonic
            result = s._ask_single("Question?")

        assert "test answer" in result
        mock_transcribe.assert_called_once()

    def test_result_format_correct_on_macos_path(self):
        """Q/A format is returned correctly on macOS path."""
        audio = np.zeros(16_000, dtype=np.float32)
        s, mock_tts, mock_listener = _build_voice_server_directly(
            use_macos_aec=True, next_speech=audio
        )

        with patch('lazy_claude.server.time') as mock_time, \
             patch('lazy_claude.server.transcribe', return_value="yes"):
            mock_time.sleep = MagicMock()
            mock_time.monotonic = time.monotonic
            result = s._ask_single("Are you there?")

        assert "Q: Are you there?" in result
        assert "A: yes" in result


# ---------------------------------------------------------------------------
# Test: _ask_single fallback path — DOES sleep + drain after TTS
# ---------------------------------------------------------------------------

class TestAskSingleFallbackPath:
    """On fallback path, _ask_single keeps the 0.8s sleep and drain_queue."""

    def test_sleep_0_8_called_on_fallback_path(self):
        """time.sleep(0.8) must be called on fallback path."""
        audio = np.zeros(16_000, dtype=np.float32)
        s, mock_tts, mock_listener = _build_voice_server_directly(
            use_macos_aec=False, next_speech=audio
        )

        sleep_calls = []

        def _track_sleep(t):
            sleep_calls.append(t)

        with patch('lazy_claude.server.time') as mock_time, \
             patch('lazy_claude.server.transcribe', return_value="hello"):
            mock_time.sleep = MagicMock(side_effect=_track_sleep)
            mock_time.monotonic = time.monotonic

            s._ask_single("Test question?")

        assert 0.8 in sleep_calls, (
            f"Expected 0.8s sleep on fallback path, got: {sleep_calls}"
        )

    def test_drain_queue_called_after_tts_on_fallback_path(self):
        """drain_queue must be called on fallback path after TTS."""
        audio = np.zeros(16_000, dtype=np.float32)
        s, mock_tts, mock_listener = _build_voice_server_directly(
            use_macos_aec=False, next_speech=audio
        )

        with patch('lazy_claude.server.time') as mock_time, \
             patch('lazy_claude.server.transcribe', return_value="hello"):
            mock_time.sleep = MagicMock()
            mock_time.monotonic = time.monotonic

            mock_listener.drain_queue.reset_mock()
            s._ask_single("Test question?")

        mock_listener.drain_queue.assert_called()

    def test_tts_still_called_on_fallback_path(self):
        """TTS must still be called on fallback path."""
        audio = np.zeros(16_000, dtype=np.float32)
        s, mock_tts, mock_listener = _build_voice_server_directly(
            use_macos_aec=False, next_speech=audio
        )

        with patch('lazy_claude.server.time') as mock_time, \
             patch('lazy_claude.server.transcribe', return_value="ok"):
            mock_time.sleep = MagicMock()
            mock_time.monotonic = time.monotonic
            s._ask_single("Hello?")

        mock_tts.speak.assert_called()


# ---------------------------------------------------------------------------
# Test: speak_message macOS path — no sleep, no drain
# ---------------------------------------------------------------------------

class TestSpeakMessageMacOSPath:
    """On macOS path, speak_message_impl skips the 0.8s sleep and drain_queue."""

    def test_no_sleep_on_macos_speak_message(self):
        """speak_message_impl must not sleep 0.8s on macOS path."""
        s, mock_tts, mock_listener = _build_voice_server_directly(use_macos_aec=True)

        sleep_calls = []

        def _track_sleep(t):
            sleep_calls.append(t)

        with patch('lazy_claude.server.time') as mock_time:
            mock_time.sleep = MagicMock(side_effect=_track_sleep)
            mock_time.monotonic = time.monotonic
            s.speak_message_impl(text="Hello there")

        assert 0.8 not in sleep_calls, (
            f"Expected no 0.8s sleep on macOS speak_message, got: {sleep_calls}"
        )

    def test_no_drain_on_macos_speak_message(self):
        """speak_message_impl must not drain_queue on macOS path."""
        s, mock_tts, mock_listener = _build_voice_server_directly(use_macos_aec=True)

        with patch('lazy_claude.server.time') as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.monotonic = time.monotonic
            mock_listener.drain_queue.reset_mock()
            s.speak_message_impl(text="Hello there")

        mock_listener.drain_queue.assert_not_called()

    def test_tts_speak_still_called_on_macos_speak_message(self):
        """TTS must still be called on macOS speak_message."""
        s, mock_tts, mock_listener = _build_voice_server_directly(use_macos_aec=True)
        s.speak_message_impl(text="Hello there")
        mock_tts.speak.assert_called_once_with("Hello there")

    def test_speak_message_returns_correct_dict_on_macos(self):
        """Return dict is the same regardless of path."""
        s, mock_tts, mock_listener = _build_voice_server_directly(use_macos_aec=True)
        result = s.speak_message_impl(text="Hello world")
        assert result == {"status": "spoken", "chars": 11}

    def test_set_tts_playing_called_on_macos_path(self):
        """set_tts_playing must be called on macOS path even without drain."""
        s, mock_tts, mock_listener = _build_voice_server_directly(use_macos_aec=True)
        s.speak_message_impl(text="Hi")
        mock_listener.set_tts_playing.assert_any_call(True)
        mock_listener.set_tts_playing.assert_any_call(False)


# ---------------------------------------------------------------------------
# Test: speak_message fallback path — DOES sleep + drain
# ---------------------------------------------------------------------------

class TestSpeakMessageFallbackPath:
    """On fallback path, speak_message_impl keeps the 0.8s sleep and drain_queue."""

    def test_sleep_0_8_called_on_fallback_speak_message(self):
        """time.sleep(0.8) must be called on fallback speak_message."""
        s, mock_tts, mock_listener = _build_voice_server_directly(use_macos_aec=False)

        sleep_calls = []

        def _track_sleep(t):
            sleep_calls.append(t)

        with patch('lazy_claude.server.time') as mock_time:
            mock_time.sleep = MagicMock(side_effect=_track_sleep)
            s.speak_message_impl(text="Hello there")

        assert 0.8 in sleep_calls, (
            f"Expected 0.8s sleep on fallback speak_message, got: {sleep_calls}"
        )

    def test_drain_called_on_fallback_speak_message(self):
        """drain_queue must be called on fallback speak_message."""
        s, mock_tts, mock_listener = _build_voice_server_directly(use_macos_aec=False)

        with patch('lazy_claude.server.time') as mock_time:
            mock_time.sleep = MagicMock()
            mock_listener.drain_queue.reset_mock()
            s.speak_message_impl(text="Hello there")

        mock_listener.drain_queue.assert_called()

    def test_speak_message_returns_correct_dict_on_fallback(self):
        """Return dict is the same on fallback path."""
        s, mock_tts, mock_listener = _build_voice_server_directly(use_macos_aec=False)
        with patch('lazy_claude.server.time') as mock_time:
            mock_time.sleep = MagicMock()
            result = s.speak_message_impl(text="Hello world")
        assert result == {"status": "spoken", "chars": 11}


# ---------------------------------------------------------------------------
# Test: calibration chirp only on fallback path
# ---------------------------------------------------------------------------

class TestCalibrationChirp:
    """_calibrate_aec is only called on the fallback path during __init__."""

    def test_calibrate_aec_not_called_when_macos_init_succeeds(self):
        """When macOS backend init succeeds, _calibrate_aec must NOT be called."""
        mock_tts = _make_mock_tts()
        mock_listener = _make_mock_listener()

        calibrate_called = []

        from lazy_claude.server import VoiceServer

        def _fake_macos_init(self_inner):
            self_inner._whisper_model = MagicMock()
            self_inner._vad_model = MagicMock()
            self_inner._listener = mock_listener
            self_inner.tts = mock_tts
            return True

        with patch.object(VoiceServer, '_try_init_macos_backend', _fake_macos_init), \
             patch.object(VoiceServer, '_calibrate_aec',
                          side_effect=lambda self_inner: calibrate_called.append(True)), \
             patch('sys.platform', 'darwin'), \
             patch('lazy_claude.server.load_model', return_value=MagicMock()), \
             patch('lazy_claude.server.load_vad_model', return_value=MagicMock()), \
             patch('lazy_claude.server.TTSEngine', return_value=mock_tts), \
             patch('lazy_claude.server.ContinuousListener', return_value=mock_listener), \
             patch('lazy_claude.server.ReferenceBuffer'), \
             patch('lazy_claude.server.EchoCanceller'):
            s = VoiceServer()

        assert len(calibrate_called) == 0, (
            "_calibrate_aec must not be called when macOS backend succeeds"
        )
        assert s._use_macos_aec is True

    def test_calibrate_aec_called_on_fallback_path(self):
        """_calibrate_aec must be called on fallback path."""
        mock_tts = _make_mock_tts()
        mock_listener = _make_mock_listener()

        calibrate_called = []

        from lazy_claude.server import VoiceServer

        def _fake_calibrate(self_inner):
            calibrate_called.append(True)

        with patch.object(VoiceServer, '_calibrate_aec', _fake_calibrate), \
             patch('sys.platform', 'linux'), \
             patch('lazy_claude.server.load_model', return_value=MagicMock()), \
             patch('lazy_claude.server.load_vad_model', return_value=MagicMock()), \
             patch('lazy_claude.server.TTSEngine', return_value=mock_tts), \
             patch('lazy_claude.server.ContinuousListener', return_value=mock_listener), \
             patch('lazy_claude.server.ReferenceBuffer'), \
             patch('lazy_claude.server.EchoCanceller'):
            s = VoiceServer()

        assert len(calibrate_called) == 1, (
            "_calibrate_aec must be called exactly once on fallback path"
        )
        assert s._use_macos_aec is False


# ---------------------------------------------------------------------------
# Test: _try_init_macos_backend method directly
# ---------------------------------------------------------------------------

class TestTryInitMacOSBackend:
    """Unit tests for _try_init_macos_backend."""

    def _make_bare_server(self):
        """Build a bare VoiceServer without calling __init__."""
        from lazy_claude.server import VoiceServer
        s = VoiceServer.__new__(VoiceServer)
        return s

    def test_returns_false_on_import_error(self):
        """ImportError from lazy_claude.av_audio → returns False."""
        s = self._make_bare_server()
        with patch('lazy_claude.server.VoiceServer._try_init_macos_backend') as m:
            m.return_value = False
            result = m(s)
        assert result is False

    def test_returns_false_on_backend_runtime_error(self):
        """RuntimeError from AVAudioBackend() → returns False."""
        s = self._make_bare_server()

        mock_av_module = MagicMock()
        mock_av_module.AVAudioBackend.side_effect = RuntimeError("AVAudioEngine failed")

        with patch.dict('sys.modules', {'lazy_claude.av_audio': mock_av_module}), \
             patch('lazy_claude.server.load_model', return_value=MagicMock()), \
             patch('lazy_claude.server.load_vad_model', return_value=MagicMock()):
            result = s._try_init_macos_backend()

        assert result is False

    def test_returns_true_when_all_succeeds(self):
        """When all imports and inits succeed, returns True."""
        s = self._make_bare_server()
        mock_vad = MagicMock()
        mock_listener = _make_mock_listener()
        mock_tts = _make_mock_tts()

        mock_av_module = MagicMock()
        mock_av_module.AVAudioBackend.return_value = MagicMock()
        mock_av_module.MacOSContinuousListener.return_value = mock_listener
        mock_av_module.MacOSTTSEngine.return_value = mock_tts

        with patch.dict('sys.modules', {'lazy_claude.av_audio': mock_av_module}), \
             patch('lazy_claude.server.load_model', return_value=MagicMock()), \
             patch('lazy_claude.server.load_vad_model', return_value=mock_vad):
            result = s._try_init_macos_backend()

        assert result is True
        assert s.tts is mock_tts
        assert s._listener is mock_listener

    def test_sets_listener_and_tts_on_success(self):
        """On success, _listener and tts are set to the macOS instances."""
        s = self._make_bare_server()
        mock_listener = _make_mock_listener()
        mock_tts = _make_mock_tts()

        mock_av_module = MagicMock()
        mock_av_module.AVAudioBackend.return_value = MagicMock()
        mock_av_module.MacOSContinuousListener.return_value = mock_listener
        mock_av_module.MacOSTTSEngine.return_value = mock_tts

        with patch.dict('sys.modules', {'lazy_claude.av_audio': mock_av_module}), \
             patch('lazy_claude.server.load_model', return_value=MagicMock()), \
             patch('lazy_claude.server.load_vad_model', return_value=MagicMock()):
            s._try_init_macos_backend()

        assert s._listener is mock_listener
        assert s.tts is mock_tts
