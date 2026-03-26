"""Tests for server.py — FastMCP server with voice tools.

These tests do NOT require audio hardware, a real TTS model, or a live MCP
client.  They verify:

- Module imports without error
- VoiceServer class is importable and instantiable (with all deps mocked)
- Tool functions exist on the server (ask_user_voice, speak_message, toggle_listening)
- toggle_listening returns correct dict
- speak_message returns correct dict
- ask_user_voice skips recording when listening is disabled (returns skipped message)
- ask_user_voice returns timeout message when get_next_speech returns None
- ask_user_voice returns formatted Q/A string when transcription succeeds
- ask_user_voice handles multiple questions in sequence
- Concurrent call protection: second call raises/returns busy message
- Graceful mic error handling: RuntimeError from get_next_speech returns error text
"""

from __future__ import annotations

import sys
import threading
import time
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
import pytest


def _stub(name: str) -> MagicMock:
    m = MagicMock()
    m.__name__ = name
    sys.modules[name] = m
    return m


for _mod in ('sounddevice', 'kokoro'):
    if _mod not in sys.modules:
        _stub(_mod)

# Patch continuation timeout to near-zero so tests don't wait 3 real seconds
import lazy_claude.server as _server_mod
_server_mod._CONTINUATION_RESPONSE_TIMEOUT = 0.01
_server_mod._INITIAL_RESPONSE_TIMEOUT = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tts():
    """Return a mock TTSEngine that does nothing."""
    mock = MagicMock()
    mock.is_speaking = False
    mock.speak = MagicMock()
    mock.stop = MagicMock()
    return mock


def _make_transcribe_result(text: str = "hello world", no_speech_prob: float = 0.1):
    """Return a TranscribeResult for use in mocks."""
    from lazy_claude.stt import TranscribeResult
    return TranscribeResult(text=text, no_speech_prob=no_speech_prob)


def _make_mock_transcribe(return_value="hello world"):
    """Return a mock transcribe function returning a TranscribeResult."""
    from lazy_claude.stt import TranscribeResult
    result = TranscribeResult(text=return_value, no_speech_prob=0.1)
    return MagicMock(return_value=result)


def _make_mock_listener(next_speech=None):
    """Return a mock ContinuousListener."""
    mock = MagicMock()
    mock.barge_in = threading.Event()
    if next_speech is not None:
        # Return next_speech exactly once, then always None thereafter
        _returned = [False]
        def _get_next_speech(**kwargs):
            if not _returned[0]:
                _returned[0] = True
                return next_speech
            return None
        mock.get_next_speech = MagicMock(side_effect=_get_next_speech)
    else:
        mock.get_next_speech = MagicMock(return_value=None)
    mock.get_last_input_at = MagicMock(return_value=None)
    mock.pop_barge_in_candidate = MagicMock(return_value=None)
    mock.set_active = MagicMock()
    mock.set_tts_playing = MagicMock()
    mock.clear_barge_in = MagicMock()
    mock.drain_queue = MagicMock()
    return mock


def _make_server(mock_tts=None, next_speech=None):
    """Build a VoiceServer with all heavy deps mocked."""
    if mock_tts is None:
        mock_tts = _make_mock_tts()
    mock_listener = _make_mock_listener(next_speech=next_speech)
    with patch('lazy_claude.server.TTSEngine', return_value=mock_tts), \
         patch('lazy_claude.server.load_model', return_value=MagicMock()), \
         patch('lazy_claude.server.load_vad_model', return_value=MagicMock()), \
         patch('lazy_claude.server.ReferenceBuffer'), \
         patch('lazy_claude.server.EchoCanceller'), \
         patch('lazy_claude.server.ContinuousListener', return_value=mock_listener):
        from lazy_claude.server import VoiceServer
        s = VoiceServer()
    s.tts = mock_tts
    # Ensure the mock listener is directly accessible
    s._listener = mock_listener
    s._try_acquire_voice_device = MagicMock(return_value=123)
    s._release_voice_device = MagicMock()
    mock_listener.reset_mock()
    return s, mock_tts, mock_listener


# ---------------------------------------------------------------------------
# VoiceServer instantiation
# ---------------------------------------------------------------------------


class TestVoiceServerInit:
    def test_server_creates_without_error(self):
        server, _, _ = _make_server()
        assert server is not None


# ---------------------------------------------------------------------------
# toggle_listening
# ---------------------------------------------------------------------------


class TestToggleListening:
    def test_toggle_off_returns_dict(self):
        server, _, _ = _make_server()
        result = server.toggle_listening_impl(enabled=False)
        assert result == {"listening": False}

    def test_toggle_on_returns_dict(self):
        server, _, _ = _make_server()
        result = server.toggle_listening_impl(enabled=True)
        assert result == {"listening": True}

    def test_toggle_off_sets_listening_false(self):
        server, _, _ = _make_server()
        server.toggle_listening_impl(enabled=False)
        assert server.listening is False

    def test_toggle_on_sets_listening_true(self):
        server, _, _ = _make_server()
        server.toggle_listening_impl(enabled=False)
        server.toggle_listening_impl(enabled=True)
        assert server.listening is True


# ---------------------------------------------------------------------------
# speak_message
# ---------------------------------------------------------------------------


class TestSpeakMessage:
    def test_speak_message_returns_correct_dict(self):
        server, mock_tts, _ = _make_server()
        result = server.speak_message_impl(text="Hello world")
        assert result == {"status": "spoken", "chars": 11}

    def test_speak_message_calls_tts_speak(self):
        server, mock_tts, _ = _make_server()
        server.speak_message_impl(text="Hi there")
        mock_tts.speak.assert_called_once_with("Hi there")

    def test_speak_message_empty_string(self):
        server, mock_tts, _ = _make_server()
        result = server.speak_message_impl(text="")
        assert result["chars"] == 0
        assert result["status"] == "spoken"

    def test_speak_message_returns_char_count(self):
        server, mock_tts, _ = _make_server()
        text = "A" * 50
        result = server.speak_message_impl(text=text)
        assert result["chars"] == 50

    def test_speak_message_calls_set_tts_playing(self):
        """speak_message_impl sets TTS flag; on fallback path also drains after."""
        server, mock_tts, mock_listener = _make_server()
        server._use_macos_aec = False  # ensure fallback path for this test
        server.speak_message_impl(text="Hello")
        mock_listener.set_tts_playing.assert_any_call(True)
        mock_listener.set_tts_playing.assert_any_call(False)
        mock_listener.drain_queue.assert_called_once()

    def test_speak_message_returns_busy_when_voice_device_locked(self):
        server, mock_tts, mock_listener = _make_server()
        with patch.object(server, "_try_acquire_voice_device", return_value=None):
            result = server.speak_message_impl(text="Hello world")
        assert result == {"status": "busy", "chars": 11}
        mock_tts.speak.assert_not_called()


# ---------------------------------------------------------------------------
# ask_user_voice — listening disabled
# ---------------------------------------------------------------------------


class TestAskUserVoiceListeningDisabled:
    def test_skips_recording_when_disabled(self):
        server, mock_tts, mock_listener = _make_server()
        server.toggle_listening_impl(enabled=False)
        result = server.ask_user_voice_impl(questions=["What is your name?"])
        # get_next_speech should never be called when listening is off
        mock_listener.get_next_speech.assert_not_called()

    def test_returns_skipped_message_when_disabled(self):
        server, mock_tts, mock_listener = _make_server()
        server.toggle_listening_impl(enabled=False)
        result = server.ask_user_voice_impl(questions=["What is your name?"])
        assert "skipped" in result.lower() or "listening paused" in result.lower()

    def test_still_speaks_question_when_disabled(self):
        server, mock_tts, mock_listener = _make_server()
        server.toggle_listening_impl(enabled=False)
        server.ask_user_voice_impl(questions=["Are you ready?"])
        mock_tts.speak.assert_called()

    def test_returns_busy_when_voice_device_locked(self):
        server, mock_tts, mock_listener = _make_server()
        with patch.object(server, "_try_acquire_voice_device", return_value=None):
            result = server.ask_user_voice_impl(questions=["What is your name?"])
        assert "voice device" in result.lower()
        mock_listener.set_active.assert_not_called()


# ---------------------------------------------------------------------------
# ask_user_voice — timeout (get_next_speech returns None)
# ---------------------------------------------------------------------------


class TestAskUserVoiceTimeout:
    def test_returns_timed_out_message(self):
        server, mock_tts, mock_listener = _make_server(next_speech=None)
        result = server.ask_user_voice_impl(questions=["Hello?"])
        assert "timed out" in result.lower() or "no response" in result.lower()

    def test_result_contains_question(self):
        server, mock_tts, mock_listener = _make_server(next_speech=None)
        result = server.ask_user_voice_impl(questions=["What time is it?"])
        assert "What time is it?" in result


# ---------------------------------------------------------------------------
# ask_user_voice — successful transcription
# ---------------------------------------------------------------------------


class TestAskUserVoiceSuccess:
    def _dummy_audio(self):
        return np.zeros(16_000, dtype=np.float32)

    def test_returns_formatted_qa_string(self):
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        with patch('lazy_claude.server.transcribe',
                   return_value=_make_transcribe_result("I am fine")):
            result = server.ask_user_voice_impl(questions=["How are you?"])
        assert "Q: How are you?" in result
        assert "A: I am fine" in result

    def test_multiple_questions_all_in_result(self):
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server()
        # Each question: primary call gets audio, continuation expires immediately
        responses = [audio, audio]
        call_count = [0]
        def _speech(**kwargs):
            call_count[0] += 1
            if responses:
                return responses.pop(0)
            return None
        mock_listener.get_next_speech.side_effect = _speech

        from lazy_claude.stt import TranscribeResult
        answers = ["Paris", "42"]
        transcribe_count = [0]

        def mock_transcribe(a, model=None, **kwargs):
            idx = min(transcribe_count[0], len(answers) - 1)
            transcribe_count[0] += 1
            return TranscribeResult(text=answers[idx], no_speech_prob=0.1)

        # Patch time.monotonic so continuation deadline expires after 1st continuation check
        _base = [time.monotonic()]
        _step = [0]
        def _fast_monotonic():
            # Each call advances time by 10s so continuation deadline expires immediately
            _step[0] += 10.0
            return _base[0] + _step[0]

        with patch('lazy_claude.server.time.monotonic', side_effect=_fast_monotonic), \
             patch('lazy_claude.server.transcribe', side_effect=mock_transcribe):
            result = server.ask_user_voice_impl(
                questions=["Capital of France?", "Answer to everything?"]
            )
        assert "Q: Capital of France?" in result
        assert "A: Paris" in result
        assert "Q: Answer to everything?" in result
        assert "A: 42" in result

    def test_trailing_over_keyword_is_removed_from_answer(self):
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        with patch('lazy_claude.server.transcribe',
                   return_value=_make_transcribe_result("deploy it now over")):
            result = server.ask_user_voice_impl(questions=["What should I do?"])
        assert "A: deploy it now" in result
        assert "A: deploy it now over" not in result

    def test_trailing_over_keyword_is_case_insensitive(self):
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        with patch('lazy_claude.server.transcribe',
                   return_value=_make_transcribe_result("ship it OVER.")):
            result = server.ask_user_voice_impl(questions=["Status?"])
        assert "A: ship it" in result
        assert "OVER" not in result

    def test_pause_without_over_waits_for_continuation_then_finishes(self):
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server()
        responses = [audio, None]
        mock_listener.get_next_speech.side_effect = lambda **_kwargs: (
            responses.pop(0) if responses else None
        )
        with patch('lazy_claude.server.transcribe',
                   return_value=_make_transcribe_result("first part")):
            result = server.ask_user_voice_impl(questions=["Continue?"])
        assert "A: first part" in result
        assert mock_listener.get_next_speech.call_args_list[0].kwargs["timeout"] == _server_mod._INITIAL_RESPONSE_TIMEOUT
        # Continuation calls use the shorter timeout
        assert mock_listener.get_next_speech.call_args_list[1].kwargs["timeout"] <= _server_mod._CONTINUATION_RESPONSE_TIMEOUT

    def test_multiple_segments_are_joined_before_timeout(self):
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server()
        responses = [audio, audio, None]
        mock_listener.get_next_speech.side_effect = lambda **_kwargs: (
            responses.pop(0) if responses else None
        )
        with patch('lazy_claude.server.transcribe', side_effect=[
            _make_transcribe_result("first part"),
            _make_transcribe_result("second part"),
        ]):
            result = server.ask_user_voice_impl(questions=["Continue?"])
        assert "A: first part second part" in result

    def test_continuation_window_extends_after_fresh_input(self):
        audio = self._dummy_audio()
        server, _, mock_listener = _make_server()
        mock_listener.get_next_speech.side_effect = [None, audio]

        # Use timestamps relative to the patched _CONTINUATION_RESPONSE_TIMEOUT (0.01s)
        # First monotonic: start time. Second: still within deadline. Third: past original deadline but fresh input extended it.
        timeout = _server_mod._CONTINUATION_RESPONSE_TIMEOUT
        t0 = 100.0
        with patch('lazy_claude.server.time.monotonic', side_effect=[t0, t0, t0 + timeout + 0.001]), \
             patch.object(server, '_get_last_input_at', side_effect=[None, t0 + timeout * 0.5]):
            result = server._wait_for_continuation_speech()

        assert result is audio
        assert mock_listener.get_next_speech.call_count == 2

    def test_listener_is_only_active_during_voice_turn(self):
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        with patch('lazy_claude.server.transcribe',
                   return_value=_make_transcribe_result("ready")):
            server.ask_user_voice_impl(questions=["Status?"])
        assert mock_listener.set_active.call_args_list[0].args == (True,)
        assert mock_listener.set_active.call_args_list[-1].args == (False,)

    def test_empty_transcription_returned_verbatim(self):
        """Empty transcription should still be returned, not dropped."""
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        with patch('lazy_claude.server.transcribe',
                   return_value=_make_transcribe_result("")):
            result = server.ask_user_voice_impl(questions=["Say something?"])
        assert "Q: Say something?" in result
        assert "A:" in result

    def test_tts_speak_called_for_each_question(self):
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        with patch('lazy_claude.server.transcribe',
                   return_value=_make_transcribe_result("yes")):
            server.ask_user_voice_impl(questions=["Q1?", "Q2?", "Q3?"])
        assert mock_tts.speak.call_count == 3

    def test_stop_barge_in_stops_tts(self):
        audio = self._dummy_audio()
        candidate_audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        mock_tts.speak.side_effect = lambda *_args, **_kwargs: time.sleep(0.2)
        mock_listener.pop_barge_in_candidate.side_effect = [candidate_audio, None, None]

        with patch(
            'lazy_claude.server.transcribe',
            side_effect=[
                _make_transcribe_result("stop"),
                _make_transcribe_result("ready over"),
            ],
        ):
            result = server.ask_user_voice_impl(questions=["Status?"])

        mock_tts.stop.assert_called_once()
        assert mock_listener.barge_in.is_set()
        assert "A: ready" in result

    def test_non_stop_barge_in_candidate_does_not_stop_tts(self):
        audio = self._dummy_audio()
        candidate_audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        mock_tts.speak.side_effect = lambda *_args, **_kwargs: time.sleep(0.1)
        mock_listener.pop_barge_in_candidate.side_effect = [candidate_audio, None, None]

        with patch(
            'lazy_claude.server.transcribe',
            side_effect=[
                _make_transcribe_result("excuse me"),
                _make_transcribe_result("ready over"),
            ],
        ):
            result = server.ask_user_voice_impl(questions=["Status?"])

        mock_tts.stop.assert_not_called()
        assert not mock_listener.barge_in.is_set()
        assert "A: ready" in result

    @pytest.mark.parametrize("phrase", ["STOP", "stop there", "please stop now"])
    def test_stop_barge_in_accepts_common_variants(self, phrase):
        audio = self._dummy_audio()
        candidate_audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        mock_tts.speak.side_effect = lambda *_args, **_kwargs: time.sleep(0.2)
        mock_listener.pop_barge_in_candidate.side_effect = [candidate_audio, None, None]

        with patch(
            'lazy_claude.server.transcribe',
            side_effect=[
                _make_transcribe_result(phrase),
                _make_transcribe_result("ready over"),
            ],
        ):
            result = server.ask_user_voice_impl(questions=["Status?"])

        mock_tts.stop.assert_called_once()
        assert mock_listener.barge_in.is_set()
        assert "A: ready" in result

    @pytest.mark.parametrize("phrase", ["don't stop", "please don't stop", "cannot stop this"])
    def test_stop_barge_in_rejects_negated_phrases(self, phrase):
        audio = self._dummy_audio()
        candidate_audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        mock_tts.speak.side_effect = lambda *_args, **_kwargs: time.sleep(0.1)
        mock_listener.pop_barge_in_candidate.side_effect = [candidate_audio, None, None]

        with patch(
            'lazy_claude.server.transcribe',
            side_effect=[
                _make_transcribe_result(phrase),
                _make_transcribe_result("ready over"),
            ],
        ):
            result = server.ask_user_voice_impl(questions=["Status?"])

        mock_tts.stop.assert_not_called()
        assert not mock_listener.barge_in.is_set()
        assert "A: ready" in result


# ---------------------------------------------------------------------------
# ask_user_voice — mic / listener error handling
# ---------------------------------------------------------------------------


class TestAskUserVoiceMicError:
    def test_get_next_speech_exception_returns_error_text_not_crash(self):
        server, mock_tts, mock_listener = _make_server()
        mock_listener.get_next_speech.side_effect = RuntimeError("mic denied")
        result = server.ask_user_voice_impl(questions=["Hello?"])
        # Should return something with error info, not raise
        assert result is not None
        assert isinstance(result, str)

    def test_get_next_speech_exception_result_contains_error_indicator(self):
        server, mock_tts, mock_listener = _make_server()
        mock_listener.get_next_speech.side_effect = RuntimeError("no mic")
        result = server.ask_user_voice_impl(questions=["Test?"])
        assert "error" in result.lower() or "mic" in result.lower() or "failed" in result.lower()


# ---------------------------------------------------------------------------
# Concurrent call protection
# ---------------------------------------------------------------------------


class TestConcurrentCallProtection:
    def test_busy_flag_rejects_concurrent_call(self):
        server, mock_tts, mock_listener = _make_server()
        # Simulate server is already busy
        server.busy = True
        result = server.ask_user_voice_impl(questions=["Are you busy?"])
        assert "busy" in result.lower() or "processing" in result.lower()

    def test_busy_flag_cleared_after_call(self):
        audio = np.zeros(16_000, dtype=np.float32)
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        with patch('lazy_claude.server.transcribe',
                   return_value=_make_transcribe_result("ok")):
            server.ask_user_voice_impl(questions=["Test?"])
        assert server.busy is False


# ---------------------------------------------------------------------------
# MCP tool registration (tools exist on the FastMCP app)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 2: Whisper noise filter (no_speech_prob threshold)
# ---------------------------------------------------------------------------


class TestNoSpeechProbFilter:
    """Phase 2: utterances with high no_speech_prob are discarded and looped."""

    def _dummy_audio(self):
        return np.zeros(16_000, dtype=np.float32)

    def _make_server_with_two_results(self, first_no_speech_prob, second_text):
        """Build server whose listener returns audio twice, with two transcribe results."""
        from lazy_claude.stt import TranscribeResult
        audio = self._dummy_audio()

        server, mock_tts, mock_listener = _make_server()
        # Override: return audio twice, then None (noise → discard → real speech → done)
        mock_listener.get_next_speech.side_effect = lambda **kwargs: (
            audio if mock_listener.get_next_speech.call_count <= 2 else None
        )

        call_count = [0]
        results = [
            TranscribeResult(text="noise", no_speech_prob=first_no_speech_prob),
            TranscribeResult(text=second_text, no_speech_prob=0.1),
        ]

        def mock_transcribe(a, model=None, **kwargs):
            idx = min(call_count[0], len(results) - 1)
            call_count[0] += 1
            return results[idx]

        return server, mock_listener, mock_transcribe

    def test_high_no_speech_prob_is_discarded_and_loops(self):
        """no_speech_prob > 0.6 → discard, call get_next_speech again, eventually return real text."""
        server, mock_listener, mock_transcribe = self._make_server_with_two_results(
            first_no_speech_prob=0.8, second_text="hello world"
        )
        with patch('lazy_claude.server.transcribe', side_effect=mock_transcribe):
            result = server.ask_user_voice_impl(questions=["Test?"])

        # Should have called get_next_speech twice (once for noise, once for real speech)
        assert mock_listener.get_next_speech.call_count >= 2
        # Final result must contain the real transcription
        assert "hello world" in result

    def test_low_no_speech_prob_is_forwarded(self):
        """no_speech_prob <= 0.6 → accepted immediately (no retry loop)."""
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        with patch('lazy_claude.server.transcribe',
                   return_value=_make_transcribe_result("all good", no_speech_prob=0.2)):
            result = server.ask_user_voice_impl(questions=["Test?"])

        # Should be called at least once (initial speech) — not retried due to discard
        assert mock_listener.get_next_speech.call_count >= 1
        assert "all good" in result

    def test_drain_queue_called_after_get_next_speech(self):
        """drain_queue() is called after each successful get_next_speech() to flush stale audio."""
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        with patch('lazy_claude.server.transcribe',
                   return_value=_make_transcribe_result("ok", no_speech_prob=0.1)):
            server.ask_user_voice_impl(questions=["Test?"])

        # drain_queue must have been called at least once during the Q&A session
        mock_listener.drain_queue.assert_called()

    def test_boundary_no_speech_prob_06_is_accepted(self):
        """Exactly 0.6 no_speech_prob → accepted (threshold is strictly > 0.6)."""
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        with patch('lazy_claude.server.transcribe',
                   return_value=_make_transcribe_result("boundary", no_speech_prob=0.6)):
            result = server.ask_user_voice_impl(questions=["Test?"])

        assert mock_listener.get_next_speech.call_count >= 1
        assert "boundary" in result


