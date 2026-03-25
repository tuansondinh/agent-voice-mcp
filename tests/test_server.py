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

import threading
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
import pytest
import asyncio


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


def _make_mock_transcribe(return_value="hello world"):
    """Return a mock transcribe function."""
    return MagicMock(return_value=return_value)


def _make_mock_listener(next_speech=None):
    """Return a mock ContinuousListener."""
    mock = MagicMock()
    mock.barge_in = threading.Event()
    mock.get_next_speech = MagicMock(return_value=next_speech)
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
    return s, mock_tts, mock_listener


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


class TestServerImport:
    def test_module_importable(self):
        import lazy_claude.server
        assert lazy_claude.server is not None

    def test_voice_server_importable(self):
        from lazy_claude.server import VoiceServer
        assert VoiceServer is not None

    def test_create_server_importable(self):
        from lazy_claude.server import create_server
        assert callable(create_server)


# ---------------------------------------------------------------------------
# VoiceServer instantiation
# ---------------------------------------------------------------------------


class TestVoiceServerInit:
    def test_server_creates_without_error(self):
        server, _, _ = _make_server()
        assert server is not None

    def test_listening_enabled_by_default(self):
        server, _, _ = _make_server()
        assert server.listening is True

    def test_busy_flag_false_by_default(self):
        server, _, _ = _make_server()
        assert server.busy is False


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

    def test_speak_message_does_not_call_set_tts_playing(self):
        """speak_message_impl must NOT call set_tts_playing (AEC handles echo)."""
        server, mock_tts, mock_listener = _make_server()
        server.speak_message_impl(text="Hello")
        mock_listener.set_tts_playing.assert_not_called()


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
        with patch('lazy_claude.server.transcribe', return_value="I am fine"):
            result = server.ask_user_voice_impl(questions=["How are you?"])
        assert "Q: How are you?" in result
        assert "A: I am fine" in result

    def test_multiple_questions_all_in_result(self):
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        answers = ["Paris", "42"]
        call_count = [0]

        def mock_transcribe(a, model=None):
            idx = call_count[0]
            call_count[0] += 1
            return answers[idx]

        with patch('lazy_claude.server.transcribe', side_effect=mock_transcribe):
            result = server.ask_user_voice_impl(
                questions=["Capital of France?", "Answer to everything?"]
            )
        assert "Q: Capital of France?" in result
        assert "A: Paris" in result
        assert "Q: Answer to everything?" in result
        assert "A: 42" in result

    def test_empty_transcription_returned_verbatim(self):
        """Empty transcription should still be returned, not dropped."""
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        with patch('lazy_claude.server.transcribe', return_value=""):
            result = server.ask_user_voice_impl(questions=["Say something?"])
        assert "Q: Say something?" in result
        assert "A:" in result

    def test_tts_speak_called_for_each_question(self):
        audio = self._dummy_audio()
        server, mock_tts, mock_listener = _make_server(next_speech=audio)
        with patch('lazy_claude.server.transcribe', return_value="yes"):
            server.ask_user_voice_impl(questions=["Q1?", "Q2?", "Q3?"])
        assert mock_tts.speak.call_count == 3


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
        with patch('lazy_claude.server.transcribe', return_value="ok"):
            server.ask_user_voice_impl(questions=["Test?"])
        assert server.busy is False


# ---------------------------------------------------------------------------
# MCP tool registration (tools exist on the FastMCP app)
# ---------------------------------------------------------------------------


class TestMCPToolRegistration:
    def test_create_server_returns_fastmcp_app(self):
        mock_tts = _make_mock_tts()
        with patch('lazy_claude.server.TTSEngine', return_value=mock_tts), \
             patch('lazy_claude.server.load_model', return_value=MagicMock()), \
             patch('lazy_claude.server.load_vad_model', return_value=MagicMock()), \
             patch('lazy_claude.server.ReferenceBuffer'), \
             patch('lazy_claude.server.EchoCanceller'), \
             patch('lazy_claude.server.ContinuousListener', return_value=_make_mock_listener()):
            from lazy_claude.server import create_server
            from mcp.server.fastmcp import FastMCP
            app, voice = create_server()
        assert isinstance(app, FastMCP)

    def test_ask_user_voice_tool_registered(self):
        mock_tts = _make_mock_tts()
        with patch('lazy_claude.server.TTSEngine', return_value=mock_tts), \
             patch('lazy_claude.server.load_model', return_value=MagicMock()), \
             patch('lazy_claude.server.load_vad_model', return_value=MagicMock()), \
             patch('lazy_claude.server.ReferenceBuffer'), \
             patch('lazy_claude.server.EchoCanceller'), \
             patch('lazy_claude.server.ContinuousListener', return_value=_make_mock_listener()):
            from lazy_claude.server import create_server
            app, voice = create_server()
        tool_names = [t.name for t in asyncio.run(app.list_tools())]
        assert "ask_user_voice" in tool_names

    def test_speak_message_tool_registered(self):
        mock_tts = _make_mock_tts()
        with patch('lazy_claude.server.TTSEngine', return_value=mock_tts), \
             patch('lazy_claude.server.load_model', return_value=MagicMock()), \
             patch('lazy_claude.server.load_vad_model', return_value=MagicMock()), \
             patch('lazy_claude.server.ReferenceBuffer'), \
             patch('lazy_claude.server.EchoCanceller'), \
             patch('lazy_claude.server.ContinuousListener', return_value=_make_mock_listener()):
            from lazy_claude.server import create_server
            app, voice = create_server()
        tool_names = [t.name for t in asyncio.run(app.list_tools())]
        assert "speak_message" in tool_names

    def test_toggle_listening_tool_registered(self):
        mock_tts = _make_mock_tts()
        with patch('lazy_claude.server.TTSEngine', return_value=mock_tts), \
             patch('lazy_claude.server.load_model', return_value=MagicMock()), \
             patch('lazy_claude.server.load_vad_model', return_value=MagicMock()), \
             patch('lazy_claude.server.ReferenceBuffer'), \
             patch('lazy_claude.server.EchoCanceller'), \
             patch('lazy_claude.server.ContinuousListener', return_value=_make_mock_listener()):
            from lazy_claude.server import create_server
            app, voice = create_server()
        tool_names = [t.name for t in asyncio.run(app.list_tools())]
        assert "toggle_listening" in tool_names
