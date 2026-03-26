"""Tests for tts.py — TTS engine with Kokoro-82M.

These tests do NOT require audio hardware or a real TTS model.
They test:
- Module and class importability
- Public API surface (speak, stop, is_speaking)
- Edge-case handling (empty text, whitespace-only text)
- stop() while not speaking does not raise
- is_speaking flag transitions
- env vars for logging suppression are set before import
"""

from __future__ import annotations

import sys
import threading
import time
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_result(audio_array: np.ndarray | None):
    """Return a mock KPipeline.Result-like object."""
    result = MagicMock()
    if audio_array is not None:
        import torch
        result.audio = torch.from_numpy(audio_array.astype(np.float32))
    else:
        result.audio = None
    return result


# ---------------------------------------------------------------------------
# API surface tests (no hardware needed)
# ---------------------------------------------------------------------------


class TestTTSModuleAPI:
    """Verify the public API surface of tts.py without hardware."""

    def test_tts_engine_importable(self):
        from lazy_claude.tts import TTSEngine
        assert TTSEngine is not None


class TestTTSEngineInit:
    """Test TTSEngine construction with mocked dependencies."""

    def _make_engine(self):
        """Return a TTSEngine with mocked Kokoro and sounddevice."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = iter([])  # empty generator by default

        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline) as mock_cls, \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None  # no exception = supported
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
            return engine, mock_cls, mock_sd

    def test_engine_creates_without_error(self):
        mock_pipeline = MagicMock()
        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
            assert engine is not None

    def test_engine_not_speaking_initially(self):
        mock_pipeline = MagicMock()
        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
            assert engine.is_speaking is False

    def test_pipeline_initialized_with_lang_code_a(self):
        """KPipeline should be initialised with lang_code='a' (American English)."""
        mock_pipeline = MagicMock()
        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline) as mock_cls, \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            from lazy_claude.tts import TTSEngine
            TTSEngine()
            # Verify KPipeline was called with lang_code='a'
            calls = mock_cls.call_args_list
            assert len(calls) == 1
            kwargs = calls[0][1] if calls[0][1] else {}
            args = calls[0][0] if calls[0][0] else ()
            lang_code = kwargs.get('lang_code') or (args[0] if args else None)
            assert lang_code == 'a', f"Expected lang_code='a', got {lang_code!r}"

    def test_check_output_settings_called_with_24khz(self):
        """On init, sd.check_output_settings should be called with samplerate=24000."""
        mock_pipeline = MagicMock()
        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            from lazy_claude.tts import TTSEngine
            TTSEngine()
            mock_sd.check_output_settings.assert_called_once_with(samplerate=24000)

    def test_fallback_to_resampling_on_unsupported_device(self):
        """If check_output_settings raises, engine should fall back without raising."""
        mock_pipeline = MagicMock()
        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.side_effect = Exception("Device does not support 24kHz")
            from lazy_claude.tts import TTSEngine
            # Should NOT raise — fall back to resampling silently
            engine = TTSEngine()
            assert engine is not None


# ---------------------------------------------------------------------------
# Edge case: empty and whitespace-only text
# ---------------------------------------------------------------------------


class TestTTSSpeakEdgeCases:
    """Test speak() edge cases that don't require hardware."""

    def _make_engine_with_mocks(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = iter([])
        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
        return engine, mock_pipeline, mock_sd

    def test_speak_empty_string_returns_immediately(self):
        """speak('') should return without calling the pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = iter([])
        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
            # Reset call count after init
            mock_pipeline.reset_mock()
            engine.speak('')
            # Pipeline should NOT be called for empty text
            mock_pipeline.assert_not_called()

    def test_speak_whitespace_only_returns_immediately(self):
        """speak('   ') should return without calling the pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = iter([])
        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
            mock_pipeline.reset_mock()
            engine.speak('   \n\t')
            mock_pipeline.assert_not_called()

    def test_speak_does_not_raise_on_empty(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = iter([])
        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
            # Should not raise
            engine.speak('')
            engine.speak('  ')


# ---------------------------------------------------------------------------
# stop() tests
# ---------------------------------------------------------------------------


class TestTTSStop:
    """Test stop() behaviour."""

    def test_stop_while_not_speaking_does_not_raise(self):
        mock_pipeline = MagicMock()
        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
            # Must not raise
            engine.stop()

    def test_stop_sets_is_speaking_false(self):
        mock_pipeline = MagicMock()
        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
            engine.stop()
            assert engine.is_speaking is False


# ---------------------------------------------------------------------------
# Env var tests
# ---------------------------------------------------------------------------


class TestTTSEnvVars:
    """Verify env vars for logging suppression are set when module loads."""

    def test_pytorch_mps_fallback_env_var_set(self):
        """PYTORCH_ENABLE_MPS_FALLBACK must be set to 1 by tts.py at import time."""
        import os
        import importlib
        import lazy_claude.tts  # trigger import
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'

    def test_transformers_verbosity_env_var_set(self):
        """TRANSFORMERS_VERBOSITY should be set to 'error' to suppress logs."""
        import os
        import lazy_claude.tts
        val = os.environ.get('TRANSFORMERS_VERBOSITY')
        assert val in ('error', 'warning', 'critical'), \
            f"TRANSFORMERS_VERBOSITY={val!r} is not a suppression level"

    def test_tokenizers_parallelism_env_var_set(self):
        """TOKENIZERS_PARALLELISM should be set to 'false' to avoid fork warnings."""
        import os
        import lazy_claude.tts
        assert os.environ.get('TOKENIZERS_PARALLELISM') == 'false'


# ---------------------------------------------------------------------------
# Speak with mock audio chunks (integration-like, no hardware)
# ---------------------------------------------------------------------------


class TestTTSSpeakWithMockAudio:
    """Test speak() processes audio chunks when pipeline returns them."""

    def test_speak_calls_pipeline_with_voice_af_heart(self):
        """speak(text) should call the pipeline with voice='af_heart'."""
        import torch
        mock_result = MagicMock()
        mock_result.audio = torch.zeros(24000, dtype=torch.float32)

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = iter([mock_result])

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.write = MagicMock()

        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline_instance), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            mock_sd.OutputStream.return_value = mock_stream
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
            engine.speak('Hello world')

        # Verify the pipeline was called with voice='af_heart'
        call_kwargs = mock_pipeline_instance.call_args[1] if mock_pipeline_instance.call_args else {}
        call_args = mock_pipeline_instance.call_args[0] if mock_pipeline_instance.call_args else ()
        voice_used = call_kwargs.get('voice') or (call_args[1] if len(call_args) > 1 else None)
        assert voice_used == 'af_heart', f"Expected voice='af_heart', got {voice_used!r}"

    def test_speak_is_speaking_false_after_completion(self):
        """is_speaking should be False after speak() returns normally."""
        import torch
        mock_result = MagicMock()
        mock_result.audio = torch.zeros(2400, dtype=torch.float32)

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = iter([mock_result])

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline_instance), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            mock_sd.OutputStream.return_value = mock_stream
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
            engine.speak('Test text')
            assert engine.is_speaking is False

    def test_speak_result_with_none_audio_skipped(self):
        """Results with audio=None should be skipped gracefully."""
        mock_result_none = MagicMock()
        mock_result_none.audio = None

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = iter([mock_result_none])

        with patch('lazy_claude.tts.KPipeline', return_value=mock_pipeline_instance), \
             patch('lazy_claude.tts.sd') as mock_sd:
            mock_sd.check_output_settings.return_value = None
            from lazy_claude.tts import TTSEngine
            engine = TTSEngine()
            # Should not raise
            engine.speak('Some text')
            assert engine.is_speaking is False
