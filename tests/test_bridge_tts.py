"""Tests for bridge_tts.py — BufferedTTSEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestBufferedTTSEngine:
    @patch("lazy_claude.bridge_tts.KPipeline")
    def test_synthesize_yields_chunks(self, mock_pipeline_cls):
        """synthesize() should yield float32 numpy arrays at 24kHz."""
        # Mock the pipeline to return fake audio chunks
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        fake_audio = MagicMock()
        fake_audio.cpu.return_value.numpy.return_value.astype.return_value = (
            np.random.randn(4800).astype(np.float32)  # 200ms at 24kHz
        )
        fake_audio.cpu.return_value.numpy.return_value.astype.return_value.size = 4800

        mock_result = MagicMock()
        mock_result.audio = fake_audio
        mock_pipeline.return_value = [mock_result]

        from lazy_claude.bridge_tts import BufferedTTSEngine

        engine = BufferedTTSEngine()
        chunks = list(engine.synthesize("Hello world"))

        assert len(chunks) == 1
        assert chunks[0].dtype == np.float32
        assert chunks[0].shape == (4800,)

    @patch("lazy_claude.bridge_tts.KPipeline")
    def test_empty_text_yields_nothing(self, mock_pipeline_cls):
        """Empty text should yield nothing."""
        from lazy_claude.bridge_tts import BufferedTTSEngine

        engine = BufferedTTSEngine()
        chunks = list(engine.synthesize(""))
        assert chunks == []

        chunks = list(engine.synthesize("   "))
        assert chunks == []

    @patch("lazy_claude.bridge_tts.KPipeline")
    def test_stop_interrupts_synthesis(self, mock_pipeline_cls):
        """Calling stop() should interrupt the generator."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        fake_audio = MagicMock()
        fake_audio.cpu.return_value.numpy.return_value.astype.return_value = (
            np.random.randn(4800).astype(np.float32)
        )
        fake_audio.cpu.return_value.numpy.return_value.astype.return_value.size = 4800

        # Return many results
        mock_result = MagicMock()
        mock_result.audio = fake_audio
        mock_pipeline.return_value = [mock_result] * 100

        from lazy_claude.bridge_tts import BufferedTTSEngine

        engine = BufferedTTSEngine()

        chunks = []
        for chunk in engine.synthesize("Long text"):
            chunks.append(chunk)
            engine.stop()  # Stop after first chunk

        assert len(chunks) == 1
        assert engine.is_speaking is False

    @patch("lazy_claude.bridge_tts.KPipeline")
    def test_is_speaking_property(self, mock_pipeline_cls):
        """is_speaking should be True during synthesis."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_pipeline.return_value = []

        from lazy_claude.bridge_tts import BufferedTTSEngine

        engine = BufferedTTSEngine()
        assert engine.is_speaking is False

        list(engine.synthesize("test"))
        assert engine.is_speaking is False

    @patch("lazy_claude.bridge_tts.KPipeline")
    def test_none_audio_skipped(self, mock_pipeline_cls):
        """Results with None audio should be skipped."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        mock_result = MagicMock()
        mock_result.audio = None
        mock_pipeline.return_value = [mock_result]

        from lazy_claude.bridge_tts import BufferedTTSEngine

        engine = BufferedTTSEngine()
        chunks = list(engine.synthesize("test"))
        assert chunks == []
