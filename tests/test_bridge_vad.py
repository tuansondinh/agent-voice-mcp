"""Tests for bridge_vad.py — RemoteVADProcessor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lazy_claude.audio import CHUNK_SAMPLES, SAMPLE_RATE


@pytest.fixture
def mock_vad():
    """Create a mock SileroVAD that returns configurable probabilities."""
    vad = MagicMock()
    vad._probs = []
    vad._call_count = 0

    def call_fn(chunk):
        idx = vad._call_count
        vad._call_count += 1
        if idx < len(vad._probs):
            return vad._probs[idx]
        return 0.0  # silence by default

    vad.__call__ = call_fn
    vad.reset = MagicMock()
    return vad


@pytest.fixture
def processor(mock_vad):
    from lazy_claude.bridge_vad import RemoteVADProcessor

    return RemoteVADProcessor(
        mock_vad,
        silence_duration=0.1,  # short for tests
        min_speech_duration=0.1,
        no_speech_timeout=2.0,
    )


class TestRemoteVADProcessor:
    def test_no_speech_returns_none(self, processor, mock_vad):
        """Feeding silence should not return an utterance."""
        mock_vad._probs = [0.0] * 20  # all silence

        silence = np.zeros(CHUNK_SAMPLES * 5, dtype=np.float32)
        utterance, speaking = processor.feed(silence)
        assert utterance is None
        assert speaking is False

    def test_speech_detected(self, processor, mock_vad):
        """Feeding speech followed by silence should return an utterance."""
        # 10 frames of speech, then 10 frames of silence
        mock_vad._probs = [0.9] * 10 + [0.0] * 10

        # Feed enough audio for all frames
        audio = np.random.randn(CHUNK_SAMPLES * 20).astype(np.float32) * 0.1
        utterance, _ = processor.feed(audio)

        # Should detect an utterance
        assert utterance is not None
        assert len(utterance) > 0
        assert utterance.dtype == np.float32

    def test_partial_chunks_accumulated(self, processor, mock_vad):
        """Audio smaller than CHUNK_SAMPLES should be buffered."""
        mock_vad._probs = [0.0] * 5

        # Feed less than one frame
        small = np.zeros(100, dtype=np.float32)
        utterance, speaking = processor.feed(small)
        assert utterance is None
        assert speaking is False

    def test_reset_clears_state(self, processor, mock_vad):
        """Reset should clear all internal state."""
        # Feed some audio
        mock_vad._probs = [0.9] * 5
        audio = np.random.randn(CHUNK_SAMPLES * 5).astype(np.float32)
        processor.feed(audio)

        processor.reset()
        assert processor.is_speaking is False

    def test_is_speaking_during_speech(self, processor, mock_vad):
        """is_speaking should be True during active speech."""
        mock_vad._probs = [0.9] * 5  # speech, no trailing silence yet

        audio = np.random.randn(CHUNK_SAMPLES * 5).astype(np.float32) * 0.1
        utterance, speaking = processor.feed(audio)

        # Still speaking (no trailing silence yet)
        assert utterance is None
        assert speaking is True

    def test_timeout_returns_none(self, processor, mock_vad):
        """No speech within timeout should return None."""
        # All silence, enough frames to exceed timeout
        n_frames = int(2.5 * SAMPLE_RATE / CHUNK_SAMPLES)  # 2.5 seconds
        mock_vad._probs = [0.0] * n_frames

        audio = np.zeros(CHUNK_SAMPLES * n_frames, dtype=np.float32)
        utterance, speaking = processor.feed(audio)

        assert utterance is None
        assert speaking is False


class TestRemoteVADProcessorMultipleFeed:
    def test_incremental_feeding(self, processor, mock_vad):
        """Feeding audio incrementally across multiple calls should work."""
        mock_vad._probs = [0.9] * 10 + [0.0] * 10

        # Feed one frame at a time
        result = None
        for i in range(20):
            audio = np.random.randn(CHUNK_SAMPLES).astype(np.float32) * 0.1
            utterance, _ = processor.feed(audio)
            if utterance is not None:
                result = utterance
                break

        assert result is not None
