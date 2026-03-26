"""bridge_vad.py — Remote audio VAD processor for the voice bridge.

Processes audio arriving over WebSocket in arbitrary-sized chunks,
rechunks into 512-sample VAD frames, and detects speech boundaries.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from lazy_claude.audio import (
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    SileroVAD,
    VadStateMachine,
    load_vad_model,
)


def _log(msg: str) -> None:
    print(f"[bridge-vad] {msg}", file=sys.stderr, flush=True)


class RemoteVADProcessor:
    """Processes remote audio through VAD to detect speech boundaries.

    Feed arbitrary-sized PCM chunks via ``feed()``. When a complete
    utterance is detected (speech followed by trailing silence), the
    concatenated speech audio is returned.

    Parameters
    ----------
    vad_model:
        A loaded SileroVAD instance.
    silence_duration:
        Seconds of trailing silence to end an utterance.
    min_speech_duration:
        Minimum speech before honouring a stop.
    no_speech_timeout:
        Seconds to wait for speech before timing out.
    """

    def __init__(
        self,
        vad_model: SileroVAD,
        silence_duration: float = 0.5,
        min_speech_duration: float = 0.3,
        no_speech_timeout: float = 30.0,
    ) -> None:
        self._vad = vad_model
        self._silence_duration = silence_duration
        self._min_speech_duration = min_speech_duration
        self._no_speech_timeout = no_speech_timeout

        self._buffer = np.array([], dtype=np.float32)
        self._speech_chunks: list[np.ndarray] = []
        self._state_machine: VadStateMachine | None = None
        self._frame_count = 0
        self._is_speaking = False

    @property
    def is_speaking(self) -> bool:
        """True if speech has been detected and we're recording."""
        return self._is_speaking

    def reset(self) -> None:
        """Reset state for a new utterance."""
        self._buffer = np.array([], dtype=np.float32)
        self._speech_chunks = []
        self._state_machine = VadStateMachine(
            silence_duration=self._silence_duration,
            min_speech_duration=self._min_speech_duration,
            no_speech_timeout=self._no_speech_timeout,
        )
        self._vad.reset()
        self._frame_count = 0
        self._is_speaking = False

    def feed(self, pcm_16khz: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """Feed audio samples and check for complete utterances.

        Parameters
        ----------
        pcm_16khz:
            1-D float32 array at 16 kHz.

        Returns
        -------
        tuple[np.ndarray | None, bool]
            (utterance, is_speaking) — utterance is the concatenated speech
            audio when detection completes, or None if still listening.
            is_speaking indicates whether speech is currently detected.
        """
        if self._state_machine is None:
            self.reset()

        self._buffer = np.concatenate([self._buffer, pcm_16khz])

        # Process all complete 512-sample frames
        while len(self._buffer) >= CHUNK_SAMPLES:
            frame = self._buffer[:CHUNK_SAMPLES]
            self._buffer = self._buffer[CHUNK_SAMPLES:]

            timestamp = self._frame_count * CHUNK_SAMPLES / SAMPLE_RATE
            self._frame_count += 1

            prob = self._vad(frame)
            done = self._state_machine.update(prob, timestamp)

            # Always collect audio once we start hearing speech
            if self._state_machine.state in ("SPEAKING", "TRAILING_SILENCE"):
                self._speech_chunks.append(frame)
                self._is_speaking = True
            elif self._state_machine.state == "WAITING" and prob >= 0.5:
                # Capture the transition frame
                self._speech_chunks.append(frame)

            if done:
                if self._state_machine.timed_out or not self._speech_chunks:
                    _log("VAD: timed out or no speech detected")
                    self.reset()
                    return None, False

                utterance = np.concatenate(self._speech_chunks)
                _log(f"VAD: utterance detected, {len(utterance)/SAMPLE_RATE:.2f}s")
                self.reset()
                return utterance, False

        return None, self._is_speaking
