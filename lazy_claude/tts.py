"""tts.py — Text-to-speech via Kokoro-82M.

All output (logging, warnings) goes to stderr.  stdout is reserved for the
MCP protocol channel and must stay clean.

Public API
----------
TTSEngine
    speak(text: str) -> None
        Synthesise text and play audio as chunks arrive (streaming playback).
        Empty / whitespace-only text returns immediately.
    stop() -> None
        Interrupt any active playback and clear the audio queue.
    is_speaking -> bool
        True while speech is being synthesised or played back.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# Environment variables — must be set before importing torch / transformers
# ---------------------------------------------------------------------------

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Suppress loguru-based kokoro/misaki output to stderr is fine; silence
# transformers and torch Python loggers to avoid cluttering stderr.
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('kokoro').setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Lazy imports (torch / kokoro are heavy — only pulled in when needed)
# ---------------------------------------------------------------------------

import sounddevice as sd  # noqa: E402  (must come after env-var block)

try:
    from kokoro import KPipeline  # noqa: E402
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "kokoro is required for TTS. Install it with: pip install kokoro"
    ) from _exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 24_000        # Hz — Kokoro outputs 24 kHz audio
_VOICE = 'af_heart'          # Default voice
_REPO_ID = 'hexgrad/Kokoro-82M'

# Type alias for ReferenceBuffer — imported lazily to avoid circular deps
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lazy_claude.aec import ReferenceBuffer


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Write a log line to stderr (never stdout)."""
    print(f"[lazy-claude tts] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------

class TTSEngine:
    """Streaming TTS engine backed by Kokoro-82M.

    Usage::

        engine = TTSEngine()
        engine.speak("Hello, world!")
        engine.stop()   # interrupt if still speaking
    """

    def __init__(self, ref_buf: "ReferenceBuffer | None" = None) -> None:
        """
        Parameters
        ----------
        ref_buf:
            Optional ReferenceBuffer shared with ContinuousListener for AEC.
            When provided, each synthesised audio chunk is also written into
            the buffer so the echo canceller can use it as a reference signal.
        """
        # Initialise Kokoro pipeline
        self._pipeline = KPipeline(lang_code='a', repo_id=_REPO_ID)

        # Check whether the output device natively supports 24 kHz.
        # If not, we enable software resampling.
        self._needs_resample = False
        try:
            sd.check_output_settings(samplerate=_SAMPLE_RATE)
        except Exception:
            _log(
                f"Output device does not natively support {_SAMPLE_RATE} Hz — "
                "will resample in software."
            )
            self._needs_resample = True

        # Shared AEC reference buffer (optional)
        self._ref_buf = ref_buf

        # Playback state
        self._speaking = False
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        """True while speech synthesis / playback is active."""
        return self._speaking

    def speak(self, text: str) -> None:
        """Synthesise *text* and play audio via sounddevice.

        Chunks are played as soon as they arrive from the generator so that
        the first audio is heard with minimal latency (streaming playback).

        Parameters
        ----------
        text:
            The text to speak.  Empty / whitespace-only strings are a no-op.
        """
        if not text or not text.strip():
            return

        self._stop_event.clear()
        self._speaking = True
        try:
            self._stream_speak(text)
        finally:
            self._speaking = False

    def stop(self) -> None:
        """Interrupt active playback.

        Safe to call at any time, including when not currently speaking.
        """
        self._stop_event.set()
        self._speaking = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stream_speak(self, text: str) -> None:
        """Run the Kokoro generator and write each audio chunk to the output
        stream as it arrives.  Returns once all chunks have been played or
        stop() has been called.
        """
        import numpy as np

        playback_rate = _SAMPLE_RATE

        try:
            with sd.OutputStream(
                samplerate=playback_rate,
                channels=1,
                dtype='float32',
            ) as stream:
                generator = self._pipeline(text, voice=_VOICE)
                for result in generator:
                    if self._stop_event.is_set():
                        break

                    audio_tensor = result.audio
                    if audio_tensor is None:
                        continue

                    # Convert torch tensor → numpy float32 array
                    try:
                        chunk = audio_tensor.cpu().numpy().astype(np.float32)
                    except Exception as exc:
                        _log(f"WARNING: could not convert audio chunk: {exc}")
                        continue

                    if chunk.size == 0:
                        continue

                    if self._needs_resample:
                        chunk = self._resample(chunk, _SAMPLE_RATE, playback_rate)

                    # sounddevice OutputStream.write expects shape (frames, channels)
                    stream.write(chunk.reshape(-1, 1))

                    # Push raw chunk (before any resampling change) into AEC reference
                    # buffer so the echo canceller can subtract the speaker signal.
                    # We push the flat float32 chunk at _SAMPLE_RATE (24 kHz); the
                    # ReferenceBuffer handles 24k→16k resampling internally.
                    if self._ref_buf is not None:
                        self._ref_buf.write(chunk)

                    if self._stop_event.is_set():
                        break

        except sd.PortAudioError as exc:
            _log(f"ERROR: PortAudio error during playback: {exc}")
        except Exception as exc:
            _log(f"ERROR: unexpected error during TTS playback: {exc}")

    @staticmethod
    def _resample(audio: 'np.ndarray', from_rate: int, to_rate: int) -> 'np.ndarray':  # noqa: F821
        """Simple linear-interpolation resample (for devices that don't support 24 kHz)."""
        import numpy as np

        if from_rate == to_rate:
            return audio
        n_out = int(len(audio) * to_rate / from_rate)
        x_old = np.linspace(0, len(audio) - 1, len(audio))
        x_new = np.linspace(0, len(audio) - 1, n_out)
        return np.interp(x_new, x_old, audio).astype(np.float32)
