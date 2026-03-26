"""bridge_tts.py — Buffered TTS engine for the remote voice bridge.

Yields 24 kHz float32 numpy chunks instead of playing them through speakers.
Reuses the same Kokoro pipeline and constants from tts.py.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Generator

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("kokoro").setLevel(logging.WARNING)

import numpy as np

try:
    from kokoro import KPipeline
except ImportError as _exc:
    raise ImportError(
        "kokoro is required for TTS. Install it with: pip install kokoro"
    ) from _exc

# ---------------------------------------------------------------------------
# Constants (same as tts.py)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24_000
_VOICE = "af_heart"
_SPEED = 1.3
_REPO_ID = "hexgrad/Kokoro-82M"


def _log(msg: str) -> None:
    print(f"[bridge-tts] {msg}", file=sys.stderr, flush=True)


class BufferedTTSEngine:
    """TTS engine that yields audio chunks instead of playing them locally.

    Usage::

        engine = BufferedTTSEngine()
        for chunk in engine.synthesize("Hello!"):
            # chunk is a 24kHz float32 numpy array
            send_to_phone(chunk)
        engine.stop()  # interrupt if needed
    """

    def __init__(self) -> None:
        _log("Initializing Kokoro pipeline...")
        self._pipeline = KPipeline(lang_code="a", repo_id=_REPO_ID)
        self._stop_event = threading.Event()
        self._speaking = False
        _log("Kokoro pipeline ready.")

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def synthesize(self, text: str) -> Generator[np.ndarray, None, None]:
        """Yield 24 kHz float32 numpy chunks as Kokoro generates them.

        Parameters
        ----------
        text:
            Text to synthesize. Empty/whitespace returns immediately.

        Yields
        ------
        np.ndarray
            1-D float32 array at 24 kHz sample rate.
        """
        if not text or not text.strip():
            return

        self._stop_event.clear()
        self._speaking = True
        try:
            for result in self._pipeline(text, voice=_VOICE, speed=_SPEED):
                if self._stop_event.is_set():
                    break

                audio_tensor = result.audio
                if audio_tensor is None:
                    continue

                try:
                    chunk = audio_tensor.cpu().numpy().astype(np.float32)
                except Exception as exc:
                    _log(f"WARNING: could not convert audio chunk: {exc}")
                    continue

                if chunk.size == 0:
                    continue

                yield chunk
        finally:
            self._speaking = False

    def stop(self) -> None:
        """Interrupt active synthesis. Safe to call at any time."""
        self._stop_event.set()
        self._speaking = False
