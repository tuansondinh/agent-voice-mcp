"""tts.py — Text-to-speech via Kokoro-82M (English) or system TTS (other languages).

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

Language support
----------------
Set LAZY_CLAUDE_LANGUAGE=de (or pass language='de' to TTSEngine) to enable
German TTS via espeak-ng (if installed) or the macOS ``say`` command.
Kokoro natively supports: a (en-us), b (en-gb), e (es), f (fr), h (hi),
i (it), j (ja), p (pt-br), z (zh).
For other languages the external TTS path is used automatically.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import tempfile
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
_SPEED = 1.2                 # TTS playback speed (1.0 = normal, higher = faster)
_REPO_ID = 'hexgrad/Kokoro-82M'

# Kokoro lang_code mapping: ISO-639-1 → Kokoro code (only supported languages)
_KOKORO_LANG_MAP: dict[str, str] = {
    'en': 'a',   # American English
    'en-us': 'a',
    'en-gb': 'b',
    'es': 'e',
    'fr': 'f',
    'hi': 'h',
    'it': 'i',
    'ja': 'j',
    'pt': 'p',
    'pt-br': 'p',
    'zh': 'z',
}

# macOS say voice for German (used when espeak-ng is unavailable).
# Preferred voices are the newer neural voices (Eddy, Flo, etc.).
# Anna is the old low-quality fallback.
_MACOS_SAY_VOICES_DE = [
    'Eddy (Deutsch (Deutschland))',   # neutral, high quality
    'Flo (Deutsch (Deutschland))',    # female, high quality
    'Anna',                           # legacy fallback
]
_MACOS_SAY_RATE = '155'   # words per minute (slightly slower = clearer)

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
# External TTS helper (for languages not supported by Kokoro)
# ---------------------------------------------------------------------------

def _generate_audio_external(text: str, lang: str) -> "tuple[np.ndarray, int]":
    """Generate speech audio using espeak-ng or macOS say.

    Tries espeak-ng first (cross-platform); falls back to the macOS ``say``
    command when espeak-ng is not installed.

    Parameters
    ----------
    text:
        Text to synthesise.
    lang:
        BCP-47 language tag, e.g. ``'de'``, ``'fr'``.

    Returns
    -------
    tuple[np.ndarray, int]
        (samples_float32, sample_rate) where samples is a 1-D float32 array.

    Raises
    ------
    RuntimeError
        If neither espeak-ng nor macOS say is available.
    """
    import numpy as np
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmpfile = f.name

    try:
        # Attempt espeak-ng first
        _espeak_ok = False
        try:
            subprocess.run(
                ['espeak-ng', '-v', lang, '-s', '150', '-w', tmpfile, text],
                check=True,
                capture_output=True,
            )
            _espeak_ok = True
        except FileNotFoundError:
            pass  # espeak-ng not installed → try say
        except subprocess.CalledProcessError as exc:
            _log(f"espeak-ng error: {exc.stderr.decode(errors='replace')}")

        if not _espeak_ok:
            # macOS say fallback
            if sys.platform != 'darwin':
                raise RuntimeError(
                    "espeak-ng is required for non-English TTS on non-macOS platforms. "
                    "Install it with: brew install espeak-ng"
                )
            # For German try the preferred voices in order; fall back on error.
            voices_to_try: list[str | None] = (
                list(_MACOS_SAY_VOICES_DE) if lang.startswith('de') else [None]
            )
            # Use AIFF output (native say format) for best quality — no --data-format flag.
            # Rename tmpfile to .aiff so soundfile detects the container correctly.
            aiff_file = tmpfile.replace('.wav', '.aiff') if tmpfile.endswith('.wav') else tmpfile + '.aiff'
            generated = False
            for voice in voices_to_try:
                cmd = ['say', '-r', _MACOS_SAY_RATE, '-o', aiff_file]
                if voice:
                    cmd += ['-v', voice]
                cmd.append(text)
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    generated = True
                    break
                except subprocess.CalledProcessError:
                    continue  # try next voice
            if not generated:
                raise RuntimeError(f"macOS say failed for all voices (lang={lang})")
            tmpfile = aiff_file  # point reader at the .aiff file

        data, sr = sf.read(tmpfile)
        return data.astype(np.float32), sr

    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------

class TTSEngine:
    """Streaming TTS engine backed by Kokoro-82M (English) or system TTS (other languages).

    Usage::

        engine = TTSEngine()
        engine.speak("Hello, world!")
        engine.stop()   # interrupt if still speaking

        # German
        engine = TTSEngine(language='de')
        engine.speak("Hallo Welt!")
    """

    def __init__(
        self,
        ref_buf: "ReferenceBuffer | None" = None,
        language: str = 'en',
    ) -> None:
        """
        Parameters
        ----------
        ref_buf:
            Optional ReferenceBuffer shared with ContinuousListener for AEC.
            When provided, each synthesised audio chunk is also written into
            the buffer so the echo canceller can use it as a reference signal.
        language:
            BCP-47 language tag, e.g. ``'en'`` (default) or ``'de'``.
            Languages supported natively by Kokoro use it directly; others
            fall back to espeak-ng / macOS say.
        """
        self._language = language.lower()
        self._use_kokoro = self._language in _KOKORO_LANG_MAP

        if self._use_kokoro:
            kokoro_lang = _KOKORO_LANG_MAP[self._language]
            # Initialise Kokoro pipeline
            self._pipeline = KPipeline(lang_code=kokoro_lang, repo_id=_REPO_ID)
        else:
            self._pipeline = None
            _log(f"Language '{language}' not supported by Kokoro — using external TTS.")

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
            if self._use_kokoro:
                self._stream_speak(text)
            else:
                self._stream_speak_external(text)
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

    def _stream_speak_external(self, text: str) -> None:
        """Speak *text* via espeak-ng / macOS say (for non-Kokoro languages)."""
        import numpy as np

        if self._stop_event.is_set():
            return
        try:
            audio, src_rate = _generate_audio_external(text, self._language)
        except Exception as exc:
            _log(f"ERROR: external TTS failed: {exc}")
            return

        if self._stop_event.is_set():
            return

        # Resample to 24kHz so the AEC reference buffer rate matches Kokoro path
        if src_rate != _SAMPLE_RATE:
            audio = self._resample(audio, src_rate, _SAMPLE_RATE)

        if self._ref_buf is not None:
            self._ref_buf.write(audio)

        try:
            with sd.OutputStream(samplerate=_SAMPLE_RATE, channels=1, dtype='float32') as stream:
                # Write in chunks so stop_event is checked periodically
                chunk_size = _SAMPLE_RATE // 10  # 100 ms chunks
                for i in range(0, len(audio), chunk_size):
                    if self._stop_event.is_set():
                        break
                    stream.write(audio[i:i + chunk_size].reshape(-1, 1))
        except sd.PortAudioError as exc:
            _log(f"ERROR: PortAudio error during external TTS playback: {exc}")
        except Exception as exc:
            _log(f"ERROR: unexpected error during external TTS playback: {exc}")

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
                generator = self._pipeline(text, voice=_VOICE, speed=_SPEED)
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

                    # Push chunk into AEC reference buffer BEFORE playing it,
                    # so the reference is available by the time the mic picks
                    # up the echo. We push the original 24kHz chunk; the
                    # ReferenceBuffer handles 24k→16k resampling internally.
                    if self._ref_buf is not None:
                        self._ref_buf.write(chunk)

                    if self._needs_resample:
                        chunk = self._resample(chunk, _SAMPLE_RATE, playback_rate)

                    # sounddevice OutputStream.write expects shape (frames, channels)
                    stream.write(chunk.reshape(-1, 1))

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
