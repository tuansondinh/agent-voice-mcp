"""audio.py — Microphone capture with Silero VAD.

All output (logging, warnings) goes to stderr.  stdout is reserved for the
MCP protocol channel and must stay clean.

Public API
----------
load_vad_model() -> SileroVAD
    Download (first run) and load the Silero VAD ONNX model.  Returns a
    callable SileroVAD object.

record_audio(...) -> np.ndarray | None
    Capture microphone audio until speech ends (or timeout).
    Returns a 16 kHz mono float32 numpy array, or None on no-speech timeout.

VadStateMachine
    State machine that drives WAITING → SPEAKING → TRAILING_SILENCE → DONE.
    Exposed publicly so tests can drive it directly.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from lazy_claude.wakeword import create_wakeword_detector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000          # Hz — the only rate Silero VAD 16kHz model supports
CHUNK_SAMPLES = 512           # samples per VAD frame at 16 kHz  (32 ms)
CONTEXT_SAMPLES = 64          # context prepended to each chunk per Silero spec

_MODELS_DIR = Path(__file__).parent / "models"
_VAD_MODEL_PATH = _MODELS_DIR / "silero_vad.onnx"
_VAD_MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master"
    "/src/silero_vad/data/silero_vad.onnx"
)

# Speech probability threshold — above → speech, below → silence
_SPEECH_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# stderr logging helper
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Write a log line to stderr (never stdout)."""
    print(f"[lazy-claude audio] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Silero VAD ONNX wrapper (no torch dependency)
# ---------------------------------------------------------------------------

class SileroVAD:
    """Thin ONNX-based wrapper around the Silero VAD model.

    Usage::

        model = SileroVAD(path)
        prob = model(chunk)   # chunk: float32 array of CHUNK_SAMPLES (512)
        model.reset()
    """

    def __init__(self, model_path: Path) -> None:
        import importlib
        import sys
        from unittest.mock import Mock

        # Some tests stub heavy native deps by placing MagicMock objects in
        # sys.modules. When the live VAD tests run later in the same process,
        # recover the real onnxruntime module here instead of using the stub.
        ort = importlib.import_module("onnxruntime")
        if isinstance(ort, Mock):
            sys.modules.pop("onnxruntime", None)
            ort = importlib.import_module("onnxruntime")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3  # suppress ort INFO messages

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.reset()

    def reset(self) -> None:
        """Reset internal GRU state and context buffer."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        """Run VAD on a single chunk.

        Parameters
        ----------
        chunk:
            1-D float32 array of exactly CHUNK_SAMPLES (512) samples.

        Returns
        -------
        float
            Speech probability in [0, 1].
        """
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        if chunk.ndim != 1 or chunk.shape[0] != CHUNK_SAMPLES:
            raise ValueError(
                f"chunk must be 1-D with {CHUNK_SAMPLES} samples, got shape {chunk.shape}"
            )

        # Prepend context: shape [1, CONTEXT_SAMPLES + CHUNK_SAMPLES]
        x = np.concatenate([self._context, chunk[np.newaxis, :]], axis=1)

        feed = {
            "input": x,
            "state": self._state,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
        }
        output, new_state = self._session.run(None, feed)

        self._state = new_state
        # Update context to last CONTEXT_SAMPLES of the input
        self._context = x[:, -CONTEXT_SAMPLES:]

        return float(output[0, 0])


# ---------------------------------------------------------------------------
# Model loader (download on first use)
# ---------------------------------------------------------------------------

def load_vad_model() -> SileroVAD:
    """Return a ready-to-use SileroVAD instance.

    Downloads the ONNX model on the first call and caches it under
    ``lazy_claude/models/silero_vad.onnx``.
    """
    if not _VAD_MODEL_PATH.exists():
        _log(f"Downloading Silero VAD model → {_VAD_MODEL_PATH} …")
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(_VAD_MODEL_URL, _VAD_MODEL_PATH)
            _log("Download complete.")
        except Exception as exc:
            _log(f"ERROR: failed to download VAD model: {exc}")
            raise

    return SileroVAD(_VAD_MODEL_PATH)


# ---------------------------------------------------------------------------
# VAD State Machine
# ---------------------------------------------------------------------------

State = Literal["WAITING", "SPEAKING", "TRAILING_SILENCE", "DONE"]


class VadStateMachine:
    """Drive audio-capture state based on per-chunk VAD probabilities.

    Parameters
    ----------
    silence_duration:
        Seconds of silence after the last speech chunk required to stop.
    min_speech_duration:
        Minimum accumulated speech seconds before we honour a stop signal.
        Prevents very short bursts (e.g. door click) from ending a recording.
    no_speech_timeout:
        If no speech has started within this many seconds, return done
        (``timed_out`` will be True).
    sample_rate:
        Audio sample rate (default 16000).
    chunk_size:
        Samples per chunk (default 512).
    speech_threshold:
        VAD probability threshold — above is speech.
    """

    def __init__(
        self,
        silence_duration: float = 0.5,
        min_speech_duration: float = 0.5,
        no_speech_timeout: float = 15.0,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = CHUNK_SAMPLES,
        speech_threshold: float = _SPEECH_THRESHOLD,
    ) -> None:
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.no_speech_timeout = no_speech_timeout
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.speech_threshold = speech_threshold

        self.state: State = "WAITING"
        self.timed_out: bool = False

        self._speech_started_at: float | None = None
        self._silence_started_at: float | None = None
        self._accumulated_speech: float = 0.0  # seconds

    def update(self, speech_prob: float, timestamp: float) -> bool:
        """Process one VAD result.

        Parameters
        ----------
        speech_prob:
            Speech probability from the VAD model (0–1).
        timestamp:
            Elapsed time in seconds (from recording start).

        Returns
        -------
        bool
            True when recording should stop (state is DONE).
        """
        is_speech = speech_prob >= self.speech_threshold

        if self.state == "WAITING":
            # No-speech timeout check
            if timestamp >= self.no_speech_timeout:
                self.state = "DONE"
                self.timed_out = True
                return True

            if is_speech:
                self.state = "SPEAKING"
                self._speech_started_at = timestamp
                self._accumulated_speech += self.chunk_size / self.sample_rate

        elif self.state == "SPEAKING":
            if is_speech:
                self._accumulated_speech += self.chunk_size / self.sample_rate
            else:
                self.state = "TRAILING_SILENCE"
                self._silence_started_at = timestamp

        elif self.state == "TRAILING_SILENCE":
            if is_speech:
                # Speech resumed — go back to SPEAKING
                self.state = "SPEAKING"
                self._accumulated_speech += self.chunk_size / self.sample_rate
                self._silence_started_at = None
            else:
                silence_elapsed = timestamp - self._silence_started_at  # type: ignore[operator]
                if (
                    silence_elapsed >= self.silence_duration
                    and self._accumulated_speech >= self.min_speech_duration
                ):
                    self.state = "DONE"
                    return True

        return self.state == "DONE"


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _get_capture_rate() -> tuple[int, bool]:
    """Return (capture_rate, needs_resample).

    Tries 16 kHz first.  If the device doesn't support it, falls back to the
    device's native rate and we'll resample in software.
    """
    import sounddevice as sd  # noqa: PLC0415

    try:
        sd.check_input_settings(samplerate=16_000, channels=1, dtype="float32")
        return 16_000, False
    except Exception:
        pass

    # Query the default input device's default sample rate
    try:
        device_info = sd.query_devices(kind="input")
        native_rate = int(device_info["default_samplerate"])
        _log(
            f"16kHz not supported by input device. "
            f"Capturing at {native_rate} Hz and resampling to 16 kHz."
        )
        return native_rate, True
    except Exception as exc:
        _log(f"WARNING: could not query native sample rate ({exc}), assuming 44100 Hz.")
        return 44_100, True


def _resample_to_16k(audio: np.ndarray, from_rate: int) -> np.ndarray:
    """Simple linear resampling to 16 kHz mono float32."""
    if from_rate == SAMPLE_RATE:
        return audio
    ratio = SAMPLE_RATE / from_rate
    n_out = int(len(audio) * ratio)
    # Use numpy for a simple polyphase-style resampling via linspace
    x_old = np.linspace(0, len(audio) - 1, len(audio))
    x_new = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Main recording function
# ---------------------------------------------------------------------------

def record_audio(
    silence_duration: float = 0.5,
    min_speech_duration: float = 0.5,
    no_speech_timeout: float = 15.0,
    vad_model: SileroVAD | None = None,
) -> np.ndarray | None:
    """Record microphone audio until the end-of-speech condition is met.

    Parameters
    ----------
    silence_duration:
        Seconds of post-speech silence to wait before stopping.
    min_speech_duration:
        Minimum seconds of speech required before a stop is honoured.
    no_speech_timeout:
        Stop and return None if no speech starts within this many seconds.
    vad_model:
        Pre-loaded SileroVAD instance.  Loaded automatically if not provided.

    Returns
    -------
    np.ndarray | None
        16 kHz mono float32 audio array, or None if no speech was detected
        before ``no_speech_timeout``.

    Raises
    ------
    RuntimeError
        If microphone permissions are denied or no input device is found.
    """
    import sounddevice as sd  # noqa: PLC0415

    if vad_model is None:
        vad_model = load_vad_model()

    capture_rate, needs_resample = _get_capture_rate()

    # Adjust chunk size proportionally if we're capturing at a different rate
    native_chunk = CHUNK_SAMPLES if not needs_resample else int(
        CHUNK_SAMPLES * capture_rate / SAMPLE_RATE
    )

    vad = VadStateMachine(
        silence_duration=silence_duration,
        min_speech_duration=min_speech_duration,
        no_speech_timeout=no_speech_timeout,
    )
    vad_model.reset()

    recorded_chunks: list[np.ndarray] = []
    done_event = threading.Event()
    start_time = time.monotonic()
    # Running buffer for VAD chunks (needed when native != 16kHz)
    _resample_buffer: list[np.ndarray] = []

    def _callback(indata: np.ndarray, frames: int, _time, status) -> None:
        nonlocal recorded_chunks, _resample_buffer

        if status:
            _log(f"sounddevice status: {status}")

        # indata shape: (frames, channels) — take mono channel
        mono = indata[:, 0].copy()

        if needs_resample:
            # Accumulate until we have enough samples for a 16kHz chunk
            _resample_buffer.append(mono)
            buf = np.concatenate(_resample_buffer)
            # How many native samples correspond to one 16kHz chunk?
            native_per_vad = int(CHUNK_SAMPLES * capture_rate / SAMPLE_RATE)
            while len(buf) >= native_per_vad:
                segment = buf[:native_per_vad]
                buf = buf[native_per_vad:]
                chunk_16k = _resample_to_16k(segment, capture_rate)
                recorded_chunks.append(chunk_16k)
                _run_vad(chunk_16k)
            _resample_buffer = [buf] if len(buf) > 0 else []
        else:
            recorded_chunks.append(mono)
            _run_vad(mono)

    def _run_vad(chunk_16k: np.ndarray) -> None:
        elapsed = time.monotonic() - start_time
        prob = vad_model(chunk_16k)
        done = vad.update(speech_prob=prob, timestamp=elapsed)
        if done:
            done_event.set()

    _log("Listening… (speak now)")

    try:
        with sd.InputStream(
            samplerate=capture_rate,
            channels=1,
            dtype="float32",
            blocksize=native_chunk,
            callback=_callback,
        ):
            done_event.wait(timeout=no_speech_timeout + 5.0)
    except sd.PortAudioError as exc:
        err_str = str(exc).lower()
        if "permission" in err_str or "access" in err_str or "denied" in err_str:
            msg = (
                "Microphone access denied. "
                "Grant permission in System Settings → Privacy & Security → Microphone."
            )
        else:
            msg = f"PortAudio error: {exc}"
        _log(f"ERROR: {msg}")
        raise RuntimeError(msg) from exc

    if vad.timed_out or not recorded_chunks:
        _log("No speech detected within timeout.")
        return None

    audio = np.concatenate(recorded_chunks)
    _log(f"Captured {len(audio) / SAMPLE_RATE:.2f}s of audio.")
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# ContinuousListener — always-on mic with barge-in support
# ---------------------------------------------------------------------------

class ContinuousListener:
    """Always-on microphone that queues user speech as numpy arrays.

    Runs a persistent background thread.  When AEC is active (ref_buf and
    echo_canceller provided), each mic frame is echo-cancelled before being
    passed to the VAD.  Barge-in is detected when VAD fires on the cleaned
    signal during TTS.  A soft fallback gate suppresses residual echo that
    slips through AEC (high residual energy during TTS).

    Usage::

        from lazy_claude.aec import ReferenceBuffer, EchoCanceller
        ref_buf = ReferenceBuffer()
        ec = EchoCanceller()
        listener = ContinuousListener(vad_model, ref_buf=ref_buf, echo_canceller=ec)
        audio = listener.get_next_speech(timeout=60.0)
        listener.stop()
    """

    # VAD probability threshold (unified — applies to both normal and barge-in detection)
    NORMAL_THRESHOLD: float = 0.5
    BARGE_IN_FRAMES: int = 3            # consecutive high-prob frames to confirm barge-in

    # Utterance segmentation
    SILENCE_DURATION: float = 0.5       # seconds of trailing silence to stop
    MIN_SPEECH_DURATION: float = 0.5    # minimum speech before a stop is honoured

    # Fallback gate: if AEC residual RMS power exceeds this during TTS → suppress chunk.
    # Acts as a safety net when the adaptive filter has not yet converged.
    # Power is mean of squared samples. 0.001 ≈ -60 dBFS (RMS ~0.03).
    # Low threshold because even quiet echo is intelligible to Whisper.
    AEC_RESIDUAL_GATE_THRESHOLD: float = 0.001

    # Post-TTS echo tail: keep the fallback gate active for this long after TTS stops.
    # Catches echo from the last TTS chunks still propagating through air/buffers.
    POST_TTS_ECHO_TAIL: float = 0.8  # seconds

    def __init__(
        self,
        vad_model: SileroVAD,
        ref_buf: "Optional[Any]" = None,
        echo_canceller: "Optional[Any]" = None,
    ) -> None:
        """
        Parameters
        ----------
        vad_model:
            Loaded SileroVAD instance.
        ref_buf:
            Optional ReferenceBuffer shared with TTSEngine.  When provided,
            the mic callback reads reference samples from it and passes them
            to the echo_canceller.
        echo_canceller:
            Optional EchoCanceller instance.  Used together with ref_buf to
            subtract the TTS speaker signal from the mic signal before VAD.
        """
        self._vad = vad_model
        self._ref_buf = ref_buf
        self._echo_canceller = echo_canceller
        self._slot_lock = threading.Condition()
        self._pending: "np.ndarray | None" = None
        self._barge_in_event = threading.Event()
        self._stop_event = threading.Event()
        # Lightweight flag: True while TTS is playing (used for fallback gate only)
        self._tts_active: bool = False
        # Timestamp when TTS stopped — used for post-TTS echo tail
        self._tts_stopped_at: float = 0.0
        # threading.Event: set = voice mode active (mic collects speech)
        self._active = threading.Event()

        # Per-utterance mutable state (written only by the mic callback thread)
        self._recording: bool = False
        self._utterance_chunks: list[np.ndarray] = []
        self._accumulated_speech: float = 0.0
        self._silence_started: float | None = None
        self._barge_in_frame_count: int = 0

        # Device change flag: set by _handle_device_change(), cleared by _run()
        # after it restarts the audio stream.
        self._device_changed: bool = False

        # Callback reference — set by _run() once the capture rate is known.
        # Exposed so tests can invoke the callback directly without starting audio.
        self._callback: "Optional[Any]" = None

        # --- Wake-word engine ---
        self._porcupine: "Optional[Any]" = None
        try:
            self._porcupine = create_wakeword_detector()
            if self._porcupine is not None:
                _log("openWakeWord wake-word engine initialised.")
        except Exception as e:
            _log(f"openWakeWord init failed: {e}")
            self._porcupine = None

        # Mode state machine: "wake_word" (waiting for keyword) or "active" (VAD listening)
        self._mode: str = "wake_word" if self._porcupine is not None else "active"
        self._active_since: "float | None" = None

        # Wake-word-only mode: after one utterance, return to wake_word mode.
        if self._porcupine is not None and os.environ.get("LAZY_CLAUDE_ALWAYS_ON") != "1":
            self._wake_word_only_mode: bool = True
        else:
            self._wake_word_only_mode = False

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="continuous-listener"
        )
        self._thread.start()
        _log("ContinuousListener started (inactive until voice mode enabled).")

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Utterance state reset (Phase 2 contract)
    # ------------------------------------------------------------------

    def _reset_utterance_state(self) -> None:
        """Reset ALL mid-utterance fields to their initial values.

        This method is the canonical way to clear per-utterance state.
        Phase 2 calls it on mode switches and set_active(False) so that no
        partial utterance audio leaks across context boundaries.

        Fields reset:
        - ``_recording``             → False
        - ``_accumulated_speech``    → 0.0
        - ``_silence_started``       → None
        - VAD state (SileroVAD GRU)  → reset via ``_vad.reset()``

        Note on ``_utterance_chunks``: the list is replaced with a new empty
        list (not cleared in-place) so that any concurrent reader of a prior
        reference is unaffected.

        Thread safety: set_active() runs on any thread; ``_make_callback``'s
        closure runs on the sounddevice callback thread.  The plain bool and
        list assignment are atomic in CPython, making this safe for Phase 1.
        Phase 2 may add a lock if needed.
        """
        self._recording = False
        self._utterance_chunks = []
        self._accumulated_speech = 0.0
        self._silence_started = None
        # Reset VAD GRU state so previous utterance's hidden state doesn't
        # contaminate the next utterance after a mode switch.
        try:
            self._vad.reset()
        except Exception:
            pass

    def set_active(self, active: bool) -> None:
        """Enable or disable speech collection (voice mode toggle).

        When inactive the background mic thread keeps running but discards
        all audio — no utterances are queued and barge-in is suppressed.
        Call set_active(True) to re-enable collection.
        """
        if active:
            self.drain_queue()
            self._active.set()
            _log("ContinuousListener: voice mode ON.")
        else:
            self._active.clear()
            self._reset_utterance_state()
            self.drain_queue()
            # On deactivation, return to wake_word mode if wake-word detection is available
            if getattr(self, '_porcupine', None) is not None:
                self._mode = "wake_word"
            _log("ContinuousListener: voice mode OFF.")

    @property
    def is_active(self) -> bool:
        return self._active.is_set()

    def set_tts_playing(self, playing: bool) -> None:
        """Set the lightweight TTS-active flag used by the fallback gate.

        This is a thin flag only — AEC handles the main echo suppression.
        """
        self._tts_active = playing
        if not playing:
            self._tts_stopped_at = time.monotonic()

    def clear_barge_in(self) -> None:
        """Reset the barge-in flag before a new TTS turn."""
        self._barge_in_event.clear()

    @property
    def barge_in(self) -> threading.Event:
        """Event that is set when barge-in speech is detected during TTS."""
        return self._barge_in_event

    def get_next_speech(self, timeout: float = 60.0) -> "np.ndarray | None":
        """Block until the next user utterance arrives, or *timeout* seconds.

        Returns a 16 kHz mono float32 array, or None on timeout.
        Uses a single-slot Condition to ensure no lost wakeups and no race
        between replacement and consumption.
        """
        with self._slot_lock:
            notified = self._slot_lock.wait_for(
                lambda: self._pending is not None, timeout=timeout
            )
            if not notified or self._pending is None:
                return None
            audio = self._pending
            self._pending = None
            return audio

    def drain_queue(self) -> None:
        """Discard the pending speech slot (e.g. TTS echo that slipped through)."""
        with self._slot_lock:
            self._pending = None

    def stop(self) -> None:
        """Stop the background listener thread."""
        self._stop_event.set()
        if getattr(self, '_porcupine', None) is not None:
            try:
                self._porcupine.delete()
            except Exception:
                pass
        _log("ContinuousListener stop requested.")

    def _handle_device_change(self) -> None:
        """Handle an audio device change.

        When PortAudio reports a device change (e.g. user unplugged headphones),
        this method:
        1. Resets the echo canceller's filter coefficients via reset_full()
           because the acoustic path has changed and learned coefficients are stale.
        2. Resets the EchoCanceller's delay estimation so it re-calibrates
           on the next few frames.
        3. Sets the _device_changed flag so _run() can restart the stream.

        This is called either from the mic callback (when a status flag indicates
        a device problem) or externally when a device change event is detected.
        """
        _log("ContinuousListener: audio device change detected — resetting AEC.")

        if self._echo_canceller is not None:
            # reset_full() zeros coefficients + clears delay state
            if hasattr(self._echo_canceller, 'reset_full'):
                self._echo_canceller.reset_full()
            else:
                # Fallback for compatibility: at least reset delay estimation
                if hasattr(self._echo_canceller, '_delay_estimated'):
                    self._echo_canceller._delay_estimated = False

        # Set flag so _run() can restart the PortAudio stream
        self._device_changed = True

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _make_callback(self, capture_rate: int, needs_resample: bool):
        """Build and store the sounddevice callback, returning it.

        Split out from _run() so tests can call self._callback directly
        without needing audio hardware.
        """
        chunk_duration = CHUNK_SAMPLES / SAMPLE_RATE  # seconds per VAD chunk

        def _play_ping_sd() -> None:
            """Play a brief 440Hz tone via sounddevice to confirm wake word."""
            try:
                import sounddevice as _sd  # noqa: PLC0415
                t = np.linspace(0, 0.08, int(0.08 * 24000), endpoint=False)
                ping = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
                self._tts_active = True
                try:
                    _sd.play(ping, samplerate=24000, blocking=True)
                finally:
                    self._tts_active = False
            except Exception as exc:
                _log(f"WARNING: ping playback failed: {exc}")
                self._tts_active = False

        def _callback(
            indata: np.ndarray, frames: int, _time_info, status
        ) -> None:
            # Drop all audio when voice mode is inactive
            if not self._active.is_set():
                return

            mono = indata[:, 0].copy()
            chunk_16k = _resample_to_16k(mono, capture_rate) if needs_resample else mono

            # Guard: VAD model expects exactly CHUNK_SAMPLES frames
            if len(chunk_16k) != CHUNK_SAMPLES:
                return

            now = time.monotonic()

            # --- Wake word mode: run the detector, not VAD ---
            _porcupine = getattr(self, '_porcupine', None)
            _mode = getattr(self, '_mode', 'active')
            if _porcupine is not None and _mode == "wake_word":
                pcm_int16 = (chunk_16k * 32767).clip(-32768, 32767).astype(np.int16)
                result = _porcupine.process(pcm_int16)
                if result >= 0:
                    _play_ping_sd()
                    self._reset_utterance_state()
                    self._mode = "active"
                    self._active_since = time.monotonic()
                    _log("ContinuousListener: wake word detected — switching to active.")
                return

            # ── AEC: pull reference and cancel echo ──
            if self._ref_buf is not None and self._echo_canceller is not None:
                ref_chunk = self._ref_buf.read(CHUNK_SAMPLES)
                processed = self._echo_canceller.cancel(chunk_16k, ref_chunk)
            else:
                processed = chunk_16k

            tts_active = self._tts_active

            # Check if we're in the post-TTS echo tail window
            in_echo_tail = (
                not tts_active
                and self._tts_stopped_at > 0
                and (now - self._tts_stopped_at) < self.POST_TTS_ECHO_TAIL
            )

            # ── Fallback gate: suppress if AEC residual energy is too high ──
            # Active during TTS and for POST_TTS_ECHO_TAIL seconds after.
            if tts_active or in_echo_tail:
                residual_power = float(np.mean(processed.astype(np.float64) ** 2))
                if residual_power > self.AEC_RESIDUAL_GATE_THRESHOLD:
                    # Echo leaked through — discard chunk; decay barge-in counter
                    self._barge_in_frame_count = max(0, self._barge_in_frame_count - 1)
                    return

            prob = self._vad(processed)

            if tts_active:
                # ── TTS playing: barge-in detection on cleaned signal ──
                # Use the same NORMAL_THRESHOLD — AEC has already attenuated echo.
                if prob >= self.NORMAL_THRESHOLD:
                    self._barge_in_frame_count += 1
                    if self._barge_in_frame_count >= self.BARGE_IN_FRAMES:
                        # Confirmed barge-in: raise event and switch to record
                        self._barge_in_event.set()
                        self._tts_active = False   # clear TTS flag
                        self._barge_in_frame_count = 0
                        # Start capturing the barge-in utterance
                        self._recording = True
                        self._accumulated_speech = chunk_duration
                        self._silence_started = None
                        self._utterance_chunks = [processed]
                else:
                    self._barge_in_frame_count = max(0, self._barge_in_frame_count - 1)
                return  # discard chunk (still in TTS, barge-in handled above)

            # ── Normal listening ──
            self._barge_in_frame_count = 0
            is_speech = prob >= self.NORMAL_THRESHOLD

            if not self._recording:
                if is_speech:
                    self._recording = True
                    self._accumulated_speech = chunk_duration
                    self._silence_started = None
                    self._utterance_chunks = [processed]
                else:
                    # Check 15s timeout while in WAITING state (not recording)
                    _active_since = getattr(self, '_active_since', None)
                    if (
                        _active_since is not None
                        and now - _active_since > 15.0
                    ):
                        _log("ContinuousListener: 15s timeout — returning to wake_word.")
                        self._reset_utterance_state()
                        if getattr(self, '_porcupine', None) is not None:
                            self._mode = "wake_word"
                            self._active_since = None
            else:
                self._utterance_chunks.append(processed)
                if is_speech:
                    self._accumulated_speech += chunk_duration
                    self._silence_started = None
                else:
                    if self._silence_started is None:
                        self._silence_started = now
                    elif (
                        now - self._silence_started >= self.SILENCE_DURATION
                        and self._accumulated_speech >= self.MIN_SPEECH_DURATION
                    ):
                        # Utterance complete — store in single-slot and notify
                        audio = np.concatenate(self._utterance_chunks)
                        with self._slot_lock:
                            self._pending = audio.astype(np.float32)
                            self._slot_lock.notify_all()
                        _log(
                            f"ContinuousListener: queued "
                            f"{len(audio) / SAMPLE_RATE:.2f}s utterance."
                        )
                        self._recording = False
                        self._utterance_chunks = []
                        self._accumulated_speech = 0.0
                        self._silence_started = None
                        # After utterance stored: if wake_word_only mode, return to wake_word
                        if getattr(self, '_wake_word_only_mode', False) and getattr(self, '_porcupine', None) is not None:
                            self._reset_utterance_state()
                            self._mode = "wake_word"
                            self._active_since = None

        self._callback = _callback
        return _callback

    def _run(self) -> None:
        import sounddevice as sd  # noqa: PLC0415

        while not self._stop_event.is_set():
            try:
                capture_rate, needs_resample = _get_capture_rate()
            except Exception as exc:
                _log(f"ERROR: ContinuousListener could not query audio device: {exc}")
                return

            native_chunk = CHUNK_SAMPLES if not needs_resample else int(
                CHUNK_SAMPLES * capture_rate / SAMPLE_RATE
            )

            callback = self._make_callback(capture_rate, needs_resample)
            self._device_changed = False

            try:
                with sd.InputStream(
                    samplerate=capture_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=native_chunk,
                    callback=callback,
                ):
                    _log("ContinuousListener: mic open.")
                    # Poll for stop or device change
                    while not self._stop_event.is_set():
                        if self._device_changed:
                            _log("ContinuousListener: device change — restarting stream.")
                            break
                        self._stop_event.wait(timeout=0.5)
            except sd.PortAudioError as exc:
                _log(f"ERROR: ContinuousListener PortAudio: {exc}")
                # If a device error occurs, treat it as a device change and retry
                if not self._stop_event.is_set():
                    _log("ContinuousListener: retrying after PortAudio error…")
                    self._handle_device_change()
                    import time as _time
                    _time.sleep(1.0)  # brief back-off before retry
                    continue
            except Exception as exc:
                _log(f"ERROR: ContinuousListener unexpected: {exc}")
                return  # Unknown error — give up
            finally:
                _log("ContinuousListener: mic closed.")

            # If we get here due to device change, loop continues and restarts
