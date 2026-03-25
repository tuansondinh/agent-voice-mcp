"""av_audio.py — macOS AVAudioEngine backend with system-level AEC.

Provides:
- AVAudioBackend: manages AVAudioEngine for both input (mic tap) and output
  (AVAudioPlayerNode) with voice processing (hardware AEC) enabled.
- MacOSContinuousListener: same public API as ContinuousListener but uses
  AVAudioBackend instead of sounddevice; no custom AEC needed.
- MacOSTTSEngine: same public API as TTSEngine but plays via AVAudioBackend.
- resample_audio(): utility for resampling float32 arrays between rates.
- AudioRechunker: buffers variable-size input and delivers fixed-size chunks.

All logging goes to stderr. stdout is reserved for the MCP protocol.

Platform note
-------------
PyObjC / AVFoundation is imported lazily and only required on macOS.
On other platforms the module imports cleanly; attempting to instantiate
AVAudioBackend will raise RuntimeError.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

# Suppress verbose ML framework logs
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('kokoro').setLevel(logging.WARNING)

# Lazy KPipeline import (heavy — only pulled when first TTS speak() is called)
try:
    from kokoro import KPipeline  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    KPipeline = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CAPTURE_RATE = 44_100   # AVAudioEngine voice processing requires 44.1 kHz
_VAD_RATE = 16_000       # Silero VAD expects 16 kHz
_TTS_RATE = 24_000       # Kokoro outputs 24 kHz
_VAD_CHUNK = 512         # samples per VAD frame at 16 kHz (32 ms)
_TAP_BUF_SIZE = 1024     # AVAudioEngine tap request size at 44.1 kHz
_TAP_QUEUE_MAXSIZE = 128  # bounded queue between CoreAudio callback and consumer


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[lazy-claude av_audio] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# resample_audio — stateless, per-call resampling
# ---------------------------------------------------------------------------

def resample_audio(
    samples: np.ndarray,
    from_rate: int,
    to_rate: int,
) -> np.ndarray:
    """Resample a 1-D float32 array from *from_rate* to *to_rate*.

    Uses linear interpolation.  Stateless — no fractional accumulator.
    For streaming use with drift correction, see :class:`AudioRechunker`.

    Parameters
    ----------
    samples:
        Input audio (1-D float32).
    from_rate, to_rate:
        Source and target sample rates in Hz.

    Returns
    -------
    np.ndarray
        Resampled float32 array.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if from_rate == to_rate:
        return samples
    n_in = len(samples)
    if n_in == 0:
        return samples
    n_out = int(round(n_in * to_rate / from_rate))
    if n_out == 0:
        return np.array([], dtype=np.float32)
    x_in = np.linspace(0.0, n_in - 1, n_in)
    x_out = np.linspace(0.0, n_in - 1, n_out)
    return np.interp(x_out, x_in, samples.astype(np.float64)).astype(np.float32)


# ---------------------------------------------------------------------------
# AudioRechunker — buffers variable-size input, delivers fixed-size chunks
# ---------------------------------------------------------------------------

class AudioRechunker:
    """Accumulate variable-size audio pushes and deliver fixed-size chunks.

    Handles fractional-sample drift by tracking exactly which samples have
    been consumed.  Thread-unsafe — call from a single thread.

    Parameters
    ----------
    chunk_size:
        Number of samples per output chunk.
    callback:
        Called with each complete chunk (1-D float32 numpy array).
    """

    def __init__(self, chunk_size: int, callback: Callable[[np.ndarray], None]) -> None:
        self._chunk_size = chunk_size
        self._callback = callback
        self._buffer: list[np.ndarray] = []
        self._buffered: int = 0  # total samples in _buffer

    def push(self, samples: np.ndarray) -> None:
        """Push new samples; emit complete chunks via callback."""
        if len(samples) == 0:
            return
        self._buffer.append(samples)
        self._buffered += len(samples)

        while self._buffered >= self._chunk_size:
            # Collect exactly _chunk_size samples
            chunk_parts: list[np.ndarray] = []
            needed = self._chunk_size
            new_buffer: list[np.ndarray] = []

            for seg in self._buffer:
                if needed == 0:
                    new_buffer.append(seg)
                elif len(seg) <= needed:
                    chunk_parts.append(seg)
                    needed -= len(seg)
                else:
                    chunk_parts.append(seg[:needed])
                    new_buffer.append(seg[needed:])
                    needed = 0

            self._buffer = new_buffer
            self._buffered -= self._chunk_size

            chunk = np.concatenate(chunk_parts) if len(chunk_parts) > 1 else chunk_parts[0].copy()
            self._callback(chunk.astype(np.float32))

    def reset(self) -> None:
        """Discard all buffered samples."""
        self._buffer = []
        self._buffered = 0


# ---------------------------------------------------------------------------
# AVAudioBackend — AVAudioEngine wrapper
# ---------------------------------------------------------------------------

class AVAudioBackend:
    """Manage one AVAudioEngine instance for both mic input and audio output.

    Init order (macOS requirement):
    1. Create AVAudioEngine.
    2. Get inputNode and mainMixerNode.
    3. Create AVAudioPlayerNode, attach to engine, connect to mainMixerNode.
    4. Enable voice processing on inputNode (system AEC).
    5. Start engine.

    The CoreAudio tap callback ONLY enqueues raw bytes into a bounded queue.
    A consumer thread drains the queue, resamples 44.1kHz→16kHz, rechunks
    into 512-sample VAD frames, and calls the user-supplied callback.

    Parameters
    ----------
    Raises
    ------
    RuntimeError
        If AVFoundation (PyObjC) is not available.
    """

    def __init__(self) -> None:
        try:
            import AVFoundation  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "pyobjc-framework-AVFoundation is required for AVAudioBackend. "
                "Install it with: pip install 'lazy-claude[macos]'"
            ) from exc

        try:
            import Foundation  # type: ignore[import-not-found]
            self._Foundation = Foundation
        except ImportError:
            self._Foundation = None

        self._AVFoundation = AVFoundation
        self._running = False
        self._tap_installed = False
        self._tap_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=_TAP_QUEUE_MAXSIZE)
        self._consumer_thread: Optional[threading.Thread] = None
        self._consumer_stop = threading.Event()
        self._tap_callback: Optional[Callable[[np.ndarray], None]] = None

        self._setup_engine()

    def _setup_engine(self) -> None:
        """Build and start the AVAudioEngine."""
        AVFoundation = self._AVFoundation

        # 1. Create engine
        self._engine = AVFoundation.AVAudioEngine.alloc().init()

        # 2. Get nodes
        self._input_node = self._engine.inputNode()
        self._mixer_node = self._engine.mainMixerNode()

        # 3. Create + attach + connect player node
        self._player = AVFoundation.AVAudioPlayerNode.alloc().init()
        self._engine.attachNode_(self._player)

        # Connect player → mainMixerNode using the mixer's input format
        mixer_format = self._mixer_node.outputFormatForBus_(0)
        self._engine.connect_to_format_(self._player, self._mixer_node, mixer_format)

        # 4. Enable voice processing (system-level AEC + NS) on inputNode
        error_ptr = None
        try:
            ok = self._input_node.setVoiceProcessingEnabled_error_(True, error_ptr)
            if not ok:
                _log("WARNING: voice processing not available on this device/macOS version.")
        except (AttributeError, TypeError):
            # Older PyObjC binding or not supported
            _log("WARNING: setVoiceProcessingEnabled:error: not available — trying simpler form.")
            try:
                self._input_node.setVoiceProcessingEnabled_(True)
            except (AttributeError, TypeError):
                _log("WARNING: voice processing unavailable — proceeding without AEC.")

        # 5. Start engine
        try:
            import objc  # type: ignore[import-not-found]
            error_ref = objc.nil
        except ImportError:
            error_ref = None

        started = self._engine.startAndReturnError_(error_ref)
        if not started:
            raise RuntimeError("AVAudioEngine failed to start.")

        self._running = True

        # Start player so it is ready to schedule buffers
        self._player.play()

        # Register for configuration-change notification (device change)
        self._register_config_notification()

        _log("AVAudioBackend: engine started with voice processing.")

    def _register_config_notification(self) -> None:
        """Register for AVAudioEngineConfigurationChangeNotification."""
        try:
            if self._Foundation is None:
                return
            nc = self._Foundation.NSNotificationCenter.defaultCenter()
            nc.addObserverForName_object_queue_usingBlock_(
                "AVAudioEngineConfigurationChangeNotification",
                self._engine,
                None,
                self._on_config_change,
            )
        except Exception as exc:
            _log(f"WARNING: could not register config change notification: {exc}")

    def _on_config_change(self, notification: Any) -> None:
        """Handle AVAudioEngine configuration change (device change)."""
        _log("AVAudioBackend: configuration change — restarting engine.")
        try:
            self._engine.stop()
            self._running = False
            import time as _time
            _time.sleep(0.1)
            started = self._engine.startAndReturnError_(None)
            if started:
                self._running = True
                self._player.play()
                _log("AVAudioBackend: engine restarted after config change.")
            else:
                _log("ERROR: AVAudioBackend: engine failed to restart after config change.")
        except Exception as exc:
            _log(f"ERROR: AVAudioBackend config change handler: {exc}")

    # ------------------------------------------------------------------
    # Mic tap
    # ------------------------------------------------------------------

    def install_mic_tap(self, callback: Callable[[np.ndarray], None]) -> None:
        """Install a tap on the input node; deliver 16kHz/512-sample chunks.

        The tap callback ONLY enqueues raw audio (bounded queue, non-blocking
        drop on overflow) to avoid Python-heavy work in the CoreAudio thread.
        A consumer thread handles resampling and rechunking.

        Parameters
        ----------
        callback:
            Called with each 512-sample float32 chunk at 16kHz.
        """
        if self._tap_installed:
            _log("WARNING: mic tap already installed — ignoring.")
            return

        self._tap_callback = callback

        AVFoundation = self._AVFoundation
        fmt = AVFoundation.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
            AVFoundation.AVAudioPCMFormatFloat32,
            float(_CAPTURE_RATE),
            1,  # mono
            True,
        )

        def _tap_block(buffer: Any, when: Any) -> None:
            """CoreAudio callback — MUST be lightweight."""
            try:
                # Extract float32 pointer from AVAudioPCMBuffer
                frame_count = buffer.frameLength()
                channel_data = buffer.floatChannelData()
                if channel_data is None or frame_count == 0:
                    return
                # channel_data[0] is a ctypes float pointer
                import ctypes
                ptr = channel_data[0]
                arr = np.ctypeslib.as_array(
                    (ctypes.c_float * frame_count).from_address(ctypes.addressof(ptr.contents))
                ).copy()
                try:
                    self._tap_queue.put_nowait(arr)
                except queue.Full:
                    pass  # drop frame — queue is bounded
            except Exception:
                pass  # never crash in CoreAudio callback

        self._input_node.installTapOnBus_bufferSize_format_block_(
            0, _TAP_BUF_SIZE, fmt, _tap_block
        )
        self._tap_installed = True

        # Start consumer thread
        self._consumer_stop.clear()
        self._consumer_thread = threading.Thread(
            target=self._tap_consumer, daemon=True, name="av-tap-consumer"
        )
        self._consumer_thread.start()
        _log("AVAudioBackend: mic tap installed.")

    def _tap_consumer(self) -> None:
        """Drain the tap queue, resample, rechunk, and call the user callback."""
        rechunker = AudioRechunker(
            chunk_size=_VAD_CHUNK,
            callback=self._tap_callback,  # type: ignore[arg-type]
        )
        while not self._consumer_stop.is_set():
            try:
                raw = self._tap_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            resampled = resample_audio(raw, _CAPTURE_RATE, _VAD_RATE)
            rechunker.push(resampled)

    def remove_mic_tap(self) -> None:
        """Remove the mic tap and stop the consumer thread."""
        if not self._tap_installed:
            return
        try:
            self._input_node.removeTapOnBus_(0)
        except Exception as exc:
            _log(f"WARNING: error removing tap: {exc}")
        self._tap_installed = False
        self._consumer_stop.set()
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=2.0)
        _log("AVAudioBackend: mic tap removed.")

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play_audio(self, chunk_24k: np.ndarray) -> None:
        """Resample a 24kHz chunk to 44.1kHz and schedule on the player node.

        Non-blocking — returns immediately after scheduling.

        Parameters
        ----------
        chunk_24k:
            1-D float32 audio at 24kHz (Kokoro output rate).
        """
        if not self._running:
            return

        AVFoundation = self._AVFoundation
        chunk_44k = resample_audio(chunk_24k, _TTS_RATE, _CAPTURE_RATE)
        n = len(chunk_44k)
        if n == 0:
            return

        fmt = AVFoundation.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
            AVFoundation.AVAudioPCMFormatFloat32,
            float(_CAPTURE_RATE),
            1,
            True,
        )
        buf = AVFoundation.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(fmt, n)
        buf.setFrameLength_(n)

        # Write samples into the buffer's float channel data
        try:
            import ctypes
            channel_data = buf.floatChannelData()
            ptr = channel_data[0]
            dest = np.ctypeslib.as_array(
                (ctypes.c_float * n).from_address(ctypes.addressof(ptr.contents))
            )
            dest[:] = chunk_44k
        except Exception as exc:
            _log(f"WARNING: could not write to AVAudioPCMBuffer: {exc}")
            return

        self._player.scheduleBuffer_completionHandler_(buf, None)

    def stop_playback(self) -> None:
        """Interrupt active playback; re-prime the player for the next utterance."""
        try:
            self._player.stop()
        except Exception:
            pass
        try:
            self._player.play()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Remove tap, stop engine, release resources."""
        if self._tap_installed:
            self.remove_mic_tap()
        try:
            self._engine.stop()
        except Exception:
            pass
        self._running = False
        _log("AVAudioBackend: engine stopped.")


# ---------------------------------------------------------------------------
# MacOSContinuousListener — same public API as ContinuousListener
# ---------------------------------------------------------------------------

class MacOSContinuousListener:
    """Always-on microphone listener backed by AVAudioEngine voice processing.

    Identical public API to :class:`~lazy_claude.audio.ContinuousListener`.
    No custom AEC — system voice processing handles echo suppression.
    `_tts_active` flag is used only for barge-in detection.

    Usage::

        listener = MacOSContinuousListener(vad_model)
        listener.set_active(True)
        audio = listener.get_next_speech(timeout=60.0)
        listener.stop()
    """

    NORMAL_THRESHOLD: float = 0.5
    BARGE_IN_FRAMES: int = 3
    SILENCE_DURATION: float = 1.5
    MIN_SPEECH_DURATION: float = 0.5

    def __init__(self, vad_model: Any) -> None:
        self._vad = vad_model
        self._speech_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._barge_in_event = threading.Event()
        self._stop_event = threading.Event()
        self._tts_active: bool = False
        self._tts_stopped_at: float = 0.0
        self._active = threading.Event()

        # Per-utterance state (written only by _process_chunk, single consumer thread)
        self._recording: bool = False
        self._utterance_chunks: list[np.ndarray] = []
        self._accumulated_speech: float = 0.0
        self._silence_started: Optional[float] = None
        self._barge_in_frame_count: int = 0

        # Build backend and install tap
        self._backend = AVAudioBackend()
        self._backend.install_mic_tap(self._process_chunk)

        _log("MacOSContinuousListener started.")

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def set_active(self, active: bool) -> None:
        """Enable or disable speech collection (voice mode toggle)."""
        if active:
            self.drain_queue()
            self._active.set()
            _log("MacOSContinuousListener: voice mode ON.")
        else:
            self._active.clear()
            self.drain_queue()
            _log("MacOSContinuousListener: voice mode OFF.")

    @property
    def is_active(self) -> bool:
        return self._active.is_set()

    def set_tts_playing(self, playing: bool) -> None:
        """Set the TTS-active flag for barge-in detection."""
        self._tts_active = playing
        if not playing:
            self._tts_stopped_at = time.monotonic()

    def clear_barge_in(self) -> None:
        """Reset the barge-in flag before a new TTS turn."""
        self._barge_in_event.clear()

    @property
    def barge_in(self) -> threading.Event:
        """Event set when barge-in speech is detected during TTS."""
        return self._barge_in_event

    def get_next_speech(self, timeout: float = 60.0) -> Optional[np.ndarray]:
        """Block until the next user utterance arrives, or *timeout* seconds.

        Returns a 16kHz mono float32 array, or None on timeout.
        """
        try:
            return self._speech_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_queue(self) -> None:
        """Discard all pending speech."""
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        """Stop the backend and release resources."""
        self._stop_event.set()
        try:
            self._backend.shutdown()
        except Exception:
            pass
        _log("MacOSContinuousListener stopped.")

    # ------------------------------------------------------------------
    # Internal: called by AVAudioBackend tap consumer thread
    # ------------------------------------------------------------------

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """Process one 16kHz/512-sample chunk from the mic tap.

        Called from the tap consumer thread.  No CoreAudio constraints here.
        """
        # Drop all audio when voice mode is inactive
        if not self._active.is_set():
            return

        # Guard: VAD model expects exactly _VAD_CHUNK frames
        if len(chunk) != _VAD_CHUNK:
            return

        chunk = chunk.astype(np.float32)
        prob = self._vad(chunk)
        is_speech = prob >= self.NORMAL_THRESHOLD
        chunk_duration = _VAD_CHUNK / _VAD_RATE  # ~0.032 s
        now = time.monotonic()

        tts_active = self._tts_active

        if tts_active:
            # Barge-in detection: consecutive high-prob frames confirm real speech
            if is_speech:
                self._barge_in_frame_count += 1
                if self._barge_in_frame_count >= self.BARGE_IN_FRAMES:
                    self._barge_in_event.set()
                    self._tts_active = False
                    self._barge_in_frame_count = 0
                    # Start capturing the barge-in utterance immediately
                    self._recording = True
                    self._accumulated_speech = chunk_duration
                    self._silence_started = None
                    self._utterance_chunks = [chunk]
            else:
                self._barge_in_frame_count = max(0, self._barge_in_frame_count - 1)
            return  # discard chunk while TTS is playing

        # Normal listening
        self._barge_in_frame_count = 0

        if not self._recording:
            if is_speech:
                self._recording = True
                self._accumulated_speech = chunk_duration
                self._silence_started = None
                self._utterance_chunks = [chunk]
        else:
            self._utterance_chunks.append(chunk)
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
                    audio = np.concatenate(self._utterance_chunks)
                    self._speech_queue.put(audio.astype(np.float32))
                    _log(
                        f"MacOSContinuousListener: queued "
                        f"{len(audio) / _VAD_RATE:.2f}s utterance."
                    )
                    self._recording = False
                    self._utterance_chunks = []
                    self._accumulated_speech = 0.0
                    self._silence_started = None


# ---------------------------------------------------------------------------
# MacOSTTSEngine — same public API as TTSEngine
# ---------------------------------------------------------------------------

class MacOSTTSEngine:
    """Streaming TTS engine backed by Kokoro-82M, playing via AVAudioBackend.

    Identical public API to :class:`~lazy_claude.tts.TTSEngine`.
    No ReferenceBuffer needed — system AEC in AVAudioBackend handles echo.

    Usage::

        engine = MacOSTTSEngine()
        engine.speak("Hello, world!")
        engine.stop()
    """

    _VOICE = 'af_heart'
    _REPO_ID = 'hexgrad/Kokoro-82M'

    def __init__(self) -> None:
        if KPipeline is None:  # pragma: no cover
            raise ImportError(
                "kokoro is required for MacOSTTSEngine. "
                "Install it with: pip install kokoro"
            )
        self._pipeline = KPipeline(lang_code='a', repo_id=self._REPO_ID)
        self._backend = AVAudioBackend()
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
        """Synthesise *text* and play via AVAudioBackend.

        Parameters
        ----------
        text:
            The text to speak. Empty / whitespace-only strings are a no-op.
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
        """Interrupt active playback."""
        self._stop_event.set()
        self._speaking = False
        try:
            self._backend.stop_playback()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stream_speak(self, text: str) -> None:
        """Run the Kokoro generator and feed each chunk to the backend."""
        generator = self._pipeline(text, voice=self._VOICE)
        for result in generator:
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

            if self._stop_event.is_set():
                break

            self._backend.play_audio(chunk)
