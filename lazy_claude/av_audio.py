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

import ctypes
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from lazy_claude.wakeword import create_wakeword_detector

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

_VAD_RATE = 16_000       # Silero VAD expects 16 kHz
_TTS_RATE = 24_000       # Kokoro outputs 24 kHz
_VAD_CHUNK = 512         # samples per VAD frame at 16 kHz (32 ms)
_TAP_BUF_SIZE = 4096     # AVAudioEngine tap request buffer size
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
        self._actual_capture_rate: int = 48_000  # updated when tap is installed
        self._vp_enabled: bool = False  # track voice processing state

        self._setup_engine()
        self._suspend_engine_if_idle()

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

        # Connect player → mainMixerNode using an explicit mono format at the
        # hardware output rate.  The mixer output format may be stereo/multi-channel,
        # but play_audio() schedules mono buffers — they must match or CoreAudio
        # raises "channelCount mismatch".
        output_rate = float(self._mixer_node.outputFormatForBus_(0).sampleRate())
        player_format = AVFoundation.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
            AVFoundation.AVAudioPCMFormatFloat32,
            output_rate,
            1,
            True,
        )
        self._engine.connect_to_format_(self._player, self._mixer_node, player_format)

        # 4. Voice processing (system-level AEC + NS) is NOT enabled here.
        #    macOS ducks all other app audio while VP is active, so we only
        #    enable it on-demand via _set_voice_processing(True) when the mic
        #    tap actually needs echo cancellation (e.g. ask_user_voice).
        #    speak_message (TTS-only / barge-in) works without VP.

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
            # ObjC blocks only receive a single notification parameter.
            # Use a lambda that captures self so _restart_engine() can be called.
            nc.addObserverForName_object_queue_usingBlock_(
                "AVAudioEngineConfigurationChangeNotification",
                self._engine,
                None,
                lambda notification: self._restart_engine(),
            )
        except Exception as exc:
            _log(f"WARNING: could not register config change notification: {exc}")

    def _ensure_engine_running(self) -> None:
        """Start the AVAudioEngine if it is currently suspended."""
        if self._running:
            return

        # After engine.stop(), node connections are invalidated.
        # Reconnect the player node before restarting — same as _restart_engine.
        AVFoundation = self._AVFoundation
        try:
            output_rate = float(self._mixer_node.outputFormatForBus_(0).sampleRate())
            player_format = AVFoundation.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
                AVFoundation.AVAudioPCMFormatFloat32,
                output_rate,
                1,
                True,
            )
            self._engine.connect_to_format_(self._player, self._mixer_node, player_format)
        except Exception as exc:
            _log(f"WARNING: could not reconnect player on resume: {exc}")

        # Do NOT re-enable voice processing here — it causes macOS audio ducking.
        # VP is only enabled on-demand by install_mic_tap(voice_processing=True).

        started = self._engine.startAndReturnError_(None)
        if not started:
            raise RuntimeError("AVAudioEngine failed to start.")
        self._running = True
        try:
            self._player.play()
        except Exception:
            pass
        _log("AVAudioBackend: engine resumed.")

    def _set_voice_processing(self, enabled: bool) -> None:
        """Toggle voice processing (AEC) on the input node.

        When voice processing is active, macOS ducks other app audio.
        Disabling it when idle prevents unwanted ducking of Safari/YouTube etc.
        """
        if self._vp_enabled == enabled:
            return
        try:
            result = self._input_node.setVoiceProcessingEnabled_error_(enabled, None)
            if isinstance(result, tuple):
                ok = result[0]
            else:
                ok = result
            if ok:
                self._vp_enabled = enabled
                _log(f"AVAudioBackend: voice processing {'enabled' if enabled else 'disabled'}.")
            else:
                _log(f"WARNING: setVoiceProcessingEnabled({enabled}) returned False.")
        except (AttributeError, TypeError):
            try:
                self._input_node.setVoiceProcessingEnabled_(enabled)
                self._vp_enabled = enabled
                _log(f"AVAudioBackend: voice processing {'enabled' if enabled else 'disabled'} (legacy API).")
            except (AttributeError, TypeError):
                _log("WARNING: voice processing toggle unavailable.")

    def _suspend_engine_if_idle(self) -> None:
        """Stop the AVAudioEngine when no mic tap is installed."""
        if not self._running or self._tap_installed:
            return
        # Disable voice processing before stopping to release macOS audio ducking
        self._set_voice_processing(False)
        try:
            self._engine.stop()
        except Exception:
            pass
        self._running = False
        _log("AVAudioBackend: engine suspended.")

    def _restart_engine(self) -> None:
        """Handle AVAudioEngine configuration change (device change).

        AVAudioEngine tears down the audio graph on device change, so we must:
        1. Stop consumer thread and flag tap as gone (engine already removed it).
        2. Stop the engine.
        3. Reconnect player with a fresh mono format at the new hardware rate.
        4. Restart the engine.
        5. Re-install the mic tap if one was active.
        """
        _log("AVAudioBackend: configuration change — restarting engine.")
        try:
            AVFoundation = self._AVFoundation

            # 1. Explicitly remove the tap before stopping — engine.stop() alone
            #    does NOT clear CoreAudio's internal tap slot, causing "nullptr == Tap()"
            #    when we try to re-install after restart.
            had_tap = self._tap_installed
            saved_callback = self._tap_callback
            if had_tap:
                try:
                    self._input_node.removeTapOnBus_(0)
                except Exception:
                    pass
                self._consumer_stop.set()
                if self._consumer_thread is not None:
                    self._consumer_thread.join(timeout=1.0)
                self._tap_installed = False
                self._consumer_stop.clear()

            # 2. Stop engine.
            self._running = False
            self._engine.stop()
            time.sleep(0.1)

            # 3. Reconnect player with a fresh mono format at the new output rate.
            #    The hardware output rate may have changed after device switch.
            try:
                output_rate = float(self._mixer_node.outputFormatForBus_(0).sampleRate())
                player_format = AVFoundation.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
                    AVFoundation.AVAudioPCMFormatFloat32,
                    output_rate,
                    1,
                    True,
                )
                self._engine.connect_to_format_(self._player, self._mixer_node, player_format)
            except Exception as exc:
                _log(f"WARNING: AVAudioBackend: could not reconnect player after device change: {exc}")

            # 4. Restart.
            started = self._engine.startAndReturnError_(None)
            if not started:
                _log("ERROR: AVAudioBackend: engine failed to restart after config change.")
                return

            self._running = True
            self._player.play()
            _log("AVAudioBackend: engine restarted after config change.")

            # 5. Re-install mic tap if one was active, preserving VP state.
            if had_tap and saved_callback is not None:
                saved_vp = self._vp_enabled
                self._vp_enabled = False  # reset so _set_voice_processing will act
                self.install_mic_tap(saved_callback, voice_processing=saved_vp)

        except Exception as exc:
            _log(f"ERROR: AVAudioBackend config change handler: {exc}")

    # ------------------------------------------------------------------
    # Mic tap
    # ------------------------------------------------------------------

    def install_mic_tap(
        self,
        callback: Callable[[np.ndarray], None],
        voice_processing: bool = True,
    ) -> None:
        """Install a tap on the input node; deliver 16kHz/512-sample chunks.

        The tap callback ONLY enqueues raw audio (bounded queue, non-blocking
        drop on overflow) to avoid Python-heavy work in the CoreAudio thread.
        A consumer thread handles resampling and rechunking.

        Parameters
        ----------
        callback:
            Called with each 512-sample float32 chunk at 16kHz.
        voice_processing:
            Enable system AEC (voice processing) on the input node.
            True for ask_user_voice (clean recording needed).
            False for speak_message barge-in (avoids macOS audio ducking).
        """
        if self._tap_installed:
            _log("WARNING: mic tap already installed — ignoring.")
            return

        self._ensure_engine_running()

        # Enable/disable voice processing BEFORE installing the tap.
        # VP changes the input node's format, so we must set it first.
        self._set_voice_processing(voice_processing)

        self._tap_callback = callback

        # Query the actual capture rate AFTER voice processing state is set.
        # VP can change the input node's format (e.g. 44.1kHz→48kHz, mono→multi-channel).
        # We pass None as the tap format to use the native format and avoid -10865 errors.
        vp_fmt = self._input_node.outputFormatForBus_(0)
        self._actual_capture_rate = int(vp_fmt.sampleRate())
        _log(f"AVAudioBackend: input node format: {self._actual_capture_rate}Hz, "
             f"{vp_fmt.channelCount()}ch")

        def _tap_block(buffer: Any, when: Any) -> None:
            """CoreAudio callback — MUST be lightweight.

            Extracts channel 0 float data via PyObjC indexing (objc.varlist)
            and enqueues the numpy array for the consumer thread.
            """
            try:
                frame_count = buffer.frameLength()
                channel_data = buffer.floatChannelData()
                if channel_data is None or frame_count == 0:
                    return
                # channel_data is objc.varlist; channel_data[0] is channel 0 pointer
                # channel_data[0][i] gives float at index i via PyObjC bridge
                ch0 = channel_data[0]
                arr = np.array([ch0[i] for i in range(frame_count)], dtype=np.float32)
                try:
                    self._tap_queue.put_nowait(arr)
                except queue.Full:
                    pass  # drop frame — queue is bounded
            except Exception:
                pass  # never crash in CoreAudio callback

        # None format = use native format (avoids format mismatch errors)
        self._input_node.installTapOnBus_bufferSize_format_block_(
            0, _TAP_BUF_SIZE, None, _tap_block
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
            resampled = resample_audio(raw, self._actual_capture_rate, _VAD_RATE)
            rechunker.push(resampled)

    def remove_mic_tap(self) -> None:
        """Remove the mic tap and stop the consumer thread."""
        if not self._tap_installed:
            return
        # Disable voice processing immediately to stop macOS audio ducking
        self._set_voice_processing(False)
        try:
            self._input_node.removeTapOnBus_(0)
        except Exception as exc:
            _log(f"WARNING: error removing tap: {exc}")
        self._tap_installed = False
        self._consumer_stop.set()
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=2.0)
        _log("AVAudioBackend: mic tap removed.")
        self._suspend_engine_if_idle()

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play_audio(
        self,
        chunk_24k: np.ndarray,
        completion_handler: Optional[Callable[[], None]] = None,
    ) -> None:
        """Resample a 24kHz chunk to the engine's output rate and schedule playback.

        Non-blocking — returns immediately after scheduling.

        Parameters
        ----------
        chunk_24k:
            1-D float32 audio at 24kHz (Kokoro output rate).
        completion_handler:
            Optional callable invoked by AVAudioPlayerNode when this buffer
            finishes playing.  Used by MacOSTTSEngine to detect when the
            last buffer has been rendered.
        """
        if not self._running:
            self._ensure_engine_running()

        AVFoundation = self._AVFoundation

        # Resample to the engine's output rate (usually 44.1kHz or 48kHz)
        output_rate = int(self._mixer_node.outputFormatForBus_(0).sampleRate())
        resampled = resample_audio(chunk_24k, _TTS_RATE, output_rate)
        n = len(resampled)
        if n == 0:
            return

        # Create a mono interleaved buffer at the output rate
        fmt = AVFoundation.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
            AVFoundation.AVAudioPCMFormatFloat32,
            float(output_rate),
            1,
            True,
        )
        buf = AVFoundation.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(fmt, n)
        buf.setFrameLength_(n)

        # Write samples into the buffer via PyObjC indexing
        try:
            channel_data = buf.floatChannelData()
            ch0 = channel_data[0]
            for i in range(n):
                ch0[i] = float(resampled[i])
        except Exception as exc:
            _log(f"WARNING: could not write to AVAudioPCMBuffer: {exc}")
            return

        # Ensure the player node is in play() state — it may have been stopped
        # by a previous barge-in or been idle since engine startup.
        if not self._player.isPlaying():
            self._player.play()

        self._player.scheduleBuffer_completionHandler_(buf, completion_handler)

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
        self._suspend_engine_if_idle()
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
    SILENCE_DURATION: float = 0.5
    MIN_SPEECH_DURATION: float = 0.2

    def __init__(self, vad_model: Any, backend: "AVAudioBackend") -> None:
        self._vad = vad_model
        self._slot_lock = threading.Condition()
        self._pending: Optional[np.ndarray] = None
        self._barge_in_candidate: Optional[np.ndarray] = None
        self._barge_in_event = threading.Event()
        self._stop_event = threading.Event()
        self._tts_active: bool = False
        self._last_input_at: Optional[float] = None
        self._active = threading.Event()

        # Per-utterance state (written only by _process_chunk, single consumer thread)
        self._recording: bool = False
        self._utterance_chunks: list[np.ndarray] = []
        self._accumulated_speech: float = 0.0
        self._silence_started: Optional[float] = None
        self._barge_in_frame_count: int = 0
        self._barge_in_recording: bool = False
        self._barge_in_chunks: list[np.ndarray] = []
        self._barge_in_accumulated_speech: float = 0.0
        self._barge_in_silence_started: Optional[float] = None

        # Use the shared backend and install tap
        self._backend = backend

        # --- Wake-word engine ---
        self._porcupine: Optional[Any] = None
        try:
            self._porcupine = create_wakeword_detector()
            if self._porcupine is not None:
                _log("openWakeWord wake-word engine initialised.")
        except Exception as e:
            _log(f"openWakeWord init failed: {e}")
            self._porcupine = None

        # Mode state machine: "wake_word" (waiting for keyword) or "active" (VAD listening)
        self._mode: str = "wake_word" if self._porcupine is not None else "active"
        self._active_since: Optional[float] = None

        # Wake-word-only mode: after one utterance, return to wake_word mode.
        # Default True when a wake-word detector is available and LAZY_CLAUDE_ALWAYS_ON != "1".
        if self._porcupine is not None and os.environ.get("LAZY_CLAUDE_ALWAYS_ON") != "1":
            self._wake_word_only_mode: bool = True
        else:
            self._wake_word_only_mode = False

        self._backend_tap_active: bool = False

        _log("MacOSContinuousListener started.")

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
        - ``_utterance_chunks``     → empty list
        - ``_silence_started``      → None
        - ``_barge_in_frame_count`` → 0
        - ``_state``-equivalent     → ``_recording = False``

        Thread safety: called from the public API (set_active) which may run
        on any thread, but ``_process_chunk`` (the writer) runs on the tap
        consumer thread.  Because ``_recording`` is a plain bool and list
        assignment is atomic in CPython, the reset is safe without a lock for
        the Phase 1 contract.  Phase 2 may add a lock if needed.
        """
        self._utterance_chunks = []
        self._silence_started = None
        self._barge_in_frame_count = 0
        self._recording = False

    def _reset_barge_in_state(self) -> None:
        """Clear any in-progress or buffered TTS interruption utterance."""
        self._barge_in_frame_count = 0
        self._barge_in_recording = False
        self._barge_in_chunks = []
        self._barge_in_accumulated_speech = 0.0
        self._barge_in_silence_started = None

    def set_active(self, active: bool, voice_processing: bool = True) -> None:
        """Enable or disable speech collection (voice mode toggle).

        Parameters
        ----------
        active:
            True to start listening, False to stop.
        voice_processing:
            When True, enable system AEC (causes macOS audio ducking).
            Set to False for barge-in-only monitoring during TTS playback.
        """
        if active:
            if not getattr(self, "_backend_tap_active", False):
                self._backend.install_mic_tap(self._process_chunk, voice_processing=voice_processing)
                self._backend_tap_active = True
            self.drain_queue()
            self._active.set()
            _log("MacOSContinuousListener: voice mode ON.")
        else:
            self._active.clear()
            self._reset_utterance_state()
            self._reset_barge_in_state()
            self.drain_queue()
            if getattr(self, "_backend_tap_active", False):
                self._backend.remove_mic_tap()
                self._backend_tap_active = False
            # On deactivation, return to wake_word mode if wake-word detection is available
            if getattr(self, '_porcupine', None) is not None:
                self._mode = "wake_word"
            _log("MacOSContinuousListener: voice mode OFF.")

    @property
    def is_active(self) -> bool:
        return self._active.is_set()

    def set_tts_playing(self, playing: bool) -> None:
        """Set the TTS-active flag for barge-in detection."""
        self._tts_active = playing
        if not playing:
            self._reset_barge_in_state()

    def clear_barge_in(self) -> None:
        """Reset the barge-in flag before a new TTS turn."""
        self._barge_in_event.clear()
        self._reset_barge_in_state()
        with self._slot_lock:
            self._barge_in_candidate = None

    @property
    def barge_in(self) -> threading.Event:
        """Event set when barge-in speech is detected during TTS."""
        return self._barge_in_event

    def get_last_input_at(self) -> Optional[float]:
        """Return the monotonic timestamp of the latest detected speech frame."""
        return self._last_input_at

    def pop_barge_in_candidate(self) -> Optional[np.ndarray]:
        """Return the next buffered interruption utterance, if any."""
        with self._slot_lock:
            audio = self._barge_in_candidate
            self._barge_in_candidate = None
            return audio

    def get_next_speech(self, timeout: float = 60.0) -> Optional[np.ndarray]:
        """Block until the next user utterance arrives, or *timeout* seconds.

        Returns a 16kHz mono float32 array, or None on timeout.
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
        """Discard the pending speech slot."""
        with self._slot_lock:
            self._pending = None

    def stop(self) -> None:
        """Stop the backend and release resources."""
        self._stop_event.set()
        if getattr(self, '_porcupine', None) is not None:
            try:
                self._porcupine.delete()
            except Exception:
                pass
        try:
            self._backend.shutdown()
        except Exception:
            pass
        _log("MacOSContinuousListener stopped.")

    # ------------------------------------------------------------------
    # Internal: called by AVAudioBackend tap consumer thread
    # ------------------------------------------------------------------

    def _play_ping(self) -> None:
        """Play a brief 440Hz confirmation tone via the backend.

        Sets _tts_active around playback to suppress VAD self-triggering.
        """
        t = np.linspace(0, 0.08, int(0.08 * _TTS_RATE), endpoint=False)
        ping = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
        self._tts_active = True
        try:
            self._backend.play_audio(ping)
        finally:
            self._tts_active = False

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
        chunk_duration = _VAD_CHUNK / _VAD_RATE  # ~0.032 s
        now = time.monotonic()

        # --- Wake word mode: run the detector, not VAD ---
        _porcupine = getattr(self, '_porcupine', None)
        _mode = getattr(self, '_mode', 'active')
        if _porcupine is not None and _mode == "wake_word":
            pcm_int16 = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
            result = _porcupine.process(pcm_int16)
            if result >= 0:
                # Keyword detected — play ping and switch to active
                self._play_ping()
                self._reset_utterance_state()
                self._mode = "active"
                self._active_since = time.monotonic()
                _log("MacOSContinuousListener: wake word detected — switching to active.")
            return  # in wake_word mode, don't run VAD

        # --- Active mode: run existing Silero VAD logic ---
        prob = self._vad(chunk)
        is_speech = prob >= self.NORMAL_THRESHOLD

        tts_active = self._tts_active

        if tts_active:
            if is_speech:
                self._last_input_at = now
                if not self._barge_in_recording:
                    self._barge_in_frame_count += 1
                    if self._barge_in_frame_count >= self.BARGE_IN_FRAMES:
                        self._barge_in_recording = True
                        self._barge_in_frame_count = 0
                        self._barge_in_accumulated_speech = chunk_duration
                        self._barge_in_silence_started = None
                        self._barge_in_chunks = [chunk]
                else:
                    self._barge_in_chunks.append(chunk)
                    self._barge_in_accumulated_speech += chunk_duration
                    self._barge_in_silence_started = None
            else:
                if self._barge_in_recording:
                    self._barge_in_chunks.append(chunk)
                    if self._barge_in_silence_started is None:
                        self._barge_in_silence_started = now
                    elif (
                        now - self._barge_in_silence_started >= self.SILENCE_DURATION
                        and self._barge_in_accumulated_speech >= self.MIN_SPEECH_DURATION
                    ):
                        audio = np.concatenate(self._barge_in_chunks).astype(np.float32)
                        with self._slot_lock:
                            self._barge_in_candidate = audio
                        self._barge_in_event.set()
                        self._reset_barge_in_state()
                else:
                    self._barge_in_frame_count = max(0, self._barge_in_frame_count - 1)
            return

        # Normal listening
        self._barge_in_frame_count = 0

        if not self._recording:
            if is_speech:
                self._last_input_at = now
                self._recording = True
                self._accumulated_speech = chunk_duration
                self._silence_started = None
                self._utterance_chunks = [chunk]
            else:
                # Check 15s timeout while in WAITING state (not recording)
                _active_since = getattr(self, '_active_since', None)
                if (
                    _active_since is not None
                    and now - _active_since > 15.0
                ):
                    _log("MacOSContinuousListener: 15s timeout — returning to wake_word.")
                    self._reset_utterance_state()
                    if getattr(self, '_porcupine', None) is not None:
                        self._mode = "wake_word"
                        self._active_since = None
        else:
            self._utterance_chunks.append(chunk)
            if is_speech:
                self._last_input_at = now
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
                    with self._slot_lock:
                        self._pending = audio.astype(np.float32)
                        self._slot_lock.notify_all()
                    _log(
                        f"MacOSContinuousListener: queued "
                        f"{len(audio) / _VAD_RATE:.2f}s utterance."
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

        # German
        engine = MacOSTTSEngine(backend=..., language='de')
        engine.speak("Hallo Welt!")
    """

    _VOICE = 'af_heart'
    _SPEED = 1.2
    _REPO_ID = 'hexgrad/Kokoro-82M'

    def __init__(self, backend: "AVAudioBackend", language: str = 'en') -> None:
        from lazy_claude.tts import _KOKORO_LANG_MAP

        self._language = language.lower()
        self._use_kokoro = self._language in _KOKORO_LANG_MAP

        if self._use_kokoro:
            if KPipeline is None:  # pragma: no cover
                raise ImportError(
                    "kokoro is required for MacOSTTSEngine. "
                    "Install it with: pip install kokoro"
                )
            kokoro_lang = _KOKORO_LANG_MAP[self._language]
            self._pipeline = KPipeline(lang_code=kokoro_lang, repo_id=self._REPO_ID)
        else:
            self._pipeline = None
            _log(f"MacOSTTSEngine: language '{language}' — using external TTS.")

        self._backend = backend
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
            if self._use_kokoro:
                self._stream_speak(text)
            else:
                self._stream_speak_external(text)
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

    def _stream_speak_external(self, text: str) -> None:
        """Speak *text* via espeak-ng / macOS say, playing through AVAudioBackend.

        Generates audio with an external TTS tool, resamples to 24kHz, then
        feeds it to the AVAudioBackend so system AEC still works correctly.
        """
        from lazy_claude.tts import _generate_audio_external, _SAMPLE_RATE as TTS_RATE

        if self._stop_event.is_set():
            return
        try:
            audio, src_rate = _generate_audio_external(text, self._language)
        except Exception as exc:
            _log(f"ERROR: external TTS failed: {exc}")
            return

        if self._stop_event.is_set():
            return

        # Resample to 24kHz (the rate play_audio expects)
        if src_rate != TTS_RATE:
            audio = resample_audio(audio, src_rate, TTS_RATE)

        # Feed in 100 ms chunks so stop_event can be checked between them
        chunk_size = TTS_RATE // 10
        done_event = threading.Event()

        chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
        if not chunks:
            return

        for i, chunk in enumerate(chunks):
            if self._stop_event.is_set():
                return
            is_last = (i == len(chunks) - 1)
            if is_last:
                self._backend.play_audio(chunk, completion_handler=lambda: done_event.set())
            else:
                self._backend.play_audio(chunk)

        done_event.wait(timeout=30.0)

    def _stream_speak(self, text: str) -> None:
        """Run the Kokoro generator and feed each chunk to the backend.

        Streams chunks to AVAudioPlayerNode as they arrive from Kokoro so
        the first audio is heard with minimal latency.  A completion handler
        on the *last* scheduled buffer signals when playback is done.
        """
        generator = self._pipeline(text, voice=self._VOICE, speed=self._SPEED)

        prev_chunk: Optional[np.ndarray] = None
        done_event = threading.Event()

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

            # Schedule the *previous* chunk without a completion handler
            # (we don't yet know if this new chunk is the last one).
            if prev_chunk is not None:
                self._backend.play_audio(prev_chunk)
            prev_chunk = chunk

        if prev_chunk is None or self._stop_event.is_set():
            return

        # Schedule the final chunk with a completion handler.
        def _on_complete() -> None:
            done_event.set()

        self._backend.play_audio(prev_chunk, completion_handler=_on_complete)

        # Wait for the last buffer to finish playing (up to 30 s safety timeout).
        done_event.wait(timeout=30.0)
