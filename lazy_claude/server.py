"""server.py — FastMCP server exposing voice I/O tools.

Transport: stdio, using the real stdout fd from stdout_guard so that
native libraries (whisper.cpp, onnxruntime, torch) can never pollute
the MCP protocol channel.

Tools exposed
-------------
ask_user_voice(questions: list[str]) -> str
    Half-duplex voice Q&A: speak each question, record answer, transcribe.

speak_message(text: str) -> dict
    TTS-only output.  Returns {"status": "spoken", "chars": len(text)}.

toggle_listening(enabled: bool) -> dict
    Enable/disable microphone recording.  When disabled ask_user_voice
    still speaks the question but skips recording.
    Returns {"listening": enabled}.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from io import TextIOWrapper

import numpy as np
from typing import Any

# stdout_guard MUST be imported first so that the real stdout fd is
# preserved before anything else writes to it.
from lazy_claude.stdout_guard import get_mcp_stdout  # noqa: E402 (intentional early import)

# ---------------------------------------------------------------------------
# Canonical environment variable names
# ---------------------------------------------------------------------------
# Set to "1" to keep the mic always-on even when wake-word detection is configured.
# Default (unset or "0"): wake-word-only mode when a detector is available.
ENV_LAZY_CLAUDE_ALWAYS_ON = "LAZY_CLAUDE_ALWAYS_ON"
_VOICE_SUBMIT_KEYWORD_RE = re.compile(r"(?i)\bover\b[\s.!?,:;]*$")
_INITIAL_RESPONSE_TIMEOUT = 60.0
_CONTINUATION_RESPONSE_TIMEOUT = 3.0


# All logging goes to stderr.
def _log(msg: str) -> None:
    print(f"[lazy-claude server] {msg}", file=sys.stderr, flush=True)


def _strip_voice_submit_keyword(text: str) -> tuple[str, bool]:
    """Strip a trailing voice submit keyword such as 'OVER'."""
    stripped = _VOICE_SUBMIT_KEYWORD_RE.sub("", text).strip()
    return stripped, stripped != text.strip()


# ---------------------------------------------------------------------------
# Lazy imports for heavy dependencies
# ---------------------------------------------------------------------------

from lazy_claude.tts import TTSEngine
from lazy_claude.stt import load_model, transcribe
from lazy_claude.audio import load_vad_model, ContinuousListener
from lazy_claude.aec import ReferenceBuffer, EchoCanceller


# ---------------------------------------------------------------------------
# VoiceServer — state holder + tool implementations
# ---------------------------------------------------------------------------

class VoiceServer:
    """Holds shared state and implements the voice tool logic.

    Attributes
    ----------
    listening : bool
        When False, ask_user_voice skips mic recording and returns a
        "skipped — listening paused" placeholder.
    busy : bool
        True while a voice turn is in progress.  Concurrent calls are
        rejected immediately.
    tts : TTSEngine | MacOSTTSEngine
        Shared TTS engine instance.
    _use_macos_aec : bool
        True when the macOS AVAudioEngine backend is active (system AEC).
        False when using the sounddevice fallback with custom AEC.
    """

    def __init__(self) -> None:
        _log("Initialising VoiceServer…")

        self._use_macos_aec: bool = False

        # --- Attempt macOS AVAudioEngine backend ---
        if sys.platform == 'darwin':
            self._use_macos_aec = self._try_init_macos_backend()

        # --- Fallback: sounddevice + custom AEC ---
        if not self._use_macos_aec:
            _log("Using sounddevice fallback backend with custom AEC.")
            self._ref_buf = ReferenceBuffer(write_sr=24_000, read_sr=16_000)
            self._echo_canceller = EchoCanceller(
                mu=0.4,
                enable_res=True,
            )
            self.tts = TTSEngine(ref_buf=self._ref_buf)
            self._whisper_model = load_model()
            self._vad_model = load_vad_model()
            self._listener = ContinuousListener(
                self._vad_model,
                ref_buf=self._ref_buf,
                echo_canceller=self._echo_canceller,
            )
            # Run AEC calibration at startup on the fallback path only
            self._calibrate_aec()

        self.listening: bool = True
        if self.listening:
            self._listener.set_active(True)
        self.busy: bool = False
        self._lock = threading.Lock()
        _log("VoiceServer ready.")

    def _try_init_macos_backend(self) -> bool:
        """Try to initialise the macOS AVAudioEngine backend.

        Returns True if successful, False if any step fails (import error,
        init error, no mic permission).  Logs a warning on failure.
        """
        try:
            from lazy_claude.av_audio import (
                AVAudioBackend,
                MacOSContinuousListener,
                MacOSTTSEngine,
            )
        except ImportError as exc:
            _log(f"WARNING: lazy_claude.av_audio import failed — falling back to sounddevice: {exc}")
            return False

        try:
            # Create ONE shared AVAudioBackend so mic capture and TTS playback
            # run through the same AVAudioEngine instance — required for system AEC.
            shared_backend = AVAudioBackend()

            # Whisper model (needed for STT regardless of audio backend)
            self._whisper_model = load_model()
            self._vad_model = load_vad_model()

            # Instantiate macOS-native listener and TTS, both sharing the same backend.
            self._listener = MacOSContinuousListener(self._vad_model, backend=shared_backend)
            self.tts = MacOSTTSEngine(backend=shared_backend)

            _log("Using macOS AVAudioEngine backend with system AEC.")
            return True

        except Exception as exc:
            _log(f"WARNING: macOS AVAudioEngine backend init failed — falling back: {exc}")
            return False

    # ------------------------------------------------------------------
    # AEC calibration (fallback path only)
    # ------------------------------------------------------------------

    def _calibrate_aec(self) -> None:
        """Play a quiet chirp through speakers to train the AEC filter.

        A 1.5-second logarithmic frequency sweep (200-4000 Hz) at low volume
        is played while the mic is active. The adaptive filter uses this known
        signal to learn the room impulse response, so echo cancellation works
        from the very first real TTS utterance.

        Only called on the sounddevice fallback path — system AEC (macOS) does
        not require calibration.
        """
        import sounddevice as sd

        _log("AEC calibration: starting chirp…")

        # Activate listener so the mic callback runs during calibration
        was_active = self._listener.is_active
        if not was_active:
            self._listener.set_active(True)

        try:
            # Generate a logarithmic chirp at 24kHz (matches TTS output rate)
            chirp_sr = 24_000
            duration = 1.5  # seconds
            t = np.linspace(0, duration, int(chirp_sr * duration), dtype=np.float32)
            f0, f1 = 200.0, 4000.0
            chirp = 0.05 * np.sin(  # low amplitude — barely audible
                2 * np.pi * f0 * duration / np.log(f1 / f0)
                * (np.exp(t / duration * np.log(f1 / f0)) - 1)
            ).astype(np.float32)

            # Push chirp into AEC reference buffer (same path as TTS)
            # and play through speakers simultaneously
            chunk_size = 1024  # samples per write at 24kHz
            with sd.OutputStream(
                samplerate=chirp_sr,
                channels=1,
                dtype='float32',
            ) as stream:
                for i in range(0, len(chirp), chunk_size):
                    chunk = chirp[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                    # Write to ref buffer BEFORE playing (same as TTS path)
                    self._ref_buf.write(chunk)
                    stream.write(chunk.reshape(-1, 1))

            # Let the last echo tail settle
            time.sleep(0.5)
            self._listener.drain_queue()

            _log("AEC calibration: done. Filter should be partially converged.")

        except Exception as exc:
            _log(f"WARNING: AEC calibration failed (non-fatal): {exc}")
        finally:
            # Restore listener state
            if not was_active:
                self._listener.set_active(False)

    # ------------------------------------------------------------------
    # Tool implementations (plain Python — called by MCP tool wrappers)
    # ------------------------------------------------------------------

    def toggle_listening_impl(self, *, enabled: bool) -> dict[str, Any]:
        """Enable or disable whether voice turns may capture microphone input."""
        self.listening = enabled
        if not enabled:
            self._listener.set_active(False)
        _log(f"Listening {'enabled' if enabled else 'disabled'}.")
        return {"listening": enabled}

    def speak_message_impl(self, *, text: str) -> dict[str, Any]:
        """Speak text via TTS and return a status dict.

        On macOS path: system AEC handles echo instantly — no sleep or drain.
        On fallback path: wait for echo tail and drain any residual.
        """
        _log(f"speak_message: {len(text)} chars")
        self._listener.set_tts_playing(True)
        try:
            self.tts.speak(text)
        except Exception as exc:
            _log(f"WARNING: TTS error during speak_message: {exc}")
        finally:
            self._listener.set_tts_playing(False)

        if not self._use_macos_aec:
            # Fallback path: wait for echo tail to pass, then drain any residual
            time.sleep(0.8)
            self._listener.drain_queue()

        return {"status": "spoken", "chars": len(text)}

    def ask_user_voice_impl(self, *, questions: list[str]) -> str:
        """Speak each question, record and transcribe each answer.

        Returns a newline-separated string of "Q: …\\nA: …" blocks.
        """
        # Concurrent call protection
        with self._lock:
            if self.busy:
                _log("ask_user_voice: rejected — already busy")
                return "A: (busy — already processing a voice turn)"
            self.busy = True

        try:
            if self.listening:
                self._listener.set_active(True)
            return self._run_qa_session(questions)
        finally:
            self._listener.set_active(False)
            with self._lock:
                self.busy = False

    def _run_qa_session(self, questions: list[str]) -> str:
        parts: list[str] = []
        for question in questions:
            qa = self._ask_single(question)
            parts.append(qa)
        return "\n\n".join(parts)

    def _ask_single(self, question: str) -> str:
        """Speak one question via TTS (with barge-in), then transcribe answer.

        The listener segments audio after ~0.5s of silence. At the server layer
        we keep the answer open for up to 3s between segments so the user can
        pause naturally. A segment ending in ``OVER`` submits immediately.

        On macOS path: system AEC handles echo instantly — no post-TTS sleep
        or drain. On fallback path: wait 0.8s for echo tail, then drain
        residual echo.
        """
        _log(f"Speaking question: {question!r}")

        # Prepare listener for this TTS turn
        self._listener.clear_barge_in()
        self._listener.set_tts_playing(True)

        # Run TTS in a background thread so barge-in can interrupt it
        tts_thread = threading.Thread(
            target=self._speak_safe, args=(question,), daemon=True
        )
        tts_thread.start()

        # Wait for TTS to finish OR barge-in to fire
        while tts_thread.is_alive():
            if self._listener.barge_in.is_set():
                _log("Barge-in detected — stopping TTS.")
                self.tts.stop()
                break
            time.sleep(0.05)

        tts_thread.join(timeout=2.0)
        self._listener.set_tts_playing(False)

        if not self._use_macos_aec:
            # Fallback path: wait for echo tail to pass, then drain any residual echo
            time.sleep(0.8)
            self._listener.drain_queue()

        if not self.listening:
            _log("Listening disabled — skipping mic recording.")
            return f"Q: {question}\nA: (skipped — listening paused)"

        answer_parts: list[str] = []
        accepted_any_segment = False
        segment_timeout = _INITIAL_RESPONSE_TIMEOUT

        # STT loop: skip utterances that are likely noise (high no_speech_prob)
        while True:
            if accepted_any_segment:
                _log(
                    f"Waiting for continuation speech (timeout={segment_timeout:.1f}s)…"
                )
            else:
                _log("Waiting for user speech…")
            try:
                audio = self._listener.get_next_speech(timeout=segment_timeout)
            except Exception as exc:
                _log(f"ERROR: mic/listener error: {exc}")
                return f"Q: {question}\nA: (error — mic failed: {exc})"

            if audio is None:
                if accepted_any_segment:
                    answer = " ".join(part for part in answer_parts if part).strip()
                    _log("No continuation detected — finalising accumulated answer.")
                    return f"Q: {question}\nA: {answer}"

                _log("No speech detected — timed out.")
                return f"Q: {question}\nA: (no response — timed out)"

            # Drain utterances that arrived while STT is running
            self._listener.drain_queue()

            _log("Transcribing…")
            result = transcribe(audio, model=self._whisper_model)
            _log(f"no_speech_prob={result.no_speech_prob:.3f}")

            if result.no_speech_prob > 0.6:
                _log(
                    f"discarded — likely noise "
                    f"(no_speech_prob={result.no_speech_prob:.3f})"
                )
                continue

            accepted_any_segment = True
            answer_text, used_submit_keyword = _strip_voice_submit_keyword(result.text)
            if answer_text or not answer_parts:
                answer_parts.append(answer_text)

            if used_submit_keyword:
                _log("Voice submit keyword detected: OVER")
                answer = " ".join(part for part in answer_parts if part).strip()
                return f"Q: {question}\nA: {answer}"

            segment_timeout = _CONTINUATION_RESPONSE_TIMEOUT

    def _speak_safe(self, text: str) -> None:
        """Run TTS, catching all exceptions so the thread never crashes."""
        try:
            self.tts.speak(text)
        except Exception as exc:
            _log(f"WARNING: TTS error: {exc}")

    # ------------------------------------------------------------------
    # set_listening_mode — Phase 3 API contract (spec only)
    # ------------------------------------------------------------------

    def set_listening_mode_impl(self, *, mode: str) -> dict:
        """Switch between 'wake_word' and 'always_on' listening modes.

        Phase 3 implements this for real (wake-word integration).  The spec
        is defined here so Phase 2 can rely on the interface.

        Parameters
        ----------
        mode:
            Requested mode: ``"wake_word"`` or ``"always_on"``.

        Returns
        -------
        dict
            ``{"mode": effective_mode, "porcupine_available": bool}``

            * ``effective_mode`` — the mode actually entered, which MAY differ
              from ``mode`` when the requested mode is unavailable.  For
              example, requesting ``"wake_word"`` when the detector is not
              installed yields ``effective_mode="always_on"``.
            * ``porcupine_available`` — compatibility field; True when a
              wake-word detector is loaded and ready.

        Notes
        -----
        On mode switch the listener's utterance state is reset via
        ``_reset_utterance_state()`` so no mid-utterance audio leaks across
        mode boundaries.
        """
        porcupine_available = (
            hasattr(self._listener, '_porcupine')
            and self._listener._porcupine is not None
        )

        if mode == "wake_word" and not porcupine_available:
            _log("WARNING: wake_word mode requested but no wake-word detector is available — falling back to always_on.")
            effective_mode = "always_on"
        else:
            effective_mode = mode

        # Update wake_word_only flag on listener
        if hasattr(self._listener, '_wake_word_only_mode'):
            self._listener._wake_word_only_mode = (effective_mode == "wake_word")

        if hasattr(self._listener, '_reset_utterance_state'):
            self._listener._reset_utterance_state()

        if hasattr(self._listener, '_mode'):
            self._listener._mode = "wake_word" if effective_mode == "wake_word" else "active"

        if hasattr(self._listener, '_active_since'):
            self._listener._active_since = None

        _log(f"set_listening_mode: effective_mode={effective_mode}, porcupine_available={porcupine_available}")
        return {"mode": effective_mode, "porcupine_available": porcupine_available}

    def shutdown(self) -> None:
        """Release resources on server stop."""
        _log("VoiceServer shutting down…")
        try:
            self._listener.stop()
        except Exception:
            pass
        try:
            self.tts.stop()
        except Exception:
            pass
        _log("VoiceServer shutdown complete.")


# ---------------------------------------------------------------------------
# FastMCP app factory
# ---------------------------------------------------------------------------

def create_server() -> "tuple[mcp.server.fastmcp.FastMCP, VoiceServer]":  # type: ignore[name-defined]
    """Build and return the FastMCP app with all voice tools registered.

    Returns a (app, voice) tuple so callers can call voice.shutdown() on exit.
    """
    from mcp.server.fastmcp import FastMCP

    voice = VoiceServer()
    app = FastMCP(
        "lazy-claude",
        log_level="WARNING",  # keep FastMCP internal logs off stderr noise
    )

    @app.tool()
    def ask_user_voice(questions: list[str]) -> str:
        """Ask the user one or more questions via voice (TTS + mic + STT).

        For each question: speaks it aloud, waits for TTS to finish, records
        the user's spoken answer with VAD, then transcribes with Whisper.

        Returns a formatted string with Q/A pairs, one per question.
        """
        return voice.ask_user_voice_impl(questions=questions)

    @app.tool()
    def speak_message(text: str) -> dict:
        """Speak a message aloud via TTS without recording.

        Returns {"status": "spoken", "chars": <number of characters spoken>}.
        """
        return voice.speak_message_impl(text=text)

    @app.tool()
    def toggle_listening(enabled: bool) -> dict:
        """Enable or disable microphone recording.

        When disabled, ask_user_voice will still speak the question but skip
        recording and return a "(skipped — listening paused)" answer.
        Call toggle_listening(true) to re-enable.

        Returns {"listening": <current state>}.
        """
        return voice.toggle_listening_impl(enabled=enabled)

    @app.tool()
    def set_listening_mode(mode: str) -> dict:
        """Switch between 'wake_word' and 'always_on' listening modes.

        In wake_word mode, the mic listens for the configured wake word
        before activating VAD and recording.  In always_on mode, VAD is always active.

        If wake-word detection is not configured, requesting wake_word falls back to always_on.

        Returns {"mode": effective_mode, "porcupine_available": bool}.
        """
        return voice.set_listening_mode_impl(mode=mode)

    return app, voice


# ---------------------------------------------------------------------------
# run_server — wire up stdio transport with guarded stdout fd
# ---------------------------------------------------------------------------

def run_server() -> None:
    """Start the MCP server on stdio, using the preserved real stdout fd."""
    import anyio
    from mcp.server.stdio import stdio_server

    app, voice = create_server()

    # Build a text-mode wrapper around the *real* stdout fd (not the
    # redirected one — stdout_guard already redirected fd 1 to stderr).
    real_stdout_binary = get_mcp_stdout()
    real_stdout_text = TextIOWrapper(real_stdout_binary, encoding="utf-8", line_buffering=True)

    async def _run() -> None:
        async with stdio_server(stdout=anyio.wrap_file(real_stdout_text)) as (
            read_stream,
            write_stream,
        ):
            await app._mcp_server.run(
                read_stream,
                write_stream,
                app._mcp_server.create_initialization_options(),
            )

    _log("Starting lazy-claude MCP server (stdio transport)…")
    try:
        anyio.run(_run)
    except KeyboardInterrupt:
        _log("Server interrupted.")
    finally:
        voice.shutdown()
        _log("Server stopped.")
