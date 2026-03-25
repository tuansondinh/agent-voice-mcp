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

import sys
import threading
import time
from io import TextIOWrapper

import numpy as np
from typing import Any

# stdout_guard MUST be imported first so that the real stdout fd is
# preserved before anything else writes to it.
from lazy_claude.stdout_guard import get_mcp_stdout  # noqa: E402 (intentional early import)

# All logging goes to stderr.
def _log(msg: str) -> None:
    print(f"[lazy-claude server] {msg}", file=sys.stderr, flush=True)


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
            # Create a shared backend instance for the listener
            backend = AVAudioBackend()

            # Whisper model (needed for STT regardless of audio backend)
            self._whisper_model = load_model()
            self._vad_model = load_vad_model()

            # Instantiate macOS-native listener and TTS
            self._listener = MacOSContinuousListener(self._vad_model)
            self.tts = MacOSTTSEngine()

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
        """Enable or disable microphone recording (voice mode toggle)."""
        self.listening = enabled
        self._listener.set_active(enabled)
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
            return self._run_qa_session(questions)
        finally:
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

        On macOS path: system AEC handles echo instantly — no post-TTS sleep or drain.
        On fallback path: wait 0.8s for echo tail, then drain residual echo.
        """
        _log(f"Speaking question: {question!r}")

        # Ensure the listener is active (mic collecting speech).
        if not self._listener.is_active:
            self._listener.set_active(True)

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

        # Wait for the user's spoken response from the always-on listener
        _log("Waiting for user speech…")
        try:
            audio = self._listener.get_next_speech(timeout=60.0)
        except Exception as exc:
            _log(f"ERROR: mic/listener error: {exc}")
            return f"Q: {question}\nA: (error — mic failed: {exc})"

        if audio is None:
            _log("No speech detected — timed out.")
            return f"Q: {question}\nA: (no response — timed out)"

        _log("Transcribing…")
        answer = transcribe(audio, model=self._whisper_model)
        return f"Q: {question}\nA: {answer}"

    def _speak_safe(self, text: str) -> None:
        """Run TTS, catching all exceptions so the thread never crashes."""
        try:
            self.tts.speak(text)
        except Exception as exc:
            _log(f"WARNING: TTS error: {exc}")

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
