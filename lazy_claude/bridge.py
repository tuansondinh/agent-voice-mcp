"""bridge.py — FastAPI voice bridge server.

WebSocket-based server that connects a phone browser to Claude Code CLI
with voice I/O. Audio is processed on the PC (Whisper STT, Kokoro TTS),
text is piped to/from Claude CLI.

Run via: agent-voice-bridge (or python -m lazy_claude.bridge_main)
"""

from __future__ import annotations

import asyncio
import re
import secrets
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from lazy_claude.audio import load_vad_model
from lazy_claude.bridge_claude import ClaudeSession
from lazy_claude.bridge_tts import SAMPLE_RATE as TTS_SAMPLE_RATE
from lazy_claude.bridge_tts import BufferedTTSEngine
from lazy_claude.bridge_vad import RemoteVADProcessor
from lazy_claude.stt import load_model as load_whisper_model
from lazy_claude.stt import transcribe

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

AUTH_TOKEN = secrets.token_hex(32)

_STATIC_DIR = Path(__file__).parent / "static"

# Sentence boundary pattern for incremental TTS
_SENTENCE_RE = re.compile(r"(?<=[.!?\n])\s+")

# Maximum chars to buffer before forcing a TTS chunk
_MAX_SENTENCE_CHARS = 150


def _log(msg: str) -> None:
    print(f"[bridge] {msg}", file=sys.stderr, flush=True)


def _get_local_ip() -> str:
    """Get the machine's local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Voice Bridge", docs_url=None, redoc_url=None)

# Shared models (loaded once at startup via lifespan or on first connect)
_models: dict[str, Any] = {}
_active_session: BridgeSession | None = None


class BridgeSession:
    """Manages one phone-to-PC voice session over WebSocket.

    Uses two concurrent asyncio tasks:
    - A reader task that continuously reads WebSocket messages and dispatches
      them to queues. This ensures control messages (e.g. stop_tts) are
      processed immediately even while the processor is busy.
    - A processor task that reads from those queues and runs STT/Claude/TTS.
    """

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self._vad_processor = RemoteVADProcessor(
            _models["vad"],
            silence_duration=0.5,
            min_speech_duration=0.3,
            no_speech_timeout=30.0,
        )
        self._tts = _models["tts"]
        self._whisper_model = _models["whisper"]
        self._claude = ClaudeSession()
        self._tts_task: asyncio.Task | None = None
        self._stop_tts = asyncio.Event()

        # Queues feeding the processor from the reader
        # audio_queue holds raw bytes; control_queue holds parsed dicts
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._control_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def run(self) -> None:
        """Main session loop: run reader and processor tasks concurrently."""
        await self._send_json({"type": "ready"})
        _log("Session started")

        reader_task = asyncio.create_task(self._reader_loop())
        processor_task = asyncio.create_task(self._processor_loop())

        try:
            # Wait for either task to finish (disconnect or fatal error)
            done, pending = await asyncio.wait(
                [reader_task, processor_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Cancel the other task
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            self._claude.cancel()
            if self._tts_task and not self._tts_task.done():
                self._tts_task.cancel()
            _log("Session ended")

    async def _reader_loop(self) -> None:
        """Continuously read from WebSocket and dispatch to queues.

        This runs independently of the processor so that control messages
        (e.g. stop_tts) are never blocked by STT/Claude/TTS work.
        """
        import json as _json

        try:
            while True:
                message = await self.ws.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                # Binary: PCM audio from phone
                if "bytes" in message and message["bytes"]:
                    await self._audio_queue.put(message["bytes"])

                # Text/JSON: control messages — handle immediately
                elif "text" in message and message["text"]:
                    try:
                        data = _json.loads(message["text"])
                        # Handle stop_tts here directly so it's never delayed
                        if data.get("type") == "stop_tts":
                            self._stop_tts.set()
                            self._tts.stop()
                            _log("TTS interrupted by client")
                        else:
                            await self._control_queue.put(data)
                    except _json.JSONDecodeError:
                        pass

        except WebSocketDisconnect:
            _log("Client disconnected")
        except Exception as exc:
            _log(f"Reader error: {exc}")

    async def _processor_loop(self) -> None:
        """Read from audio queue and run VAD/STT/Claude/TTS pipeline."""
        try:
            while True:
                data = await self._audio_queue.get()
                await self._handle_audio(data)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            _log(f"Processor error: {exc}")

    async def _handle_audio(self, data: bytes) -> None:
        """Process incoming PCM audio from the phone."""
        # Convert Int16LE bytes to float32
        pcm_int16 = np.frombuffer(data, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32768.0

        # Feed to VAD
        utterance, is_speaking = self._vad_processor.feed(pcm_float)

        # Send VAD state to phone for visual feedback
        await self._send_json({"type": "vad_state", "speaking": is_speaking})

        if utterance is not None:
            await self._process_utterance(utterance)

    async def _process_utterance(self, audio: np.ndarray) -> None:
        """Transcribe speech, send to Claude, stream TTS back."""
        loop = asyncio.get_event_loop()

        # 1. Transcribe with Whisper (blocking, run in executor)
        result = await loop.run_in_executor(
            None, lambda: transcribe(audio, model=self._whisper_model)
        )

        if not result.text or result.no_speech_prob > 0.6:
            _log(f"Discarding noise (no_speech_prob={result.no_speech_prob:.2f})")
            return

        _log(f"User said: {result.text}")

        # 2. Send transcript to phone
        await self._send_json({"type": "transcript", "text": result.text})

        # 3. Send to Claude and stream response with incremental TTS
        await self._stream_claude_response(result.text)

    async def _stream_claude_response(self, user_text: str) -> None:
        """Send to Claude, stream text to phone, and do incremental TTS."""
        self._stop_tts.clear()
        full_response: list[str] = []
        sentence_buffer = ""
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Start TTS consumer task
        tts_task = asyncio.create_task(self._tts_consumer(tts_queue))

        try:
            async for chunk in self._claude.send_message(user_text):
                full_response.append(chunk)
                await self._send_json({"type": "assistant_chunk", "text": chunk})

                # Accumulate for sentence-level TTS
                sentence_buffer += chunk
                sentences = _SENTENCE_RE.split(sentence_buffer)

                if len(sentences) > 1:
                    # All but last are complete sentences
                    for sentence in sentences[:-1]:
                        sentence = sentence.strip()
                        if sentence:
                            await tts_queue.put(sentence)
                    sentence_buffer = sentences[-1]
                elif len(sentence_buffer) > _MAX_SENTENCE_CHARS:
                    # Force TTS on long chunks without sentence boundaries
                    await tts_queue.put(sentence_buffer.strip())
                    sentence_buffer = ""

            # Flush remaining text
            if sentence_buffer.strip():
                await tts_queue.put(sentence_buffer.strip())

            # Signal TTS consumer to finish
            await tts_queue.put(None)

            response_text = "".join(full_response)
            await self._send_json(
                {"type": "assistant_done", "text": response_text}
            )
            _log(f"Claude responded: {response_text[:100]}...")

        except Exception as exc:
            _log(f"Error streaming Claude response: {exc}")
            await tts_queue.put(None)
        finally:
            await tts_task

    async def _tts_consumer(self, queue: asyncio.Queue[str | None]) -> None:
        """Consume sentences from queue and stream TTS audio to phone.

        Audio chunks are yielded one-by-one from the synthesize generator
        via a thread executor, so the stop event is checked between each
        chunk and interruption takes effect immediately.
        """
        loop = asyncio.get_event_loop()

        await self._send_json({"type": "tts_start"})

        try:
            while True:
                sentence = await queue.get()
                if sentence is None:
                    break

                if self._stop_tts.is_set():
                    # Drain remaining items
                    while not queue.empty():
                        queue.get_nowait()
                    break

                # Run the blocking synthesize generator in a thread executor,
                # but yield chunks back to the async loop one at a time so
                # the stop event can interrupt mid-sentence.
                chunk_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

                def _run_synthesis(s: str) -> None:
                    """Run synthesis in thread, push chunks to async queue."""
                    try:
                        for c in self._tts.synthesize(s):
                            # Use call_soon_threadsafe to safely cross threads
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, c)
                    finally:
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

                synth_future = loop.run_in_executor(
                    None, lambda s=sentence: _run_synthesis(s)
                )

                # Stream chunks as they arrive from the synthesis thread
                interrupted = False
                while True:
                    chunk = await chunk_queue.get()
                    if chunk is None:
                        break
                    if self._stop_tts.is_set():
                        interrupted = True
                        # Signal TTS engine to stop early
                        self._tts.stop()
                        break
                    # Convert float32 to Int16LE for phone
                    int16_data = (chunk * 32767).clip(-32768, 32767).astype(
                        np.int16
                    )
                    await self.ws.send_bytes(int16_data.tobytes())

                await synth_future  # ensure thread completes

                if interrupted or self._stop_tts.is_set():
                    # Drain remaining sentences from queue
                    while not queue.empty():
                        queue.get_nowait()
                    break

        except Exception as exc:
            _log(f"TTS consumer error: {exc}")
        finally:
            await self._send_json({"type": "tts_end"})

    async def _send_json(self, data: dict) -> None:
        """Send a JSON message to the phone, ignoring errors."""
        try:
            import json
            await self.ws.send_text(json.dumps(data))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def serve_ui(token: str = Query("")):
    """Serve the mobile web UI."""
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Voice Bridge</h1><p>index.html not found</p>")
    html = index_path.read_text()
    return HTMLResponse(html)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ready" if _models.get("whisper") else "loading",
        "models_loaded": list(_models.keys()),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, token: str = Query("")):
    """WebSocket endpoint for voice communication."""
    global _active_session

    # Auth check
    if token != AUTH_TOKEN:
        await ws.close(code=4001, reason="Invalid token")
        _log("Rejected connection: invalid token")
        return

    # Origin validation — allow localhost and the server's own LAN IP.
    # This prevents cross-site WebSocket hijacking from untrusted pages.
    origin = ws.headers.get("origin", "")
    if origin:
        allowed_prefixes = [
            "http://localhost",
            "http://127.0.0.1",
        ]
        local_ip = _get_local_ip()
        if local_ip != "127.0.0.1":
            allowed_prefixes.append(f"http://{local_ip}")

        if not any(origin.startswith(p) for p in allowed_prefixes):
            await ws.close(code=1008, reason="Invalid origin")
            _log(f"Rejected connection: invalid origin '{origin}'")
            return

    # Single session enforcement
    if _active_session is not None:
        _log("Closing existing session for new connection")
        try:
            await _active_session.ws.close(code=4002, reason="Replaced by new connection")
        except Exception:
            pass

    await ws.accept()
    _log("WebSocket connected")

    session = BridgeSession(ws)
    _active_session = session

    try:
        await session.run()
    finally:
        # Only clear if we still own the reference; a newer connection may
        # have already replaced _active_session before our finally runs.
        if _active_session is session:
            _active_session = None


def load_models() -> None:
    """Eagerly load all ML models. Call at startup."""
    _log("Loading models (this may take 10-15 seconds on first run)...")

    _log("Loading VAD model...")
    _models["vad"] = load_vad_model()

    _log("Loading Whisper model...")
    _models["whisper"] = load_whisper_model()

    _log("Loading TTS engine...")
    _models["tts"] = BufferedTTSEngine()

    _log("All models loaded and ready.")
