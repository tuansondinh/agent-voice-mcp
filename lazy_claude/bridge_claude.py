"""bridge_claude.py — Claude CLI subprocess wrapper for the voice bridge.

Spawns ``claude -p "text" --output-format stream-json`` per message,
parses the streaming JSON output, and yields text chunks.

Supports multi-turn conversations via --resume with a conversation file.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator


def _log(msg: str) -> None:
    print(f"[bridge-claude] {msg}", file=sys.stderr, flush=True)


def _find_claude_binary() -> str | None:
    """Find the claude CLI binary on PATH."""
    return shutil.which("claude")


class ClaudeSession:
    """Manages interaction with Claude CLI via subprocess.

    Each call to ``send_message`` spawns a new ``claude -p`` process.
    Multi-turn context is maintained via a conversation file.

    Parameters
    ----------
    conversation_dir:
        Directory to store conversation files. Defaults to a temp dir.
    """

    def __init__(self, conversation_dir: Path | None = None) -> None:
        self._claude_bin = _find_claude_binary()
        if not self._claude_bin:
            raise RuntimeError(
                "Claude CLI not found on PATH. "
                "Install it from https://docs.anthropic.com/en/docs/claude-code"
            )
        _log(f"Using Claude CLI: {self._claude_bin}")

        if conversation_dir is None:
            conversation_dir = Path(tempfile.mkdtemp(prefix="voice-bridge-"))
        self._conversation_dir = conversation_dir
        self._conversation_dir.mkdir(parents=True, exist_ok=True)

        self._conversation_file = self._conversation_dir / "conversation.json"
        self._active_process: asyncio.subprocess.Process | None = None

    async def send_message(self, text: str) -> AsyncGenerator[str, None]:
        """Send text to Claude CLI, yield text chunks as they stream back.

        Parameters
        ----------
        text:
            User message to send to Claude.

        Yields
        ------
        str
            Text chunks from Claude's response as they arrive.
        """
        if not text.strip():
            return

        cmd = [
            self._claude_bin,
            "-p", text,
            "--output-format", "stream-json",
        ]

        # Resume conversation if file exists
        if self._conversation_file.exists():
            cmd.extend(["--resume", str(self._conversation_file)])

        _log(f"Sending to Claude: {text[:80]}...")

        try:
            self._active_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            assert self._active_process.stdout is not None

            async for line in self._active_process.stdout:
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded:
                    continue

                try:
                    event = json.loads(decoded)
                except json.JSONDecodeError:
                    continue

                # Extract text from stream-json events
                text_chunk = self._extract_text(event)
                if text_chunk:
                    yield text_chunk

            await asyncio.wait_for(self._active_process.wait(), timeout=10)

            if self._active_process.returncode != 0:
                stderr_data = b""
                if self._active_process.stderr:
                    stderr_data = await self._active_process.stderr.read()
                _log(
                    f"Claude exited with code {self._active_process.returncode}: "
                    f"{stderr_data.decode('utf-8', errors='replace')[:200]}"
                )

        except asyncio.TimeoutError:
            _log("Claude CLI timed out")
            if self._active_process:
                self._active_process.kill()
        except Exception as exc:
            _log(f"Error communicating with Claude: {exc}")
        finally:
            self._active_process = None

    def cancel(self) -> None:
        """Kill the active Claude process if running."""
        if self._active_process and self._active_process.returncode is None:
            _log("Cancelling active Claude process")
            self._active_process.kill()

    @staticmethod
    def _extract_text(event: dict) -> str | None:
        """Extract text content from a stream-json event.

        Claude CLI stream-json emits various event types. We extract text from:
        - content_block_delta with text_delta
        - assistant message content blocks
        - result messages
        """
        event_type = event.get("type")

        # content_block_delta events (streaming)
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                return delta.get("text", "")

        # Direct assistant content
        if event_type == "assistant":
            content = event.get("message", {}).get("content", [])
            texts = []
            for block in content:
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            if texts:
                return "".join(texts)

        # Result with text
        if event_type == "result":
            result_text = event.get("result", "")
            if isinstance(result_text, str) and result_text:
                return result_text

        return None

    @staticmethod
    def check_available() -> bool:
        """Check if the claude CLI is available on PATH."""
        return _find_claude_binary() is not None
