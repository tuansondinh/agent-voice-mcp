"""Tests for bridge_claude.py — ClaudeSession."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestClaudeSession:
    @patch("lazy_claude.bridge_claude._find_claude_binary", return_value="/usr/bin/claude")
    def test_init_finds_claude(self, mock_find):
        from lazy_claude.bridge_claude import ClaudeSession

        session = ClaudeSession()
        assert session._claude_bin == "/usr/bin/claude"

    @patch("lazy_claude.bridge_claude._find_claude_binary", return_value=None)
    def test_init_raises_without_claude(self, mock_find):
        from lazy_claude.bridge_claude import ClaudeSession

        with pytest.raises(RuntimeError, match="Claude CLI not found"):
            ClaudeSession()

    @patch("lazy_claude.bridge_claude._find_claude_binary", return_value="/usr/bin/claude")
    def test_check_available(self, mock_find):
        from lazy_claude.bridge_claude import ClaudeSession

        assert ClaudeSession.check_available() is True

    @patch("lazy_claude.bridge_claude._find_claude_binary", return_value=None)
    def test_check_not_available(self, mock_find):
        from lazy_claude.bridge_claude import ClaudeSession

        assert ClaudeSession.check_available() is False


class TestExtractText:
    def test_content_block_delta(self):
        from lazy_claude.bridge_claude import ClaudeSession

        event = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        assert ClaudeSession._extract_text(event) == "Hello"

    def test_content_block_delta_non_text(self):
        from lazy_claude.bridge_claude import ClaudeSession

        event = {
            "type": "content_block_delta",
            "delta": {"type": "tool_use_delta"},
        }
        assert ClaudeSession._extract_text(event) is None

    def test_assistant_message(self):
        from lazy_claude.bridge_claude import ClaudeSession

        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ]
            },
        }
        assert ClaudeSession._extract_text(event) == "Hello world"

    def test_result_event(self):
        from lazy_claude.bridge_claude import ClaudeSession

        event = {"type": "result", "result": "Done!"}
        assert ClaudeSession._extract_text(event) == "Done!"

    def test_unknown_event(self):
        from lazy_claude.bridge_claude import ClaudeSession

        event = {"type": "ping"}
        assert ClaudeSession._extract_text(event) is None

    def test_empty_result(self):
        from lazy_claude.bridge_claude import ClaudeSession

        event = {"type": "result", "result": ""}
        assert ClaudeSession._extract_text(event) is None


class TestSendMessage:
    @patch("lazy_claude.bridge_claude._find_claude_binary", return_value="/usr/bin/claude")
    @pytest.mark.asyncio
    async def test_send_empty_message(self, mock_find):
        from lazy_claude.bridge_claude import ClaudeSession

        session = ClaudeSession()
        chunks = []
        async for chunk in session.send_message(""):
            chunks.append(chunk)
        assert chunks == []

    @patch("lazy_claude.bridge_claude._find_claude_binary", return_value="/usr/bin/claude")
    @pytest.mark.asyncio
    async def test_send_message_streams_text(self, mock_find):
        from lazy_claude.bridge_claude import ClaudeSession

        session = ClaudeSession()

        # Mock subprocess
        mock_proc = AsyncMock()
        mock_proc.returncode = 0

        # Simulate stream-json output
        lines = [
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}).encode() + b"\n",
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}).encode() + b"\n",
        ]

        async def mock_stdout_iter():
            for line in lines:
                yield line

        mock_proc.stdout.__aiter__ = mock_stdout_iter
        mock_proc.stderr = AsyncMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            chunks = []
            async for chunk in session.send_message("Hi"):
                chunks.append(chunk)

        assert chunks == ["Hello", " world"]
