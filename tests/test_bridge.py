"""Tests for bridge.py — FastAPI voice bridge server."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestBridgeModule:
    def test_auth_token_generated(self):
        """AUTH_TOKEN should be a non-empty hex string."""
        from lazy_claude.bridge import AUTH_TOKEN

        assert isinstance(AUTH_TOKEN, str)
        assert len(AUTH_TOKEN) == 64  # 32 bytes hex

    def test_sentence_splitting(self):
        """Sentence regex should split on sentence boundaries."""
        from lazy_claude.bridge import _SENTENCE_RE

        text = "Hello world. How are you? Fine! Thanks."
        parts = _SENTENCE_RE.split(text)
        assert len(parts) == 4
        assert parts[0] == "Hello world."
        assert parts[1] == "How are you?"

    def test_sentence_no_split_on_single(self):
        """Single sentence should not be split."""
        from lazy_claude.bridge import _SENTENCE_RE

        text = "Hello world"
        parts = _SENTENCE_RE.split(text)
        assert len(parts) == 1


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_before_models(self):
        """Health endpoint should report loading status."""
        from lazy_claude.bridge import app

        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data


class TestWebSocketAuth:
    @pytest.mark.asyncio
    async def test_invalid_token_rejected(self):
        """WebSocket with invalid token should be closed."""
        from lazy_claude.bridge import app

        from fastapi.testclient import TestClient

        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws?token=invalid"):
                pass

    @pytest.mark.asyncio
    async def test_missing_token_rejected(self):
        """WebSocket without token should be closed."""
        from lazy_claude.bridge import app

        from fastapi.testclient import TestClient

        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws?token="):
                pass
