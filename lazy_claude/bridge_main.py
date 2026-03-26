"""bridge_main.py — Entry point for the remote voice bridge server.

Usage::

    agent-voice-bridge                    # default: 0.0.0.0:8787
    agent-voice-bridge --port 9000        # custom port
    agent-voice-bridge --host 127.0.0.1   # localhost only
"""

from __future__ import annotations

import argparse
import socket
import sys


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


def run_bridge() -> None:
    """Start the voice bridge server."""
    parser = argparse.ArgumentParser(
        description="Remote voice bridge for Claude Code"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8787, help="Port (default: 8787)"
    )
    args = parser.parse_args()

    # Check Claude CLI availability first
    from lazy_claude.bridge_claude import ClaudeSession

    if not ClaudeSession.check_available():
        _log("ERROR: Claude CLI not found on PATH.")
        _log("Install it from: https://docs.anthropic.com/en/docs/claude-code")
        sys.exit(1)

    # Load models eagerly
    from lazy_claude.bridge import AUTH_TOKEN, app, load_models

    load_models()

    # Print access URL
    local_ip = _get_local_ip()
    _log("")
    _log("=" * 60)
    _log("  Voice Bridge is ready!")
    _log("")
    _log(f"  Local:   http://localhost:{args.port}/?token={AUTH_TOKEN}")
    _log(f"  Network: http://{local_ip}:{args.port}/?token={AUTH_TOKEN}")
    _log("")
    _log("  Open the Network URL on your phone to start.")
    _log("  (Both devices must be on the same WiFi network,")
    _log("   or use Tailscale/ngrok for remote access)")
    _log("=" * 60)
    _log("")

    # Start uvicorn
    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        # Disable access logs (noisy with WebSocket)
        access_log=False,
    )


if __name__ == "__main__":
    run_bridge()
