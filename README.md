# agent-voice-mcp: Voice MCP Server

A Model Context Protocol (MCP) server that provides voice input/output integration for MCP clients such as Claude Code and Codex, enabling hands-free interaction and voice-driven workflows on macOS.

Talk to Claude Code or Codex by voice.

Typical flow:

```text
You: Enable voice mode

Claude: Now in voice mode. What would you like to do?

You: Open the auth flow and fix the login bug OVER

Claude: I’ll inspect the auth flow and the login path now. Anything else?

You: No, continue OVER
```

How it works:
- Claude asks questions aloud
- you answer by voice
- OPTIONAL: say `OVER` when you want your reply submitted right away, otherwise wait 3 seconds.
- if you pause briefly, Claude can still wait for continuation before finalizing the turn

## Quick Install

### 1. Install system dependencies on macOS
```bash
brew install espeak-ng ffmpeg
```

### 2. Register the MCP server in Claude Code
```bash
claude mcp add agent-voice-mcp npx @tuan_son.dinh/agent-voice-mcp
```

This is the main install path. No repo clone is required.

Or add manually to `~/.claude.json` under `mcpServers`:
```json
{
  "agent-voice-mcp": {
    "command": "npx",
    "args": ["@tuan_son.dinh/agent-voice-mcp"]
  }
}
```

> On first run the server auto-installs its Python dependencies (~500MB). Subsequent starts are instant.

### 3. Restart Claude Code

If Claude Code is already running, restart it so the new MCP server is loaded.

That is enough for installation. No `CLAUDE.md` changes are required just to make the MCP server available.


## How To Use

Once the MCP server is installed, there are two common ways to start using it:

### 1. Ask Claude to enable voice mode

If your prompt setup includes voice-mode instructions, tell Claude something like:

```text
Enable voice mode
```

or:

```text
Switch to voice mode
```

Claude should then start using `ask_user_voice` for spoken interaction instead of asking for typed replies.

### 2. Trigger voice mode with a hook or keybinding

If you have a local hook configured, you can bind a key such as `V` to trigger voice mode.

Typical behavior:
- press `V`
- your hook tells Claude to enter voice mode
- Claude calls `ask_user_voice` with an initial spoken prompt such as: "Now in voice mode. What would you like to do?"

This is optional client-side setup. The MCP server does not create the `V` keybinding by itself; your local Claude workflow or hook needs to do that.

### Leaving voice mode

If your prompt setup mirrors the example in this README, exit by saying something like:
- `exit voice mode`
- `stop voice mode`
- `back to text`

### How Voice Replies Are Submitted

Users should be told explicitly how reply submission works.

The important rule is:

```text
Say OVER to submit your response.
```

More precisely, the current behavior is:
- the listener closes one spoken segment after about `500ms` of silence
- after that, the server waits up to `3 seconds` for more speech
- if the last spoken segment ends with `OVER`, the response is submitted immediately after that segment closes
- `OVER` is removed from the final transcript and is treated as a submit keyword, not part of the answer

Practical examples:

```text
"Please update the login page OVER"
```

This submits the reply as soon as the segment closes.

```text
"Please update the login page"
```

This does not submit immediately. The server will still wait briefly for continuation before finalizing the reply.

```text
"Please update the login page ... and also fix the header OVER"
```

This can span multiple spoken segments. The reply is finalized when the final segment ends with `OVER`.

Recommended wording for users:
- say your answer normally
- end with `OVER` when you want Claude to submit it
- if you do not say `OVER`, Claude may wait for a short continuation window before treating the response as finished

### Other MCP Clients

Any stdio MCP client can run this server. For local development, the generic command is:

```json
{
  "agent-voice-mcp": {
    "command": "uv",
    "args": ["--directory", "/absolute/path/to/agent-voice-mcp", "run", "python", "-m", "lazy_claude"]
  }
}
```

Use your MCP client's normal server-registration flow and substitute the correct local path.

---

## Features

- **Voice Input**: Speak questions and receive transcribed responses automatically submitted
- **Text-to-Speech**: Convert text responses to spoken audio with natural-sounding output
- **Real-Time Recording**: Stream audio input with automatic silence detection and submission
- **Fast Model Loading**: Efficient ML model management for speech recognition and synthesis
- **Fully Integrated**: Registered as `agent-voice-mcp` MCP server in Claude Code

## Prerequisites

### System Requirements
- **macOS** (Apple Silicon or Intel) or Linux/Windows with fallback
- **Python** 3.12+
- **System Tools**:
  ```bash
  brew install espeak-ng ffmpeg
  ```

### Dependencies
Automatically installed via `pyproject.toml`:
- `kokoro` — High-quality text-to-speech engine
- `pywhispercpp` — Fast speech-to-text (Whisper.cpp binding)
- `sounddevice` — Audio input/output (fallback path)
- `soundfile` — Audio file handling
- `onnxruntime` — ML inference for TTS and VAD
- `mcp[cli]` — Model Context Protocol server framework
- `numpy` — Numerical computing

**macOS-specific (optional, for system AEC)**:
- `pyobjc-framework-AVFoundation` — AVAudioEngine bindings
- `pyobjc-framework-Foundation` — Device change notifications

Install with macOS support:
```bash
pip install -e '.[macos]'
# or
uv sync --all-extras
```

## Setup

### 1. Clone or Navigate to Project
```bash
cd /Users/sonwork/Workspace/agent-voice-mcp
```

### 2. Install System Dependencies
```bash
brew install espeak-ng ffmpeg
```

### 3. Set Up Python Environment
```bash
# Sync dependencies using uv
uv sync

# Or use pip
pip install -e .
```

### 4. Register with Your MCP Client

#### Claude Code
The server can be registered in `~/.claude.json`:
```json
{
  "agent-voice-mcp": {
    "command": "uv",
    "args": ["--directory", "/Users/sonwork/Workspace/agent-voice-mcp", "run", "python", "-m", "lazy_claude"]
  }
}
```

If not present, add this entry to the `mcpServers` object in `~/.claude.json`.

#### Codex and Other MCP Clients

Register the same stdio command in your client's MCP config:

```json
{
  "agent-voice-mcp": {
    "command": "uv",
    "args": ["--directory", "/absolute/path/to/agent-voice-mcp", "run", "python", "-m", "lazy_claude"]
  }
}
```

No `CLAUDE.md` update is required. If your client supports project instructions, you can optionally add the short voice-tool guidance shown above.

## Usage

### Starting the Server
The server runs automatically when Claude Code loads the `agent-voice-mcp` MCP. No manual startup needed.

### Available Tools

#### `ask_user_voice(questions: list[str]) → str`
Reads questions aloud and records voice replies with automatic submission.

**Example**:
```python
result = ask_user_voice([
    "What is your name?",
    "What task should I help you with?"
])
# Output: "Q: What is your name?\nA: John\nQ: What task should I help you with?\nA: Fix the bug in the login page"
```

**Parameters**:
- `questions` — List of questions to ask the user

**Returns**: Formatted string with Q/A pairs (one per line): `Q: question\nA: answer`

---

#### `speak_message(text: str) → dict`
Converts text to speech and plays it aloud.

**Example**:
```python
result = speak_message("Your task has been completed successfully")
# Output: {"status": "spoken", "chars": 45}
```

**Parameters**:
- `text` — Message to speak

**Returns**: `{"status": "spoken", "chars": <character count>}`

---

#### `toggle_listening(enabled: bool) → dict`
Enable or disable the voice input listening mode (useful for background noise or multi-tasking).

**Example**:
```python
result = toggle_listening(enabled=False)
# Output: {"listening": False}
```

**Parameters**:
- `enabled` — `True` to enable, `False` to disable

**Returns**: `{"listening": <bool>}` — the current listening state after the change

---

## Voice Workflow Integration

### In Claude Code
If you want Claude Code to behave like a hands-free voice assistant, add the global prompt section above. A minimal version is:

```markdown
Use the ask_user_voice tool to get the user's voice input on whether to proceed with deployment.
```

### Best Practices
1. **Ask Single Questions**: Use clear, concise questions (one per call)
2. **Batch When Appropriate**: Group related questions in a single call to reduce back-and-forth
3. **Mic Is Turn-Based**: Voice capture is activated only during `ask_user_voice(...)`, not as an always-listening input mode
4. **Handle Pauses**: The listener segments after about `500ms` of silence, but the server keeps the reply open for up to `3s` waiting for continuation
5. **Use `OVER` To Submit**: Ending a segment with `OVER` followed by the normal `500ms` segment-close pause submits immediately and removes `OVER` from the returned transcript
6. **Provide Context**: Speak response confirmations back using `speak_message` for feedback loops

## Architecture

The server supports **two audio backends**, automatically selected at startup:

### macOS System AEC Path (Preferred)
On macOS, the server uses `AVAudioEngine` with the Voice Processing IO audio unit
(same tech as FaceTime) for instant, hardware-level echo cancellation:

- **Single `AVAudioBackend` instance** manages both mic input and TTS output through
  one `AVAudioEngine` with voice processing enabled
- **Instant echo cancellation** — no convergence time, system AEC handles it all
- **Mic tap** at native hardware rate (44.1 or 48 kHz) → resample to 16 kHz → deliver
  512-sample chunks to VAD
- **No custom AEC code** — system provides automatic noise suppression + AGC
- **No ReferenceBuffer** — system AEC sees the output automatically
- **Device change recovery** — engine restarts on speaker/headphone/Bluetooth switches

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed macOS path flow and AVAudioBackend internals.

### Fallback Path (Non-macOS or on Failure)
When macOS AVAudioEngine is unavailable or fails, the server falls back to `sounddevice`
with a custom adaptive AEC pipeline:

1. **PBFDLMS adaptive filter** — Partitioned-block frequency-domain LMS filter that
   learns and cancels the speaker-to-microphone acoustic path in real time.
2. **Residual Echo Suppression (RES)** — Spectral subtraction post-filter that catches
   echo components the adaptive filter misses (especially during early convergence).
3. **Geigel Double-Talk Detector (DTD)** — Freezes filter adaptation when user speech
   and TTS overlap, preventing filter divergence.
4. **Fallback gate** — Suppresses chunks with high residual power during TTS playback
   when AEC has not yet converged (safety net for the first few seconds).
5. **AEC calibration chirp** — At startup, a quiet 1.5-second chirp (200–4000 Hz)
   plays to train the filter before real TTS utterances.

The mic stays open throughout TTS playback — no hard muting. Barge-in works by running
VAD on the AEC-cleaned signal; when the user speaks, VAD fires and TTS stops.

**Fallback components:**
- **`audio.py`** — ContinuousListener with sounddevice + VAD + barge-in
- **`tts.py`** — TTSEngine with sounddevice output + ReferenceBuffer for AEC
- **`aec.py`** — PBFDLMS filter + RES + DTD + device change handling

### Shared Components
- **`server.py`** — MCP server setup, path detection, tool registration
- **`av_audio.py`** — AVAudioBackend, MacOSContinuousListener, MacOSTTSEngine (macOS only)
- **`stt.py`** — Speech-to-text using Whisper.cpp (both paths)
- **`stdout_guard.py`** — Prevents audio interference with Claude Code output

### Manual AEC Test Script

To generate a before/after comparison of the echo cancellation:

```bash
# Uses a synthetic 440 Hz tone (no WAV needed):
python scripts/test_aec_manual.py

# Or pass your own WAV file:
python scripts/test_aec_manual.py /path/to/speech.wav --duration 10

# Output written to ./aec_test_output/
#   raw_mic.wav        - unprocessed microphone signal
#   cleaned_aec.wav    - after adaptive filter only
#   cleaned_res.wav    - after adaptive filter + residual echo suppression
```

### Model Downloads
Models are cached locally:
- **Whisper model** → `~/.cache/huggingface/...`
- **Kokoro TTS** → `~/.cache/kokoro/...`

First-run setup will download models (~500MB total). Subsequent runs use cached models.

## Troubleshooting

### Issue: No audio input detected
**Solution**:
1. Check microphone permissions: Settings > Security & Privacy > Microphone
2. Test audio device: `python -m sounddevice`
3. Verify `espeak-ng` and `ffmpeg`:
   ```bash
   espeak-ng --version
   ffmpeg -version
   ```

### Issue: Server fails to start
**Solution**:
1. Verify Python 3.12+: `python --version`
2. Check dependencies: `uv sync` or `pip install -e .`
3. Test manual startup: `uv run python -m lazy_claude`

### Issue: Poor speech recognition
**Solution**:
1. Speak clearly and at normal volume
2. Reduce background noise
3. Use `toggle_listening(False)` to disable during loud environments

### Issue: TTS playback issues
**Solution**:
1. Check speaker volume and mute status
2. Verify audio device: `python -m sounddevice`
3. Test playback: `speak_message("Test")`

### First-Run Model Download Warnings
Warnings about missing HF_TOKEN or CUDA are normal. Models download from Hugging Face automatically.

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Running with Logging
```bash
uv run python -m lazy_claude
```

### Project Structure
```
agent-voice-mcp/
├── lazy_claude/
│   ├── __main__.py              # Entry point
│   ├── server.py                # MCP server, path detection, tool definitions
│   ├── av_audio.py              # macOS: AVAudioBackend, MacOSContinuousListener, MacOSTTSEngine
│   ├── audio.py                 # Fallback: ContinuousListener with sounddevice + VAD
│   ├── tts.py                   # Fallback: TTSEngine with sounddevice + ReferenceBuffer
│   ├── aec.py                   # Fallback: PBFDLMS + RES + DTD echo cancellation
│   ├── stt.py                   # Speech-to-text (both paths)
│   ├── stdout_guard.py          # Output safety wrapper
│   └── models/                  # Model files and utilities
├── scripts/
│   ├── test_aec_manual.py       # Manual AEC before/after comparison (fallback path)
│   └── test_macos_aec.py        # Manual macOS AEC test (system path)
├── tests/                       # Test suite (unit + integration)
├── ARCHITECTURE.md              # Detailed architecture guide
├── pyproject.toml               # Project metadata and dependencies
└── README.md                    # This file
```

## Requirements Specification

### Hardware
- Microphone and speakers (or headphones)
- Minimum 4GB RAM for model loading

### Software
- macOS 10.15+
- Python 3.12+
- espeak-ng and ffmpeg

### Network
- Internet connection (first-run model downloads)
- No ongoing network usage after model caching

## License

Part of the agent-voice-mcp project. See LICENSE for details.

## Support

For issues or feature requests, refer to the project repository or contact the maintainer.
