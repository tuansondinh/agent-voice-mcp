# Architecture: lazy-claude (claude-voice)

## Overview

lazy-claude provides voice input/output integration for Claude Code with **two separate audio backends**, each optimized for its platform:

1. **macOS System AEC Path** — Uses `AVAudioEngine` with Voice Processing IO (same tech as FaceTime)
   - Instant, system-level echo cancellation (no convergence delay)
   - Single shared `AVAudioEngine` manages both mic input and TTS output
   - Mic tap delivers clean 16 kHz audio to VAD/STT
   - No custom AEC implementation needed — OS handles it

2. **Fallback Path** (non-macOS or macOS failures) — Uses `sounddevice` + custom adaptive filter
   - Custom PBFDLMS echo cancellation with residual suppression
   - Requires convergence time (6–12 seconds) but works on any platform
   - Graceful fallback if macOS Voice Processing is unavailable

Both paths expose the same MCP tool API: `ask_user_voice`, `speak_message`, `toggle_listening`.

---

## Component Architecture

### Data Flow: macOS System AEC Path

```
┌─────────────────────────────────────────────────────────────────┐
│ MCP Tool: ask_user_voice() / speak_message()                    │
│ (server.py → VoiceServer)                                        │
└────────────┬────────────────────────────────────────────────────┘
             │
      ┌──────▼──────────────────────────────────────────────┐
      │ AVAudioBackend (single shared instance)            │
      │ └─ AVAudioEngine (Voice Processing IO enabled)     │
      │    ├─ inputNode (voice processing + tap)          │
      │    └─ AVAudioPlayerNode (for TTS playback)         │
      └──────┬─────────────────────────┬────────────────────┘
             │                         │
             │ Mic tap (44.1kHz raw)   │ TTS output (24→44.1kHz)
             │                         │
      ┌──────▼──────┐         ┌────────▼──────┐
      │ Resample    │         │ Kokoro        │
      │ 44.1k→16k   │         │ TTS Pipeline  │
      └──────┬──────┘         └────────┬──────┘
             │                         │
      ┌──────▼──────────┐      ┌───────▼─────────┐
      │ AudioRechunker  │      │ Resample        │
      │ → 512-sample    │      │ 24k→44.1kHz     │
      │   VAD frames    │      │ & schedule on   │
      └──────┬──────────┘      │ AVAudioPlayerNode
             │                 └─────────────────┘
      ┌──────▼──────────────┐
      │ Silero VAD          │
      │ (16kHz, 32ms frame) │
      └─────────────────────┘
             │
      ┌──────▼──────────────┐
      │ Barge-in detector   │
      │ (during TTS)        │
      └─────────────────────┘
```

**Key properties:**
- **Single engine**: Both input and output run on the same `AVAudioEngine` instance → system AEC sees both and cancels echo automatically
- **No ReferenceBuffer**: System AEC doesn't need the speaker signal as a reference — it's all handled inside the hardware Voice Processing node
- **Tap callback is minimal**: CoreAudio callback only enqueues raw audio into a bounded queue; all Python work (resampling, rechunking, VAD) happens in a consumer thread
- **Device change recovery**: Engine registers for `AVAudioEngineConfigurationChangeNotification` and restarts on device switch

### Data Flow: Fallback Path (sounddevice + custom AEC)

```
┌─────────────────────────────────────────────────────────────────┐
│ MCP Tool: ask_user_voice() / speak_message()                    │
│ (server.py → VoiceServer)                                        │
└────────────┬────────────────────────────────────────────────────┘
             │
      ┌──────▼────────────────────────────────────────┐
      │ TTSEngine (sounddevice OutputStream)           │
      │ └─ Kokoro TTS Pipeline (24kHz output)         │
      │    ├─ Write to ReferenceBuffer (AEC ref path) │
      │    └─ Write to speaker stream                  │
      └──────┬────────────────────────────────────────┘
             │
             │ Reference signal (24kHz) → ReferenceBuffer
             │
      ┌──────▼──────────────────────────────┐
      │ ContinuousListener (sounddevice)    │
      │ ├─ Mic stream (16kHz)                │
      │ ├─ ReferenceBuffer reader (16kHz)   │
      │ └─ EchoCanceller (PBFDLMS+RES+DTD) │
      └──────┬──────────────────────────────┘
             │
      ┌──────▼──────────────────┐
      │ Cleaned audio (16kHz)   │
      │ AEC + fallback gate      │
      └──────┬──────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Silero VAD                  │
      │ (16kHz, 32ms frame)         │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Barge-in detector           │
      │ (during TTS)                │
      └─────────────────────────────┘
```

**Key properties:**
- **Dual stream**: Mic and speaker signals flow through separate `sounddevice` streams
- **ReferenceBuffer bridge**: TTS output is copied to a resampling buffer that the listener reads
- **Custom AEC**: PBFDLMS adaptive filter learns the acoustic path from TTS → mic, with residual echo suppression and double-talk detection
- **AEC convergence**: Filter needs ~6–12 seconds to converge; fallback gate suppresses residual echo during early convergence
- **AEC calibration**: At server startup, a quiet chirp (200–4000 Hz) plays to train the filter before real TTS utterances

---

## Server Initialization & Path Detection

### VoiceServer startup (server.py)

```python
class VoiceServer:
    def __init__(self):
        _log("Initialising VoiceServer…")

        self._use_macos_aec = False

        # --- Attempt macOS AVAudioEngine backend ---
        if sys.platform == 'darwin':
            self._use_macos_aec = self._try_init_macos_backend()

        # --- Fallback: sounddevice + custom AEC ---
        if not self._use_macos_aec:
            _log("Using sounddevice fallback backend with custom AEC.")
            self._ref_buf = ReferenceBuffer(write_sr=24_000, read_sr=16_000)
            self._echo_canceller = EchoCanceller(mu=0.4, enable_res=True)
            self.tts = TTSEngine(ref_buf=self._ref_buf)
            self._listener = ContinuousListener(
                self._vad_model,
                ref_buf=self._ref_buf,
                echo_canceller=self._echo_canceller,
            )
            self._calibrate_aec()  # 1.5s chirp at startup
```

### macOS backend detection (_try_init_macos_backend)

**Success path:**
1. Platform is `darwin`
2. Import `av_audio.AVAudioBackend` succeeds
3. Create single shared `AVAudioBackend()` instance
4. Instantiate `MacOSContinuousListener` and `MacOSTTSEngine`, both sharing the backend
5. Return `True`

**Fallback path:**
- If platform is not `darwin` OR import fails OR voice processing unavailable OR mic permission denied → return `False`
- VoiceServer then initializes the sounddevice path

---

## AVAudioBackend Internals (macOS Path)

### Initialization Order (Critical for system AEC)

```
1. AVAudioEngine.alloc().init()                              # Create engine
2. Get inputNode and mainMixerNode                           # Fetch node references
3. AVAudioPlayerNode().init() → attach to engine             # Create player
4. Connect player → mainMixerNode with explicit mono format  # Wire playback
5. inputNode.setVoiceProcessingEnabled_(True)                # Enable Voice Processing IO AEC
6. AVAudioEngine.startAndReturnError_()                      # Start engine
7. AVAudioPlayerNode.play()                                  # Start player
8. Register for AVAudioEngineConfigurationChangeNotification # Device change handling
```

**Why this order matters:**
- Voice processing must be enabled **before** starting the engine
- Player must be created and connected **before** voice processing (quiet output otherwise)
- Both input and output must be on the same engine for system AEC to work

### Mic Tap Pipeline

```
CoreAudio Callback (44.1kHz)
    │ [real-time thread — minimal work]
    │
    ├─ Enqueue raw audio into bounded Queue(maxsize=128)
    │ [Only enqueue operation; drop if full]
    │
Consumer Thread (sleeps on queue.get())
    │
    ├─ Dequeue audio batch
    ├─ Resample 44.1kHz → 16kHz (linear interpolation)
    ├─ AudioRechunker: accumulate samples → emit exactly 512-sample chunks
    └─ Call user tap_callback(16kHz_512_samples)
         │
         └─ Silero VAD processes chunk
            (VAD expects 16kHz, 512-sample = 32ms frames)
```

**Threading boundary:**
- CoreAudio callback (real-time) ↔ Queue ↔ Consumer thread (Python)
- Tap callback is completely decoupled from CoreAudio; consumer thread runs at Python pace

**Sample rate note:**
- AVAudioEngine tap operates at native hardware rate (typically 44.1 kHz or 48 kHz)
- Actual capture rate stored in `_actual_capture_rate` after tap is installed
- Resample dynamically to 16 kHz using the detected rate

### Playback Pipeline (TTS)

```
Kokoro TTS Pipeline (generates 24kHz audio)
    │
    ├─ Iterate generator for each chunk (variable size)
    ├─ Resample 24kHz → 44.1kHz (linear interpolation)
    └─ Create AVAudioPCMBuffer from resampled chunk
         │
         ├─ Schedule on AVAudioPlayerNode
         │ (non-blocking; buffer queued for playback)
         │
         └─ Wait for playback completion (completionHandler semaphore)
            [Blocks inside speak() until playback finishes]
         │
    After all chunks scheduled and played:
         └─ TTS.speak() returns
```

**Blocking behavior:**
- Each `scheduleBuffer_completionHandler_` call blocks via semaphore until that buffer is done playing
- `speak()` doesn't return until the entire TTS utterance is audible
- This enables reliable barge-in: `set_tts_playing(False)` only fires after playback actually ends

### Device Change Recovery (_restart_engine)

When user switches audio device (headphones ↔ speakers ↔ Bluetooth), the system sends `AVAudioEngineConfigurationChangeNotification`:

```
Notification fires
    │
    ├─ Remove mic tap (if installed)
    ├─ Stop AVAudioEngine
    ├─ Remove AVAudioPlayerNode
    │
    ├─ Rebuild engine:
    │  ├─ Get fresh inputNode and mainMixerNode references
    │  ├─ Create new AVAudioPlayerNode
    │  ├─ Connect player → mixer (with new format)
    │  └─ Re-enable voice processing
    │
    ├─ Start engine
    ├─ Reinstall mic tap (returns updated _actual_capture_rate)
    │
    └─ Resume operation
```

**Audio during device switch:**
- Any TTS currently playing is cancelled (system stops the engine)
- VAD continues to monitor mic on the new device
- Future TTS utterances play on the new device

---

## Threading Model

### macOS Path

```
┌─────────────────────────────────────────────────────────────┐
│ Main Thread (MCP server + tool handlers)                   │
│ ├─ ask_user_voice() → speak() → TTS scheduler              │
│ ├─ barge_in event ← MacOSContinuousListener                │
│ └─ set_tts_playing(True/False)                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CoreAudio Realtime Thread (in AVAudioEngine tap callback)  │
│ └─ Put raw audio into bounded queue [non-blocking]         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Consumer Thread (AVAudioBackend._consumer_thread)           │
│ ├─ Drain bounded queue (blocks on queue.get())             │
│ ├─ Resample 44.1k → 16k                                    │
│ ├─ Rechunk into 512-sample VAD frames                      │
│ └─ Call tap_callback (routes to MacOSContinuousListener)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Whisper STT Thread (in ask_user_voice)                     │
│ └─ Transcribe VAD output to text                           │
└─────────────────────────────────────────────────────────────┘
```

**Synchronization:**
- `barge_in` Event: set by consumer thread (VAD detects speech during TTS), checked by main thread
- `set_tts_playing()`: main thread sets flag, read by consumer thread (for barge-in gating)
- `stop_playback()`: main thread → AVAudioBackend stop → TTS generator halts

### Fallback Path

```
┌─────────────────────────────────────────────────────────────┐
│ Main Thread (MCP server + tool handlers)                   │
│ ├─ ask_user_voice() → speak() → OutputStream.write()       │
│ ├─ barge_in event ← ContinuousListener                     │
│ └─ set_tts_playing(True/False)                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ sounddevice Callback Threads                               │
│ ├─ InputStream callback: mic samples → input queue         │
│ └─ OutputStream callback: write scheduler (TTS chunks)     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ ContinuousListener Thread (audio processing loop)           │
│ ├─ Drain input queue                                       │
│ ├─ Read from ReferenceBuffer (TTS reference)               │
│ ├─ EchoCanceller.process (PBFDLMS + RES + DTD)             │
│ ├─ Silero VAD                                              │
│ └─ Set barge_in event if speech detected during TTS        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Whisper STT Thread (in ask_user_voice)                     │
│ └─ Transcribe VAD output to text                           │
└─────────────────────────────────────────────────────────────┘
```

---

## MCP Tool Flow

### ask_user_voice(questions: list[str]) → str

**macOS path:**

```
MCP tool call: ask_user_voice(["What is your name?"])
    │
    ├─ Speak question via self.tts.speak()
    │  └─ MacOSTTSEngine.speak() blocks until audible ✓
    │
    ├─ Set self.listening = True
    ├─ Start MacOSContinuousListener
    │  └─ Tap installs; consumer thread begins draining queue
    │
    ├─ Wait for either:
    │  ├─ User speech detected + silence tail → utterance complete
    │  └─ Timeout (no speech)
    │
    ├─ Stop listener
    │  └─ Tap removed; consumer thread exits
    │
    ├─ Transcribe VAD audio via Whisper.cpp
    │
    └─ Return "Q: What is your name?\nA: <transcription>"
```

**Fallback path:**

```
MCP tool call: ask_user_voice(["What is your name?"])
    │
    ├─ Speak question via self.tts.speak()
    │  └─ TTSEngine.speak() blocks until sounddevice OutputStream drains ✓
    │
    ├─ Wait 0.8s post-TTS sleep (echo tail settlement)
    │
    ├─ Set self.listening = True
    ├─ Start ContinuousListener
    │  └─ Mic stream + AEC loop begins
    │
    ├─ Wait for either:
    │  ├─ User speech detected + silence tail → utterance complete
    │  └─ Timeout (no speech)
    │
    ├─ Stop listener
    │  └─ Mic stream closes
    │
    ├─ Transcribe VAD audio via Whisper.cpp
    │
    └─ Return "Q: What is your name?\nA: <transcription>"
```

### speak_message(text: str) → dict

```
MCP tool call: speak_message("Your task is complete")
    │
    ├─ self._listener.set_tts_playing(True)
    │  [Barge-in gate: ignore VAD during TTS]
    │
    ├─ Call self.tts.speak(text)
    │  ├─ macOS:   blocks until playback finishes
    │  └─ Fallback: blocks until OutputStream drains
    │
    └─ self._listener.set_tts_playing(False)
       [Barge-in gate: resume VAD monitoring]

    Return {"status": "spoken", "chars": len(text)}
```

### toggle_listening(enabled: bool) → dict

```
MCP tool call: toggle_listening(enabled=False)
    │
    ├─ self.listening = enabled
    ├─ self._listener.set_active(enabled)
    │  ├─ macOS:   stops/starts tap
    │  └─ Fallback: stops/starts mic stream
    │
    └─ Return {"listening": enabled}
```

---

## Barge-in Flow

**Scenario: User speaks during TTS playback**

```
Main Thread                          Consumer/Listener Thread
─────────────────────────────────────────────────────────────
speak_message("Hello world")
  ├─ set_tts_playing(True)  ────────→  _tts_active = True
  │
  ├─ tts.speak() blocks
  │  [TTS output plays...]
  │                                      VAD processes 16kHz audio
  │                                      User starts speaking...
  │                                      VAD detects speech
  │
  │                                      ✓ VAD fired
  │                                      ✓ _tts_active == True
  │                                      → barge_in.set()
  │
  │                                      (Transcribe continues)
  │
  │     ← ask_user_voice detects barge_in event
  │
  ├─ ask_user_voice() wakes up
  ├─ Calls tts.stop()
  │  ├─ macOS:   cancel pending buffers on AVAudioPlayerNode
  │  └─ Fallback: close OutputStream (stop playback)
  │
  └─ tts.speak() unblocks / returns
     (TTS output silenced; user speech continues)
```

**Key:** The barge-in flag is `_tts_active` in both paths, set only during `set_tts_playing(True/False)` calls. VAD fires continuously, but barge-in event is only set when `_tts_active == True`.

---

## Old Modules (Preserved)

The fallback path uses three legacy modules that remain **unchanged**:

- **`audio.py`** — `ContinuousListener` with sounddevice + VAD + barge-in
  - Entry point: `get_next_speech()` (blocking)
  - Barge-in: set `barge_in` event when VAD fires during `_tts_active`
  - Device change: automatic via sounddevice

- **`tts.py`** — `TTSEngine` with Kokoro TTS + sounddevice OutputStream
  - Entry point: `speak(text)` (blocking)
  - Reference signal: pushes output to ReferenceBuffer for AEC

- **`aec.py`** — `EchoCanceller` + `ReferenceBuffer`
  - PBFDLMS adaptive filter (Partitioned Block Frequency Domain LMS)
  - Residual Echo Suppression (spectral subtraction)
  - Geigel Double-Talk Detector (freeze filter during overlap)
  - Fallback gate (suppress during early convergence)
  - ReferenceBuffer: ring buffer bridging TTS output (24 kHz) to listener (16 kHz)

These modules are **fully functional** when macOS path is unavailable.

---

## New Modules (macOS Path)

- **`av_audio.py`** — AVAudioEngine backend + macOS listener and TTS
  - `AVAudioBackend` — Single engine, mic tap, playback
  - `MacOSContinuousListener` — Tap-based VAD, barge-in
  - `MacOSTTSEngine` — Kokoro → AVAudioBackend playback
  - `AudioRechunker` — Fixed-size chunk delivery
  - `resample_audio()` — Stateless resampling utility

---

## Dependencies

### System requirements (macOS)
- `espeak-ng` — for fallback TTS prosody
- `ffmpeg` — audio resampling utility

### Python dependencies

**Always installed:**
- `kokoro` — TTS engine (both paths)
- `pywhispercpp` — STT (both paths)
- `sounddevice` — Fallback audio I/O
- `soundfile` — Audio file utilities
- `onnxruntime` — Silero VAD inference (both paths)
- `numpy` — Numerical computing
- `mcp[cli]` — MCP server framework

**macOS optional** (required for AVAudioEngine path):
- `pyobjc-framework-AVFoundation` — AVAudioEngine bindings
- `pyobjc-framework-Foundation` — NSNotificationCenter for device change

Specified in `pyproject.toml` as extras:
```toml
[project.optional-dependencies]
macos = ["pyobjc-framework-AVFoundation", "pyobjc-framework-Foundation"]
```

Install for macOS path:
```bash
pip install -e '.[macos]'
# or
uv sync --all-extras
```

---

## Error Handling & Fallback

The system is **fail-safe**: if any step of macOS initialization fails, the server seamlessly switches to the sounddevice fallback.

### macOS path failures
- ❌ Import `av_audio` fails → fallback
- ❌ AVAudioBackend() init fails → fallback
- ❌ Voice processing unavailable → fallback with warning, but continue (no AEC)
- ❌ Mic permission denied → fallback
- ❌ Device change recovery fails → log warning, try to restart on next tap

### Fallback path
- Always available (pure Python + sounddevice)
- AEC calibration failure is non-fatal (logs warning, proceeds)
- Mic/speaker device changes handled automatically by sounddevice

---

## Testing Strategy

**Unit tests** (`tests/test_av_audio.py`, `tests/test_phase2_aec_integration.py`):
- Resample accuracy (44.1k↔16k, 24k↔44.1k, no drift)
- AudioRechunker 512-sample chunking
- AVAudioBackend init/shutdown with mocked PyObjC
- Voice processing enabled on input node
- MacOSContinuousListener VAD from synthetic speech
- MacOSTTSEngine delegation to backend
- 16kHz/512-chunk delivery to Silero VAD
- Fallback path selection when macOS unavailable

**Integration tests** (`scripts/test_macos_aec.py`):
- Manual: TTS phrase spoken, user interrupts, transcription contains only user speech
- Manual: device switch recovery (engine restarts, tap reinstalls)
- E2E: ask_user_voice() round-trip with zero echo in transcription

**Regression tests**:
- All existing tests for sounddevice path still pass
- Old modules (audio.py, tts.py, aec.py) untouched

---

## Performance Characteristics

### macOS Path
- **Echo latency**: 0–1 frame (~22ms) — system AEC is causal within the Voice Processing node
- **Echo attenuation**: 40–60 dB (hardware dependent, typically excellent)
- **Memory**: ~50 MB (engine buffers, Kokoro model)
- **CPU**: ~5–10% (one thread draining queue + VAD)
- **AEC convergence**: Instant (no training period)
- **Barge-in latency**: ~32ms (one VAD frame)

### Fallback Path
- **Echo latency**: 100–500ms (PBFDLMS convergence time)
- **Echo attenuation**: 20–40 dB (after convergence; earlier is worse)
- **Memory**: ~100 MB (sounddevice buffers, Kokoro model, AEC filter)
- **CPU**: ~15–20% (mic callback + AEC convolution + VAD)
- **AEC convergence**: 6–12 seconds (first few seconds use fallback gate)
- **Barge-in latency**: ~32ms + echo tail (~200–500ms post-TTS)

---

## Future Improvements

1. **Resume TTS after device change** — Currently cancelled; could store state and retry
2. **Fallback path low-pass filter** — `scipy.signal.resample_poly` for better downsampling quality
3. **Adaptive barge-in threshold** — Detect speaker volume and adjust VAD sensitivity
4. **Multiple mic support** — Allow user to select mic device at runtime
5. **Noise suppression tuning** — Expose Voice Processing noise suppression level on macOS
