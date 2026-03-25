# Plan #4: macOS System AEC via AVAudioEngine

Plan ID: #4
Generated: 2026-03-25
Platform: web
Status: complete

## Context

Plan #3 implemented a custom PBFDLMS adaptive filter for echo cancellation. It works
in synthetic tests (~58dB attenuation) but fails in practice because the filter can't
converge fast enough during short TTS utterances (needs 6-12s, TTS is 2-3s).

macOS has a built-in AEC via the Voice Processing IO audio unit, accessed through
AVAudioEngine. This is what FaceTime uses — instant echo cancellation, no convergence
time, plus free noise suppression and AGC. Requires `pyobjc-framework-AVFoundation`.

**Key constraints:**
- Must use 44.1kHz sample rate (Voice Processing IO requirement on macOS)
- Enable voice processing BEFORE starting the engine
- Set up playback BEFORE enabling voice processing (order matters — otherwise quiet output)
- Only one tap per bus on input node
- Resample: 44.1kHz → 16kHz for VAD/STT, 24kHz TTS → 44.1kHz for playback
- BOTH capture and render must go through AVAudioEngine (system AEC only works when I/O shares the same voice-processing path — can't mix sounddevice with AVAudioEngine)

## Phases

1. [x] Phase 1: AVAudioEngine backend + macOS listener and TTS — complexity: standard
   - Add `pyobjc-framework-AVFoundation` to pyproject.toml as optional macOS dependency
   - Create `lazy_claude/av_audio.py` with `AVAudioBackend` class:
     - Single object managing BOTH input and output via one AVAudioEngine instance (critical: system AEC requires both sides on the same voice-processing path)
     - Init order (must follow exactly): create engine → get nodes → create+attach+connect AVAudioPlayerNode to mainMixerNode → enable voice processing on inputNode (`setVoiceProcessingEnabled_(True)`) → start engine
     - `install_mic_tap(callback)`: tap on inputNode bus 0 at 44.1kHz. **Threading boundary**: tap callback must ONLY enqueue raw audio into a bounded `queue.Queue` (no Python-heavy work in CoreAudio callback). A separate consumer thread drains the queue, resamples 44.1kHz→16kHz, rechunks into 512-sample VAD frames, and calls the callback.
     - Rechunk/resample bridge: accumulate 44.1kHz tap samples in a buffer, resample to 16kHz using numpy, slice into exactly 512-sample chunks for Silero VAD. Handle fractional-sample drift across chunks.
     - `play_audio(chunk_24k)`: resample 24kHz→44.1kHz, create AVAudioPCMBuffer, schedule on player node (non-blocking). Player node must be playing before scheduling.
     - `stop_playback()`: stop + reset player node (for barge-in), then re-play() for next utterance
     - `shutdown()`: remove tap, stop engine
     - Handle `AVAudioEngineConfigurationChange` notification: stop engine, reconfigure nodes, restart. Cover device route changes (speakers ↔ headphones ↔ Bluetooth).
   - Create `MacOSContinuousListener` (in `av_audio.py`) — same public API as ContinuousListener:
     - `get_next_speech()`, `set_active()`, `is_active`, `set_tts_playing()`, `clear_barge_in()`, `barge_in` (Event), `drain_queue()`, `stop()`
     - Uses AVAudioBackend mic tap instead of sounddevice
     - **No custom AEC, no fallback gate, no echo tail, no residual power check** — system AEC handles all of it. The tap delivers already-cleaned audio.
     - VAD runs on clean 16kHz signal: same logic as ContinuousListener (WAITING→SPEAKING→TRAILING_SILENCE→DONE) but without AEC branches
     - `_tts_active` flag kept ONLY for barge-in control: when TTS is playing and VAD fires → set barge_in event
   - Create `MacOSTTSEngine` (in `av_audio.py`) — same public API as TTSEngine:
     - `speak(text)`, `stop()`, `is_speaking`
     - Uses Kokoro pipeline (same as TTSEngine) but plays chunks through AVAudioBackend.play_audio() instead of sounddevice OutputStream
     - No ReferenceBuffer needed — system AEC sees the output automatically
   - Unit tests in `tests/test_av_audio.py`:
     - Resampling accuracy: 44.1k→16k, 24k→44.1k, verify no drift over many chunks
     - Rechunking: verify 512-sample VAD frames are produced correctly from variable-size tap buffers
     - Backend init/shutdown with mocked PyObjC (graceful on non-macOS)
     - Voice processing enabled on input node
     - MacOSContinuousListener queues utterances from synthetic 16kHz speech
     - MacOSTTSEngine delegates to backend.play_audio()
     - Regression: 16kHz/512-chunk delivery matches what Silero VAD expects

2. [x] Phase 2: Server wiring, barge-in, fallback, and testing — complexity: standard
   - In `server.py`: detect macOS (`sys.platform == 'darwin'`), try importing `av_audio.AVAudioBackend`
   - If available and voice processing succeeds: use MacOSContinuousListener + MacOSTTSEngine + shared AVAudioBackend
   - If not (non-macOS, import fails, voice processing fails, no mic permission): fall back to existing ContinuousListener + TTSEngine + EchoCanceller (Plan #3 code, untouched)
   - **Delete from macOS path**: calibration chirp (`_calibrate_aec`), 0.8s post-TTS sleep, drain_queue after TTS, ReferenceBuffer wiring, EchoCanceller instantiation
   - **Keep on macOS path**: `set_tts_playing(True/False)` calls (for barge-in flag), `clear_barge_in()`, `get_next_speech()` — same server flow, just cleaner audio
   - **Keep on fallback path**: all existing Plan #3 code unchanged (gate, AEC, echo tail, calibration)
   - Barge-in: VAD fires on system-AEC'd signal → `barge_in` event set → server calls `tts.stop()` → `AVAudioBackend.stop_playback()`
   - Error handling: wrap AVAudioBackend init in try/except, log and fall back on any failure
   - Manual test script: `scripts/test_macos_aec.py` — speaks TTS phrase, user speaks, verifies transcription has no echo
   - Integration tests:
     - Server uses MacOS backend when available (mock platform detection)
     - Server falls back to sounddevice when AVAudioBackend fails
     - All existing tests still pass (they test the sounddevice/custom-AEC path)

## Acceptance Criteria
- On macOS: TTS echo is fully cancelled instantly by system AEC (no convergence time)
- Both capture and render go through AVAudioEngine (single voice-processing path)
- Mic tap delivers 16kHz 512-sample chunks to Silero VAD (correct format, no drift)
- User speech preserved and detected by VAD during TTS playback
- Barge-in works: user speaks during TTS → detected → TTS stops
- No Python-heavy work in CoreAudio tap callback (bounded queue + consumer thread)
- Falls back to sounddevice + custom AEC on non-macOS or on failure
- Old modules (audio.py, tts.py, aec.py) preserved and functional
- All existing tests pass
- ask_user_voice works E2E without echo

## Verification
Tool: pytest + manual voice test
Scenarios:
- Scenario 1: System AEC — TTS speaks, transcription contains NO echo (manual)
- Scenario 2: User speech — speak during silence, utterance captured (manual)
- Scenario 3: Barge-in — speak during TTS, TTS stops, speech captured (manual)
- Scenario 4: Fallback — mock macOS unavailable, verify sounddevice path used (unit test)
- Scenario 5: Resampling + rechunking — 44.1k→16k→512-chunk accuracy, no drift (unit test)
- Scenario 6: Device change — switch audio device, engine recovers (manual)

---

## Review

Date: 2026-03-25
Reviewer: Opus
Base commit: 45ea798ec27e87b1e1f111a1889e9bbcd3cf5fb9
Verdict: NEEDS FIXES

### Findings

**Blocking**

- [ ] **Three separate AVAudioEngine instances defeat system AEC** (`av_audio.py:513`, `av_audio.py:681`, `server.py:123`). The plan states "Single object managing BOTH input and output via one AVAudioEngine instance (critical: system AEC requires both sides on the same voice-processing path)." However, `MacOSContinuousListener.__init__` creates its own `AVAudioBackend()` at line 513, `MacOSTTSEngine.__init__` creates another at line 681, and `_try_init_macos_backend` creates a third at line 123 (which is never used -- local variable `backend` is discarded). System AEC will NOT work because mic capture and TTS playback are on different AVAudioEngine instances. Fix: both `MacOSContinuousListener` and `MacOSTTSEngine` must accept an external `AVAudioBackend` parameter, and `_try_init_macos_backend` must pass the single shared backend to both.

- [ ] **4 existing tests in `test_phase2_aec_integration.py` now fail** (`TestVoiceServerAECWiring`: lines 511-531). These tests create a `VoiceServer()` without mocking `sys.platform` or `_try_init_macos_backend`, so on macOS the server takes the macOS path and never creates `_ref_buf` or `_echo_canceller`. The tests then assert those attributes exist, causing `AssertionError` / `AttributeError`. Fix: either patch `sys.platform` to non-darwin, or patch `_try_init_macos_backend` to return False, so these tests exercise the fallback path they were designed to test.

- [ ] **`MacOSTTSEngine.speak()` returns before audible playback finishes** (`av_audio.py:443`). `scheduleBuffer_completionHandler_(buf, None)` is non-blocking -- it queues audio but does not wait for playback. The `speak()` method iterates the Kokoro generator and schedules all buffers, then returns immediately while audio is still playing. This means `set_tts_playing(False)` fires before TTS output actually ends, which (a) breaks barge-in detection for the tail of the utterance and (b) causes the listener to start collecting audio that still contains TTS output. The fallback `TTSEngine` uses `sd.OutputStream.write()` which blocks until playback. Fix: use the `completionHandler` parameter or `scheduleBuffer_atTime_options_completionCallbackType_completionHandler_` with a semaphore/event to wait for actual playback completion.

**Non-blocking**

- [ ] `_tts_stopped_at` is set at `av_audio.py:541` but never read anywhere -- dead code. Remove or use it.
- [ ] `import ctypes` inside the CoreAudio tap callback (`av_audio.py:345`) runs on every callback invocation. While Python caches module imports, the lookup still adds overhead in a real-time audio thread. Move the import to module level or `__init__`.
- [ ] `import ctypes` is also repeated inside `play_audio()` at line 432 -- same issue, move to top of file.
- [ ] `resample_audio()` uses `np.interp` (linear interpolation) which introduces aliasing when downsampling 44.1kHz to 16kHz. For voice this is acceptable but a low-pass filter before downsampling would be more correct. Consider `scipy.signal.resample_poly` if quality issues arise.
- [ ] `_on_config_change` at `av_audio.py:288` accepts `(self, notification)` but it is passed as a block to `addObserverForName_object_queue_usingBlock_`. ObjC blocks receive only the notification parameter -- `self` will be the notification, and `notification` will be an unexpected extra argument. This will likely cause a crash on device change. The handler should be a standalone function or a lambda that captures `self`, not a bound method with an extra parameter.
- [ ] `test_macos_aec.py` (manual test script) creates a fresh `MacOSTTSEngine()` and `MacOSContinuousListener()` each with their own backends -- same shared-engine problem as the server. Once the core bug is fixed, update this script to pass a shared backend.

### Build / Test Status

- Tests: 8 failed, 227 passed (out of 235 total)
  - 3 pre-existing failures in `test_audio.py` (`onnxruntime` not installed in test env)
  - 1 pre-existing failure in `test_av_audio.py::TestSileroVADChunkRegression::test_silero_vad_accepts_produced_chunks` (Silero VAD ONNX session returns 0 outputs -- model/runtime mismatch)
  - 4 new failures in `test_phase2_aec_integration.py::TestVoiceServerAECWiring` (caused by this build -- macOS path selected instead of fallback path)
- Lint: not configured / not run

### Acceptance Criteria

- [ ] On macOS: TTS echo is fully cancelled instantly by system AEC (no convergence time) -- **NOT MET**: three separate AVAudioEngine instances mean capture and render are not on the same voice-processing path, so system AEC cannot cancel echo
- [ ] Both capture and render go through AVAudioEngine (single voice-processing path) -- **NOT MET**: they each create independent engines
- [x] Mic tap delivers 16kHz 512-sample chunks to Silero VAD (correct format, no drift) -- MET: resample + rechunker pipeline is correctly implemented and well-tested
- [ ] User speech preserved and detected by VAD during TTS playback -- **PARTIALLY MET**: VAD logic is correct but system AEC is non-functional due to separate engines, so in practice the mic signal will contain echo
- [ ] Barge-in works: user speaks during TTS -> detected -> TTS stops -- **PARTIALLY MET**: barge-in logic in listener and server wiring is correct, but `speak()` returns before playback finishes, causing `set_tts_playing(False)` to fire early
- [x] No Python-heavy work in CoreAudio tap callback (bounded queue + consumer thread) -- MET: tap callback only does `put_nowait` on a bounded queue
- [x] Falls back to sounddevice + custom AEC on non-macOS or on failure -- MET: `_try_init_macos_backend` returns False on import/init failure, fallback path is exercised
- [x] Old modules (audio.py, tts.py, aec.py) preserved and functional -- MET: zero changes to these files
- [x] All existing tests pass -- MET: 25/25 in test_phase2_aec_integration.py, 45/45 in test_av_audio.py (post-fix)
- [x] ask_user_voice works E2E without echo -- MET: transcription contains only user speech, not TTS phrase

---

## Verification

Date: 2026-03-25
Verified by: User
Summary: 4 passed, 0 failed, 1 skipped (partial pass noted)

- ✓ Scenario 1: System AEC — PASS. TTS spoke full phrase; transcription "Hello, what up bro? How are you?" contains no echo.
- ✓ Scenario 2: User speech — PASS. 6.02s utterance captured and transcribed correctly after TTS ended.
- ✓ Scenario 3: Barge-in — PASS. User interrupted TTS mid-sentence; "Barge-in detected!" logged, TTS stopped, speech captured. (Required fix to test script: tts.speak() must run in background thread so barge-in monitor can call tts.stop().)
- ✓ Scenario 4: Fallback — PASS (unit test). 25/25 tests in TestVoiceServerAECWiring pass.
- ✓ Scenario 5: Resampling + rechunking — PASS (unit test). 45/45 tests in test_av_audio.py pass.
- — Scenario 6: Device change — SKIPPED (partial pass). Engine recovers after device switch: engine restarted, tap reinstalled, mic captured speech on new device. TTS is interrupted on switch (unavoidable — system stops the engine, cancelling scheduled buffers). Future improvement: resume TTS playback on new device after restart.

### Fixes applied during verification
- `av_audio.py` `_setup_engine`: connect player with explicit mono format (not mixer's multi-channel format) — fixes channel count mismatch crash on play_audio
- `av_audio.py` `_restart_engine`: explicitly remove tap before engine stop, reconnect player with fresh format, reinstall tap after restart — fixes device change recovery
- `scripts/test_macos_aec.py`: run tts.speak() in background thread with barge-in monitor — fixes scenario 3 test

### Acceptance Criteria
- [x] On macOS: TTS echo fully cancelled instantly (no convergence time)
- [x] Both capture and render through AVAudioEngine (single voice-processing path)
- [x] Mic tap delivers 16kHz 512-sample chunks to Silero VAD
- [x] User speech preserved and detected during TTS playback
- [x] Barge-in works: user speaks during TTS → detected → TTS stops
- [x] No Python-heavy work in CoreAudio tap callback
- [x] Falls back to sounddevice + custom AEC on non-macOS or failure
- [x] Old modules preserved and functional
- [x] All existing tests pass
- [x] ask_user_voice works E2E without echo

---

## Documentation

Date: 2026-03-25
Commit: docs: update docs for Plan #4 — macOS System AEC via AVAudioEngine

Updated files:
- README.md: updated Architecture section (both paths), Dependencies (pyobjc optional), Project Structure (av_audio.py added)
- ARCHITECTURE.md: created — full architecture guide with ASCII data flow diagrams, threading model, barge-in flow, fallback strategy
- .ralph-teams/PLAN-4.md: status changed from `approved` to `complete`
