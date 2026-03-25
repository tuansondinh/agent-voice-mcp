#!/usr/bin/env python3
"""test_macos_aec.py — Manual test for macOS AVAudioEngine AEC backend.

Usage::

    python scripts/test_macos_aec.py

What it does:
1. Creates AVAudioBackend (starts AVAudioEngine with voice processing / system AEC).
2. Creates MacOSTTSEngine (shares the same backend).
3. Creates MacOSContinuousListener (listens on mic tap).
4. Speaks a test phrase via TTS.
5. Waits for user speech (up to 30 seconds).
6. Transcribes and prints the result.

How to verify no echo:
- The transcription should contain ONLY what you said, not the TTS phrase.
- If you hear the TTS phrase echoed back in the transcription, AEC is not working.

Requirements:
- macOS only (uses AVAudioEngine).
- pyobjc-framework-AVFoundation must be installed.
- Microphone permission must be granted to the terminal / Python process.
"""

from __future__ import annotations

import sys
import time

if sys.platform != 'darwin':
    print("ERROR: This script is macOS-only.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    print("[test_macos_aec] Starting macOS AEC backend test…", flush=True)

    # Step 1: Import macOS backend components
    try:
        from lazy_claude.av_audio import AVAudioBackend, MacOSTTSEngine, MacOSContinuousListener
    except ImportError as exc:
        print(f"ERROR: Could not import av_audio: {exc}", file=sys.stderr)
        print("Install pyobjc: pip install 'lazy-claude[macos]'", file=sys.stderr)
        sys.exit(1)

    # Step 2: Create AVAudioBackend
    print("[test_macos_aec] Initialising AVAudioBackend…", flush=True)
    try:
        backend = AVAudioBackend()
    except RuntimeError as exc:
        print(f"ERROR: AVAudioBackend init failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Step 3: Load VAD model for the listener
    print("[test_macos_aec] Loading VAD model…", flush=True)
    from lazy_claude.audio import load_vad_model
    vad_model = load_vad_model()

    # Step 4: Create TTS engine and listener
    print("[test_macos_aec] Creating TTS engine…", flush=True)
    tts = MacOSTTSEngine()

    print("[test_macos_aec] Creating listener…", flush=True)
    listener = MacOSContinuousListener(vad_model)
    listener.set_active(True)

    # Step 5: Speak a test phrase
    test_phrase = (
        "Hello, I am testing echo cancellation. "
        "Please say something after this message ends."
    )
    print(f"[test_macos_aec] Speaking: {test_phrase!r}", flush=True)

    # Signal TTS playing so barge-in detection works
    listener.set_tts_playing(True)
    tts.speak(test_phrase)
    listener.set_tts_playing(False)

    print("[test_macos_aec] TTS finished. Waiting for your speech…", flush=True)
    print("[test_macos_aec] (Speak now — you have 30 seconds)", flush=True)

    # Step 6: Wait for user speech
    audio = listener.get_next_speech(timeout=30.0)

    if audio is None:
        print("[test_macos_aec] No speech detected within 30 seconds.", flush=True)
    else:
        # Transcribe
        print(f"[test_macos_aec] Captured {len(audio) / 16_000:.2f}s of audio.", flush=True)
        print("[test_macos_aec] Transcribing…", flush=True)
        from lazy_claude.stt import load_model, transcribe
        whisper_model = load_model()
        result = transcribe(audio, model=whisper_model)
        print(f"\n[TRANSCRIPTION] {result!r}\n", flush=True)
        print("[test_macos_aec] Verify: the transcription should NOT contain the TTS phrase.", flush=True)

    # Cleanup
    listener.stop()
    print("[test_macos_aec] Done.", flush=True)


if __name__ == '__main__':
    main()
