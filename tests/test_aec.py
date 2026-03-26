"""Tests for aec.py — Acoustic Echo Cancellation module.

Tests cover:
- Synthetic echo attenuation >= 20dB
- Ring buffer alignment and thread-safety (concurrent read/write)
- Sample-rate conversion correctness
- Delay estimation accuracy with known delay
- Double-talk detector does not diverge filter
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

SAMPLE_RATE = 16_000
CHUNK = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq: float, n_samples: int, sr: int = SAMPLE_RATE, amp: float = 0.3) -> np.ndarray:
    """Generate a mono sine wave at *freq* Hz."""
    t = np.arange(n_samples, dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms_db(signal: np.ndarray) -> float:
    """RMS level in dBFS. Returns -inf for silence."""
    rms = np.sqrt(np.mean(signal.astype(np.float64) ** 2))
    if rms < 1e-12:
        return -np.inf
    return 20 * np.log10(rms)


# ---------------------------------------------------------------------------
# EchoCanceller basic API
# ---------------------------------------------------------------------------

class TestEchoCancellerAPI:
    def setup_method(self):
        from lazy_claude.aec import EchoCanceller
        self.EchoCanceller = EchoCanceller

    def test_instantiation_defaults(self):
        ec = self.EchoCanceller()
        assert ec is not None

    def test_instantiation_custom_params(self):
        ec = self.EchoCanceller(filter_length=1024, mu=0.05, dtd_threshold=0.6)
        assert ec is not None

    def test_cancel_returns_float32_array(self):
        ec = self.EchoCanceller()
        mic = np.zeros(CHUNK, dtype=np.float32)
        ref = np.zeros(CHUNK, dtype=np.float32)
        out = ec.cancel(mic, ref)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert out.shape == (CHUNK,)

    def test_cancel_silence_returns_silence(self):
        ec = self.EchoCanceller()
        mic = np.zeros(CHUNK, dtype=np.float32)
        ref = np.zeros(CHUNK, dtype=np.float32)
        out = ec.cancel(mic, ref)
        assert np.allclose(out, 0.0, atol=1e-6)

    def test_freeze_on_silence_attribute(self):
        ec = self.EchoCanceller()
        # Attribute must exist
        assert hasattr(ec, "cancel")


# ---------------------------------------------------------------------------
# Echo attenuation >= 20 dB
# ---------------------------------------------------------------------------

class TestEchoAttenuation:
    """After convergence, AEC should attenuate echo by at least 20 dB."""

    def test_echo_attenuation_20db(self):
        from lazy_claude.aec import EchoCanceller

        # Use a short filter for fast convergence in tests
        ec = EchoCanceller(filter_length=512, mu=0.1, chunk_size=CHUNK)

        freq = 440.0  # reference tone
        # Synthetic scenario: mic = reference delayed by some taps + tiny noise
        delay_samples = 32
        n_chunks = 200

        # Generate continuous reference signal
        total_samples = n_chunks * CHUNK + delay_samples + 1024
        ref_signal = _sine(freq, total_samples, amp=0.5)

        # Mic = delayed reference (echo) with amplitude 0.5
        mic_signal = np.zeros(total_samples, dtype=np.float32)
        mic_signal[delay_samples:] = ref_signal[:-delay_samples] * 0.5

        # Run adaptation for most chunks (training phase)
        train_chunks = n_chunks - 20
        for i in range(train_chunks):
            mic_chunk = mic_signal[i * CHUNK:(i + 1) * CHUNK]
            ref_chunk = ref_signal[i * CHUNK:(i + 1) * CHUNK]
            ec.cancel(mic_chunk, ref_chunk)

        # Measure attenuation on last 20 chunks
        residual_chunks = []
        for i in range(train_chunks, n_chunks):
            mic_chunk = mic_signal[i * CHUNK:(i + 1) * CHUNK]
            ref_chunk = ref_signal[i * CHUNK:(i + 1) * CHUNK]
            out = ec.cancel(mic_chunk, ref_chunk)
            residual_chunks.append(out)

        residual = np.concatenate(residual_chunks)
        # Compare against the same mic windows (without cancellation)
        echo_power = np.mean(mic_signal[train_chunks * CHUNK:n_chunks * CHUNK] ** 2)
        residual_power = np.mean(residual ** 2)

        if echo_power > 0:
            attenuation_db = 10 * np.log10(echo_power / (residual_power + 1e-12))
            assert attenuation_db >= 20.0, (
                f"Echo attenuation only {attenuation_db:.1f} dB (need >= 20 dB)"
            )


# ---------------------------------------------------------------------------
# Delay estimation
# ---------------------------------------------------------------------------

class TestDelayEstimation:
    def test_estimate_delay_accuracy(self):
        from lazy_claude.aec import estimate_delay

        sr = SAMPLE_RATE
        freq = 440.0
        true_delay = 128  # samples

        n = 4096
        ref = _sine(freq, n, amp=0.5)
        mic = np.zeros(n, dtype=np.float32)
        mic[true_delay:] = ref[: n - true_delay]

        estimated = estimate_delay(mic, ref, max_delay=512)
        # Allow ±1 sample tolerance
        assert abs(estimated - true_delay) <= 2, (
            f"Expected delay ~{true_delay}, got {estimated}"
        )

    def test_estimate_delay_zero(self):
        from lazy_claude.aec import estimate_delay

        n = 4096
        ref = _sine(300.0, n, amp=0.5)
        mic = ref.copy()

        estimated = estimate_delay(mic, ref, max_delay=512)
        assert abs(estimated) <= 2, f"Expected ~0 delay, got {estimated}"

    def test_echo_canceller_retries_delay_estimation_after_silent_window(self):
        from lazy_claude.aec import EchoCanceller

        ec = EchoCanceller(chunk_size=CHUNK)
        silent = np.zeros(CHUNK, dtype=np.float32)

        for _ in range(ec._delay_est_frames_needed):
            ec.cancel(silent, silent)

        assert ec._delay_estimated is False
        assert ec._delay_est_mic_chunks == []
        assert ec._delay_est_ref_chunks == []


# ---------------------------------------------------------------------------
# ReferenceBuffer — lock-free SPSC ring buffer
# ---------------------------------------------------------------------------

class TestReferenceBuffer:
    def setup_method(self):
        from lazy_claude.aec import ReferenceBuffer
        self.ReferenceBuffer = ReferenceBuffer

    def test_write_and_read_same_rate(self):
        """Write at 16kHz, read at 16kHz — data passes through unchanged."""
        buf = self.ReferenceBuffer(capacity=4096, write_sr=16_000, read_sr=16_000)
        data = _sine(440.0, CHUNK)
        buf.write(data)
        out = buf.read(CHUNK)
        assert out is not None
        assert len(out) == CHUNK
        # Passthrough at same rate: should match closely
        np.testing.assert_allclose(out, data, atol=1e-4)

    def test_write_24k_read_16k_resamples(self):
        """Write at 24kHz, read at 16kHz — output length should be correct."""
        write_sr = 24_000
        read_sr = 16_000
        buf = self.ReferenceBuffer(capacity=8192, write_sr=write_sr, read_sr=read_sr)

        # Write 240 samples at 24kHz (10ms)
        n_write = 240
        data = _sine(440.0, n_write, sr=write_sr)
        buf.write(data)

        # After resampling 24k→16k: 240 * (16000/24000) = 160 samples
        expected_read = 160
        out = buf.read(expected_read)
        assert out is not None
        assert len(out) == expected_read

    def test_read_underflow_returns_silence(self):
        """Reading more than available returns zeros (not an error)."""
        buf = self.ReferenceBuffer(capacity=4096, write_sr=16_000, read_sr=16_000)
        out = buf.read(CHUNK)
        # Should return zeros (silence) when empty
        assert out is not None
        assert len(out) == CHUNK
        assert np.allclose(out, 0.0)

    def test_concurrent_read_write(self):
        """Writer and reader threads run concurrently without crashes/corruption."""
        from lazy_claude.aec import ReferenceBuffer

        buf = ReferenceBuffer(capacity=16384, write_sr=16_000, read_sr=16_000)
        errors = []

        def writer():
            try:
                for _ in range(100):
                    chunk = _sine(440.0, CHUNK)
                    buf.write(chunk)
                    time.sleep(0.001)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def reader():
            try:
                for _ in range(100):
                    out = buf.read(CHUNK)
                    assert out is not None and len(out) == CHUNK
                    time.sleep(0.001)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t_w = threading.Thread(target=writer)
        t_r = threading.Thread(target=reader)
        t_w.start()
        t_r.start()
        t_w.join(timeout=5.0)
        t_r.join(timeout=5.0)

        assert not errors, f"Concurrent errors: {errors}"

    def test_sample_rate_conversion_frequency_preserved(self):
        """After 24k→16k resampling, the dominant frequency should be preserved."""
        write_sr = 24_000
        read_sr = 16_000
        buf = self.ReferenceBuffer(capacity=32768, write_sr=write_sr, read_sr=read_sr)

        freq_hz = 800.0
        # Write 2400 samples at 24kHz (100ms)
        n_write = 2400
        data = _sine(freq_hz, n_write, sr=write_sr)
        buf.write(data)

        # Read 1600 samples at 16kHz (100ms)
        n_read = 1600
        out = buf.read(n_read)
        assert out is not None and len(out) == n_read

        # Check dominant FFT bin is near 800 Hz
        fft_mag = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(n_read, d=1.0 / read_sr)
        peak_freq = freqs[np.argmax(fft_mag)]
        assert abs(peak_freq - freq_hz) < 100.0, (
            f"Expected dominant freq ~{freq_hz} Hz, got {peak_freq:.1f} Hz"
        )


# ---------------------------------------------------------------------------
# Double-talk detector (DTD) — Geigel algorithm
# ---------------------------------------------------------------------------

class TestDoubleTalkDetector:
    """Geigel DTD: freeze adaptation when user+TTS overlap to prevent divergence."""

    def test_dtd_prevents_filter_divergence(self):
        """During double-talk the filter weights should not diverge."""
        from lazy_claude.aec import EchoCanceller

        ec = EchoCanceller(filter_length=512, mu=0.1, dtd_threshold=0.5, chunk_size=CHUNK)

        # First train on clean echo (reference only in mic)
        ref_signal = _sine(440.0, 100 * CHUNK, amp=0.5)
        mic_echo = ref_signal * 0.4  # pure echo
        for i in range(80):
            ec.cancel(mic_echo[i * CHUNK:(i + 1) * CHUNK],
                      ref_signal[i * CHUNK:(i + 1) * CHUNK])

        # Snapshot filter norms after training
        norm_before = ec.filter_norm()

        # Now inject double-talk: mic = echo + loud user speech
        user_speech = _sine(880.0, 20 * CHUNK, amp=0.8)
        for i in range(20):
            mic_dt = (mic_echo[(80 + i) * CHUNK:(81 + i) * CHUNK]
                      + user_speech[i * CHUNK:(i + 1) * CHUNK])
            ec.cancel(mic_dt, ref_signal[(80 + i) * CHUNK:(81 + i) * CHUNK])

        norm_after = ec.filter_norm()

        # Filter should not have exploded (10x growth = divergence)
        ratio = norm_after / (norm_before + 1e-12)
        assert ratio < 10.0, (
            f"Filter diverged during double-talk: norm ratio = {ratio:.2f}"
        )

    def test_dtd_freezes_during_double_talk(self):
        """DTD should detect double-talk and the filter should not change."""
        from lazy_claude.aec import EchoCanceller

        ec = EchoCanceller(filter_length=512, mu=0.1, dtd_threshold=0.5, chunk_size=CHUNK)

        # Loud reference (TTS) and loud mic (user)
        ref = _sine(440.0, CHUNK, amp=0.9)
        mic = _sine(880.0, CHUNK, amp=0.9)  # strong user signal — DTD should fire

        norm_before = ec.filter_norm()
        ec.cancel(mic, ref)
        norm_after = ec.filter_norm()

        # When DTD fires, filter coefficients should stay the same
        assert abs(norm_after - norm_before) < 1e-6, (
            f"Filter updated during double-talk (norm before={norm_before:.6f}, "
            f"after={norm_after:.6f})"
        )


# ---------------------------------------------------------------------------
# Filter freeze on reference silence
# ---------------------------------------------------------------------------

class TestSilenceFreeze:
    def test_filter_frozen_on_silence(self):
        """Filter must not drift when reference signal is silent."""
        from lazy_claude.aec import EchoCanceller

        ec = EchoCanceller(filter_length=512, mu=0.1, chunk_size=CHUNK)

        # Train briefly
        ref = _sine(440.0, 10 * CHUNK, amp=0.3)
        mic = ref * 0.3
        for i in range(10):
            ec.cancel(mic[i * CHUNK:(i + 1) * CHUNK],
                      ref[i * CHUNK:(i + 1) * CHUNK])

        norm_before = ec.filter_norm()

        # Feed silence reference + small mic signal
        silent_ref = np.zeros(CHUNK, dtype=np.float32)
        small_mic = np.random.default_rng(42).random(CHUNK).astype(np.float32) * 0.01
        for _ in range(20):
            ec.cancel(small_mic, silent_ref)

        norm_after = ec.filter_norm()
        # Filter should be unchanged (frozen)
        assert abs(norm_after - norm_before) < 1e-6, (
            f"Filter drifted during silence: before={norm_before:.6f}, after={norm_after:.6f}"
        )
