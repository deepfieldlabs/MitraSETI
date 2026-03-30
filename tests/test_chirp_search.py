"""
Tests for Chirp Rate (Doppler Acceleration) Search — MitraSETI v0.3.0

Validates that quadratic frequency drift is correctly removed and that
chirp-only signals are properly identified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_gpu.chirp_search import (
    ChirpSearchResult,
    dechirp_spectrogram,
    run_chirp_search,
)


def _make_header(n_chans: int = 1024) -> dict:
    return {
        "fch1": 1500.0,
        "foff": -0.00028,
        "tsamp": 18.253611,
        "tstart": 59000.0,
        "source_name": "TEST",
    }


# ─── De-chirp Function ───────────────────────────────────────────────────


class TestDechirp:
    def test_zero_chirp_is_identity(self):
        data = np.random.randn(8, 64).astype(np.float32)
        result = dechirp_spectrogram(data, 0.0, 18.0, 280.0)
        np.testing.assert_array_equal(result, data)

    def test_output_shape_preserved(self):
        data = np.random.randn(16, 128).astype(np.float32)
        result = dechirp_spectrogram(data, 0.05, 18.0, 280.0)
        assert result.shape == data.shape

    def test_first_row_unchanged(self):
        """t=0 means elapsed=0, so no shift is applied to row 0."""
        data = np.random.randn(8, 32).astype(np.float32)
        result = dechirp_spectrogram(data, 0.1, 18.0, 280.0)
        np.testing.assert_array_equal(result[0], data[0])

    def test_positive_chirp_shifts_later_rows(self):
        """With positive chirp, later time steps should be shifted."""
        data = np.zeros((8, 32), dtype=np.float32)
        data[4, 16] = 100.0  # place signal at channel 16, time 4
        result = dechirp_spectrogram(data, 0.5, 18.0, 280.0)
        # After de-chirp, the signal at t=4 should have moved
        # shift = 0.5 * 0.5 * (4*18)^2 / 280 ≈ 4.6 channels
        assert result[4, 16] == 0.0  # signal moved away from ch 16


# ─── Chirp Search ─────────────────────────────────────────────────────────


class TestChirpSearch:
    def test_returns_result_structure(self):
        np.random.seed(42)
        data = np.random.randn(8, 256).astype(np.float32)
        header = _make_header(256)
        result = run_chirp_search(
            data, header,
            chirp_max=0.05, chirp_steps=3,
            min_snr=20.0,
        )
        assert isinstance(result, ChirpSearchResult)
        assert result.processing_time_ms > 0
        assert len(result.chirp_rates_tested) == 3

    def test_detects_linear_signal_at_baseline(self):
        """A strong linear signal should appear in the baseline."""
        np.random.seed(42)
        data = np.random.randn(8, 512).astype(np.float32)
        for t in range(8):
            ch = 200 + t
            if 0 <= ch < 512:
                data[t, ch] += 50
        header = _make_header(512)
        result = run_chirp_search(
            data, header,
            chirp_max=0.05, chirp_steps=3,
            min_snr=8.0,
        )
        assert result.baseline_hits > 0

    def test_chirp_signal_detected(self):
        """Inject a quadratic-drift signal and verify chirp search finds it."""
        np.random.seed(42)
        n_times, n_chans = 16, 1024
        data = np.random.randn(n_times, n_chans).astype(np.float32)
        tsamp = 18.253611
        foff_hz = 280.0
        accel = 0.05  # Hz/s^2

        for t in range(n_times):
            elapsed = t * tsamp
            # Quadratic drift: 0.5 * a * t^2
            shift_ch = int(round(0.5 * accel * elapsed ** 2 / foff_hz))
            linear_ch = t  # +1 channel per time step linear drift
            ch = 500 + linear_ch + shift_ch
            if 0 <= ch < n_chans:
                data[t, ch] += 40

        header = _make_header(n_chans)
        result = run_chirp_search(
            data, header,
            chirp_max=0.1, chirp_steps=5,
            min_snr=5.0,
        )
        assert len(result.candidates) > 0

    def test_noise_only_no_chirp_candidates(self):
        """Pure noise should produce no chirp-only candidates at high SNR."""
        np.random.seed(123)
        data = np.random.randn(8, 256).astype(np.float32)
        header = _make_header(256)
        result = run_chirp_search(
            data, header,
            chirp_max=0.05, chirp_steps=3,
            min_snr=25.0,
        )
        assert result.chirp_only_count == 0
