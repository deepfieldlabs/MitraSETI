"""
Tests for Matched Filter Bank — MitraSETI v0.3.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_gpu.matched_filter import (
    MatchedFilterResult,
    _template_broadband_chirp,
    _template_comb,
    _template_modulated,
    _template_narrowband_drift,
    _template_pulsed,
    generate_template_bank,
    run_matched_filter_search,
)


def _make_header(n_chans: int = 1024) -> dict:
    return {
        "fch1": 1500.0,
        "foff": -0.00028,
        "tsamp": 18.253611,
        "tstart": 59000.0,
        "source_name": "TEST",
    }


# ─── Template Generation ─────────────────────────────────────────────────


class TestTemplates:
    def test_narrowband_shape(self):
        t = _template_narrowband_drift(8, 32, 2)
        assert t.shape == (8, 32)
        assert t.sum() > 0

    def test_narrowband_zero_drift(self):
        t = _template_narrowband_drift(8, 32, 0)
        nonzero = np.argwhere(t > 0)
        channels = nonzero[:, 1]
        assert len(set(channels)) == 1

    def test_pulsed_has_gaps(self):
        t = _template_pulsed(12, 32, period=4, duty_cycle=0.25)
        power_per_row = t.sum(axis=1)
        assert np.any(power_per_row == 0)

    def test_broadband_chirp_spans_channels(self):
        t = _template_broadband_chirp(8, 64, 16)
        nonzero = np.argwhere(t > 0)
        channel_range = nonzero[:, 1].max() - nonzero[:, 1].min()
        assert channel_range > 10

    def test_comb_has_multiple_tones(self):
        t = _template_comb(8, 64, n_tones=5, spacing=10)
        mean_per_channel = t.mean(axis=0)
        active_channels = np.sum(mean_per_channel > 0)
        assert active_channels == 5

    def test_modulated_amplitude_varies(self):
        t = _template_modulated(8, 32, mod_period=4)
        center = 16
        amplitudes = t[:, center]
        assert amplitudes.max() > amplitudes.min()

    def test_template_bank_size(self):
        bank = generate_template_bank(8, 32)
        assert len(bank) > 15
        for name, template, params in bank:
            assert isinstance(name, str)
            assert template.ndim == 2
            assert isinstance(params, dict)


# ─── Matched Filter Search ───────────────────────────────────────────────


class TestMatchedFilterSearch:
    def test_returns_result(self):
        np.random.seed(42)
        data = np.random.randn(8, 256).astype(np.float32)
        header = _make_header(256)
        result = run_matched_filter_search(data, header, min_snr=15.0)
        assert isinstance(result, MatchedFilterResult)
        assert result.processing_time_ms > 0
        assert result.templates_tested > 0

    def test_detects_narrowband(self):
        """A strong narrowband drifting signal should be detected."""
        np.random.seed(42)
        n_times, n_chans = 8, 512
        data = np.random.randn(n_times, n_chans).astype(np.float32)
        for t in range(n_times):
            data[t, 200 + t] += 30
        header = _make_header(n_chans)
        result = run_matched_filter_search(data, header, min_snr=5.0)
        assert len(result.candidates) > 0

    def test_detects_comb(self):
        """A comb signal (equally spaced tones) should be detected."""
        np.random.seed(42)
        n_times, n_chans = 8, 512
        data = np.random.randn(n_times, n_chans).astype(np.float32)
        spacing = 20
        for i in range(5):
            ch = 200 + i * spacing
            if ch < n_chans:
                data[:, ch] += 25
        header = _make_header(n_chans)
        result = run_matched_filter_search(data, header, min_snr=5.0)
        assert len(result.candidates) > 0

    def test_noise_only_high_threshold(self):
        """Pure noise should have no detections at high SNR threshold."""
        np.random.seed(42)
        data = np.random.randn(8, 256).astype(np.float32)
        header = _make_header(256)
        result = run_matched_filter_search(data, header, min_snr=50.0)
        assert len(result.candidates) == 0

    def test_per_template_timing(self):
        np.random.seed(42)
        data = np.random.randn(8, 128).astype(np.float32)
        header = _make_header(128)
        result = run_matched_filter_search(data, header, min_snr=20.0)
        assert len(result.per_template_times_ms) > 0
        assert all(v >= 0 for v in result.per_template_times_ms.values())
