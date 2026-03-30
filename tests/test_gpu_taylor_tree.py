"""
Tests for GPU-accelerated Taylor Tree De-Doppler Search — MitraSETI v0.3.0

Validates the GPU (CuPy) and NumPy fallback implementations against known
signal injections. Tests correctness, not speed — benchmarks are separate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_gpu.taylor_tree_gpu import (
    GPUSearchResult,
    GPUSignalCandidate,
    _build_taylor_tree,
    _build_taylor_tree_vectorised,
    _cluster_candidates,
    _normalise,
    gpu_taylor_tree_search,
    is_gpu_available,
)


def _make_header(n_chans: int = 1024, foff: float = -0.00028) -> dict:
    return {
        "fch1": 1500.0,
        "foff": foff,
        "tsamp": 18.253611,
        "tstart": 59000.0,
        "source_name": "TEST",
    }


def _inject_drifting_signal(
    n_times: int,
    n_chans: int,
    start_ch: int,
    drift_chans: int,
    snr: float = 30.0,
) -> np.ndarray:
    """Create noise + a drifting narrowband signal."""
    np.random.seed(42)
    data = np.random.randn(n_times, n_chans).astype(np.float32)
    for t in range(n_times):
        ch = start_ch + int(drift_chans * t / max(n_times - 1, 1))
        if 0 <= ch < n_chans:
            data[t, ch] += snr
    return data


# ─── Normalisation ────────────────────────────────────────────────────────


class TestNormalisation:
    def test_output_shape(self):
        data = np.random.randn(8, 64).astype(np.float32)
        result = _normalise(data)
        assert result.shape == data.shape

    def test_zero_mean_channels(self):
        data = np.random.randn(100, 32).astype(np.float32)
        result = _normalise(data)
        medians = np.median(result, axis=0)
        assert np.all(np.abs(medians) < 1.0)

    def test_handles_constant_channel(self):
        data = np.ones((8, 4), dtype=np.float32) * 5.0
        result = _normalise(data)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# ─── Tree Construction ────────────────────────────────────────────────────


class TestTreeConstruction:
    def test_single_timestep(self):
        data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        tree = _build_taylor_tree(data, 1, 4, 1, 0, 1)
        assert tree.shape == (1, 4)
        np.testing.assert_array_equal(tree[0], data[0])

    def test_two_timesteps_shape(self):
        data = np.random.randn(2, 8).astype(np.float32)
        tree = _build_taylor_tree(data, 2, 8, 2, 1, 1)
        assert tree.shape == (2, 8)

    def test_four_timesteps_shape(self):
        data = np.random.randn(4, 16).astype(np.float32)
        tree = _build_taylor_tree(data, 4, 16, 4, 2, 1)
        assert tree.shape == (4, 16)

    def test_positive_negative_symmetry(self):
        """Drift=0 should give the same result for both directions."""
        data = np.random.randn(4, 32).astype(np.float32)
        tree_pos = _build_taylor_tree(data, 4, 32, 4, 2, 1)
        tree_neg = _build_taylor_tree(data, 4, 32, 4, 2, -1)
        np.testing.assert_allclose(tree_pos[0], tree_neg[0], atol=1e-5)

    def test_vectorised_matches_basic(self):
        """Vectorised builder should produce identical results to basic."""
        data = np.random.randn(8, 64).astype(np.float32)
        tree_basic = _build_taylor_tree(data, 8, 64, 8, 3, 1)
        tree_vec = _build_taylor_tree_vectorised(data, 8, 64, 8, 3, 1)
        np.testing.assert_allclose(tree_basic, tree_vec, atol=1e-4)

    def test_vectorised_matches_negative(self):
        data = np.random.randn(4, 32).astype(np.float32)
        tree_basic = _build_taylor_tree(data, 4, 32, 4, 2, -1)
        tree_vec = _build_taylor_tree_vectorised(data, 4, 32, 4, 2, -1)
        np.testing.assert_allclose(tree_basic, tree_vec, atol=1e-4)


# ─── End-to-End Search ────────────────────────────────────────────────────


class TestGPUSearch:
    def test_detects_strong_signal(self):
        """A strong (SNR=50) drifting signal should be detected."""
        data = _inject_drifting_signal(16, 1024, 500, 3, snr=50)
        header = _make_header(1024)
        result = gpu_taylor_tree_search(data, header, min_snr=8.0)

        assert isinstance(result, GPUSearchResult)
        assert len(result.candidates) > 0
        assert result.backend in ("cuda", "numpy_fallback", "numba_jit")

        best = result.candidates[0]
        freq_mhz = best.frequency_hz / 1e6
        assert abs(freq_mhz - (1500.0 + 500 * (-0.00028))) < 0.5

    def test_no_signal_no_detection(self):
        """Pure noise should produce no candidates above SNR 25."""
        np.random.seed(123)
        data = np.random.randn(16, 512).astype(np.float32)
        header = _make_header(512)
        result = gpu_taylor_tree_search(data, header, min_snr=25.0)
        assert len(result.candidates) == 0

    def test_negative_drift(self):
        """Signal with negative drift should be detected."""
        data = _inject_drifting_signal(16, 1024, 700, -4, snr=40)
        header = _make_header(1024)
        result = gpu_taylor_tree_search(data, header, min_snr=8.0)
        assert len(result.candidates) > 0
        found_negative = any(c.drift_rate < 0 for c in result.candidates)
        assert found_negative

    def test_zero_drift(self):
        """Signal with zero drift should be detected.

        Zero-drift signals are tricky because the per-channel median
        normalisation subtracts a constant signal. We use a large SNR
        and few time steps so that the median doesn't fully wash it out.
        """
        np.random.seed(99)
        n_times, n_chans = 4, 512
        data = np.random.randn(n_times, n_chans).astype(np.float32)
        # Only some time steps have the signal — avoids median suppression
        data[0, 250] += 80
        data[1, 250] += 80
        data[3, 250] += 80
        header = _make_header(n_chans)
        result = gpu_taylor_tree_search(data, header, min_snr=3.0)
        assert len(result.candidates) > 0

    def test_result_has_timing(self):
        data = _inject_drifting_signal(8, 256, 100, 1, snr=30)
        header = _make_header(256)
        result = gpu_taylor_tree_search(data, header, min_snr=8.0)
        assert result.processing_time_ms > 0

    def test_multiple_signals(self):
        """Two signals at different frequencies should both be detected."""
        np.random.seed(42)
        data = np.random.randn(8, 512).astype(np.float32)
        for t in range(8):
            data[t, 100 + t] += 40
            data[t, 400 - t] += 40
        header = _make_header(512)
        result = gpu_taylor_tree_search(data, header, min_snr=8.0)
        assert len(result.candidates) >= 2


# ─── Clustering ───────────────────────────────────────────────────────────


class TestClustering:
    def test_merges_nearby(self):
        candidates = [
            GPUSignalCandidate(1e9, 0.5, 20.0, 0, 1, 100),
            GPUSignalCandidate(1e9 + 1000, 0.5, 18.0, 0, 1, 100),
            GPUSignalCandidate(1e9 + 2000, 0.5, 15.0, 0, 1, 100),
        ]
        result = _cluster_candidates(candidates)
        assert len(result) == 1
        assert result[0].snr == 20.0

    def test_keeps_distant(self):
        candidates = [
            GPUSignalCandidate(1e9, 0.5, 20.0, 0, 1, 100),
            GPUSignalCandidate(2e9, 0.5, 18.0, 0, 1, 100),
        ]
        result = _cluster_candidates(candidates)
        assert len(result) == 2


# ─── GPU-specific (skipped if no GPU) ─────────────────────────────────────


@pytest.mark.skipif(not is_gpu_available(), reason="No CUDA GPU available")
class TestGPUSpecific:
    def test_gpu_backend_label(self):
        data = _inject_drifting_signal(8, 256, 100, 1, snr=30)
        header = _make_header(256)
        result = gpu_taylor_tree_search(data, header, min_snr=8.0)
        assert result.backend == "cuda"

    def test_gpu_info_populated(self):
        data = _inject_drifting_signal(8, 256, 100, 1, snr=30)
        header = _make_header(256)
        result = gpu_taylor_tree_search(data, header, min_snr=8.0)
        assert result.gpu_info.get("available") is True
        assert "name" in result.gpu_info
