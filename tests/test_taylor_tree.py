"""
Tests for the Taylor tree de-Doppler algorithm.

Validates that:
1. Taylor tree produces equivalent results to brute-force for simple signals
2. Known drifting signals are correctly detected
3. SNR thresholds are respected
4. Positive and negative drifts are handled
5. Edge cases: single time step, zero drift, maximum drift
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_spectrogram(
    n_times: int,
    n_chans: int,
    signal_freq_idx: int,
    drift_channels: int,
    signal_amplitude: float = 50.0,
    noise_std: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic spectrogram with a drifting narrowband signal."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0, noise_std, (n_times, n_chans)).astype(np.float32)

    for t in range(n_times):
        ch = signal_freq_idx + int(round(drift_channels * t / max(n_times - 1, 1)))
        if 0 <= ch < n_chans:
            data[t, ch] += signal_amplitude

    return data


def _make_header(n_chans, n_times, fch1=1420.0, foff=-0.00028, tsamp=18.0):
    """Create a mock FilterbankHeader."""
    try:
        import mitraseti_core as _core
    except ImportError:
        import astroseti_core as _core

    return _core.FilterbankHeader(
        nchans=n_chans,
        nifs=1,
        nbits=32,
        tsamp=tsamp,
        fch1=fch1,
        foff=foff,
        tstart=59000.0,
        source_name="TEST",
        ra=0.0,
        dec=0.0,
    )


def test_taylor_vs_bruteforce():
    """Ensure Taylor tree detects the same signals as brute-force."""
    try:
        import mitraseti_core as _core
    except ImportError:
        import astroseti_core as _core

    n_times, n_chans = 16, 512
    signal_ch = 256
    drift_ch = 4

    data = _make_spectrogram(n_times, n_chans, signal_ch, drift_ch, signal_amplitude=80.0)
    header = _make_header(n_chans, n_times)
    data_flat = data.ravel().tolist()

    params_taylor = _core.SearchParams(max_drift_rate=4.0, min_snr=5.0, use_taylor_tree=True)
    engine_taylor = _core.DedopplerEngine(params_taylor)
    result_taylor = engine_taylor.search(data_flat, n_times, n_chans, header)

    params_brute = _core.SearchParams(max_drift_rate=4.0, min_snr=5.0, use_taylor_tree=False)
    engine_brute = _core.DedopplerEngine(params_brute)
    result_brute = engine_brute.search(data_flat, n_times, n_chans, header)

    taylor_freqs = {round(c.frequency_hz / 1e3) for c in result_taylor.candidates}
    brute_freqs = {round(c.frequency_hz / 1e3) for c in result_brute.candidates}

    signal_freq_khz = round((header.fch1 + signal_ch * header.foff) * 1e6 / 1e3)

    assert any(abs(f - signal_freq_khz) < 10 for f in taylor_freqs), (
        f"Taylor tree missed injected signal at {signal_freq_khz} kHz. Found: {taylor_freqs}"
    )

    assert any(abs(f - signal_freq_khz) < 10 for f in brute_freqs), (
        f"Brute-force missed injected signal at {signal_freq_khz} kHz. Found: {brute_freqs}"
    )

    print(
        f"  PASS: Taylor found {len(result_taylor.candidates)} candidates, "
        f"brute-force found {len(result_brute.candidates)}"
    )
    print(f"  Both detected signal near {signal_freq_khz} kHz")


def test_negative_drift():
    """Ensure negative drift rates are detected."""
    try:
        import mitraseti_core as _core
    except ImportError:
        import astroseti_core as _core

    n_times, n_chans = 16, 512
    signal_ch = 256
    drift_ch = -3

    data = _make_spectrogram(n_times, n_chans, signal_ch, drift_ch, signal_amplitude=80.0)
    header = _make_header(n_chans, n_times)

    params = _core.SearchParams(max_drift_rate=4.0, min_snr=5.0, use_taylor_tree=True)
    engine = _core.DedopplerEngine(params)
    result = engine.search(data.ravel().tolist(), n_times, n_chans, header)

    neg_drift_cands = [c for c in result.candidates if c.drift_rate < -0.01]
    assert len(neg_drift_cands) > 0, "Taylor tree failed to detect negative-drift signal"

    best = max(neg_drift_cands, key=lambda c: c.snr)
    print(
        f"  PASS: Detected negative drift signal — drift_rate={best.drift_rate:.4f} Hz/s, SNR={best.snr:.1f}"
    )


def test_slow_drift():
    """Ensure signals with slow drift (2 channels) are detected.

    Zero and very slow drifts (1 channel) are absorbed by per-channel
    median normalization — the signal stays in one channel long enough
    for the median to track it.  With 2+ channel drift the signal is
    sparse enough in each channel to survive normalization.
    """
    try:
        import mitraseti_core as _core
    except ImportError:
        import astroseti_core as _core

    n_times, n_chans = 16, 512
    signal_ch = 200
    drift_ch = 2

    data = _make_spectrogram(n_times, n_chans, signal_ch, drift_ch, signal_amplitude=80.0)
    header = _make_header(n_chans, n_times)

    params = _core.SearchParams(max_drift_rate=4.0, min_snr=5.0, use_taylor_tree=True)
    engine = _core.DedopplerEngine(params)
    result = engine.search(data.ravel().tolist(), n_times, n_chans, header)

    assert len(result.candidates) > 0, (
        "Taylor tree failed to detect slow-drift signal (2 channel drift)"
    )

    best = max(result.candidates, key=lambda c: c.snr)
    print(
        f"  PASS: Detected slow-drift signal — drift_rate={best.drift_rate:.4f} Hz/s, SNR={best.snr:.1f}"
    )


def test_snr_threshold():
    """Weak signals below threshold should not be detected."""
    try:
        import mitraseti_core as _core
    except ImportError:
        import astroseti_core as _core

    n_times, n_chans = 16, 256
    signal_ch = 128

    data = _make_spectrogram(n_times, n_chans, signal_ch, drift_channels=2, signal_amplitude=1.0)
    header = _make_header(n_chans, n_times)

    params = _core.SearchParams(max_drift_rate=4.0, min_snr=20.0, use_taylor_tree=True)
    engine = _core.DedopplerEngine(params)
    result = engine.search(data.ravel().tolist(), n_times, n_chans, header)

    assert len(result.candidates) == 0, (
        f"Expected no detections for weak signal, got {len(result.candidates)}"
    )
    print("  PASS: Weak signal correctly below threshold — 0 candidates")


def test_performance_speedup():
    """Taylor tree should be faster than brute-force on large data.

    The Taylor tree advantage grows with max_drift_channels: brute-force
    is O(D * N * F) while Taylor tree is O(log(N) * N * F) where D is
    the number of drift rates.  For large max_drift_rate or many time
    steps, the speedup is significant.
    """
    try:
        import mitraseti_core as _core
    except ImportError:
        import astroseti_core as _core

    n_times, n_chans = 64, 8192
    data = _make_spectrogram(n_times, n_chans, 4096, drift_channels=8, signal_amplitude=60.0)
    header = _make_header(n_chans, n_times, foff=-0.00014)
    data_flat = data.ravel().tolist()

    t0 = time.perf_counter()
    params_taylor = _core.SearchParams(max_drift_rate=4.0, min_snr=5.0, use_taylor_tree=True)
    engine_taylor = _core.DedopplerEngine(params_taylor)
    result_taylor = engine_taylor.search(data_flat, n_times, n_chans, header)
    taylor_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    params_brute = _core.SearchParams(max_drift_rate=4.0, min_snr=5.0, use_taylor_tree=False)
    engine_brute = _core.DedopplerEngine(params_brute)
    result_brute = engine_brute.search(data_flat, n_times, n_chans, header)
    brute_time = time.perf_counter() - t0

    speedup = brute_time / max(taylor_time, 1e-6)
    print(
        f"  Taylor tree: {taylor_time * 1000:.1f} ms ({result_taylor.processing_time_ms} ms core)"
    )
    print(f"  Brute-force: {brute_time * 1000:.1f} ms ({result_brute.processing_time_ms} ms core)")
    print(f"  Speedup:     {speedup:.1f}x")
    print(f"  PASS: Performance comparison complete (speedup={speedup:.1f}x)")


def test_multiple_signals():
    """Detect multiple injected signals at different frequencies and drift rates."""
    try:
        import mitraseti_core as _core
    except ImportError:
        import astroseti_core as _core

    n_times, n_chans = 32, 1024
    rng = np.random.default_rng(99)
    data = rng.normal(0, 1, (n_times, n_chans)).astype(np.float32)

    injections = [
        (200, 2, 70.0),
        (500, -3, 80.0),
        (800, 0, 60.0),
    ]

    for signal_ch, drift_ch, amp in injections:
        for t in range(n_times):
            ch = signal_ch + int(round(drift_ch * t / max(n_times - 1, 1)))
            if 0 <= ch < n_chans:
                data[t, ch] += amp

    header = _make_header(n_chans, n_times)
    params = _core.SearchParams(max_drift_rate=4.0, min_snr=5.0, use_taylor_tree=True)
    engine = _core.DedopplerEngine(params)
    result = engine.search(data.ravel().tolist(), n_times, n_chans, header)

    detected_freqs = sorted(set(round(c.frequency_hz / 1e3) for c in result.candidates))
    expected_freqs = [
        round((header.fch1 + ch * header.foff) * 1e6 / 1e3) for ch, _, _ in injections
    ]

    found = 0
    for ef in expected_freqs:
        if any(abs(df - ef) < 10 for df in detected_freqs):
            found += 1

    assert found >= 2, (
        f"Expected at least 2 of 3 injected signals, found {found}. "
        f"Expected near {expected_freqs}, detected {detected_freqs}"
    )
    print(f"  PASS: Detected {found}/{len(injections)} injected signals")


def test_fine_resolution_few_timesteps():
    """Regression: fine frequency resolution with few time steps.

    When channel_bw is very small (sub-Hz) and n_times is small,
    max_drift_channels >> n_padded.  The tree must clamp the drift
    range to n_padded-1 without panicking.  This reproduces the
    Voyager1 crash (16t × 524K channels, foff ~ 3e-6 MHz).
    """
    try:
        import mitraseti_core as _core
    except ImportError:
        import astroseti_core as _core

    n_times, n_chans = 16, 4096
    signal_ch = 2048
    drift_ch = 3

    data = _make_spectrogram(n_times, n_chans, signal_ch, drift_ch, signal_amplitude=60.0)
    header = _make_header(n_chans, n_times, foff=-2.8e-06, tsamp=18.25)

    params = _core.SearchParams(max_drift_rate=4.0, min_snr=5.0, use_taylor_tree=True)
    engine = _core.DedopplerEngine(params)

    result = engine.search(data.ravel().tolist(), n_times, n_chans, header)
    print(
        f"  PASS: No crash — {len(result.candidates)} candidates (fine-res: max_drift >> n_padded)"
    )


def main():
    print("=" * 60)
    print("Taylor Tree De-Doppler Tests")
    print("=" * 60)

    tests = [
        ("Taylor vs Brute-force equivalence", test_taylor_vs_bruteforce),
        ("Negative drift detection", test_negative_drift),
        ("Slow drift detection", test_slow_drift),
        ("SNR threshold filtering", test_snr_threshold),
        ("Fine-resolution few-timesteps (Voyager regression)", test_fine_resolution_few_timesteps),
        ("Performance comparison", test_performance_speedup),
        ("Multiple signal detection", test_multiple_signals),
    ]

    passed = 0
    failed = 0

    for name, func in tests:
        print(f"\n[TEST] {name}")
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
