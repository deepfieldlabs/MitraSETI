"""
Integrated Chirp Rate (Doppler Acceleration) Search — MitraSETI v0.3.0

Extends the linear de-Doppler search to detect signals with second-order
frequency drift (acceleration).

The Physics
-----------
A transmitter on a planet orbiting its star has a time-varying radial
velocity relative to Earth. To first order, this produces a linear
frequency drift (the standard de-Doppler problem):

    f(t) = f_0 + f_dot * t

But the acceleration of the planet in its orbit adds a quadratic term:

    f(t) = f_0 + f_dot * t + 0.5 * f_ddot * t^2

where f_ddot is the "chirp rate" or "frequency acceleration" in Hz/s^2.

For a planet at 1 AU orbiting a Sun-like star:
    f_ddot ~ 10^-3 Hz/s^2 at L-band (1.4 GHz)

Over a 5-minute observation at BL resolution, this produces a drift of
~0.1 channels — negligible. But for longer observations (30 min+) or
closer orbits, the quadratic term smears the signal across channels and
reduces detection SNR.

The Algorithm
-------------
1. For each trial chirp rate in [-a_max, +a_max]:
   a. De-chirp the spectrogram: shift each row by -0.5*a*t^2 channels
   b. Run the standard Taylor tree on the straightened data
   c. Collect candidates
2. Subtract baseline (a=0) candidates to find chirp-only detections
3. For candidates found at multiple chirp rates, keep the one with the
   highest SNR (that chirp rate best matches the signal's true acceleration)

This module provides a clean API that can be called from the pipeline or
the CLI, unlike the standalone script in scripts/chirp_search.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChirpCandidate:
    """A candidate detected through chirp-rate search."""

    frequency_hz: float
    drift_rate: float
    chirp_rate: float  # Hz/s^2
    snr: float
    is_chirp_only: bool  # True if only detectable WITH chirp correction


@dataclass
class ChirpSearchResult:
    """Results from a chirp-rate search."""

    candidates: List[ChirpCandidate]
    baseline_hits: int
    chirp_only_count: int
    best_chirp_rate: Optional[float]
    processing_time_ms: float
    chirp_rates_tested: List[float]


def dechirp_spectrogram(
    data: np.ndarray,
    chirp_rate: float,
    tsamp: float,
    foff_hz: float,
) -> np.ndarray:
    """Remove quadratic frequency drift from a spectrogram.

    For each time step t, shifts the spectrum by -0.5*a*t^2 channels.
    Uses np.roll for integer-channel shifts.

    Args:
        data: shape (n_times, n_chans), float32
        chirp_rate: frequency acceleration in Hz/s^2
        tsamp: time resolution in seconds
        foff_hz: channel width in Hz (absolute value used)

    Returns:
        De-chirped spectrogram of the same shape.
    """
    n_times, n_chans = data.shape
    result = np.empty_like(data)
    abs_foff = abs(foff_hz) if abs(foff_hz) > 0 else 1.0

    for t in range(n_times):
        elapsed = t * tsamp
        shift_hz = 0.5 * chirp_rate * elapsed * elapsed
        shift_ch = int(round(shift_hz / abs_foff))

        if shift_ch == 0:
            result[t] = data[t]
        else:
            result[t] = np.roll(data[t], -shift_ch)
            # Zero out wrapped elements to prevent false edge signals
            if shift_ch > 0 and shift_ch < n_chans:
                result[t, -shift_ch:] = 0.0
            elif shift_ch < 0 and abs(shift_ch) < n_chans:
                result[t, : abs(shift_ch)] = 0.0

    return result


def run_chirp_search(
    spectrogram: np.ndarray,
    header: Dict[str, Any],
    chirp_max: float = 0.1,
    chirp_steps: int = 9,
    max_drift_rate: float = 4.0,
    min_snr: float = 10.0,
) -> ChirpSearchResult:
    """Run a chirp-rate search on a spectrogram.

    Tests a range of quadratic drift corrections and identifies signals
    that are only detectable (or significantly stronger) with the chirp
    correction applied.

    Args:
        spectrogram: (n_times, n_chans) power spectrogram.
        header: Dict with fch1, foff, tsamp, tstart, source_name.
        chirp_max: Maximum chirp rate magnitude in Hz/s^2.
        chirp_steps: Number of trial chirp rates (spread over [-max, +max]).
        max_drift_rate: Maximum linear drift rate for Taylor tree.
        min_snr: Minimum SNR threshold.

    Returns:
        ChirpSearchResult with all candidates.
    """
    from core_gpu.taylor_tree_gpu import gpu_taylor_tree_search

    t_start = time.perf_counter()

    n_times, n_chans = spectrogram.shape
    tsamp = header["tsamp"]
    foff_hz = abs(header["foff"]) * 1e6

    chirp_rates = list(np.linspace(-chirp_max, chirp_max, chirp_steps))

    # Baseline: zero chirp rate (standard linear search)
    baseline = gpu_taylor_tree_search(
        spectrogram,
        header,
        max_drift_rate=max_drift_rate,
        min_snr=min_snr,
    )
    baseline_keys = {
        (round(c.frequency_hz, 1), round(c.drift_rate, 3)) for c in baseline.candidates
    }
    baseline_snrs = {
        (round(c.frequency_hz, 1), round(c.drift_rate, 3)): c.snr for c in baseline.candidates
    }

    all_candidates: List[ChirpCandidate] = []
    # Include baseline candidates with chirp_rate=0
    for c in baseline.candidates:
        all_candidates.append(
            ChirpCandidate(
                frequency_hz=c.frequency_hz,
                drift_rate=c.drift_rate,
                chirp_rate=0.0,
                snr=c.snr,
                is_chirp_only=False,
            )
        )

    chirp_only_candidates: List[ChirpCandidate] = []

    for accel in chirp_rates:
        if accel == 0.0:
            continue

        dechirped = dechirp_spectrogram(spectrogram, accel, tsamp, foff_hz)
        result = gpu_taylor_tree_search(
            dechirped,
            header,
            max_drift_rate=max_drift_rate,
            min_snr=min_snr,
        )

        for c in result.candidates:
            key = (round(c.frequency_hz, 1), round(c.drift_rate, 3))
            is_new = key not in baseline_keys
            is_stronger = not is_new and c.snr > baseline_snrs.get(key, 0) * 1.1

            if is_new or is_stronger:
                cc = ChirpCandidate(
                    frequency_hz=c.frequency_hz,
                    drift_rate=c.drift_rate,
                    chirp_rate=accel,
                    snr=c.snr,
                    is_chirp_only=is_new,
                )
                all_candidates.append(cc)
                if is_new:
                    chirp_only_candidates.append(cc)

    # Deduplicate: keep highest SNR per (freq, drift) pair
    best_by_key: Dict[Tuple[float, float], ChirpCandidate] = {}
    for c in all_candidates:
        key = (round(c.frequency_hz, 0), round(c.drift_rate, 2))
        if key not in best_by_key or c.snr > best_by_key[key].snr:
            best_by_key[key] = c

    final = sorted(best_by_key.values(), key=lambda c: -c.snr)

    # Find the chirp rate that produced the most unique detections
    chirp_counts: Dict[float, int] = {}
    for c in chirp_only_candidates:
        chirp_counts[c.chirp_rate] = chirp_counts.get(c.chirp_rate, 0) + 1
    best_chirp = max(chirp_counts, key=chirp_counts.get) if chirp_counts else None

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        "Chirp search: %d trials, %d baseline, %d chirp-only, %.1f ms",
        len(chirp_rates),
        len(baseline.candidates),
        len(chirp_only_candidates),
        elapsed_ms,
    )

    return ChirpSearchResult(
        candidates=final,
        baseline_hits=len(baseline.candidates),
        chirp_only_count=len(chirp_only_candidates),
        best_chirp_rate=best_chirp,
        processing_time_ms=round(elapsed_ms, 2),
        chirp_rates_tested=chirp_rates,
    )
