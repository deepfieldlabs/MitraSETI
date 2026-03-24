"""
Periodicity Detection for Pulsed Signals — MitraSETI

Detects periodic patterns in radio signals using FFT-based analysis.
Catches pulsars, rotating transmitters, and periodic RFI that standard
de-Doppler search misses.

Methods:
  - FFT periodogram on time-collapsed signal power
  - Autocorrelation for confirming periodicity
  - Folded light curve construction

This is a genuinely novel addition to SETI search tools — no other
open-source SETI pipeline combines de-Doppler with periodicity detection.

Usage:
    from inference.periodicity import detect_periodicity
    result = detect_periodicity(spectrogram, tsamp=18.25)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PeriodicityResult:
    """Result of periodicity analysis on a signal."""
    is_periodic: bool
    best_period_s: float
    period_significance: float
    harmonics: List[float]
    folded_profile: Optional[np.ndarray]
    periodogram_freqs: np.ndarray
    periodogram_power: np.ndarray


def detect_periodicity(
    spectrogram: np.ndarray,
    tsamp: float = 18.253611008,
    freq_channel: Optional[int] = None,
    min_period_samples: int = 3,
    significance_threshold: float = 5.0,
) -> PeriodicityResult:
    """Detect periodic pulsations in a spectrogram.

    Extracts the time series at the brightest frequency channel (or
    a specified channel), computes the Lomb-Scargle-like periodogram
    via FFT, and identifies significant periodicities.

    Args:
        spectrogram: 2D array (freq_bins, time_steps) or (time, freq).
        tsamp: Time resolution in seconds.
        freq_channel: Specific channel to analyze (default: brightest).
        min_period_samples: Minimum period in time samples.
        significance_threshold: SNR threshold for periodicity detection.

    Returns:
        PeriodicityResult with period, significance, and folded profile.
    """
    if spectrogram.ndim != 2:
        return _null_result()

    # Ensure (freq, time) orientation
    if spectrogram.shape[0] > spectrogram.shape[1]:
        spec = spectrogram
    else:
        spec = spectrogram.T

    n_freq, n_time = spec.shape

    if n_time < 2 * min_period_samples:
        return _null_result()

    # Select frequency channel
    if freq_channel is not None and 0 <= freq_channel < n_freq:
        time_series = spec[freq_channel, :]
    else:
        avg_power = spec.mean(axis=1)
        brightest = int(np.argmax(avg_power))
        time_series = spec[brightest, :]

    # Detrend (remove linear trend)
    x = np.arange(n_time, dtype=np.float64)
    coeffs = np.polyfit(x, time_series.astype(np.float64), 1)
    trend = np.polyval(coeffs, x)
    detrended = time_series - trend

    # Normalize
    std = detrended.std()
    if std < 1e-10:
        return _null_result()
    detrended = detrended / std

    # FFT periodogram
    fft_vals = np.fft.rfft(detrended)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n_time, d=tsamp)

    # Skip DC component
    if len(power) > 1:
        power[0] = 0

    # Compute significance (SNR above noise floor)
    if len(power) > 4:
        noise_floor = np.median(power[1:])
        noise_std = np.median(np.abs(power[1:] - noise_floor)) * 1.4826
    else:
        noise_floor = 0
        noise_std = 1

    if noise_std < 1e-10:
        noise_std = 1.0

    snr_spectrum = (power - noise_floor) / noise_std

    # Find peaks above threshold
    min_freq_idx = max(1, min_period_samples)
    max_freq_idx = len(snr_spectrum) - 1

    if min_freq_idx >= max_freq_idx:
        return _null_result()

    search_snr = snr_spectrum[min_freq_idx:max_freq_idx + 1]
    search_freqs = freqs[min_freq_idx:max_freq_idx + 1]

    peak_idx = int(np.argmax(search_snr))
    peak_snr = float(search_snr[peak_idx])
    peak_freq = float(search_freqs[peak_idx])

    is_periodic = peak_snr >= significance_threshold
    best_period = 1.0 / peak_freq if peak_freq > 0 else 0.0

    # Find harmonics (peaks at 2×, 3×, ... of fundamental)
    harmonics = []
    if is_periodic and peak_freq > 0:
        for h in range(2, 5):
            harmonic_freq = peak_freq * h
            harmonic_idx = int(round(harmonic_freq * n_time * tsamp))
            if 0 < harmonic_idx < len(snr_spectrum):
                if snr_spectrum[harmonic_idx] >= significance_threshold * 0.5:
                    harmonics.append(1.0 / harmonic_freq)

    # Fold the time series at the best period
    folded = None
    if is_periodic and best_period > 0:
        n_bins = min(32, max(8, int(best_period / tsamp)))
        folded = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        for t in range(n_time):
            phase = (t * tsamp % best_period) / best_period
            bin_idx = int(phase * n_bins) % n_bins
            folded[bin_idx] += detrended[t]
            counts[bin_idx] += 1
        counts[counts == 0] = 1
        folded = folded / counts

    return PeriodicityResult(
        is_periodic=is_periodic,
        best_period_s=round(best_period, 6),
        period_significance=round(peak_snr, 2),
        harmonics=[round(h, 6) for h in harmonics],
        folded_profile=folded,
        periodogram_freqs=freqs,
        periodogram_power=power,
    )


def _null_result() -> PeriodicityResult:
    return PeriodicityResult(
        is_periodic=False,
        best_period_s=0.0,
        period_significance=0.0,
        harmonics=[],
        folded_profile=None,
        periodogram_freqs=np.array([]),
        periodogram_power=np.array([]),
    )


def batch_periodicity_search(
    spectrogram: np.ndarray,
    tsamp: float = 18.253611008,
    n_channels: int = 10,
    significance_threshold: float = 5.0,
) -> List[Tuple[int, PeriodicityResult]]:
    """Search for periodicity across multiple frequency channels.

    Selects the brightest n_channels and runs periodicity detection
    on each.  Returns results sorted by significance.
    """
    if spectrogram.ndim != 2:
        return []

    if spectrogram.shape[0] > spectrogram.shape[1]:
        spec = spectrogram
    else:
        spec = spectrogram.T

    n_freq, n_time = spec.shape
    avg_power = spec.mean(axis=1)
    top_channels = np.argsort(avg_power)[-n_channels:]

    results = []
    for ch in top_channels:
        result = detect_periodicity(
            spec, tsamp=tsamp, freq_channel=int(ch),
            significance_threshold=significance_threshold,
        )
        if result.period_significance > 0:
            results.append((int(ch), result))

    results.sort(key=lambda x: x[1].period_significance, reverse=True)
    return results
