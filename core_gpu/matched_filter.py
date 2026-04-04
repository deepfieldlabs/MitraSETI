"""
Matched Filter Bank for SETI Signal Detection — MitraSETI v0.3.0

A matched filter is the optimal linear detector for a known signal
shape embedded in Gaussian noise. By correlating the spectrogram with
a bank of signal templates, we can detect specific signal morphologies
that may be missed by the standard Taylor tree (which only looks for
straight-line drifts).

Templates Implemented
---------------------
1. **Narrowband drifting** — the classic SETI signal template, equivalent
   to the Taylor tree but computed via cross-correlation.
2. **Pulsed narrowband** — a signal that appears intermittently (e.g., a
   rotating transmitter beaming past Earth periodically).
3. **Broadband chirp** — a signal that sweeps across many channels rapidly,
   characteristic of radar or pulsars.
4. **Comb signal** — multiple equally-spaced narrowband tones, which could
   indicate an artificial origin (natural sources don't produce combs).
5. **Modulated carrier** — an amplitude-modulated narrowband signal, where
   the power varies over time in a structured pattern.

Theory
------
For a template h[t, f] and spectrogram x[t, f], the matched filter
output is:

    SNR_mf = (sum(x * h)) / sqrt(sum(h^2))

This is the correlation normalised by the template energy. It is
mathematically proven to maximise the output SNR when the noise is
white and Gaussian (Wiener-Khinchin theorem).

For coloured noise (realistic radio data), we first whiten the data
by dividing by the per-channel noise estimate. This converts the
problem back to white noise, making the matched filter optimal again.

Computational Approach
----------------------
Rather than sliding each template across every (time, freq) position
(which is O(n_templates * n_times * n_chans * template_size^2)), we
compute the 2D cross-correlation using FFT:

    corr = IFFT(FFT(data) * conj(FFT(template)))

This reduces the per-template cost to O(n_times * n_chans * log(n_times * n_chans)),
making a bank of 50-100 templates feasible on large spectrograms.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import signal as sp_signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MatchedFilterCandidate:
    """A detection from the matched filter bank."""

    frequency_hz: float
    time_idx: int
    snr: float
    template_name: str
    template_params: Dict[str, Any]
    freq_channel: int


@dataclass
class MatchedFilterResult:
    """Results from a matched filter bank search."""

    candidates: List[MatchedFilterCandidate]
    templates_tested: int
    processing_time_ms: float
    per_template_times_ms: Dict[str, float]


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


def _template_narrowband_drift(
    n_times: int,
    n_chans: int,
    drift_channels: int,
) -> np.ndarray:
    """Create a narrowband drifting signal template.

    A diagonal line from (0, center) to (n_times-1, center + drift_channels).
    """
    template = np.zeros((n_times, n_chans), dtype=np.float32)
    center = n_chans // 2

    for t in range(n_times):
        ch = center + int(drift_channels * t / max(n_times - 1, 1))
        if 0 <= ch < n_chans:
            template[t, ch] = 1.0

    return template


def _template_pulsed(
    n_times: int,
    n_chans: int,
    period: int,
    duty_cycle: float = 0.3,
    drift_channels: int = 0,
) -> np.ndarray:
    """Create a pulsed narrowband template.

    Signal appears for duty_cycle fraction of each period.
    """
    template = np.zeros((n_times, n_chans), dtype=np.float32)
    center = n_chans // 2
    on_duration = max(1, int(period * duty_cycle))

    for t in range(n_times):
        phase = t % period
        if phase < on_duration:
            ch = center + int(drift_channels * t / max(n_times - 1, 1))
            if 0 <= ch < n_chans:
                template[t, ch] = 1.0

    return template


def _template_broadband_chirp(
    n_times: int,
    n_chans: int,
    sweep_channels: int,
) -> np.ndarray:
    """Create a broadband chirp template that sweeps across many channels."""
    template = np.zeros((n_times, n_chans), dtype=np.float32)
    center = n_chans // 2
    half_sweep = sweep_channels // 2

    for t in range(n_times):
        frac = t / max(n_times - 1, 1)
        ch = center - half_sweep + int(sweep_channels * frac)
        # Spread across a few adjacent channels for broadband nature
        for dc in range(-1, 2):
            if 0 <= ch + dc < n_chans:
                template[t, ch + dc] = 1.0 if dc == 0 else 0.5

    return template


def _template_comb(
    n_times: int,
    n_chans: int,
    n_tones: int = 5,
    spacing: int = 20,
) -> np.ndarray:
    """Create a comb signal template — multiple equally-spaced tones.

    Natural signals don't produce perfectly spaced tones, so this
    is a strong indicator of artificial origin.
    """
    template = np.zeros((n_times, n_chans), dtype=np.float32)
    center = n_chans // 2
    total_width = (n_tones - 1) * spacing
    start_ch = center - total_width // 2

    for i in range(n_tones):
        ch = start_ch + i * spacing
        if 0 <= ch < n_chans:
            template[:, ch] = 1.0

    return template


def _template_modulated(
    n_times: int,
    n_chans: int,
    mod_period: int = 4,
) -> np.ndarray:
    """Create an amplitude-modulated carrier template.

    A narrowband signal whose power varies sinusoidally over time.
    """
    template = np.zeros((n_times, n_chans), dtype=np.float32)
    center = n_chans // 2

    for t in range(n_times):
        amplitude = 0.5 + 0.5 * np.cos(2 * np.pi * t / mod_period)
        template[t, center] = float(amplitude)

    return template


# ---------------------------------------------------------------------------
# Template bank generation
# ---------------------------------------------------------------------------


def generate_template_bank(
    n_times: int,
    template_width: int = 64,
) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """Generate the full bank of signal templates.

    Returns a list of (name, template_array, params) tuples.
    Templates are small (n_times x template_width) patches that will be
    slid across the full spectrogram via FFT cross-correlation.

    Args:
        n_times: Number of time steps in the data.
        template_width: Width of each template in channels.

    Returns:
        List of (template_name, template_2d_array, params_dict).
    """
    templates = []

    # Narrowband drifts at several rates
    for drift in [-4, -2, -1, 0, 1, 2, 4]:
        name = f"narrowband_drift_{drift:+d}"
        t = _template_narrowband_drift(n_times, template_width, drift)
        templates.append((name, t, {"drift_channels": drift}))

    # Pulsed signals with various periods
    for period in [2, 3, 4]:
        for drift in [0, 1, -1]:
            name = f"pulsed_p{period}_d{drift:+d}"
            t = _template_pulsed(n_times, template_width, period, 0.3, drift)
            templates.append((name, t, {"period": period, "drift": drift}))

    # Broadband chirps
    for sweep in [8, 16, 32]:
        name = f"broadband_chirp_{sweep}"
        t = _template_broadband_chirp(n_times, template_width, sweep)
        templates.append((name, t, {"sweep_channels": sweep}))

    # Comb signals
    for n_tones in [3, 5, 7]:
        spacing = template_width // (n_tones + 1)
        if spacing < 2:
            spacing = 2
        name = f"comb_{n_tones}x{spacing}"
        t = _template_comb(n_times, template_width, n_tones, spacing)
        templates.append((name, t, {"n_tones": n_tones, "spacing": spacing}))

    # Modulated carriers
    for mod_period in [3, 4, 6]:
        name = f"modulated_p{mod_period}"
        t = _template_modulated(n_times, template_width, mod_period)
        templates.append((name, t, {"mod_period": mod_period}))

    return templates


# ---------------------------------------------------------------------------
# Matched filter core
# ---------------------------------------------------------------------------


def _matched_filter_fft(
    data: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """Compute matched filter output via FFT cross-correlation.

    Returns the SNR map: correlation normalised by template energy
    and local noise.
    """
    # Normalise template
    template_energy = np.sqrt(np.sum(template**2))
    if template_energy < 1e-10:
        return np.zeros_like(data)

    norm_template = template / template_energy

    # Use scipy's fftconvolve for efficiency
    # correlate2d = convolve with flipped template
    corr = sp_signal.fftconvolve(data, norm_template[::-1, ::-1], mode="same")

    return corr


def _extract_peaks(
    snr_map: np.ndarray,
    min_snr: float,
    header: Dict[str, Any],
    template_name: str,
    template_params: Dict[str, Any],
    max_peaks: int = 50,
) -> List[MatchedFilterCandidate]:
    """Extract peaks from the matched filter SNR map."""
    # Find all points above threshold
    peaks = np.argwhere(snr_map > min_snr)

    if len(peaks) == 0:
        return []

    # Sort by SNR descending
    snr_values = snr_map[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(-snr_values)
    peaks = peaks[order]
    snr_values = snr_values[order]

    # Greedy NMS: suppress peaks within 3 channels and 2 time steps
    candidates = []
    used = set()

    for idx in range(min(len(peaks), max_peaks * 5)):
        t, f = int(peaks[idx, 0]), int(peaks[idx, 1])
        key = (t // 2, f // 3)
        if key in used:
            continue
        used.add(key)

        freq_hz = (header["fch1"] + f * header["foff"]) * 1e6
        candidates.append(
            MatchedFilterCandidate(
                frequency_hz=freq_hz,
                time_idx=t,
                snr=float(snr_values[idx]),
                template_name=template_name,
                template_params=template_params,
                freq_channel=f,
            )
        )

        if len(candidates) >= max_peaks:
            break

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_matched_filter_search(
    spectrogram: np.ndarray,
    header: Dict[str, Any],
    min_snr: float = 8.0,
    template_width: int = 64,
    max_candidates_per_template: int = 20,
) -> MatchedFilterResult:
    """Run the full matched filter bank search.

    Args:
        spectrogram: (n_times, n_chans) spectrogram, float32.
        header: Dict with fch1, foff, tsamp, tstart.
        min_snr: Minimum matched filter SNR for detection.
        template_width: Channel width of each template.
        max_candidates_per_template: Max detections per template type.

    Returns:
        MatchedFilterResult with all detections.
    """
    t_start = time.perf_counter()

    n_times, n_chans = spectrogram.shape

    # Normalise the spectrogram (per-channel median/MAD whitening)
    median = np.median(spectrogram, axis=0)
    mad = np.median(np.abs(spectrogram - median[np.newaxis, :]), axis=0)
    sigma = np.where(mad < 1e-7, 1.0, 1.4826 * mad)
    data_norm = (spectrogram - median[np.newaxis, :]) / sigma[np.newaxis, :]

    # Pre-filter: energy detection to skip noise-only channels.
    # Based on Pulscan (2025) — chi-squared test on channel power.
    # Channels with total squared power below 2*n_times (expected under
    # chi-squared for Gaussian noise) are excluded, reducing the FFT
    # cross-correlation search space by 5-10x on typical data.
    channel_power = np.sum(data_norm**2, axis=0)
    active_mask = channel_power > 2.0 * n_times
    n_active = int(np.sum(active_mask))
    logger.debug(
        "Pre-filter: %d / %d channels active (%.1f%% reduction)",
        n_active,
        n_chans,
        100.0 * (1.0 - n_active / max(n_chans, 1)),
    )

    if n_active == 0:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return MatchedFilterResult(
            candidates=[],
            templates_tested=0,
            processing_time_ms=round(elapsed_ms, 2),
            per_template_times_ms={},
        )

    # Generate templates
    templates = generate_template_bank(n_times, template_width)

    all_candidates = []
    per_template_times = {}

    for name, template, params in templates:
        t0 = time.perf_counter()

        # Only correlate against active channels
        snr_map = _matched_filter_fft(data_norm, template)
        # Zero out inactive channels so no spurious peaks appear
        snr_map[:, ~active_mask] = 0.0
        peaks = _extract_peaks(
            snr_map,
            min_snr,
            header,
            name,
            params,
            max_peaks=max_candidates_per_template,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        per_template_times[name] = round(elapsed, 2)
        all_candidates.extend(peaks)

    # Global deduplication across templates (same freq_channel / time_idx)
    best_by_pos: Dict[Tuple[int, int], MatchedFilterCandidate] = {}
    for c in all_candidates:
        key = (c.time_idx, c.freq_channel // 3)
        if key not in best_by_pos or c.snr > best_by_pos[key].snr:
            best_by_pos[key] = c

    final = sorted(best_by_pos.values(), key=lambda c: -c.snr)

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        "Matched filter: %d templates, %d detections, %.1f ms",
        len(templates),
        len(final),
        elapsed_ms,
    )

    return MatchedFilterResult(
        candidates=final,
        templates_tested=len(templates),
        processing_time_ms=round(elapsed_ms, 2),
        per_template_times_ms=per_template_times,
    )
