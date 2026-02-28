"""
Radio Signal Feature Extractor for MitraSETI

Extracts physical and statistical features from radio spectrograms.
These features feed into both the signal classifier and the OOD anomaly
detector, and are useful for downstream analysis and filtering.

Features extracted:
- SNR (signal-to-noise ratio)
- Drift rate (via Hough transform on the spectrogram)
- Bandwidth
- Central frequency
- Duration
- Spectral index
- Polarisation ratio (if dual-pol data available)
- Modulation index
- Kurtosis (spectral peakedness)
- Skewness (spectral asymmetry)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RadioFeatures:
    """Physical and statistical features of a radio signal."""

    snr: float
    drift_rate: float  # Hz/s
    bandwidth: float  # Hz
    central_freq: float  # Hz
    duration: float  # seconds
    spectral_index: float
    polarization_ratio: float  # 0–1 (1 = fully polarised)
    modulation_index: float
    kurtosis: float
    skewness: float


class FeatureExtractor:
    """
    Extract physical features from radio SETI spectrograms.

    The extractor operates on 2D spectrogram arrays (frequency x time) and
    an optional metadata header that supplies observing parameters such as
    frequency resolution, time resolution, and reference frequency.

    All compute methods can be called independently, but the high-level
    :meth:`extract` method runs them all and returns a single
    :class:`RadioFeatures` dataclass.
    """

    # Default observing parameters (used when the header is missing keys)
    DEFAULT_FREQ_RESOLUTION: float = 2.7939677238464355  # Hz per channel (BL default)
    DEFAULT_TIME_RESOLUTION: float = 18.253611008  # seconds per time step
    DEFAULT_REF_FREQ: float = 1420.405751e6  # Hydrogen line (Hz)

    def __init__(self) -> None:
        """Initialise the feature extractor."""
        pass

    # ------------------------------------------------------------------
    # High-level interface
    # ------------------------------------------------------------------

    def extract(
        self,
        spectrogram: np.ndarray,
        header: Optional[Dict] = None,
    ) -> RadioFeatures:
        """
        Extract all features from a spectrogram.

        Args:
            spectrogram: 2D numpy array of shape (frequency_bins, time_steps).
                         Values should be power/intensity (linear scale preferred).
            header: Optional metadata dictionary with observing parameters.
                    Recognised keys:
                    - ``foff``  : frequency resolution in Hz (per channel)
                    - ``tsamp`` : time resolution in seconds (per sample)
                    - ``fch1``  : reference frequency of first channel (Hz)

        Returns:
            RadioFeatures dataclass with all extracted features.
        """
        if spectrogram.ndim != 2:
            raise ValueError(
                f"Expected 2D spectrogram (freq x time), got shape {spectrogram.shape}"
            )

        header = header or {}
        freq_res = abs(float(header.get("foff", self.DEFAULT_FREQ_RESOLUTION)))
        time_res = float(header.get("tsamp", self.DEFAULT_TIME_RESOLUTION))
        ref_freq = float(header.get("fch1", self.DEFAULT_REF_FREQ))

        spec = spectrogram.astype(np.float64)

        snr = self.compute_snr(spec)
        drift_rate = self.compute_drift_rate(spec, freq_res, time_res)
        bandwidth = self.compute_bandwidth(spec, freq_res)
        central_freq = self.compute_central_freq(spec, freq_res, ref_freq)
        duration = self.compute_duration(spec, time_res)
        spectral_index = self.compute_spectral_index(spec, freq_res, ref_freq)
        polarization_ratio = self.compute_polarization_ratio(header)
        modulation_index = self.compute_modulation_index(spec)
        kurtosis = self.compute_kurtosis(spec)
        skewness = self.compute_skewness(spec)

        features = RadioFeatures(
            snr=snr,
            drift_rate=drift_rate,
            bandwidth=bandwidth,
            central_freq=central_freq,
            duration=duration,
            spectral_index=spectral_index,
            polarization_ratio=polarization_ratio,
            modulation_index=modulation_index,
            kurtosis=kurtosis,
            skewness=skewness,
        )
        logger.debug(f"Extracted features: SNR={snr:.1f}, drift={drift_rate:.3f} Hz/s")
        return features

    # ------------------------------------------------------------------
    # Individual feature methods
    # ------------------------------------------------------------------

    @staticmethod
    def compute_snr(spectrogram: np.ndarray) -> float:
        """
        Compute the signal-to-noise ratio of the spectrogram.

        Uses the ratio of the peak value to the median absolute deviation
        (MAD) of the background, which is robust to outliers.

        Args:
            spectrogram: 2D array (frequency x time).

        Returns:
            Estimated SNR (dimensionless).
        """
        flat = spectrogram.ravel()
        median = np.median(flat)
        mad = np.median(np.abs(flat - median))
        # MAD to std conversion for Gaussian noise
        sigma = mad * 1.4826
        if sigma < 1e-12:
            return 0.0
        peak = float(np.max(flat))
        return (peak - median) / sigma

    @staticmethod
    def compute_drift_rate(
        spectrogram: np.ndarray,
        freq_res: float,
        time_res: float,
    ) -> float:
        """
        Estimate the drift rate of a narrowband signal using the Hough transform.

        Projects the spectrogram along a set of candidate drift-rate angles and
        selects the one that maximises the integrated power.  The drift rate is
        returned in Hz/s.

        Args:
            spectrogram: 2D array (frequency x time).
            freq_res: Frequency resolution (Hz per channel).
            time_res: Time resolution (seconds per sample).

        Returns:
            Estimated drift rate in Hz/s.
        """
        n_freq, n_time = spectrogram.shape
        if n_time < 2:
            return 0.0

        # Candidate drift rates: from -max_drift to +max_drift
        max_drift_channels = n_freq / 2  # max channels drifted over full obs
        total_time = n_time * time_res
        max_drift_hz_per_s = (max_drift_channels * freq_res) / total_time if total_time > 0 else 0.0

        n_trials = min(401, 2 * n_freq + 1)
        trial_rates = np.linspace(-max_drift_hz_per_s, max_drift_hz_per_s, n_trials)

        best_power = -np.inf
        best_rate = 0.0

        for rate in trial_rates:
            # Shift each time step by the appropriate number of channels
            shift_per_step = (rate * time_res) / freq_res if freq_res > 0 else 0.0
            power = 0.0
            for t in range(n_time):
                shift = int(round(shift_per_step * t))
                shifted_col = np.roll(spectrogram[:, t], -shift)
                power += shifted_col.max()
            if power > best_power:
                best_power = power
                best_rate = rate

        return float(best_rate)

    @staticmethod
    def compute_bandwidth(spectrogram: np.ndarray, freq_res: float) -> float:
        """
        Estimate the bandwidth of the signal.

        Computes the bandwidth as the range of frequency channels that
        exceed 50 % of the peak power in the time-averaged spectrum.

        Args:
            spectrogram: 2D array (frequency x time).
            freq_res: Frequency resolution (Hz per channel).

        Returns:
            Estimated bandwidth in Hz.
        """
        avg_spectrum = spectrogram.mean(axis=1)
        peak = avg_spectrum.max()
        if peak <= 0:
            return 0.0
        above_half = np.where(avg_spectrum > 0.5 * peak)[0]
        if len(above_half) == 0:
            return 0.0
        bw_channels = above_half[-1] - above_half[0] + 1
        return float(bw_channels * freq_res)

    @staticmethod
    def compute_central_freq(spectrogram: np.ndarray, freq_res: float, ref_freq: float) -> float:
        """
        Compute the central frequency of the signal.

        Weighted centroid of the time-averaged spectrum relative to the
        reference frequency.

        Args:
            spectrogram: 2D array (frequency x time).
            freq_res: Frequency resolution (Hz per channel).
            ref_freq: Reference frequency of the first channel (Hz).

        Returns:
            Central frequency in Hz.
        """
        avg_spectrum = spectrogram.mean(axis=1)
        total_power = avg_spectrum.sum()
        if total_power <= 0:
            return ref_freq
        channels = np.arange(len(avg_spectrum))
        centroid_channel = float(np.sum(channels * avg_spectrum) / total_power)
        return ref_freq + centroid_channel * freq_res

    @staticmethod
    def compute_duration(spectrogram: np.ndarray, time_res: float) -> float:
        """
        Estimate the signal duration.

        Computes the time span over which the frequency-averaged power
        exceeds 50 % of its peak value.

        Args:
            spectrogram: 2D array (frequency x time).
            time_res: Time resolution (seconds per sample).

        Returns:
            Estimated duration in seconds.
        """
        time_profile = spectrogram.mean(axis=0)
        peak = time_profile.max()
        if peak <= 0:
            return 0.0
        above_half = np.where(time_profile > 0.5 * peak)[0]
        if len(above_half) == 0:
            return 0.0
        duration_steps = above_half[-1] - above_half[0] + 1
        return float(duration_steps * time_res)

    @staticmethod
    def compute_spectral_index(spectrogram: np.ndarray, freq_res: float, ref_freq: float) -> float:
        """
        Estimate the spectral index (power-law slope).

        Fits a power law S ~ f^alpha to the time-averaged spectrum in
        log-log space using linear regression.

        Args:
            spectrogram: 2D array (frequency x time).
            freq_res: Frequency resolution (Hz per channel).
            ref_freq: Reference frequency (Hz).

        Returns:
            Spectral index alpha.
        """
        avg_spectrum = spectrogram.mean(axis=1)
        n_freq = len(avg_spectrum)
        freqs = ref_freq + np.arange(n_freq) * freq_res

        # Filter positive values for log-log fit
        mask = (avg_spectrum > 0) & (freqs > 0)
        if mask.sum() < 2:
            return 0.0

        log_f = np.log10(freqs[mask])
        log_s = np.log10(avg_spectrum[mask])

        # Simple linear regression: log_s = alpha * log_f + c
        try:
            coeffs = np.polyfit(log_f, log_s, 1)
            return float(coeffs[0])
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    @staticmethod
    def compute_polarization_ratio(header: Optional[Dict]) -> float:
        """
        Extract polarisation ratio from the header.

        If the observation includes Stokes parameters (I, Q, U, V), the
        polarisation ratio is computed as sqrt(Q^2 + U^2 + V^2) / I.
        If polarisation data is not available, returns 0.0.

        Args:
            header: Metadata dictionary (may contain ``stokes_i``, ``stokes_q``,
                    ``stokes_u``, ``stokes_v``).

        Returns:
            Polarisation ratio in [0, 1].
        """
        if not header:
            return 0.0

        stokes_i = header.get("stokes_i")
        stokes_q = header.get("stokes_q", 0.0)
        stokes_u = header.get("stokes_u", 0.0)
        stokes_v = header.get("stokes_v", 0.0)

        if stokes_i is None or stokes_i <= 0:
            return 0.0

        pol_intensity = math.sqrt(stokes_q**2 + stokes_u**2 + stokes_v**2)
        ratio = pol_intensity / stokes_i
        return float(min(ratio, 1.0))

    @staticmethod
    def compute_modulation_index(spectrogram: np.ndarray) -> float:
        """
        Compute the modulation index of the signal.

        The modulation index measures the relative variation of the
        frequency-averaged power over time:
            m = std(power_over_time) / mean(power_over_time)

        A high modulation index indicates a pulsed or variable signal.

        Args:
            spectrogram: 2D array (frequency x time).

        Returns:
            Modulation index (dimensionless, >= 0).
        """
        time_profile = spectrogram.mean(axis=0)
        mean_power = time_profile.mean()
        if mean_power <= 0:
            return 0.0
        return float(time_profile.std() / mean_power)

    @staticmethod
    def compute_kurtosis(spectrogram: np.ndarray) -> float:
        """
        Compute the excess kurtosis of the time-averaged spectrum.

        Kurtosis measures the "peakedness" of the spectral distribution.
        A Gaussian distribution has excess kurtosis of 0; a strong
        narrowband signal produces high kurtosis.

        Args:
            spectrogram: 2D array (frequency x time).

        Returns:
            Excess kurtosis (dimensionless).
        """
        avg_spectrum = spectrogram.mean(axis=1)
        n = len(avg_spectrum)
        if n < 4:
            return 0.0

        mean = avg_spectrum.mean()
        std = avg_spectrum.std()
        if std < 1e-12:
            return 0.0

        z = (avg_spectrum - mean) / std
        # Excess kurtosis = E[z^4] - 3
        kurt = float(np.mean(z**4) - 3.0)
        return kurt

    @staticmethod
    def compute_skewness(spectrogram: np.ndarray) -> float:
        """
        Compute the skewness of the time-averaged spectrum.

        Skewness measures the asymmetry of the spectral distribution.
        Positive skewness indicates a tail towards higher frequencies;
        negative skewness indicates a tail towards lower frequencies.

        Args:
            spectrogram: 2D array (frequency x time).

        Returns:
            Skewness (dimensionless).
        """
        avg_spectrum = spectrogram.mean(axis=1)
        n = len(avg_spectrum)
        if n < 3:
            return 0.0

        mean = avg_spectrum.mean()
        std = avg_spectrum.std()
        if std < 1e-12:
            return 0.0

        z = (avg_spectrum - mean) / std
        skew = float(np.mean(z**3))
        return skew
