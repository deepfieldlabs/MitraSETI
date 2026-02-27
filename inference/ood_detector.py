"""
Out-of-Distribution (OOD) Detector for Radio SETI Signals

Ensemble detection using multiple methods for robust anomaly detection
in radio spectrograms:
- MSP (Maximum Softmax Probability)
- Energy-based detection
- Spectral distance to known signal templates

Adapted from AstroLens OOD detector for radio signal domain.

References:
- MSP: "A Baseline for Detecting Misclassified and OOD Examples" (Hendrycks & Gimpel, 2017)
- Energy: "Energy-based Out-of-distribution Detection" (Liu et al., NeurIPS 2020)
- Spectral Distance: compares spectral shape to known signal templates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OODResult:
    """Result of OOD detection on a radio signal."""
    ood_score: float
    is_anomaly: bool
    threshold: float
    method_scores: Dict[str, float] = field(default_factory=dict)


class RadioOODDetector:
    """
    Ensemble out-of-distribution detector for radio SETI signals.

    Combines three OOD detection methods:
    - MSP: Low max softmax probability indicates uncertainty / possible OOD.
    - Energy: High energy score indicates low model confidence / possible OOD.
    - Spectral distance: Compares the signal's spectral shape to known signal
      templates. Large distance indicates a novel or unseen signal type.

    Final decision uses voting: if ``voting_threshold`` or more methods flag
    the signal as OOD, it is reported as anomalous.
    """

    def __init__(
        self,
        msp_threshold: float = 0.5,
        energy_threshold: float = 2.5,
        spectral_threshold: float = 1.5,
        temperature: float = 1.0,
        voting_threshold: int = 2,
    ):
        """
        Args:
            msp_threshold: Confidence below which MSP flags as OOD.
                           Lower values make detection less sensitive.
            energy_threshold: Energy score above which the signal is flagged.
                              Lower values make detection more sensitive.
            spectral_threshold: Spectral distance above which the signal is flagged.
            temperature: Softmax temperature for MSP and energy computation.
            voting_threshold: Minimum number of methods that must agree to flag
                              the signal as anomalous (1–3).
        """
        self.msp_threshold = msp_threshold
        self.energy_threshold = energy_threshold
        self.spectral_threshold = spectral_threshold
        self.temperature = temperature
        self.voting_threshold = max(1, min(3, voting_threshold))

        # Reference statistics computed during calibration
        self._reference_spectra: Optional[np.ndarray] = None  # (N, freq_bins)
        self._reference_mean_spectrum: Optional[np.ndarray] = None
        self._reference_std_spectrum: Optional[np.ndarray] = None
        self._calibrated: bool = False

    # ------------------------------------------------------------------
    # Individual detection methods
    # ------------------------------------------------------------------

    def compute_msp(self, logits: np.ndarray) -> float:
        """
        Maximum Softmax Probability.

        Lower MSP means the model is less confident, suggesting OOD.
        Returns ``1 - max_prob`` so that *higher* values indicate more
        anomalous inputs (consistent with the other scores).
        """
        scaled = logits / self.temperature
        # Numerically stable softmax
        exp_logits = np.exp(scaled - np.max(scaled))
        probs = exp_logits / np.sum(exp_logits)
        max_prob = float(np.max(probs))
        return 1.0 - max_prob

    def compute_energy(self, logits: np.ndarray) -> float:
        """
        Energy-based OOD score.

        E(x) = -T * log( sum( exp( logit_i / T ) ) )

        Higher energy indicates lower model confidence and a greater
        likelihood of being OOD.
        """
        scaled = logits / self.temperature
        max_logit = np.max(scaled)
        energy = -self.temperature * (
            max_logit + np.log(np.sum(np.exp(scaled - max_logit)))
        )
        return float(energy)

    def compute_spectral_distance(self, spectrogram: np.ndarray) -> float:
        """
        Spectral distance to reference signal templates.

        Compares the time-averaged spectrum of the input to the mean reference
        spectrum using a normalized Euclidean distance. If no reference data
        has been calibrated, returns 0.0.

        Args:
            spectrogram: 2D array of shape (frequency, time).

        Returns:
            Normalised distance (higher = more different from references).
        """
        if self._reference_mean_spectrum is None:
            return 0.0

        # Time-averaged power spectrum of the input
        input_spectrum = spectrogram.mean(axis=1).astype(np.float64)

        # Resize if needed
        if input_spectrum.shape[0] != self._reference_mean_spectrum.shape[0]:
            from scipy.ndimage import zoom
            factor = self._reference_mean_spectrum.shape[0] / input_spectrum.shape[0]
            input_spectrum = zoom(input_spectrum, factor, order=1)

        # Normalise input spectrum
        std = input_spectrum.std()
        if std > 0:
            input_spectrum = (input_spectrum - input_spectrum.mean()) / std

        # Z-score distance relative to reference distribution
        ref_std = self._reference_std_spectrum.copy()
        ref_std[ref_std < 1e-10] = 1.0  # avoid division by zero
        z_scores = np.abs(input_spectrum - self._reference_mean_spectrum) / ref_std
        distance = float(np.mean(z_scores))

        return distance

    # ------------------------------------------------------------------
    # Main detection interface
    # ------------------------------------------------------------------

    def detect(
        self,
        spectrogram: np.ndarray,
        classifier: "SignalClassifier",
    ) -> OODResult:
        """
        Detect if a radio spectrogram is out-of-distribution.

        Runs the spectrogram through the classifier to obtain logits, then
        applies all three detection methods and combines them by voting.

        Args:
            spectrogram: 2D numpy array (frequency x time).
            classifier: A ``SignalClassifier`` instance used to obtain logits.

        Returns:
            OODResult with combined score, per-method scores, and anomaly flag.
        """
        # Get classifier outputs
        result = classifier.classify(spectrogram)
        logits = np.array(
            [result.all_scores[st.name.lower()] for st in _signal_types()],
            dtype=np.float32,
        )

        # Compute individual scores
        msp_score = self.compute_msp(logits)
        energy_score = self.compute_energy(logits)
        spectral_score = self.compute_spectral_distance(spectrogram)

        method_scores: Dict[str, float] = {
            "msp": msp_score,
            "energy": energy_score,
            "spectral_distance": spectral_score,
        }

        # Voting
        votes = 0
        if msp_score > (1.0 - self.msp_threshold):
            votes += 1
        if energy_score > self.energy_threshold:
            votes += 1
        if spectral_score > self.spectral_threshold:
            votes += 1

        is_anomaly = votes >= self.voting_threshold

        # Combined score (weighted average, normalised to ~[0, 1])
        combined = (
            msp_score * 2.0
            + energy_score * 0.5
            + spectral_score * 0.5
        ) / 3.0

        return OODResult(
            ood_score=combined,
            is_anomaly=is_anomaly,
            threshold=self.energy_threshold,
            method_scores=method_scores,
        )

    def detect_from_scores(
        self,
        spectrogram: np.ndarray,
        all_scores: Dict[str, float],
    ) -> OODResult:
        """Detect OOD using pre-computed class scores (avoids duplicate forward pass).

        Same logic as detect() but skips the classifier.classify() call.

        Args:
            spectrogram: 2D numpy array (frequency x time) -- used only for
                         spectral distance computation.
            all_scores: Per-class probability dictionary from a prior classify()
                        call.  Values are softmax probabilities, so we convert
                        back to log-space (approximate logits) before applying
                        MSP and energy scoring.
        """
        probs = np.nan_to_num(
            np.array(
                [all_scores.get(st.name.lower(), 0.0) for st in _signal_types()],
                dtype=np.float32,
            ),
            nan=0.0,
        )
        logits = np.log(np.clip(probs, 1e-10, 1.0))

        msp_score = self.compute_msp(logits)
        energy_score = self.compute_energy(logits)
        spectral_score = self.compute_spectral_distance(spectrogram)

        method_scores: Dict[str, float] = {
            "msp": msp_score,
            "energy": energy_score,
            "spectral_distance": spectral_score,
        }

        votes = 0
        if msp_score > (1.0 - self.msp_threshold):
            votes += 1
        if energy_score > self.energy_threshold:
            votes += 1
        if spectral_score > self.spectral_threshold:
            votes += 1

        is_anomaly = votes >= self.voting_threshold

        combined = (
            msp_score * 2.0
            + energy_score * 0.5
            + spectral_score * 0.5
        ) / 3.0

        return OODResult(
            ood_score=combined,
            is_anomaly=is_anomaly,
            threshold=self.energy_threshold,
            method_scores=method_scores,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, reference_data: List[np.ndarray]) -> None:
        """
        Calibrate the detector using a set of known in-distribution spectrograms.

        Computes reference spectral statistics (mean and std of time-averaged
        power spectra) and optionally adjusts thresholds based on the
        observed score distributions.

        Args:
            reference_data: List of 2D numpy arrays (frequency x time), each
                            representing a known in-distribution signal.
        """
        if not reference_data:
            logger.warning("calibrate() called with empty reference data")
            return

        logger.info(
            f"Calibrating OOD detector with {len(reference_data)} reference signals"
        )

        # Compute time-averaged spectra for all reference signals
        spectra: List[np.ndarray] = []
        target_freq_bins: int = reference_data[0].shape[0]

        for spec in reference_data:
            avg = spec.mean(axis=1).astype(np.float64)
            # Resize to common frequency grid if needed
            if avg.shape[0] != target_freq_bins:
                from scipy.ndimage import zoom
                avg = zoom(avg, target_freq_bins / avg.shape[0], order=1)
            # Normalise
            std = avg.std()
            if std > 0:
                avg = (avg - avg.mean()) / std
            spectra.append(avg)

        self._reference_spectra = np.stack(spectra)
        self._reference_mean_spectrum = self._reference_spectra.mean(axis=0)
        self._reference_std_spectrum = self._reference_spectra.std(axis=0)

        # Auto-calibrate spectral threshold at 95th percentile of reference distances
        ref_distances = [
            float(
                np.mean(
                    np.abs(s - self._reference_mean_spectrum)
                    / np.clip(self._reference_std_spectrum, 1e-10, None)
                )
            )
            for s in spectra
        ]
        if ref_distances:
            self.spectral_threshold = float(np.percentile(ref_distances, 95))

        self._calibrated = True
        logger.info(
            f"Calibration complete. Spectral threshold set to {self.spectral_threshold:.3f}"
        )

    @property
    def is_calibrated(self) -> bool:
        """Whether the detector has been calibrated with reference data."""
        return self._calibrated


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _signal_types():
    """Lazy import to avoid circular dependency."""
    from .signal_classifier import SignalType
    return list(SignalType)
