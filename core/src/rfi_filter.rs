//! Radio Frequency Interference (RFI) rejection filters.
//!
//! The radio spectrum is crowded with terrestrial signals — GPS, WiFi,
//! cellular networks, satellite downlinks, radar, and countless other
//! emitters.  Any real SETI search must distinguish genuine
//! extraterrestrial technosignatures from this ocean of human-made
//! interference.
//!
//! This module applies a battery of heuristic and statistical tests to
//! each [`SignalCandidate`] and assigns an `rfi_score` between 0.0
//! (almost certainly not RFI) and 1.0 (almost certainly RFI).  Candidates
//! whose score exceeds a configurable threshold are removed from the
//! final result set.
//!
//! # Filters
//!
//! 1. **Known-band filter** — rejects signals whose frequency falls inside
//!    a known terrestrial allocation (GPS L1/L2, WiFi 2.4/5 GHz, …).
//! 2. **Zero-drift filter** — a transmitter on another planet would exhibit
//!    a non-zero Doppler drift; signals at exactly 0 Hz/s are suspicious.
//! 3. **Broadband filter** — genuine narrowband technosignatures are, by
//!    definition, narrow.  Very broadband detections are almost always RFI.
//! 4. **Persistence filter** — a signal present in the majority of
//!    observations is more likely to be local interference than an
//!    astrophysical transient.
//! 5. **Composite scoring** — the individual sub-scores are combined into a
//!    single `rfi_score`.

use pyo3::prelude::*;

use crate::types::{FilterbankHeader, SignalCandidate};

// ---------------------------------------------------------------------------
// Known RFI bands (frequency ranges in Hz)
// ---------------------------------------------------------------------------

/// Pre-populated list of known terrestrial RFI bands.
///
/// Each entry is `(lower_bound_hz, upper_bound_hz, label)`.
/// Sources: ITU Radio Regulations, FCC allocations, common experience from
/// Breakthrough Listen and the Green Bank Telescope.
const KNOWN_RFI_BANDS: &[(f64, f64, &str)] = &[
    // GPS
    (1_575_420_000.0, 1_575_420_000.0 + 20_000_000.0, "GPS L1"),
    (1_227_600_000.0, 1_227_600_000.0 + 20_000_000.0, "GPS L2"),
    (1_176_450_000.0, 1_176_450_000.0 + 24_000_000.0, "GPS L5"),
    // GLONASS
    (1_598_000_000.0, 1_610_000_000.0, "GLONASS L1"),
    (1_242_000_000.0, 1_252_000_000.0, "GLONASS L2"),
    // Iridium satellite downlink
    (1_616_000_000.0, 1_626_500_000.0, "Iridium"),
    // WiFi 2.4 GHz
    (2_400_000_000.0, 2_500_000_000.0, "WiFi 2.4 GHz"),
    // WiFi 5 GHz
    (5_150_000_000.0, 5_850_000_000.0, "WiFi 5 GHz"),
    // Cellular (LTE downlink bands — simplified)
    (729_000_000.0, 756_000_000.0, "LTE Band 12/17"),
    (869_000_000.0, 894_000_000.0, "LTE Band 5"),
    (1_930_000_000.0, 1_995_000_000.0, "LTE Band 2 (PCS)"),
    (2_110_000_000.0, 2_170_000_000.0, "LTE Band 1"),
    // Satellite TV (Ku-band downlink, approximate)
    (10_700_000_000.0, 12_750_000_000.0, "Ku-band sat TV"),
    // Aeronautical radar (L-band)
    (1_215_000_000.0, 1_240_000_000.0, "L-band radar"),
    // Hydrogen line — not RFI, but galactic emission can mimic a hit
    (1_420_405_751.0 - 500_000.0, 1_420_405_751.0 + 500_000.0, "HI 21 cm (galactic)"),
];

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

/// Maximum |drift rate| (Hz/s) considered "zero drift" — likely terrestrial.
const ZERO_DRIFT_THRESHOLD: f64 = 0.05;

/// Maximum bandwidth (Hz) for a plausible narrowband technosignature.
/// Anything wider than this is flagged as broadband RFI.
const MAX_NARROWBAND_BW: f64 = 500.0;

/// Overall RFI score threshold: candidates scoring above this are rejected.
const RFI_SCORE_THRESHOLD: f64 = 0.70;

// ---------------------------------------------------------------------------
// RFIFilter
// ---------------------------------------------------------------------------

/// Composite RFI rejection filter.
///
/// Holds a list of known RFI frequency bands and exposes a [`filter`]
/// method that scores and removes likely interference from a candidate
/// list.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RFIFilter {
    /// User-supplied extra RFI bands (lower Hz, upper Hz) merged with the
    /// built-in list.
    #[pyo3(get)]
    pub known_rfi_bands: Vec<(f64, f64)>,
}

#[pymethods]
impl RFIFilter {
    /// Create a new RFI filter.
    ///
    /// `extra_bands` is an optional list of `(lower_hz, upper_hz)` tuples
    /// for additional known RFI sources beyond the built-in catalogue.
    #[new]
    #[pyo3(signature = (extra_bands=None))]
    pub fn new(extra_bands: Option<Vec<(f64, f64)>>) -> Self {
        let mut bands: Vec<(f64, f64)> = KNOWN_RFI_BANDS.iter().map(|&(lo, hi, _)| (lo, hi)).collect();
        if let Some(extra) = extra_bands {
            bands.extend(extra);
        }
        Self {
            known_rfi_bands: bands,
        }
    }

    /// Score and filter a list of signal candidates.
    ///
    /// Returns only those candidates whose composite RFI score is below
    /// the rejection threshold (i.e., likely *not* RFI).  Each returned
    /// candidate has its `rfi_score` and `is_candidate` fields updated.
    #[pyo3(name = "filter")]
    pub fn py_filter(
        &self,
        candidates: Vec<SignalCandidate>,
        header: FilterbankHeader,
    ) -> Vec<SignalCandidate> {
        self.filter(&candidates, &header)
    }

    fn __repr__(&self) -> String {
        format!("RFIFilter(known_bands={})", self.known_rfi_bands.len())
    }
}

impl RFIFilter {
    /// Score and filter candidates (Rust-native interface).
    ///
    /// Each candidate is evaluated by every sub-filter.  The individual
    /// scores (each in [0, 1]) are combined via a weighted average into
    /// the final `rfi_score`.
    pub fn filter(
        &self,
        candidates: &[SignalCandidate],
        _header: &FilterbankHeader,
    ) -> Vec<SignalCandidate> {
        candidates
            .iter()
            .filter_map(|c| {
                let mut scored = c.clone();
                let score = self.composite_score(&scored);
                scored.rfi_score = score;
                scored.is_candidate = score < RFI_SCORE_THRESHOLD;

                if scored.is_candidate {
                    Some(scored)
                } else {
                    None
                }
            })
            .collect()
    }

    // ======================================================================
    // Individual sub-filters
    // ======================================================================

    /// Check whether the candidate frequency lies inside any known RFI band.
    ///
    /// Returns 1.0 if the frequency is inside a known band, 0.0 otherwise.
    fn known_band_score(&self, candidate: &SignalCandidate) -> f64 {
        let freq = candidate.frequency_hz;
        for &(lo, hi) in &self.known_rfi_bands {
            if freq >= lo && freq <= hi {
                return 1.0;
            }
        }
        0.0
    }

    /// Score based on drift rate proximity to zero.
    ///
    /// A perfectly zero drift rate almost always indicates a local (co-
    /// rotating) transmitter.  We assign a score of 1.0 for |drift| = 0
    /// that falls off linearly to 0.0 at [`ZERO_DRIFT_THRESHOLD`] Hz/s.
    fn zero_drift_score(&self, candidate: &SignalCandidate) -> f64 {
        let abs_drift = candidate.drift_rate.abs();
        if abs_drift >= ZERO_DRIFT_THRESHOLD {
            0.0
        } else {
            1.0 - (abs_drift / ZERO_DRIFT_THRESHOLD)
        }
    }

    /// Score based on signal bandwidth.
    ///
    /// Genuine extraterrestrial technosignatures are expected to be
    /// extremely narrowband (< ~10 Hz).  Signals broader than
    /// [`MAX_NARROWBAND_BW`] Hz receive a high RFI score.
    fn broadband_score(&self, candidate: &SignalCandidate) -> f64 {
        if candidate.bandwidth <= MAX_NARROWBAND_BW {
            0.0
        } else {
            ((candidate.bandwidth - MAX_NARROWBAND_BW) / MAX_NARROWBAND_BW)
                .min(1.0)
        }
    }

    /// Persistence filter placeholder.
    ///
    /// In a full pipeline this would compare the candidate against a
    /// database of previously observed signals.  If a signal appears in
    /// more than 50 % of observations it is likely persistent RFI.
    ///
    /// TODO: Implement cross-observation persistence check.  This requires
    /// access to a database or an in-memory history of prior search runs.
    fn persistence_score(&self, _candidate: &SignalCandidate) -> f64 {
        // Without cross-observation context we cannot evaluate persistence,
        // so we conservatively return 0.0 (no evidence of RFI).
        0.0
    }

    /// Combine individual sub-scores into a single composite RFI score.
    ///
    /// Weights reflect empirical importance:
    /// - Known-band match is nearly definitive (weight 0.40).
    /// - Zero drift is a strong indicator (weight 0.30).
    /// - Broadband is moderately indicative (weight 0.20).
    /// - Persistence (when implemented) rounds out the score (weight 0.10).
    fn composite_score(&self, candidate: &SignalCandidate) -> f64 {
        let w_band = 0.40;
        let w_drift = 0.30;
        let w_broad = 0.20;
        let w_persist = 0.10;

        let score = w_band * self.known_band_score(candidate)
            + w_drift * self.zero_drift_score(candidate)
            + w_broad * self.broadband_score(candidate)
            + w_persist * self.persistence_score(candidate);

        score.clamp(0.0, 1.0)
    }
}
