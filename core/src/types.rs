//! Common types shared across the MitraSETI signal-processing pipeline.
//!
//! These structures represent the fundamental data objects in a SETI
//! narrowband search: observation metadata (filterbank headers), search
//! configuration, individual signal candidates, and aggregated results.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SignalCandidate
// ---------------------------------------------------------------------------

/// A single candidate signal detected by the de-Doppler search.
///
/// In radio SETI, a "hit" is a narrowband signal that drifts in frequency
/// over time — consistent with a transmitter on another planet whose
/// relative radial velocity to Earth is changing.  Each candidate records
/// the detected frequency, drift rate, signal-to-noise ratio, and an RFI
/// (Radio Frequency Interference) score assigned by post-detection filters.
#[pyclass(get_all, set_all)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignalCandidate {
    /// Topocentric frequency of the detection at the start of the
    /// observation, in Hz.
    pub frequency_hz: f64,

    /// Observed drift rate in Hz/s.  A non-zero drift rate is the hallmark
    /// of an extraterrestrial technosignature: terrestrial RFI typically
    /// shows zero drift because the transmitter is co-rotating with the
    /// receiver.
    pub drift_rate: f64,

    /// Signal-to-noise ratio of the integrated power along the drift
    /// trajectory.  Higher SNR means a more confident detection.
    pub snr: f64,

    /// Unix timestamp (seconds) of the first time sample contributing to
    /// the detection.
    pub start_time: f64,

    /// Unix timestamp (seconds) of the last time sample contributing to
    /// the detection.
    pub end_time: f64,

    /// Estimated bandwidth of the signal in Hz.
    pub bandwidth: f64,

    /// RFI score assigned by [`crate::rfi_filter::RFIFilter`].
    /// 0.0 = almost certainly a genuine signal of interest,
    /// 1.0 = almost certainly RFI.
    pub rfi_score: f64,

    /// Whether this candidate survived all RFI rejection filters and is
    /// considered worthy of follow-up observation.
    pub is_candidate: bool,
}

#[pymethods]
impl SignalCandidate {
    /// Create a new SignalCandidate from Python.
    #[new]
    #[pyo3(signature = (frequency_hz, drift_rate, snr, start_time, end_time, bandwidth, rfi_score=0.0, is_candidate=true))]
    pub fn new(
        frequency_hz: f64,
        drift_rate: f64,
        snr: f64,
        start_time: f64,
        end_time: f64,
        bandwidth: f64,
        rfi_score: f64,
        is_candidate: bool,
    ) -> Self {
        Self {
            frequency_hz,
            drift_rate,
            snr,
            start_time,
            end_time,
            bandwidth,
            rfi_score,
            is_candidate,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SignalCandidate(freq={:.6} MHz, drift={:.4} Hz/s, SNR={:.2}, rfi={:.2}, candidate={})",
            self.frequency_hz / 1e6,
            self.drift_rate,
            self.snr,
            self.rfi_score,
            self.is_candidate,
        )
    }
}

// ---------------------------------------------------------------------------
// FilterbankHeader
// ---------------------------------------------------------------------------

/// Metadata header for a filterbank observation file.
///
/// Filterbank files (`.fil`) and their HDF5 equivalents (`.h5`) store
/// channelised radio-telescope data as a 2-D spectrogram
/// (time × frequency).  The header describes the shape and physical
/// coordinates of that spectrogram so downstream algorithms can convert
/// array indices into real-world frequencies and timestamps.
#[pyclass(get_all, set_all)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterbankHeader {
    /// Number of frequency channels.
    pub nchans: usize,

    /// Number of intermediate frequency (IF) streams (polarisations).
    pub nifs: usize,

    /// Bits per sample in the raw data (commonly 8, 16, or 32).
    pub nbits: u32,

    /// Time between successive spectra, in seconds.
    pub tsamp: f64,

    /// Centre frequency of the *first* channel, in MHz.  Together with
    /// `foff` this defines the full frequency axis.
    pub fch1: f64,

    /// Channel bandwidth in MHz (negative when channels are ordered from
    /// high to low frequency, which is the common sigproc convention).
    pub foff: f64,

    /// Start time of the observation as MJD (Modified Julian Date).
    pub tstart: f64,

    /// Name of the observed source (e.g. "Kepler-442").
    pub source_name: String,

    /// Right Ascension of the pointing, in degrees.
    pub ra: f64,

    /// Declination of the pointing, in degrees.
    pub dec: f64,
}

#[pymethods]
impl FilterbankHeader {
    #[new]
    #[pyo3(signature = (nchans, nifs, nbits, tsamp, fch1, foff, tstart, source_name, ra, dec))]
    pub fn new(
        nchans: usize,
        nifs: usize,
        nbits: u32,
        tsamp: f64,
        fch1: f64,
        foff: f64,
        tstart: f64,
        source_name: String,
        ra: f64,
        dec: f64,
    ) -> Self {
        Self {
            nchans,
            nifs,
            nbits,
            tsamp,
            fch1,
            foff,
            tstart,
            source_name,
            ra,
            dec,
        }
    }

    /// Return the frequency of channel `i` in MHz.
    pub fn channel_freq(&self, i: usize) -> f64 {
        self.fch1 + i as f64 * self.foff
    }

    /// Return the total observation bandwidth in MHz.
    pub fn total_bandwidth(&self) -> f64 {
        (self.nchans as f64 * self.foff).abs()
    }

    fn __repr__(&self) -> String {
        format!(
            "FilterbankHeader(source='{}', nchans={}, fch1={:.4} MHz, foff={:.6} MHz, tsamp={:.6} s)",
            self.source_name, self.nchans, self.fch1, self.foff, self.tsamp,
        )
    }
}

// ---------------------------------------------------------------------------
// SearchParams
// ---------------------------------------------------------------------------

/// Configuration parameters for a de-Doppler search.
///
/// These parameters control the sensitivity, speed, and RFI-handling
/// behaviour of the search algorithm.
#[pyclass(get_all, set_all)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchParams {
    /// Maximum drift rate to search, in Hz/s.  Typical SETI searches use
    /// values between 2 and 10 Hz/s — higher rates capture faster relative
    /// accelerations but increase computational cost linearly.
    pub max_drift_rate: f64,

    /// Minimum signal-to-noise ratio for a detection to be reported.
    /// Values of 10–25 are common; lower thresholds yield more candidates
    /// (including more false positives).
    pub min_snr: f64,

    /// Number of rayon worker threads for parallel processing.  Set to 0
    /// to use all available CPU cores.
    pub n_workers: usize,

    /// Whether to apply RFI rejection filters to the candidate list.
    pub rfi_rejection: bool,
}

#[pymethods]
impl SearchParams {
    #[new]
    #[pyo3(signature = (max_drift_rate=4.0, min_snr=10.0, n_workers=0, rfi_rejection=true))]
    pub fn new(max_drift_rate: f64, min_snr: f64, n_workers: usize, rfi_rejection: bool) -> Self {
        Self {
            max_drift_rate,
            min_snr,
            n_workers,
            rfi_rejection,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchParams(max_drift={:.2} Hz/s, min_snr={:.1}, workers={}, rfi={})",
            self.max_drift_rate, self.min_snr, self.n_workers, self.rfi_rejection,
        )
    }
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// Aggregated results of a de-Doppler search run.
#[pyclass(get_all)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// Candidate signals that passed all thresholds and (optionally) RFI
    /// filters.
    pub candidates: Vec<SignalCandidate>,

    /// Total number of raw threshold-crossings before RFI rejection.
    pub total_signals: usize,

    /// Number of signals removed by RFI filters.
    pub rfi_rejected: usize,

    /// Wall-clock processing time for the search, in milliseconds.
    pub processing_time_ms: u64,
}

#[pymethods]
impl SearchResult {
    #[new]
    #[pyo3(signature = (candidates, total_signals, rfi_rejected, processing_time_ms))]
    pub fn new(
        candidates: Vec<SignalCandidate>,
        total_signals: usize,
        rfi_rejected: usize,
        processing_time_ms: u64,
    ) -> Self {
        Self {
            candidates,
            total_signals,
            rfi_rejected,
            processing_time_ms,
        }
    }

    /// Serialise the result to a JSON string for interoperability.
    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(candidates={}, total={}, rfi_rejected={}, time={}ms)",
            self.candidates.len(),
            self.total_signals,
            self.rfi_rejected,
            self.processing_time_ms,
        )
    }
}
