//! De-Doppler search engine — the heart of narrowband SETI signal detection.
//!
//! # Background
//!
//! A transmitter on a distant planet will be Doppler-shifted with respect to
//! an Earth-based receiver.  Because the relative radial velocity changes over
//! the course of an observation (due to Earth's rotation and the target's
//! orbital motion), the received frequency *drifts* linearly (to first order)
//! with time.  The *de-Doppler* search algorithm tries every plausible drift
//! rate and, for each one, integrates the spectrogram power along that
//! diagonal trajectory.  A genuine drifting signal will add coherently and
//! stand out above the noise, while random noise integrates incoherently and
//! grows only as √N.
//!
//! # Algorithm overview
//!
//! 1. **Noise estimation** — compute per-channel median and MAD (median
//!    absolute deviation) to establish the noise floor.
//! 2. **Normalisation** — convert raw power to SNR units:
//!    `snr[t][f] = (power[t][f] - median[f]) / (1.4826 * MAD[f])`
//! 3. **Drift-rate sweep** — for each candidate drift rate from −`max_drift`
//!    to +`max_drift` (in steps set by the channel resolution and integration
//!    time), integrate power along that diagonal.  This step is embarrassingly
//!    parallel and executed with **rayon**.
//! 4. **Peak detection** — find frequency channels where the integrated SNR
//!    exceeds `min_snr`.
//! 5. **Clustering** — merge nearby detections that belong to the same
//!    physical signal.
//! 6. **Result** — return candidates sorted by descending SNR.
//!
//! ## Taylor Tree (future)
//!
//! A naïve drift-rate search is O(N_d × N_t × N_f) where N_d is the number
//! of drift rates, N_t the number of time steps, and N_f the number of
//! frequency channels.  The **Taylor tree** algorithm (Taylor 1974) reduces
//! this to O(N_d × N_f × log₂ N_t) by recursively combining partial sums —
//! analogous to an FFT butterfly.  The placeholder in this module marks where
//! the tree should be inserted.

use std::time::Instant;

use log::info;
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::rfi_filter::RFIFilter;
use crate::types::{FilterbankHeader, SearchParams, SearchResult, SignalCandidate};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during a de-Doppler search.
#[derive(Debug, thiserror::Error)]
pub enum DedopplerError {
    #[error("input spectrogram has zero-length axis (shape: {0}×{1})")]
    EmptyData(usize, usize),

    #[error("drift-rate step is zero or negative — check header.tsamp and header.foff")]
    InvalidDriftStep,

    #[error("search parameters invalid: {0}")]
    BadParams(String),
}

// ---------------------------------------------------------------------------
// DedopplerEngine
// ---------------------------------------------------------------------------

/// The main de-Doppler search engine.
///
/// Construct with a [`SearchParams`] and call [`DedopplerEngine::search`] on a
/// spectrogram to obtain a [`SearchResult`] containing any candidate signals.
#[pyclass]
#[derive(Clone, Debug)]
pub struct DedopplerEngine {
    /// Configuration controlling drift-rate range, SNR threshold, etc.
    params: SearchParams,
}

#[pymethods]
impl DedopplerEngine {
    /// Create a new engine with the given search parameters.
    #[new]
    #[pyo3(signature = (params=None))]
    pub fn new(params: Option<SearchParams>) -> Self {
        Self {
            params: params.unwrap_or_else(|| SearchParams::new(4.0, 10.0, 0, true)),
        }
    }

    /// Run the de-Doppler search from Python.
    ///
    /// `data` is a flattened (row-major) spectrogram with shape
    /// `(n_timesteps, n_channels)`.  Returns a [`SearchResult`].
    #[pyo3(name = "search")]
    pub fn py_search(
        &self,
        data: Vec<f32>,
        n_times: usize,
        n_chans: usize,
        header: FilterbankHeader,
    ) -> PyResult<SearchResult> {
        let array = Array2::from_shape_vec((n_times, n_chans), data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.search(&array, &header)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("DedopplerEngine({:?})", self.params)
    }
}

impl DedopplerEngine {
    /// Run the de-Doppler search on a 2-D spectrogram.
    ///
    /// # Arguments
    /// * `data`   — `Array2<f32>` with shape `(n_timesteps, n_channels)`.
    /// * `header` — observation metadata used to convert indices to physical
    ///              units.
    ///
    /// # Returns
    /// A [`SearchResult`] with all candidates found (optionally RFI-filtered).
    pub fn search(
        &self,
        data: &Array2<f32>,
        header: &FilterbankHeader,
    ) -> Result<SearchResult, DedopplerError> {
        let start = Instant::now();

        let (n_times, n_chans) = data.dim();
        if n_times == 0 || n_chans == 0 {
            return Err(DedopplerError::EmptyData(n_times, n_chans));
        }

        info!(
            "Starting de-Doppler search: {}×{} spectrogram, max_drift={:.2} Hz/s, min_snr={:.1}",
            n_times, n_chans, self.params.max_drift_rate, self.params.min_snr,
        );

        // -- configure rayon thread pool -----------------------------------
        if self.params.n_workers > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.params.n_workers)
                .build_global()
                .ok(); // ignore if already initialised
        }

        // ------------------------------------------------------------------
        // Step 1 & 2: Noise estimation and normalisation
        // ------------------------------------------------------------------
        let normalised = self.normalise(data, n_times, n_chans);

        // ------------------------------------------------------------------
        // Step 3: Enumerate candidate drift rates
        // ------------------------------------------------------------------
        let drift_rates = self.enumerate_drift_rates(header, n_times)?;
        info!("Searching {} drift rates", drift_rates.len());

        // ------------------------------------------------------------------
        // Step 4: Parallel drift-rate search
        // ------------------------------------------------------------------
        let raw_candidates: Vec<SignalCandidate> = drift_rates
            .par_iter()
            .flat_map(|&drift_rate| {
                self.search_single_drift(&normalised, header, drift_rate, n_times, n_chans)
            })
            .collect();

        info!("Raw threshold crossings: {}", raw_candidates.len());

        // ------------------------------------------------------------------
        // Step 5: Cluster nearby detections
        // ------------------------------------------------------------------
        let clustered = Self::cluster_candidates(raw_candidates, header);

        // ------------------------------------------------------------------
        // Step 6 (optional): RFI rejection
        // ------------------------------------------------------------------
        let total_signals = clustered.len();
        let candidates = if self.params.rfi_rejection {
            let rfi = RFIFilter::new(None);
            rfi.filter(&clustered, header)
        } else {
            clustered
        };
        let rfi_rejected = total_signals - candidates.len();

        let elapsed = start.elapsed().as_millis() as u64;
        info!(
            "Search complete in {} ms — {} candidates ({} RFI-rejected)",
            elapsed,
            candidates.len(),
            rfi_rejected,
        );

        Ok(SearchResult {
            candidates,
            total_signals,
            rfi_rejected,
            processing_time_ms: elapsed,
        })
    }

    // ======================================================================
    // Internal helpers
    // ======================================================================

    /// Compute per-channel median and MAD, then return an SNR-normalised copy
    /// of the spectrogram.
    fn normalise(&self, data: &Array2<f32>, n_times: usize, n_chans: usize) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros((n_times, n_chans));

        for ch in 0..n_chans {
            // Collect the column for this channel.
            let mut col: Vec<f32> = (0..n_times).map(|t| data[[t, ch]]).collect();
            col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = col[col.len() / 2];

            // MAD = median(|x_i - median|)
            let mut abs_devs: Vec<f32> = col.iter().map(|&v| (v - median).abs()).collect();
            abs_devs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = abs_devs[abs_devs.len() / 2];

            // σ ≈ 1.4826 × MAD  (for a Gaussian distribution)
            let sigma = 1.4826 * mad;
            let sigma = if sigma < f32::EPSILON { 1.0 } else { sigma };

            for t in 0..n_times {
                out[[t, ch]] = (data[[t, ch]] - median) / sigma;
            }
        }

        out
    }

    /// Build the list of drift rates to search, evenly spaced between
    /// ±`max_drift_rate`.  The step size is determined by the frequency and
    /// time resolution of the observation.
    fn enumerate_drift_rates(
        &self,
        header: &FilterbankHeader,
        n_times: usize,
    ) -> Result<Vec<f64>, DedopplerError> {
        // Drift-rate resolution: one channel width over the full obs length.
        // Δν / T_obs  (both in Hz / s)
        let obs_length = header.tsamp * n_times as f64;
        let channel_bw_hz = header.foff.abs() * 1e6; // foff is in MHz
        if obs_length <= 0.0 || channel_bw_hz <= 0.0 {
            return Err(DedopplerError::InvalidDriftStep);
        }
        let drift_step = channel_bw_hz / obs_length;

        let n_steps = (self.params.max_drift_rate / drift_step).ceil() as i64;
        let rates: Vec<f64> = (-n_steps..=n_steps)
            .map(|i| i as f64 * drift_step)
            .collect();

        Ok(rates)
    }

    /// Integrate power along a single drift-rate trajectory and return any
    /// peaks above the SNR threshold.
    ///
    /// For each frequency channel, the algorithm walks through time and
    /// accumulates the normalised power at the channel offset predicted by
    /// the drift rate.  This is the *brute-force* version; a Taylor tree
    /// would be significantly faster for large data.
    fn search_single_drift(
        &self,
        data: &Array2<f32>,
        header: &FilterbankHeader,
        drift_rate: f64,
        n_times: usize,
        n_chans: usize,
    ) -> Vec<SignalCandidate> {
        // TODO: Replace this brute-force integration with a Taylor tree.
        //
        // The Taylor tree (Taylor 1974) works as follows:
        //   - Layer 0: the raw spectrogram rows.
        //   - Layer k: each row is the sum of two rows from layer k-1,
        //     with one of them shifted by 2^(k-1) channels.
        //   - After log₂(N_t) layers the final row contains the fully
        //     integrated power for each starting channel and total drift.
        //
        // This reduces the O(N_t) per-channel sum to O(log₂ N_t) additions
        // reused across adjacent drift rates.

        let channel_bw_hz = header.foff.abs() * 1e6;
        let mut candidates = Vec::new();

        for start_ch in 0..n_chans {
            let mut integrated_snr: f64 = 0.0;

            for t in 0..n_times {
                // Predicted channel offset at time step t.
                let dt = header.tsamp * t as f64;
                let freq_offset_hz = drift_rate * dt;
                let chan_offset = (freq_offset_hz / channel_bw_hz).round() as isize;

                let ch = start_ch as isize + chan_offset;
                if ch < 0 || ch >= n_chans as isize {
                    break; // trajectory has left the band
                }
                integrated_snr += data[[t, ch as usize]] as f64;
            }

            // Normalise by √N_t (noise grows as √N under incoherent addition).
            let snr = integrated_snr / (n_times as f64).sqrt();

            if snr >= self.params.min_snr as f64 {
                let freq_hz = (header.fch1 + start_ch as f64 * header.foff) * 1e6;
                let obs_length = header.tsamp * n_times as f64;

                candidates.push(SignalCandidate {
                    frequency_hz: freq_hz,
                    drift_rate,
                    snr,
                    start_time: header.tstart,
                    end_time: header.tstart + obs_length,
                    bandwidth: channel_bw_hz,
                    rfi_score: 0.0,
                    is_candidate: true,
                });
            }
        }

        candidates
    }

    /// Cluster detections that are close in frequency and drift rate.
    ///
    /// Multiple drift-rate trials will often trigger on the same physical
    /// signal.  We keep only the highest-SNR detection within each cluster
    /// (defined as detections within ±5 channels and ±1 drift-rate step).
    fn cluster_candidates(
        mut candidates: Vec<SignalCandidate>,
        _header: &FilterbankHeader,
    ) -> Vec<SignalCandidate> {
        if candidates.is_empty() {
            return candidates;
        }

        // Sort by frequency then by descending SNR.
        candidates.sort_by(|a, b| {
            a.frequency_hz
                .partial_cmp(&b.frequency_hz)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    b.snr
                        .partial_cmp(&a.snr)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        let mut clustered: Vec<SignalCandidate> = Vec::new();

        for c in candidates {
            let dominated = clustered.iter().any(|existing| {
                // Two detections belong to the same cluster if they are
                // within 5 kHz and 0.5 Hz/s of each other.
                let freq_close = (existing.frequency_hz - c.frequency_hz).abs() < 5_000.0;
                let drift_close = (existing.drift_rate - c.drift_rate).abs() < 0.5;
                freq_close && drift_close
            });

            if !dominated {
                clustered.push(c);
            }
        }

        // Final sort: descending SNR (best candidates first).
        clustered.sort_by(|a, b| {
            b.snr
                .partial_cmp(&a.snr)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        clustered
    }
}
