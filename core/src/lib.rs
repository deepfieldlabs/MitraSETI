//! # mitraseti-core
//!
//! High-performance SETI signal-processing library written in Rust with
//! Python bindings via [PyO3](https://pyo3.rs).
//!
//! ## Overview
//!
//! This crate provides the computational core of the **MitraSETI** pipeline:
//!
//! | Module        | Purpose                                                  |
//! |---------------|----------------------------------------------------------|
//! | [`dedoppler`] | De-Doppler search — detects drifting narrowband signals  |
//! | [`filterbank`]| Reads `.fil` (sigproc) and `.h5` (HDF5) spectrogram files|
//! | [`rfi_filter`]| Rejects Radio Frequency Interference from candidate lists|
//! | [`types`]     | Shared data structures (candidates, headers, params)     |
//!
//! ## Python usage
//!
//! When compiled as a `cdylib` (the default crate-type), the library
//! exposes a Python module called `mitraseti_core`:
//!
//! ```python
//! import mitraseti_core
//!
//! reader = mitraseti_core.FilterbankReader()
//! header, data, n_times, n_chans = reader.read("observation.fil")
//!
//! params = mitraseti_core.SearchParams(max_drift_rate=4.0, min_snr=10.0)
//! engine = mitraseti_core.DedopplerSearch(params)
//! result = engine.search(data, n_times, n_chans, header)
//!
//! for candidate in result.candidates:
//!     print(candidate)
//! ```

pub mod dedoppler;
pub mod filterbank;
pub mod rfi_filter;
pub mod types;

// Re-export the most commonly used items at crate root for convenience.
pub use dedoppler::DedopplerEngine;
pub use filterbank::FilterbankReader;
pub use rfi_filter::RFIFilter;
pub use types::{FilterbankHeader, SearchParams, SearchResult, SignalCandidate};

use pyo3::prelude::*;

/// The top-level Python module exposed by this crate.
///
/// Registered classes:
/// - `DedopplerSearch` — the de-Doppler search engine
/// - `FilterbankReader` — auto-detecting file reader
/// - `RFIFilter` — RFI rejection filter
/// - `SignalCandidate` — a single detection result
/// - `FilterbankHeader` — observation metadata
/// - `SearchParams` — search configuration
/// - `SearchResult` — aggregated search output
#[pymodule]
fn mitraseti_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Expose DedopplerEngine under the name "DedopplerSearch" for a
    // friendlier Python API.
    m.add_class::<DedopplerEngine>()?;
    m.add_class::<FilterbankReader>()?;
    m.add_class::<RFIFilter>()?;
    m.add_class::<SignalCandidate>()?;
    m.add_class::<FilterbankHeader>()?;
    m.add_class::<SearchParams>()?;
    m.add_class::<SearchResult>()?;

    // Module-level metadata
    m.add("__version__", "0.1.0")?;
    m.add("__doc__", "High-performance SETI signal processing core.")?;

    Ok(())
}
