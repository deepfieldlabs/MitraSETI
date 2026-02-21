//! Filterbank and HDF5 file readers for radio-telescope spectrogram data.
//!
//! Radio SETI observations are typically stored in one of two formats:
//!
//! - **Sigproc filterbank** (`.fil`) — a simple binary format with a
//!   self-describing header followed by raw spectral data.
//! - **HDF5** (`.h5`) — the Breakthrough Listen variant stores the same
//!   data in an HDF5 container with the spectrogram in a dataset called
//!   `"data"` and metadata as HDF5 attributes.
//!
//! This module provides a [`FilterbankIO`] trait and two concrete
//! implementations ([`SigprocReader`] and [`Hdf5Reader`]) so that
//! additional formats can be added in the future.  The high-level
//! [`FilterbankReader`] struct auto-detects the format from the file
//! extension and delegates to the appropriate backend.

use std::fs::File;
use std::io::{BufReader, Read as IoRead};
use std::path::Path;

use ndarray::Array2;
use pyo3::prelude::*;

use crate::types::FilterbankHeader;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur while reading filterbank or HDF5 files.
#[derive(Debug, thiserror::Error)]
pub enum FilterbankError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("invalid filterbank header: {0}")]
    BadHeader(String),

    #[error("HDF5 error: {0}")]
    Hdf5(String),

    #[error("data shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Abstract interface for reading a spectrogram and its header from a file.
///
/// Implement this trait to add support for new file formats.
pub trait FilterbankIO {
    /// Read the file at `path` and return the header plus a 2-D array of
    /// shape `(n_timesteps, n_channels)` in `f32` power units.
    fn read(&self, path: &str) -> Result<(FilterbankHeader, Array2<f32>), FilterbankError>;
}

// ---------------------------------------------------------------------------
// Sigproc .fil reader
// ---------------------------------------------------------------------------

/// Reader for the classic sigproc filterbank binary format.
///
/// The format starts with the string `"HEADER_START"`, followed by a
/// sequence of keyword–value pairs, and terminated by `"HEADER_END"`.
/// The remainder of the file is raw spectral data in channel-major order.
pub struct SigprocReader;

impl SigprocReader {
    /// Read a null-terminated keyword string from the buffer.
    fn read_string(reader: &mut BufReader<File>) -> Result<String, FilterbankError> {
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;

        if len > 1024 {
            return Err(FilterbankError::BadHeader(format!(
                "keyword length {} exceeds sanity limit",
                len
            )));
        }

        let mut str_buf = vec![0u8; len];
        reader.read_exact(&mut str_buf)?;
        Ok(String::from_utf8_lossy(&str_buf).to_string())
    }

    /// Read a single `f64` value from the buffer.
    fn read_f64(reader: &mut BufReader<File>) -> Result<f64, FilterbankError> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    /// Read a single `i32` value from the buffer.
    fn read_i32(reader: &mut BufReader<File>) -> Result<i32, FilterbankError> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }
}

impl FilterbankIO for SigprocReader {
    fn read(&self, path: &str) -> Result<(FilterbankHeader, Array2<f32>), FilterbankError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // --- Parse header --------------------------------------------------
        let magic = Self::read_string(&mut reader)?;
        if magic != "HEADER_START" {
            return Err(FilterbankError::BadHeader(
                "file does not start with HEADER_START".into(),
            ));
        }

        let mut nchans: usize = 0;
        let mut nifs: usize = 1;
        let mut nbits: u32 = 32;
        let mut tsamp: f64 = 0.0;
        let mut fch1: f64 = 0.0;
        let mut foff: f64 = 0.0;
        let mut tstart: f64 = 0.0;
        let mut source_name = String::new();
        let mut ra: f64 = 0.0;
        let mut dec: f64 = 0.0;

        loop {
            let keyword = Self::read_string(&mut reader)?;
            match keyword.as_str() {
                "HEADER_END" => break,
                "nchans" => nchans = Self::read_i32(&mut reader)? as usize,
                "nifs" => nifs = Self::read_i32(&mut reader)? as usize,
                "nbits" => nbits = Self::read_i32(&mut reader)? as u32,
                "tsamp" => tsamp = Self::read_f64(&mut reader)?,
                "fch1" => fch1 = Self::read_f64(&mut reader)?,
                "foff" => foff = Self::read_f64(&mut reader)?,
                "tstart" => tstart = Self::read_f64(&mut reader)?,
                "source_name" => source_name = Self::read_string(&mut reader)?,
                "src_raj" => ra = Self::read_f64(&mut reader)?,
                "src_dej" => dec = Self::read_f64(&mut reader)?,
                // Skip unknown keywords — they may be ints or doubles.
                // In a production implementation we would use the sigproc
                // keyword table to decide the value type.  For now we
                // attempt to skip 8 bytes (most values are doubles).
                _ => {
                    // TODO: Use the official sigproc keyword→type mapping to
                    // decide whether to skip 4 or 8 bytes.  For now we
                    // optimistically try 8 (f64).
                    let mut skip = [0u8; 8];
                    let _ = reader.read_exact(&mut skip);
                }
            }
        }

        if nchans == 0 {
            return Err(FilterbankError::BadHeader("nchans is 0".into()));
        }

        let header = FilterbankHeader {
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
        };

        // --- Read spectral data --------------------------------------------
        let bytes_per_sample = (nbits / 8) as usize;
        let mut raw_bytes = Vec::new();
        reader.read_to_end(&mut raw_bytes)?;

        let n_samples = raw_bytes.len() / bytes_per_sample;
        let n_times = n_samples / nchans;

        let data_f32: Vec<f32> = match nbits {
            8 => raw_bytes.iter().map(|&b| b as f32).collect(),
            16 => raw_bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
                .collect(),
            32 => raw_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            _ => {
                return Err(FilterbankError::BadHeader(format!(
                    "unsupported nbits={}",
                    nbits
                )))
            }
        };

        let array = Array2::from_shape_vec((n_times, nchans), data_f32).map_err(|e| {
            FilterbankError::ShapeMismatch {
                expected: format!("({}, {})", n_times, nchans),
                actual: e.to_string(),
            }
        })?;

        Ok((header, array))
    }
}

// ---------------------------------------------------------------------------
// HDF5 .h5 reader
// ---------------------------------------------------------------------------

/// HDF5 reader is available when compiled with the `hdf5-support` feature.
/// For standard use, .h5 files are read via Python's h5py library instead.
#[cfg(feature = "hdf5-support")]
pub struct Hdf5Reader;

#[cfg(feature = "hdf5-support")]
impl FilterbankIO for Hdf5Reader {
    fn read(&self, path: &str) -> Result<(FilterbankHeader, Array2<f32>), FilterbankError> {
        let file = hdf5::File::open(path).map_err(|e| FilterbankError::Hdf5(e.to_string()))?;
        let root = file.group("/").map_err(|e| FilterbankError::Hdf5(e.to_string()))?;

        let nchans = read_attr_usize(&root, "nchans")?;
        let nifs = read_attr_usize(&root, "nifs").unwrap_or(1);
        let nbits = read_attr_u32(&root, "nbits").unwrap_or(32);
        let tsamp = read_attr_f64(&root, "tsamp")?;
        let fch1 = read_attr_f64(&root, "fch1")?;
        let foff = read_attr_f64(&root, "foff")?;
        let tstart = read_attr_f64(&root, "tstart")?;
        let source_name = read_attr_string(&root, "source_name").unwrap_or_default();
        let ra = read_attr_f64(&root, "src_raj").unwrap_or(0.0);
        let dec = read_attr_f64(&root, "src_dej").unwrap_or(0.0);

        let header = FilterbankHeader { nchans, nifs, nbits, tsamp, fch1, foff, tstart, source_name, ra, dec };

        let dataset = file.dataset("data").map_err(|e| FilterbankError::Hdf5(e.to_string()))?;
        let flat: Vec<f32> = dataset.read_raw().map_err(|e| FilterbankError::Hdf5(e.to_string()))?;
        let n_times = flat.len() / (nifs * nchans);
        let data: Vec<f32> = if nifs == 1 { flat } else {
            flat.chunks(nifs * nchans).flat_map(|frame| frame[..nchans].iter().copied()).collect()
        };

        let array = Array2::from_shape_vec((n_times, nchans), data).map_err(|e| {
            FilterbankError::ShapeMismatch { expected: format!("({}, {})", n_times, nchans), actual: e.to_string() }
        })?;
        Ok((header, array))
    }
}

#[cfg(feature = "hdf5-support")]
fn read_attr_f64(group: &hdf5::Group, name: &str) -> Result<f64, FilterbankError> {
    let attr = group.attr(name).map_err(|e| FilterbankError::Hdf5(format!("missing attribute '{}': {}", name, e)))?;
    attr.read_scalar::<f64>().map_err(|e| FilterbankError::Hdf5(format!("cannot read '{}' as f64: {}", name, e)))
}

#[cfg(feature = "hdf5-support")]
fn read_attr_usize(group: &hdf5::Group, name: &str) -> Result<usize, FilterbankError> {
    read_attr_f64(group, name).map(|v| v as usize)
}

#[cfg(feature = "hdf5-support")]
fn read_attr_u32(group: &hdf5::Group, name: &str) -> Result<u32, FilterbankError> {
    read_attr_f64(group, name).map(|v| v as u32)
}

#[cfg(feature = "hdf5-support")]
fn read_attr_string(group: &hdf5::Group, name: &str) -> Result<String, FilterbankError> {
    let attr = group.attr(name).map_err(|e| FilterbankError::Hdf5(format!("missing attribute '{}': {}", name, e)))?;
    attr.read_scalar::<hdf5::types::VarLenUnicode>()
        .map(|s| s.to_string())
        .or_else(|_| attr.read_scalar::<hdf5::types::FixedUnicode<256>>().map(|s| s.to_string()))
        .map_err(|e| FilterbankError::Hdf5(format!("cannot read '{}' as string: {}", name, e)))
}

// ---------------------------------------------------------------------------
// High-level reader with format auto-detection (Python-exposed)
// ---------------------------------------------------------------------------

/// Auto-detecting filterbank reader.
///
/// Given a file path, [`FilterbankReader`] inspects the extension (`.fil`
/// or `.h5`) and delegates to the appropriate backend.  This is the
/// primary interface exposed to Python.
#[pyclass]
#[derive(Clone, Debug)]
pub struct FilterbankReader;

#[pymethods]
impl FilterbankReader {
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Read a filterbank file and return `(header, flat_data, n_times, n_chans)`.
    ///
    /// The spectrogram is returned as a flat `Vec<f32>` in row-major order
    /// so it can easily be reshaped on the Python side with numpy.
    #[pyo3(name = "read")]
    pub fn py_read(
        &self,
        path: &str,
    ) -> PyResult<(FilterbankHeader, Vec<f32>, usize, usize)> {
        let (header, array) = self
            .read(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let (n_times, n_chans) = array.dim();
        let flat = array.into_raw_vec();
        Ok((header, flat, n_times, n_chans))
    }

    fn __repr__(&self) -> String {
        "FilterbankReader()".to_string()
    }
}

impl FilterbankReader {
    /// Read a filterbank or HDF5 file and return the header and data array.
    pub fn read(&self, path: &str) -> Result<(FilterbankHeader, Array2<f32>), FilterbankError> {
        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match ext {
            "fil" => SigprocReader.read(path),
            #[cfg(feature = "hdf5-support")]
            "h5" | "hdf5" => Hdf5Reader.read(path),
            #[cfg(not(feature = "hdf5-support"))]
            "h5" | "hdf5" => Err(FilterbankError::UnsupportedFormat(
                "HDF5 support not compiled in; use Python h5py instead".to_string(),
            )),
            other => Err(FilterbankError::UnsupportedFormat(other.to_string())),
        }
    }
}
