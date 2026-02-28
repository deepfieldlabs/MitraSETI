# Rust Core

The `mitraseti-core` crate (`core/`) implements the compute-intensive stages of the MitraSETI pipeline in Rust. It is not a thin wrapper — it contains the de-Doppler search engine, filterbank file reader, and RFI filter, all exposed to Python through PyO3 bindings with zero-copy data transfer.

---

## Why Rust

MitraSETI chose Rust for its core processing engine for specific, measurable reasons:

### Memory Safety Without Garbage Collection

Filterbank files routinely exceed 1 GB. Rust's ownership model ensures zero-copy data handling and deterministic memory usage. There are no GC pauses during long processing runs — critical for streaming observation mode where the pipeline runs continuously for days.

### Zero-Cost Abstractions

Iterator combinators, pattern matching, and trait dispatch compile down to the same machine code as hand-written loops. The code remains expressive and maintainable without sacrificing performance.

### rayon Parallelism

The `rayon` crate provides work-stealing parallel iterators that distribute computation across all CPU cores with a single-line change from sequential to parallel. The de-Doppler drift-rate search, which is embarrassingly parallel, gains near-linear speedup with core count.

### Ecosystem

Rust's crate ecosystem provides high-quality, zero-dependency alternatives to C libraries:

| Crate | Purpose | Replaces |
|-------|---------|----------|
| `rayon` 1.10 | Data-parallel iterators | OpenMP / multiprocessing |
| `ndarray` 0.16 | N-dimensional arrays | NumPy (C-backed) |
| `rustfft` 6.2 | FFT engine | FFTW (C library) |
| `pyo3` 0.22 | Python bindings | ctypes / CFFI |
| `hdf5` 0.8 | HDF5 file I/O (optional) | h5py (C-backed) |
| `serde` / `serde_json` | Serialization | json module |
| `thiserror` | Structured errors | — |

---

## Crate Structure

```
core/
├── Cargo.toml          # Dependencies and feature flags
└── src/
    ├── lib.rs           # PyO3 module registration
    ├── types.rs         # Shared data structures
    ├── filterbank.rs    # File readers (.fil, .h5)
    ├── dedoppler.rs     # De-Doppler search engine
    └── rfi_filter.rs    # RFI rejection filters
```

### `lib.rs` — Module Registration

Registers the `mitraseti_core` Python module via PyO3 and exposes all public types:

- `DedopplerEngine` — drift-rate search
- `FilterbankReader` — file I/O
- `RFIFilter` — interference rejection
- `SignalCandidate`, `FilterbankHeader`, `SearchParams`, `SearchResult` — data types

### `types.rs` — Data Structures

Core data types shared across all modules:

**`SignalCandidate`** — a detected signal:

| Field | Type | Description |
|-------|------|-------------|
| `frequency_hz` | `f64` | Center frequency in Hz |
| `drift_rate` | `f64` | Drift rate in Hz/s |
| `snr` | `f64` | Signal-to-noise ratio |
| `start_time` | `f64` | Start time (seconds) |
| `end_time` | `f64` | End time (seconds) |
| `bandwidth` | `f64` | Signal bandwidth in Hz |
| `rfi_score` | `f64` | Composite RFI score (0.0–1.0) |
| `is_candidate` | `bool` | Passed all filters |

**`FilterbankHeader`** — file metadata:

| Field | Type | Description |
|-------|------|-------------|
| `nchans` | `u32` | Number of frequency channels |
| `nifs` | `u32` | Number of intermediate frequencies |
| `nbits` | `u32` | Bits per sample (8, 16, or 32) |
| `tsamp` | `f64` | Sampling time in seconds |
| `fch1` | `f64` | Frequency of channel 1 in MHz |
| `foff` | `f64` | Channel bandwidth in MHz |
| `tstart` | `f64` | MJD of first sample |
| `source_name` | `String` | Source name from header |
| `ra` | `f64` | Right ascension |
| `dec` | `f64` | Declination |

**`SearchParams`** — de-Doppler configuration:

| Field | Type | Default |
|-------|------|---------|
| `max_drift_rate` | `f64` | 4.0 Hz/s |
| `min_snr` | `f64` | 10.0 |
| `n_workers` | `usize` | all cores |
| `rfi_rejection` | `bool` | true |

---

## DedopplerEngine

The de-Doppler engine (`core/src/dedoppler.rs`) is the most performance-critical component. It searches for narrowband signals drifting in frequency over time.

### Algorithm Detail

```
for each drift_rate in [-max_drift .. +max_drift]:     ← rayon parallel
    for each channel in [0 .. n_channels]:
        integrated_snr = 0
        for each time_step in [0 .. n_time]:
            shifted_channel = channel + drift_rate × time_step / channel_width
            integrated_snr += normalized_power[time_step][shifted_channel]
        if integrated_snr > min_snr:
            record(channel, drift_rate, integrated_snr)
```

### Noise Estimation

Per-channel noise is estimated using the median and MAD (Median Absolute Deviation), which are robust to outliers (unlike mean/std):

```
median[f] = median_over_time(power[t][f])
mad[f]    = median_over_time(|power[t][f] - median[f]|)
snr[t][f] = (power[t][f] - median[f]) / (1.4826 × mad[f])
```

The factor 1.4826 normalizes MAD to be consistent with standard deviation for Gaussian distributions.

### Parallelization

The drift-rate loop is parallelized using rayon's `par_iter()`:

```rust
drift_rates.par_iter()
    .flat_map(|&drift_rate| {
        // Each thread independently integrates along its assigned drift rate
        search_at_drift_rate(&normalized, drift_rate, min_snr, &header)
    })
    .collect()
```

Each drift rate is independent — no synchronization, no shared mutable state. This achieves near-linear scaling with core count.

### Post-Processing

After the parallel search, results are clustered to merge duplicate detections:

- Frequency tolerance: ±5 kHz
- Drift rate tolerance: ±0.5 Hz/s
- Keep highest SNR per cluster

> **Roadmap:** The brute-force algorithm is O(N²) in channel count. The Taylor tree algorithm (v0.2.0) will reduce this to O(N log N) for an additional order-of-magnitude speedup.

---

## FilterbankReader

The filterbank reader (`core/src/filterbank.rs`) supports two radio astronomy file formats through a common `FilterbankIO` trait.

### Sigproc `.fil` Reader

Reads the binary Sigproc filterbank format:

1. **Header parsing:** Scans for `HEADER_START` and `HEADER_END` markers. Between them, reads key-value pairs where each key is a Pascal-style string (length prefix + characters) followed by the value.

2. **Data reading:** After the header, raw samples are read according to `nbits` (8, 16, or 32 bit) and arranged into an `Array2<f32>` with shape `(n_time_steps, n_channels)`.

3. **Channel order:** Handles both ascending and descending frequency order (determined by the sign of `foff`).

### HDF5 `.h5` Reader

Reads the Breakthrough Listen HDF5 format (requires the `hdf5` feature flag):

1. Opens the HDF5 file and reads the `"data"` dataset
2. Extracts header attributes (channel count, sampling time, frequencies)
3. Returns the same `(FilterbankHeader, Array2<f32>)` tuple as the Sigproc reader

### Format Auto-Detection

`FilterbankReader` inspects the file extension and delegates to the appropriate reader:

```
.fil → SigprocReader
.h5  → Hdf5Reader (if compiled with hdf5 feature)
      → Error (if hdf5 feature not enabled)
```

Python-side fallback to `blimpy` handles edge cases where the Rust reader fails or the format is unusual.

---

## RFIFilter

The RFI filter (`core/src/rfi_filter.rs`) applies four independent sub-filters and combines their scores with configurable weights.

### Known-Band Filter (weight: 0.40)

Checks signal frequency against a database of known interference sources:

| Band | Frequency Range | Source |
|------|----------------|--------|
| GPS L1 | 1575.42 ± 10 MHz | Navigation satellite |
| GPS L2 | 1227.60 ± 10 MHz | Navigation satellite |
| GPS L5 | 1176.45 ± 10 MHz | Navigation satellite |
| GLONASS | 1602.0 ± 10 MHz | Navigation satellite |
| Iridium | 1616–1626.5 MHz | Communication satellite |
| WiFi 2.4 GHz | 2400–2500 MHz | Terrestrial wireless |
| WiFi 5 GHz | 5150–5850 MHz | Terrestrial wireless |
| LTE | Various bands | Cellular network |
| Satellite TV | 10.7–12.75 GHz | Broadcast satellite |
| HI 21cm | 1420.405 ± 1 MHz | Hydrogen line (special) |
| Radar | Various | Military/weather radar |

Returns 1.0 if the signal falls within a known band, 0.0 otherwise.

### Zero-Drift Filter (weight: 0.30)

Signals with |drift rate| < 0.05 Hz/s are likely terrestrial — a true ET signal from a rotating/orbiting body would exhibit non-zero Doppler drift. Score scales inversely with drift rate magnitude.

### Broadband Filter (weight: 0.20)

Signals with bandwidth > 500 Hz receive a high broadband score. ET signals are expected to be narrowband (concentrated in a few Hz), while broadband emission is typically natural astrophysical sources or wideband RFI.

### Persistence Filter (weight: 0.10)

Placeholder for cross-observation persistence scoring. Signals that appear in every observation (regardless of pointing direction) are likely local RFI. Currently returns 0.0; cross-observation tracking is planned for v0.2.0.

### Composite Scoring

```
rfi_score = 0.40 × band + 0.30 × zero_drift + 0.20 × broadband + 0.10 × persistence
```

Threshold: **rfi_score > 0.70 → rejected**

---

## PyO3 Bindings

The Rust core is exposed to Python through [PyO3](https://pyo3.rs/), Rust's mature Python binding framework.

### Zero-Copy Data Transfer

NumPy arrays pass between Python and Rust without serialization. PyO3's `numpy` integration maps `ndarray::Array2<f32>` directly to/from NumPy `ndarray` using shared memory:

```rust
#[pyfunction]
fn process(data: PyReadonlyArray2<f32>) -> PyResult<Vec<SignalCandidate>> {
    let array = data.as_array();  // Zero-copy view into NumPy memory
    // ... process ...
}
```

### Exposed Classes

All major components are annotated with `#[pyclass]` and their methods with `#[pymethods]`:

```python
import mitraseti_core

engine = mitraseti_core.DedopplerEngine()
reader = mitraseti_core.FilterbankReader()
rfi    = mitraseti_core.RFIFilter()

header, data = reader.read("observation.fil")

params = mitraseti_core.SearchParams(
    max_drift_rate=4.0,
    min_snr=10.0,
)

result = engine.search(data, header, params)

filtered = rfi.filter(result.candidates)
```

### Building

The Rust extension is built using [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
maturin develop --release    # Development build with optimizations
maturin build --release      # Create wheel for distribution
```

The `--release` flag is important — debug builds are significantly slower due to disabled optimizations and enabled bounds checking.

---

## Performance Characteristics

### Parallel Scaling

The de-Doppler search scales near-linearly with CPU core count:

```
Cores    Relative speedup (Voyager-1, 1M channels)
  1      1.0x    (baseline)
  4      3.8x
  8      7.2x
 16     13.5x
```

### Memory Usage

- Filterbank data: `n_time × n_channels × 4 bytes` (f32)
- Normalized SNR array: same size as data
- Candidates: negligible (typically <1000 entries)
- Total: approximately `2 × data_size` peak memory

### Key Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `ZERO_DRIFT_THRESHOLD` | 0.05 Hz/s | Below this, signal considered stationary |
| `MAX_NARROWBAND_BW` | 500 Hz | Above this, signal considered broadband |
| `RFI_SCORE_THRESHOLD` | 0.70 | Composite score above this → rejected |
| Cluster frequency tolerance | ±5 kHz | Merge detections within this range |
| Cluster drift tolerance | ±0.5 Hz/s | Merge detections within this range |

---

## Cargo.toml

```toml
[package]
name = "mitraseti-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "mitraseti_core"
crate-type = ["cdylib"]

[dependencies]
rayon = "1.10"           # Data-parallel iterators
ndarray = "0.16"         # N-dimensional arrays
rustfft = "6.2"          # FFT engine
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"           # NumPy interop for PyO3
serde = { version = "1", features = ["derive"] }
serde_json = "1"
log = "0.4"
thiserror = "1"

[dependencies.hdf5]
version = "0.8"
optional = true

[features]
default = []
hdf5 = ["dep:hdf5"]
```
