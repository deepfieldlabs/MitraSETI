# Pipeline Deep Dive

The MitraSETI processing pipeline transforms raw filterbank observation data into classified, cross-referenced signal candidates. Each stage feeds into the next, with the Rust core handling compute-intensive operations and Python handling ML inference and orchestration.

```
.fil/.h5 ──→ Read ──→ De-Doppler ──→ RFI Filter ──→ Cluster ──→ Rules ──→ ML ──→ Catalog ──→ Export
              1          2              3             4          5a       5b       6           7
```

---

## Stage 1: File Reading

**Component:** `FilterbankReader` (Rust — `core/src/filterbank.rs`)

The pipeline begins by reading raw radio observation data. MitraSETI supports two formats used by Breakthrough Listen and the wider radio astronomy community.

### Sigproc `.fil` Format

Binary format with a header delimited by `HEADER_START` / `HEADER_END` markers. The Rust `SigprocReader` parses header fields (channel count, sampling time, frequency metadata) and reads the data payload into an `ndarray::Array2<f32>` with shape `(n_time_steps, n_channels)`.

Supports 8-bit, 16-bit, and 32-bit sample widths.

### HDF5 `.h5` Format

The Breakthrough Listen standard file format. Read via the `hdf5` Rust crate (when compiled with the `hdf5` feature) or through `blimpy` as a Python fallback. The data lives in a dataset named `"data"` with attributes carrying header metadata.

### Data Handling

- **3D → 2D conversion:** Files with multiple IFs (intermediate frequencies) arrive as `(n_time, n_ifs, n_chans)`. The pipeline selects IF 0 and reshapes to `(n_time, n_chans)`.
- **Large file subsetting:** To keep brute-force de-Doppler tractable, data is capped at ~16 million data points. Larger files are subsetted by selecting a frequency range from the center of the band.
- **Fallback chain:** Rust reader → blimpy → error. The pipeline always tries the fastest path first.

### Output

`(FilterbankHeader, Array2<f32>)` — header metadata and a 2D power array ready for de-Doppler search.

---

## Stage 2: De-Doppler Search

**Component:** `DedopplerEngine` (Rust — `core/src/dedoppler.rs`)

The de-Doppler search is the most compute-intensive stage. It searches for narrowband signals that drift in frequency over time due to the relative acceleration between a transmitter and Earth.

### Algorithm

1. **Noise estimation:** For each frequency channel, compute the median and MAD (Median Absolute Deviation) across time steps.

2. **Normalization:** Convert raw power to SNR:
   ```
   snr[t][f] = (power[t][f] - median[f]) / (1.4826 × MAD[f])
   ```

3. **Drift-rate enumeration:** Generate evenly spaced trial drift rates from `-max_drift` to `+max_drift`, where the step size is determined by the channel resolution and observation duration.

4. **Parallel integration:** For each drift rate (parallelized via rayon), integrate the normalized power along the corresponding diagonal through the time-frequency plane. If the integrated SNR exceeds `min_snr`, record a detection.

5. **Peak detection:** Identify local maxima in the integrated SNR across frequency for each drift rate.

6. **Clustering:** Merge detections within ±5 kHz in frequency and ±0.5 Hz/s in drift rate, keeping the highest-SNR detection per cluster.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_drift_rate` | 4.0 Hz/s | Maximum drift rate to search |
| `min_snr` | 10.0 | Minimum SNR threshold |
| `n_workers` | all cores | Rayon thread pool size |

### Performance

The rayon parallel iterator distributes drift-rate trials across all available CPU cores. On a 16-core machine processing a million-channel observation, this yields the 45x speedup over turboSETI's single-threaded Python loop.

> **Future:** Taylor tree algorithm (O(N log N) vs current brute-force O(N²)) is on the v0.2.0 roadmap for an additional order-of-magnitude improvement.

### Output

`Vec<SignalCandidate>` — each with `frequency_hz`, `drift_rate`, `snr`, `start_time`, `end_time`, `bandwidth`.

---

## Stage 3: RFI Filtering

**Component:** `RFIFilter` (Rust — `core/src/rfi_filter.rs`)

Radio frequency interference dominates real observation data. The RFI filter applies four independent sub-filters, combines their scores, and rejects signals above a composite threshold.

### Sub-Filters

| Filter | Weight | What It Catches |
|--------|--------|-----------------|
| **Known-band** | 0.40 | Signals in documented RFI bands (GPS L1/L2/L5, GLONASS, Iridium, WiFi 2.4/5 GHz, LTE, satellite TV, radar, HI 21cm) |
| **Zero-drift** | 0.30 | Signals with \|drift\| < 0.05 Hz/s — likely terrestrial origin (no Doppler shift from relative motion) |
| **Broadband** | 0.20 | Signals with bandwidth > 500 Hz — real ET signals are expected to be narrowband |
| **Persistence** | 0.10 | Signals that appear consistently across multiple observations (placeholder for cross-observation tracking) |

### Composite Score

```
rfi_score = 0.40 × band_score + 0.30 × drift_score + 0.20 × broadband_score + 0.10 × persistence_score
```

Signals with `rfi_score > 0.70` are rejected.

### Known RFI Bands

The known-band database includes major interference sources:

- **Navigation:** GPS L1 (1575.42 MHz), L2 (1227.60 MHz), L5 (1176.45 MHz), GLONASS (1602.0 MHz)
- **Communication:** Iridium (1616–1626.5 MHz), WiFi 2.4 GHz, WiFi 5 GHz, LTE bands
- **Broadcast:** Satellite TV (10.7–12.75 GHz)
- **Science:** HI 21cm line (1420.405 MHz) — not RFI, but flagged for special handling
- **Radar:** Various military and weather radar bands

### Output

Filtered `Vec<SignalCandidate>` with `rfi_score` populated. Rejected signals are counted for statistics but not passed forward.

---

## Stage 4: Hit Clustering

**Component:** `_cluster_hits()` (Python — `pipeline.py`)

After RFI filtering, many remaining hits are duplicate detections of the same physical signal at slightly different frequencies or drift rates.

### Algorithm

1. Sort hits by SNR (descending)
2. For each hit, check if any existing cluster center is within tolerance:
   - Frequency tolerance: ±64 channels
   - Drift rate tolerance: ±0.5 Hz/s
3. If within tolerance, merge into existing cluster (keep higher-SNR detection)
4. Otherwise, start a new cluster

### Output

Deduplicated list of `SignalCandidate`, one per physical signal, with the highest SNR representative kept for each cluster.

---

## Stage 5: Two-Stage Classification

The classification system is designed for real-world data where >99% of detections are RFI. Running ML inference on every detection would be wasteful. Instead, a fast rule-based filter eliminates the obvious cases, and ML runs only on the survivors.

### Stage 5a: Rule-Based Filter

**Component:** Stage 1 classifier (Python — `pipeline.py`)

A fast pass over all signals that runs in microseconds per signal:

| Check | Criteria | Rationale |
|-------|----------|-----------|
| Drift rate range | 0.05–10.0 Hz/s | Too low = terrestrial, too high = unphysical |
| SNR thresholds | ≥25 for high, ≥50 for exceptional | Prioritizes strong signals |
| Zero-drift rejection | \|drift\| < 0.05 Hz/s | No Doppler shift = not an ET transmitter in relative motion |
| Boundary artifacts | drift ≥ 98% of max | Edge effects in the de-Doppler search |

Signals that fail any check are classified as RFI and excluded from ML inference. This eliminates the vast majority of detections.

### Stage 5b: ML Inference

**Component:** `SignalClassifier` (Python — `inference/signal_classifier.py`)

Survivors from Stage 1 pass through the CNN + Transformer model (see [ML Model Architecture](ML-Model-Architecture) for full details):

1. **Spectrogram extraction:** 256×64 patch centered on the candidate frequency
2. **Batch inference:** Sub-batches of 128 through the model
3. **Classification:** 9-class probability distribution + confidence score
4. **Feature extraction:** SNR, drift rate, bandwidth, spectral index, kurtosis, skewness
5. **OOD detection:** Ensemble score using logits from the same forward pass (no duplicate computation)
6. **Spectrogram caching:** Saved for future auto-training in streaming mode

Only the top 500 candidates by SNR are sent to ML inference to cap computation time.

### Stage 5c: OOD Detection

**Component:** `RadioOODDetector` (Python — `inference/ood_detector.py`)

Signals that don't match any of the 9 training classes are flagged rather than forced into the nearest bucket. The OOD detector uses an ensemble of three methods:

- **Maximum Softmax Probability (MSP):** Low max probability → likely OOD
- **Energy Score:** Free energy of the logit vector
- **Spectral Distance:** Z-score distance from training distribution in feature space

An anomaly is flagged when ≥2 methods agree.

### Output

Each surviving candidate receives: `signal_type`, `confidence`, `rfi_probability`, `ood_score`, `is_anomaly`, `feature_vector` (128-dim).

---

## Stage 6: Catalog Cross-Reference

**Component:** `RadioCatalogQuery` (Python — `catalog/radio_catalogs.py`)

Candidates with sky coordinates (RA/Dec) are cross-referenced against four astronomical catalogs and the AstroLens optical anomaly database.

### Catalogs

| Catalog | Coverage | Query Method |
|---------|----------|--------------|
| **SIMBAD** | All known astronomical objects | TAP/ADQL via astroquery |
| **NVSS** (NRAO VLA Sky Survey) | 1.4 GHz radio continuum, δ > −40° | VizieR TAP |
| **FIRST** (Faint Images of the Radio Sky at Twenty-cm) | 1.4 GHz, high-resolution | VizieR TAP |
| **ATNF Pulsar Catalogue** | Known pulsars worldwide | Web API |

### Features

- **24-hour caching:** Results cached to `mitraseti_artifacts/data/catalog_cache/` to reduce API load
- **Angular separation:** Vincenty formula for accurate distance calculation
- **Composite check:** `is_known_source()` queries all catalogs and returns the closest match
- **Configurable radius:** Default search radius of 5 arcminutes

### AstroLens Integration

Signals near optical anomalies detected by [AstroLens](https://github.com/SamanTabworlds/astroLens) receive additional cross-reference data. A narrowband drifting signal coincident with an optically anomalous star is far more interesting than one near a known satellite.

---

## Stage 7: Result Export

Results flow to multiple outputs simultaneously:

### JSON Output

Full pipeline results as a structured JSON file:

```json
{
  "file_info": {
    "path": "observation.fil",
    "channels": 1048576,
    "time_steps": 16
  },
  "candidates": [...],
  "timing": {
    "dedoppler_ms": 45,
    "rfi_filter_ms": 12,
    "ml_inference_ms": 230,
    "total_ms": 320
  },
  "summary": {
    "raw_hits": 847,
    "rfi_rejected": 824,
    "ml_classified": 23,
    "candidates": 1,
    "anomalies": 0
  }
}
```

### SQLite Database

Signals, observations, and candidates stored in async SQLite for querying through the API and UI.

### HTML Reports

Daily streaming reports with charts, candidate rankings, and processing statistics saved to `mitraseti_artifacts/streaming_reports/`.

### WebSocket Broadcast

During processing, live updates are broadcast to connected WebSocket clients at `/ws/live` for real-time monitoring in the web UI.

---

## Pipeline Summary Table

| Stage | Component | Technology | Typical Time (Voyager-1) |
|-------|-----------|------------|--------------------------|
| 1. File Reading | `FilterbankReader` | Rust, h5py, blimpy | ~5 ms |
| 2. De-Doppler | `DedopplerEngine` | Rust, rayon | ~30 ms |
| 3. RFI Filtering | `RFIFilter` | Rust | ~5 ms |
| 4. Clustering | `_cluster_hits` | Python | <1 ms |
| 5a. Rule Filter | Stage 1 | Python | <1 ms |
| 5b. ML Inference | `SignalClassifier` | PyTorch | ~15 ms |
| 5c. OOD Detection | `RadioOODDetector` | PyTorch | ~2 ms |
| 6. Catalog Xref | `RadioCatalogQuery` | astroquery, HTTP | ~100 ms (cached) |
| 7. Export | DB + JSON + WS | aiosqlite, JSON | ~5 ms |
| **Total** | | | **~60 ms** |
