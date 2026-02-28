<p align="center">
  <img src="assets/mitraseti_logo.png" alt="MitraSETI Logo" width="200">
</p>

<h1 align="center">MitraSETI</h1>

<p align="center">
  <strong>Intelligent SETI Signal Analysis — Rust-Accelerated Processing with Machine Learning Classification</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.70%2B-orange?style=flat-square&logo=rust&logoColor=white" alt="Rust 1.70+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT"></a>
  <a href="https://pypi.org/project/mitraseti/"><img src="https://img.shields.io/badge/version-0.1.0-purple?style=flat-square" alt="Version 0.1.0"></a>
  <a href="https://github.com/deepfieldlabs/MitraSETI/stargazers"><img src="https://img.shields.io/github/stars/deepfieldlabs/MitraSETI?style=flat-square&color=yellow" alt="GitHub Stars"></a>
</p>

---

The Search for Extraterrestrial Intelligence generates terabytes of radio observation data, yet the standard analysis tool — [turboSETI](https://github.com/UCBerkeleySETI/turbo_seti) — is a pure-Python, batch-only pipeline with no machine learning, no RFI learning, and no way to distinguish a drifting ET signal from a drifting satellite. MitraSETI replaces this bottleneck with a **Rust-powered de-Doppler engine** (up to **45x faster** on real Breakthrough Listen data), a **CNN + Transformer classifier** that automatically rejects RFI and flags anomalies, and a **streaming observation mode** that can run multi-day campaigns unattended — complete with desktop and web interfaces for real-time monitoring.

---

## Key Features

- **45x faster processing** on million-channel observations via parallel Rust de-Doppler search with rayon
- **Two-stage ML classification** — rule-based filtering eliminates 99%+ obvious RFI; CNN+Transformer inference runs only on survivors
- **Out-of-distribution detection** — ensemble of MSP, Energy, and Spectral distance methods flags anomalous signals
- **9-class signal taxonomy** — NARROWBAND_DRIFTING, NARROWBAND_STATIONARY, BROADBAND, PULSED, CHIRP, RFI_TERRESTRIAL, RFI_SATELLITE, NOISE, CANDIDATE_ET
- **Streaming observation mode** — continuous multi-day campaigns with auto-training, self-correcting thresholds, and daily HTML reports
- **Catalog cross-matching** — SIMBAD, NVSS, FIRST, and ATNF Pulsar catalogs with 24-hour caching
- **AstroLens integration** — optical + radio cross-reference for multi-modal discovery
- **Desktop + Web UI** — PyQt5 desktop app and FastAPI + Jinja2 web interface with animated starfield and glass theme
- **Format support** — Sigproc `.fil` and HDF5 `.h5` (Breakthrough Listen format) with blimpy fallback

---

## Benchmark

Measured on real Breakthrough Listen observation data. No synthetic files, no cherry-picked runs.

| Dataset | File | Channels | Time Steps | MitraSETI | turboSETI | Speedup |
|---|---|---|---|---|---|---|
| **Voyager-1** | `.h5` (48 MB) | 1,048,576 | 16 | **0.06 s** | 2.53 s | **45x** |
| **TRAPPIST-1** | `.fil` (14 MB) | 65,536 | — | **0.43–0.54 s** | 0.08–0.64 s | comparable |

**Detection comparison on Voyager-1:**

| Tool | Detections | Notes |
|---|---|---|
| turboSETI | 3 hits, SNR 245.7 | Raw hit list, no classification |
| MitraSETI | 1 candidate | ML-classified, OOD-scored, RFI-filtered |

turboSETI reports raw hits. MitraSETI reports *classified candidates* — signals that survived rule-based filtering, CNN+Transformer inference, and out-of-distribution analysis. One high-confidence candidate is more actionable than three unscreened hits.

> Benchmarks run on an Apple Silicon workstation. Rust core compiled with `--release`. turboSETI 2.x via pip.

---

## Pipeline

<p align="center">
  <img src="assets/mitraseti_pipeline.png" alt="MitraSETI Pipeline" width="900">
</p>

---

## Architecture

```
                         ┌──────────────────────────────┐
                         │    Filterbank  /  HDF5 Input  │
                         │      (.fil / .h5 files)       │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │           RUST CORE (mitraseti-core)  │
                    │                                       │
                    │  ┌─────────────┐  ┌────────────────┐  │
                    │  │ Filterbank  │  │  De-Doppler    │  │
                    │  │ Reader      │──│  Engine        │  │
                    │  │ .fil / .h5  │  │  rayon ∥       │  │
                    │  └─────────────┘  └───────┬────────┘  │
                    │                           │           │
                    │                  ┌────────▼────────┐  │
                    │                  │   RFI Filter    │  │
                    │                  │   known-band,   │  │
                    │                  │   zero-drift,   │  │
                    │                  │   broadband,    │  │
                    │                  │   persistence   │  │
                    │                  └────────┬────────┘  │
                    └──────────────────────────┼────────────┘
                                               │
                          ┌────────────────────▼────────────────────┐
                          │          PYTHON ML LAYER                │
                          │                                        │
                          │  Stage 1: Rule-Based Filter (all hits) │
                          │              │                         │
                          │              ▼ survivors only           │
                          │  Stage 2: CNN+Transformer Inference    │
                          │           + OOD Detection              │
                          └────────────────────┬───────────────────┘
                                               │
                     ┌─────────────────────────┼─────────────────────────┐
                     │                         │                         │
                     ▼                         ▼                         ▼
          ┌────────────────┐       ┌────────────────┐       ┌────────────────┐
          │    Catalog     │       │   AstroLens    │       │    Signal      │
          │  Cross-Match   │       │   Optical      │       │    Export      │
          │  SIMBAD/NVSS/  │       │   Cross-Ref    │       │  JSON / DB     │
          │  FIRST/Pulsar  │       │                │       │                │
          └───────┬────────┘       └───────┬────────┘       └───────┬────────┘
                  └────────────────────────┼────────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │       UI LAYER          │
                              │  Desktop (PyQt5)        │
                              │  Web (FastAPI + Jinja2)  │
                              │  WebSocket live stream   │
                              └─────────────────────────┘
```

---

## Pipeline Deep Dive

| Stage | Component | What It Does | Technology |
|---|---|---|---|
| **1. Ingest** | `FilterbankReader` | Reads `.fil` (Sigproc) and `.h5` (HDF5/BL) files; handles 3D data, large-file subsetting, blimpy fallback | Rust (`core/src/filterbank.rs`), h5py, blimpy |
| **2. De-Doppler** | `DedopplerEngine` | Brute-force drift-rate search across all channels; finds narrowband signals drifting due to relative acceleration | Rust (`core/src/dedoppler.rs`), rayon parallelism |
| **3. RFI Rejection** | `RFIFilter` | Eliminates known-band interference, zero-drift signals, broadband events; persistence scoring across observations | Rust (`core/src/rfi_filter.rs`) |
| **4. Clustering** | `_cluster_hits` | Deduplicates nearby hits in frequency/drift space; keeps highest-SNR per cluster | Python (pipeline.py) |
| **5a. Rule Filter** | Stage 1 classifier | Fast pass over all signals: checks drift rate range, SNR thresholds, boundary artifacts; eliminates 99%+ obvious RFI | Python (pipeline.py) |
| **5b. ML Inference** | Stage 2 classifier | CNN+Transformer on surviving candidates; produces 9-class probabilities, confidence, RFI probability | PyTorch (`inference/signal_classifier.py`) |
| **5c. OOD Detection** | `RadioOODDetector` | Ensemble of MSP + Energy + Spectral distance; flags signals outside training distribution as anomalies | PyTorch (`inference/ood_detector.py`) |
| **6. Features** | `FeatureExtractor` | Extracts SNR, drift rate, bandwidth, spectral index, kurtosis, skewness per signal | NumPy/SciPy (`inference/feature_extractor.py`) |
| **7. Catalog** | `RadioCatalogQuery` | Cross-references coordinates against SIMBAD, NVSS, FIRST, ATNF Pulsar catalogs | astroquery (`catalog/radio_catalogs.py`) |
| **8. Storage** | `SignalDB` | Async SQLite storage for signals, observations, and candidates | aiosqlite (`api/database.py`) |

---

## Rust Core Advantages

The `mitraseti-core` crate (`core/`) is not a thin wrapper — it implements the compute-intensive stages of the pipeline in Rust for concrete, measurable gains:

**Parallelism with rayon.** The de-Doppler search distributes drift-rate trials across all available CPU cores. On a 16-core machine processing a million-channel observation, this alone accounts for the 45x speedup over turboSETI's single-threaded Python loop.

**Memory safety without garbage collection.** Filterbank files routinely exceed 1 GB. Rust's ownership model ensures zero-copy data handling and deterministic memory usage — no GC pauses during long processing runs.

**Zero-copy Python interop via PyO3.** The Rust core exposes `DedopplerEngine`, `RFIFilter`, `FilterbankReader`, and data types directly to Python through PyO3 bindings. NumPy arrays pass through without serialization overhead.

**FFT via rustfft.** Spectral analysis uses the pure-Rust `rustfft` crate, avoiding C library dependencies while maintaining competitive throughput.

**ndarray for tensor operations.** The Rust equivalent of NumPy, providing familiar n-dimensional array semantics with SIMD-friendly memory layout.

```toml
# core/Cargo.toml — key dependencies
rayon = "1.10"        # data-parallel iterators
ndarray = "0.16"      # n-dimensional arrays
rustfft = "6.2"       # FFT engine
pyo3 = "0.22"         # Python bindings
hdf5 = "0.8"          # HDF5 support (optional feature)
```

---

## ML Model Architecture

### CNN + Transformer Hybrid

MitraSETI's classifier (`inference/signal_classifier.py`) uses a **SpectralCNNBackbone** for frequency-domain feature extraction combined with a **2-layer, 4-head Transformer encoder** for temporal pattern recognition:

```
Input spectrogram (256 freq × 64 time)
    │
    ▼
Conv1d backbone (spectral features along frequency axis)
    │
    ▼
Transformer encoder (2 layers, 4 heads — temporal patterns)
    │
    ▼
MLP classification head → 9 classes
```

**Signal classes:**

| Class | Description |
|---|---|
| `NARROWBAND_DRIFTING` | Narrowband signal with non-zero drift rate — the primary ET signature |
| `NARROWBAND_STATIONARY` | Narrowband, zero drift — likely local RFI |
| `BROADBAND` | Wideband emission |
| `PULSED` | Periodic pulsed signal |
| `CHIRP` | Frequency-swept signal |
| `RFI_TERRESTRIAL` | Terrestrial radio frequency interference |
| `RFI_SATELLITE` | Satellite downlink interference |
| `NOISE` | Background noise, no signal present |
| `CANDIDATE_ET` | Passes all filters — requires human review |

### Two-Stage Classification

The two-stage design is critical for performance on real data where >99% of detections are RFI:

1. **Stage 1 (rule-based, all signals):** Checks drift rate range (0.05–10.0 Hz/s), SNR thresholds, drift boundary artifacts, and zero-drift patterns. Runs in microseconds per signal. Eliminates the vast majority.

2. **Stage 2 (ML, survivors only):** Extracts 256×64 spectrograms around each candidate frequency, runs batch CNN+Transformer inference, and applies OOD detection. Only invoked on the small fraction that pass Stage 1.

### Out-of-Distribution Detection

Signals that don't match any of the 9 training classes are flagged rather than forced into the nearest bucket. The OOD detector uses an ensemble of three methods:

- **Maximum Softmax Probability (MSP):** Low max probability → likely OOD
- **Energy Score:** Free energy of the logit vector
- **Spectral Distance:** Distance from training distribution in feature space

### Trained Model

| File | Size | Description |
|---|---|---|
| `signal_classifier_v1.pt` | ~3 MB | CNN+Transformer weights trained on Breakthrough Listen data |
| `ood_calibration.json` | ~5 MB | OOD detector calibration thresholds |

Inference supports **MPS** (Apple Silicon), **CUDA** (NVIDIA), and **CPU** backends automatically.

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Rust 1.70+** (install via [rustup](https://rustup.rs/))
- **maturin** (for building the Rust extension)

### Installation

<details>
<summary><strong>macOS</strong></summary>

```bash
git clone https://github.com/deepfieldlabs/MitraSETI.git
cd MitraSETI

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Build the Rust core (requires Rust toolchain)
pip install maturin
maturin develop --release

# Install MitraSETI in editable mode
pip install -e .
```

</details>

<details>
<summary><strong>Linux</strong></summary>

```bash
# Install system dependencies (Debian/Ubuntu)
sudo apt-get update
sudo apt-get install -y python3-dev libhdf5-dev pkg-config

git clone https://github.com/deepfieldlabs/MitraSETI.git
cd MitraSETI

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Build the Rust core
pip install maturin
maturin develop --release

pip install -e .
```

</details>

<details>
<summary><strong>Windows</strong></summary>

```powershell
git clone https://github.com/deepfieldlabs/MitraSETI.git
cd MitraSETI

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

# Build the Rust core (requires Rust toolchain from rustup.rs)
pip install maturin
maturin develop --release

pip install -e .
```

</details>

### Verify Installation

```bash
python -c "import mitraseti_core; print('Rust core loaded')"
python -c "from inference.signal_classifier import SignalClassifier; print('ML layer ready')"
```

---

## Web Interface

MitraSETI includes a FastAPI-powered web interface with interactive signal browsing, sky map, reports, and catalog cross-reference lookups. The web UI runs standalone — no separate API process is needed.

```bash
# Start the web server
cd MitraSETI
python -m web.app --port 9090

# Open in browser
open http://localhost:9090
```

Pages: **Dashboard** · **Waterfall Viewer** · **Signal Gallery** · **RFI Dashboard** · **Sky Map** · **Streaming Monitor** · **Reports** · **About**

The REST API (for programmatic access) is also available:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Swagger docs: http://localhost:8000/docs
```

---

## Desktop App

The PyQt5 desktop application provides a native interface with waterfall viewer, signal gallery, sky map, RFI dashboard, and streaming observation panel.

```bash
python main.py
```

<p align="center">
  <img src="screenshots/waterfall_viewer.png" alt="Waterfall Viewer" width="900"><br>
  <em>Waterfall Viewer — spectrogram display with drift-rate overlay and zoom controls</em>
</p>

<p align="center">
  <img src="screenshots/signal_gallery.png" alt="Signal Gallery" width="900"><br>
  <em>Signal Gallery — browse detections with ML classification and confidence scores</em>
</p>

<p align="center">
  <img src="screenshots/sky_radar.png" alt="Sky Radar" width="900"><br>
  <em>Sky Radar — interactive celestial map with radio + optical source overlay</em>
</p>

The desktop app includes:
- **Waterfall Viewer** — zoom, pan, drift line overlay, ON/OFF comparison
- **Signal Gallery** — browse detections with ML classification badges and confidence scores
- **Sky Map Panel** — interactive sky map with AstroLens optical overlay
- **RFI Dashboard** — real-time rejection statistics
- **Streaming Panel** — control and monitor continuous observation runs

---

## CLI Usage

Process filterbank files directly from the command line:

```bash
# Process a single file
python pipeline.py observation.fil

# Process multiple files
python pipeline.py data/*.fil data/*.h5

# Specify model weights and output
python pipeline.py observation.h5 \
    --model models/signal_classifier_v1.pt \
    --ood-cal models/ood_calibration.json \
    --json-output results.json

# Process with custom database
python pipeline.py observation.fil --db results.sqlite
```

**Example output:**

```
============================================================
File: Voyager1.h5
  Status:     success
  Raw hits:   847
  Filtered:   23
  RFI:        19
  Candidates: 1
  Anomalies:  0
  Time:       0.060s

  Top candidates:
    freq=8419921066.000000 Hz  drift=0.3928 Hz/s  SNR=245.7  class=narrowband_drifting  conf=0.982
```

---

## Docker

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# Process files via the containerized API
curl -X POST http://localhost:8000/process \
    -F "file=@observation.fil" \
    -F "source_name=Voyager-1"
```

The Docker setup includes the Rust core pre-compiled with HDF5 support, all Python dependencies, and the web server.

---

## Streaming Mode

MitraSETI's streaming observation engine runs multi-day campaigns unattended, continuously processing filterbank files as they arrive.

```bash
# Run a 7-day continuous observation
python scripts/streaming_observation.py --days 7

# Aggressive mode (lower SNR thresholds, more candidates)
python scripts/streaming_observation.py --days 3 --mode aggressive

# Generate a report from existing data without processing
python scripts/streaming_observation.py --report-only

# Reset state and start fresh
python scripts/streaming_observation.py --reset
```

### Streaming Features

- **Auto-training:** Initial model training at 5 files, fine-tuning every 500 processing cycles
- **Self-correcting:** Adjusts SNR thresholds based on candidate rates — too many candidates triggers stricter filtering, too few relaxes it
- **ON-OFF cadence analysis:** ABACAD pattern RFI rejection for source/reference observation pairs
- **Daily HTML reports:** Charts, candidate rankings, processing statistics, and performance metrics saved to `mitraseti_artifacts/streaming_reports/`
- **State persistence:** Saves/resumes across restarts via JSON state files
- **Health monitoring:** Tracks API status, disk space, and model state

---

## API Reference

MitraSETI exposes a REST API via FastAPI. Key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health: GPU status, models loaded, disk space |
| `POST` | `/process` | Upload and process a `.fil` / `.h5` file through the full pipeline |
| `GET` | `/signals` | List detected signals with filters (SNR, drift rate, classification, RFI score) |
| `GET` | `/signals/{id}` | Get a single signal by ID |
| `PATCH` | `/signals/{id}` | Update signal classification, verification status, or notes |
| `GET` | `/candidates` | List signals promoted to ET candidate status, ordered by SNR |
| `GET` | `/stats` | Aggregate processing statistics across all observations |
| `GET` | `/catalog/crossref` | Cross-reference coordinates against SIMBAD, NVSS, FIRST, ATNF Pulsar catalogs |
| `GET` | `/astrolens/crossref` | Check for AstroLens optical anomalies near a radio position |
| `WS` | `/ws/live` | WebSocket stream for real-time signal updates during processing |

### Example: Process and Query

```bash
# Upload a file for processing
curl -X POST http://localhost:8000/process \
    -F "file=@data/voyager1.h5" \
    -F "source_name=Voyager-1" \
    -F "ra=286.86" \
    -F "dec=12.17"

# List candidates with SNR > 20
curl "http://localhost:8000/signals?min_snr=20&is_candidate=true"

# Cross-reference a detection against radio catalogs
curl "http://localhost:8000/catalog/crossref?ra=286.86&dec=12.17&radius_arcmin=5"
```

---

## Comparison with turboSETI

MitraSETI is **not** a fork of turboSETI — it is a ground-up reimagination of the SETI signal analysis pipeline, built for the era of machine learning and massive data volumes.

<p align="center">
  <img src="assets/mitraseti_comparison.png" alt="MitraSETI vs turboSETI" width="800">
</p>

| Capability | turboSETI | MitraSETI |
|---|---|---|
| **Core language** | Pure Python | Rust core + Python ML layer |
| **Parallelism** | Single-threaded | Multi-core via rayon (all CPU cores) |
| **Processing speed** | 2.53 s (Voyager-1 H5) | **0.06 s (45x faster)** |
| **ML classification** | None | CNN + Transformer (9 classes) |
| **RFI handling** | Manual SNR threshold | Learned rejection + rule-based + persistence scoring |
| **Out-of-distribution** | None | Ensemble OOD detection (MSP + Energy + Spectral) |
| **Signal taxonomy** | Hit / no hit | 9 classes with confidence scores |
| **Streaming mode** | Batch only | Multi-day continuous with auto-training |
| **Self-correcting** | No | Adjusts thresholds based on candidate rates |
| **ON-OFF cadence** | External script | Built-in ABACAD pattern analysis |
| **Optical cross-ref** | None | AstroLens integration |
| **Catalog matching** | Basic | SIMBAD, NVSS, FIRST, ATNF Pulsar (24h cache) |
| **Web interface** | None | FastAPI + Jinja2 + WebSocket live stream |
| **Desktop app** | None | PyQt5 with waterfall viewer, sky map, gallery |
| **API** | None | REST API with Swagger docs |
| **Daily reports** | None | Auto-generated HTML with charts |

**The intelligence layer is the key difference.** turboSETI finds signals. MitraSETI *understands* them — automatically classifying, scoring, cross-referencing, and prioritizing so researchers focus on what matters.

---

## AstroLens Integration

MitraSETI includes first-class integration with [AstroLens](https://github.com/deepfieldlabs/astroLens), enabling a unique **optical + radio cross-reference** workflow:

1. **Detect** narrowband signals in radio filterbank data with MitraSETI
2. **Cross-reference** signal coordinates against AstroLens optical catalog via the `/astrolens/crossref` API
3. **Overlay** optical imagery on the interactive sky map in the desktop app
4. **Correlate** radio detections with known optical sources, transients, and anomalies
5. **Flag** signals near stars, galaxies, or optical events that lack obvious radio counterparts

This multi-modal approach opens discovery pathways that single-wavelength analysis cannot achieve. A narrowband drifting signal coincident with an optically anomalous star is far more interesting than one coincident with a known satellite.

---

## What's Coming Next

**v0.2.0 Roadmap:**

| Feature | Description |
|---|---|
| **Taylor tree de-Doppler** | O(N log N) algorithm replacing brute-force O(N²) — order-of-magnitude speedup on large files |
| **CLI tool** | Standalone `mitraseti` command for headless operation on HPC clusters |
| **REST API extensions** | Batch upload, observation scheduling, webhook notifications |
| **Pre-trained model zoo** | Models trained on GBT, Parkes, MeerKAT, and ATA datasets |
| **GPU-accelerated de-Doppler** | CUDA and Metal compute shaders for the de-Doppler search |
| **Real-time SDR input** | Direct ingestion from software-defined radios (RTL-SDR, USRP) |
| **Cloud deployment** | AWS / GCP Terraform modules for scalable processing |

---

## Contributing

Contributions are welcome. Whether it's a bug fix, new feature, or documentation improvement:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/taylor-tree-dedoppler`)
3. **Commit** your changes (`git commit -m 'Implement Taylor tree de-Doppler'`)
4. **Push** to the branch (`git push origin feature/taylor-tree-dedoppler`)
5. **Open** a Pull Request

Please make sure to:
- Write tests for new features (`pytest` with the existing test suite in `tests/`)
- Follow the existing code style (enforced via `ruff`)
- Update documentation as needed

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <img src="assets/deepfieldlabs_profile_768.png" alt="Deep Field Labs" width="80" style="border-radius: 50%;">
  <br>
  <sub>Created by <a href="https://www.linkedin.com/in/samantabatabaeian/">Saman Tabatabaeian</a> · <a href="https://github.com/deepfieldlabs">Deep Field Labs</a></sub>
  <br><br>
  <strong>Built for the search for extraterrestrial intelligence.</strong>
  <br>
  If you find this useful, please <strong>star the repo</strong> — it helps the project reach more researchers.
</p>
