# MitraSETI Wiki

**Intelligent SETI Signal Analysis — Rust-Accelerated Processing with Machine Learning Classification**

> **☁️ [MitraSETI Cloud](https://mitraseti.deepfieldlabs.dev)** — The managed cloud version is available. Upload your data, run automated pipelines, and visualise results in your browser. Free Explorer tier — no installation, no credit card required. Built by [DeepField Labs](https://deepfieldlabs.dev).

MitraSETI is a ground-up reimagination of the SETI signal analysis pipeline, combining a **Rust-powered de-Doppler engine** (up to 45x faster on real Breakthrough Listen data), a **CNN + Transformer classifier** that automatically rejects RFI and flags anomalies, and a **streaming observation mode** for multi-day unattended campaigns — with desktop and web interfaces for real-time monitoring.

---

## Table of Contents

### Core Documentation

| Page | Description |
|------|-------------|
| **[Architecture](Architecture)** | System overview, data flow diagrams, component interaction, threading model, and state management |
| **[Pipeline Deep Dive](Pipeline-Deep-Dive)** | Stage-by-stage walkthrough of the full processing pipeline from file ingest to result export |
| **[ML Model Architecture](ML-Model-Architecture)** | CNN + Transformer signal classifier, OOD detection ensemble, 9-class taxonomy, and training pipeline |
| **[Rust Core](Rust-Core)** | The `mitraseti-core` crate — Taylor tree de-Doppler engine, filterbank reader, RFI filter, and PyO3 bindings |

### Operations & Performance

| Page | Description |
|------|-------------|
| **[Benchmark Results](Benchmark-Results)** | Real-data benchmarks against turboSETI with methodology and analysis |
| **[Streaming Mode](Streaming-Mode)** | Continuous observation guide — auto-training, self-correcting thresholds, cadence analysis, daily reports |
| **[API Reference](API-Reference)** | Full FastAPI endpoint documentation with curl examples |
| **[CLI Reference](CLI-Reference)** | Click CLI commands — `mitraseti search`, `stream`, `benchmark`, `export`, `crossmatch`, `report` |

### v0.2.0 Modules

| Page | Description |
|------|-------------|
| **[Taylor Tree Algorithm](Taylor-Tree)** | O(N log N) de-Doppler search — implementation, theory (Taylor 1974), and performance benchmarks |
| **[RFI Management](RFI-Management)** | Spectral kurtosis, known RFI database (27 sources), ON/OFF cadence filtering |
| **[Signal Ranking](Signal-Ranking)** | HDBSCAN clustering, interestingness scoring, cross-epoch persistence tracking |
| **[AstroLens Integration](AstroLens-Integration)** | Unified sky map, astropy SkyCoord cross-matching, FITS catalog export |

### Context

| Page | Description |
|------|-------------|
| **[Comparison with turboSETI](Comparison-with-turboSETI)** | Feature-by-feature comparison, architectural differences, when to use each tool, and migration guide |

---

## Quick Links

- **GitHub Repository:** [deepfieldlabs/MitraSETI](https://github.com/deepfieldlabs/MitraSETI)
- **License:** MIT
- **Python:** 3.10+ | **Rust:** 1.70+
- **Author:** [Saman Tabatabaeian](https://www.linkedin.com/in/samantabatabaeian/) · [Deep Field Labs](https://github.com/deepfieldlabs)

---

## Demo

![MitraSETI Demo](https://raw.githubusercontent.com/deepfieldlabs/MitraSETI/main/assets/gifs/MitraSETI.gif)

---

## Screenshots

### Waterfall Viewer
![Waterfall Viewer](https://raw.githubusercontent.com/deepfieldlabs/MitraSETI/main/screenshots/waterfall_viewer.png)

### Signal Gallery
![Signal Gallery](https://raw.githubusercontent.com/deepfieldlabs/MitraSETI/main/screenshots/signal_gallery.png)

### Sky Radar
![Sky Radar](https://raw.githubusercontent.com/deepfieldlabs/MitraSETI/main/screenshots/sky_radar.png)

---

## Key Features at a Glance

- **45x faster processing** on million-channel observations via parallel Rust de-Doppler search
- **Taylor tree de-Doppler** — O(N log N) algorithm, 4.3x faster than brute-force (v0.2.0)
- **Two-stage ML classification** — rule-based filtering + CNN+Transformer inference
- **Out-of-distribution detection** — ensemble of MSP, Energy, and Spectral distance methods
- **9-class signal taxonomy** — from NARROWBAND_DRIFTING to CANDIDATE_ET
- **HDBSCAN density clustering** — replaces greedy deduplication for robust false-positive reduction (v0.2.0)
- **Adaptive spectral kurtosis** — RFI excision using median + MAD thresholds (v0.2.0)
- **Known RFI database** — 27 cataloged terrestrial interference sources (v0.2.0)
- **FITS catalog export** — standard astronomical format via astropy (v0.2.0)
- **Cross-epoch persistence** — track recurring signals across observations (v0.2.0)
- **Astropy cross-matching** — SkyCoord KD-tree matching with AstroLens optical data (v0.2.0)
- **Click CLI** — `mitraseti search`, `stream`, `benchmark`, `export`, `crossmatch`, `report` (v0.2.0)
- **Interestingness scoring** — composite ranking by SNR, drift, RFI probability, OOD score (v0.2.0)
- **Periodicity detection** — FFT-based search for periodic/pulsed signals (v0.2.0)
- **ON/OFF cadence filter** — standard SETI RFI rejection using observation patterns (v0.2.0)
- **Streaming observation mode** — multi-day campaigns with auto-training and daily HTML reports
- **Catalog cross-matching** — SIMBAD, NVSS, FIRST, and ATNF Pulsar catalogs
- **AstroLens integration** — optical + radio unified sky map
- **Desktop + Web UI** — PyQt5 desktop app and FastAPI web interface with WebSocket live streaming
- **Format support** — Sigproc `.fil` and HDF5 `.h5` (Breakthrough Listen format)
