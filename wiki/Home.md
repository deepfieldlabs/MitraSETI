# MitraSETI Wiki

**Intelligent SETI Signal Analysis — Rust-Accelerated Processing with Machine Learning Classification**

MitraSETI is a ground-up reimagination of the SETI signal analysis pipeline, combining a **Rust-powered de-Doppler engine** (up to 45x faster on real Breakthrough Listen data), a **CNN + Transformer classifier** that automatically rejects RFI and flags anomalies, and a **streaming observation mode** for multi-day unattended campaigns — with desktop and web interfaces for real-time monitoring.

---

## Table of Contents

### Core Documentation

| Page | Description |
|------|-------------|
| **[Architecture](Architecture)** | System overview, data flow diagrams, component interaction, threading model, and state management |
| **[Pipeline Deep Dive](Pipeline-Deep-Dive)** | Stage-by-stage walkthrough of the full processing pipeline from file ingest to result export |
| **[ML Model Architecture](ML-Model-Architecture)** | CNN + Transformer signal classifier, OOD detection ensemble, 9-class taxonomy, and training pipeline |
| **[Rust Core](Rust-Core)** | The `mitraseti-core` crate — de-Doppler engine, filterbank reader, RFI filter, and PyO3 bindings |

### Operations & Performance

| Page | Description |
|------|-------------|
| **[Benchmark Results](Benchmark-Results)** | Real-data benchmarks against turboSETI with methodology and analysis |
| **[Streaming Mode](Streaming-Mode)** | Continuous observation guide — auto-training, self-correcting thresholds, cadence analysis, daily reports |
| **[API Reference](API-Reference)** | Full FastAPI endpoint documentation with curl examples |

### Context

| Page | Description |
|------|-------------|
| **[Comparison with turboSETI](Comparison-with-turboSETI)** | Feature-by-feature comparison, architectural differences, when to use each tool, and migration guide |

---

## Quick Links

- **GitHub Repository:** [SamanTabworlds/MitraSETI](https://github.com/SamanTabworlds/MitraSETI)
- **License:** MIT
- **Python:** 3.10+ | **Rust:** 1.70+
- **Author:** [Saman Tabatabaeian](https://github.com/SamanTabworlds) · Deep Field Labs

---

## Key Features at a Glance

- **45x faster processing** on million-channel observations via parallel Rust de-Doppler search
- **Two-stage ML classification** — rule-based filtering + CNN+Transformer inference
- **Out-of-distribution detection** — ensemble of MSP, Energy, and Spectral distance methods
- **9-class signal taxonomy** — from NARROWBAND_DRIFTING to CANDIDATE_ET
- **Streaming observation mode** — multi-day campaigns with auto-training and daily HTML reports
- **Catalog cross-matching** — SIMBAD, NVSS, FIRST, and ATNF Pulsar catalogs
- **AstroLens integration** — optical + radio cross-reference
- **Desktop + Web UI** — PyQt5 desktop app and FastAPI web interface with WebSocket live streaming
- **Format support** — Sigproc `.fil` and HDF5 `.h5` (Breakthrough Listen format)
