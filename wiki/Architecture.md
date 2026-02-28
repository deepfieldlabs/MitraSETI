# Architecture

MitraSETI is a layered system with a Rust acceleration core, a Python ML layer, and dual UI frontends. Each layer is independently testable and communicates through well-defined interfaces.

---

## System Overview

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

## Data Flow

The primary data flow follows a strictly sequential pipeline from raw observation data to classified results:

```
  .fil / .h5 file
       │
       ▼
  ┌──────────────┐     Rust core reads binary filterbank or HDF5
  │ FilterbankReader │──→ Returns (FilterbankHeader, Array2<f32>)
  └──────┬───────┘       shape: (n_time_steps, n_channels)
         │
         ▼
  ┌──────────────┐     Parallel drift-rate search across all channels
  │ DedopplerEngine │──→ Returns Vec<SignalCandidate>
  └──────┬───────┘       with frequency, drift_rate, snr
         │
         ▼
  ┌──────────────┐     Composite scoring: known-band, zero-drift,
  │   RFIFilter   │──→ broadband, persistence → rfi_score 0.0–1.0
  └──────┬───────┘     Rejects signals with score > 0.70
         │
         ╔═══════════════════════════════╗
         ║  Crosses PyO3 boundary into   ║
         ║  Python via zero-copy arrays  ║
         ╚══════════════╤════════════════╝
                        │
         ▼              ▼
  ┌──────────────┐     Deduplicates nearby hits in frequency/drift
  │ Hit Clustering │──→ space (±64 channels, ±0.5 Hz/s)
  └──────┬───────┘     Keeps highest-SNR per cluster
         │
         ▼
  ┌──────────────┐     Fast checks: drift range, SNR thresholds,
  │ Stage 1: Rules │──→ boundary artifacts, zero-drift patterns
  └──────┬───────┘     Eliminates >99% of signals
         │
         ▼ (survivors only)
  ┌──────────────┐     CNN+Transformer on 256×64 spectrograms
  │ Stage 2: ML   │──→ 9-class probabilities + OOD ensemble score
  └──────┬───────┘     Caches spectrograms for future training
         │
         ▼
  ┌──────────────┐     SIMBAD, NVSS, FIRST, ATNF Pulsar catalogs
  │ Catalog Xref  │──→ AstroLens optical anomaly cross-reference
  └──────┬───────┘     24-hour caching to reduce API load
         │
         ▼
  ┌──────────────┐     Async SQLite (signals, observations, candidates)
  │ Storage/Export │──→ JSON result files, HTML reports
  └──────────────┘     WebSocket broadcast to connected clients
```

---

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        pipeline.py                          │
│                   MitraSETIPipeline                         │
│                                                             │
│  Orchestrates the full processing flow. Calls Rust core     │
│  via PyO3, then Python ML, then catalog and storage.        │
├─────────┬───────────┬──────────────┬──────────────┬─────────┤
│         │           │              │              │         │
│         ▼           ▼              ▼              ▼         │
│   ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐   │
│   │ mitraseti│ │inference/│ │ catalog/  │ │  api/     │   │
│   │ _core    │ │signal_   │ │ radio_    │ │ database  │   │
│   │ (Rust)   │ │classifier│ │ catalogs  │ │ .py       │   │
│   │          │ │.py       │ │ .py       │ │           │   │
│   │ Dedoppler│ │          │ │           │ │ SignalDB  │   │
│   │ Engine   │ │ Signal   │ │ RadioCat  │ │ (SQLite)  │   │
│   │ Filterbnk│ │ Classifier│ │ alogQuery│ │           │   │
│   │ Reader   │ │          │ │           │ │           │   │
│   │ RFIFilter│ │ OOD      │ │ SkyPos   │ │           │   │
│   │          │ │ Detector │ │ ition    │ │           │   │
│   │          │ │          │ │           │ │           │   │
│   │          │ │ Feature  │ │           │ │           │   │
│   │          │ │ Extractor│ │           │ │           │   │
│   └──────────┘ └──────────┘ └───────────┘ └───────────┘   │
└─────────────────────────────────────────────────────────────┘
         │                                       │
         │              ┌────────────────────────┘
         │              │
         ▼              ▼
┌──────────────┐ ┌──────────────┐
│  Desktop UI  │ │   Web UI     │
│  (PyQt5)     │ │  (FastAPI)   │
│              │ │              │
│ main.py      │ │ api/main.py  │
│ ui/*.py      │ │ web/app.py   │
│              │ │ templates/   │
│ 7 panels:    │ │              │
│ - Dashboard  │ │ Endpoints:   │
│ - Waterfall  │ │ /health      │
│ - Gallery    │ │ /process     │
│ - RFI Panel  │ │ /signals     │
│ - Sky Map    │ │ /candidates  │
│ - Streaming  │ │ /ws/live     │
│ - Settings   │ │              │
└──────────────┘ └──────────────┘
```

---

## Threading Model

MitraSETI uses multiple execution contexts to maximize throughput without blocking user interfaces.

### Main Thread

- **Desktop app:** PyQt5 event loop — handles UI rendering, user input, signal/slot dispatch
- **Web server:** Uvicorn async event loop — handles HTTP requests, WebSocket connections
- **CLI:** Synchronous pipeline execution in the main thread

### Rust Worker Threads (rayon thread pool)

The Rust core uses rayon's work-stealing thread pool for data-parallel operations. When `DedopplerEngine.search()` is called, rayon automatically distributes drift-rate trials across all available CPU cores.

```
Main thread ──→ PyO3 call ──→ DedopplerEngine.search()
                                    │
                              rayon::par_iter()
                              ┌─────┼─────┐
                              ▼     ▼     ▼
                          Thread  Thread  Thread  ... (one per core)
                          drift₁  drift₂  drift₃
                              │     │     │
                              └─────┼─────┘
                                    ▼
                              Merged results
                                    │
                              ◄─── Return to Python
```

The thread pool is initialized once and reused across calls. Thread count defaults to the number of logical CPUs.

### ML Inference Thread

When running in the desktop app or streaming mode, ML inference can run asynchronously to avoid blocking the UI:

- **Batch inference:** Signals are grouped into sub-batches of 128 for efficient GPU utilization
- **Device selection:** Automatic — CUDA > MPS (Apple Silicon) > CPU
- **Feature reuse:** OOD detection reuses logits from the classifier forward pass (no duplicate computation)

### Streaming Worker

The streaming observation engine (`scripts/streaming_observation.py`) runs a continuous processing loop:

```
┌──────────────────────────────────────────┐
│           Streaming Main Loop            │
│                                          │
│  while running:                          │
│    1. Discover new .fil/.h5 files        │
│    2. Process via MitraSETIPipeline      │
│    3. Update state (JSON persistence)    │
│    4. Check auto-training trigger        │
│    5. Self-correct SNR thresholds        │
│    6. Generate daily report (if due)     │
│    7. Health check (API, disk, model)    │
│    8. Sleep(interval)                    │
│                                          │
│  Interval varies by mode:               │
│    normal=30s, aggressive=10s, turbo=2s  │
└──────────────────────────────────────────┘
```

---

## State Management

MitraSETI uses JSON-based persistence for lightweight, human-readable state that survives restarts.

### Streaming State (`mitraseti_artifacts/data/streaming_state.json`)

```json
{
  "start_time": "2025-01-15T08:00:00",
  "files_processed": 1247,
  "total_signals": 89432,
  "total_candidates": 15,
  "total_rfi_rejected": 87210,
  "corrections_applied": 3,
  "current_mode": "normal",
  "current_snr_threshold": 10.0,
  "mode_history": ["normal", "aggressive", "normal"],
  "processed_files": ["obs_001.fil", "obs_002.fil", "..."],
  "best_candidates": [
    {
      "frequency_hz": 8419921066.0,
      "drift_rate": 0.3928,
      "snr": 245.7,
      "classification": "narrowband_drifting",
      "confidence": 0.982
    }
  ]
}
```

### Database State (`mitraseti_artifacts/data/mitraseti.db`)

Async SQLite via `aiosqlite` for structured signal storage:

| Table | Purpose |
|-------|---------|
| `signals` | All detected signals with classification, confidence, coordinates |
| `observations` | Processing metadata per file — timing, counts, status |
| `candidates` | Promoted ET candidates with catalog matches and notes |

Indexed on: `snr`, `is_candidate`, `observation_id`, `classification`, `status`.

### Catalog Cache (`mitraseti_artifacts/data/catalog_cache/`)

JSON files with 24-hour TTL for SIMBAD, NVSS, FIRST, and ATNF Pulsar query results. Keyed by `(ra, dec, radius)` to avoid redundant API calls.

### Configuration (`paths.py`)

All artifact paths are centralized in `paths.py` with environment variable overrides:

| Path | Default | Override |
|------|---------|----------|
| Artifacts root | `../mitraseti_artifacts` | `MITRASETI_ARTIFACTS_DIR` |
| Database | `data/mitraseti.db` | `DATABASE_URL` |
| Models | `models/` | `MODELS_DIR` |
| Filterbank data | `data/filterbank/` | — |
| Streaming state | `data/streaming_state.json` | — |
| Reports | `streaming_reports/` | — |
