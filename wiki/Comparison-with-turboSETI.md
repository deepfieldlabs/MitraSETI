# Comparison with turboSETI

MitraSETI is **not** a fork of turboSETI — it is a ground-up reimagination of the SETI signal analysis pipeline, built for the era of machine learning and massive data volumes. This page provides a detailed comparison to help researchers decide when to use each tool.

---

## Feature-by-Feature Comparison

| Capability | turboSETI | MitraSETI |
|---|---|---|
| **Core language** | Pure Python | Rust core + Python ML layer |
| **Parallelism** | Single-threaded | Multi-core via rayon (all CPU cores) |
| **De-Doppler algorithm** | Taylor tree (O(N log N)) | Brute-force (O(N²)), Taylor tree planned v0.2.0 |
| **Processing speed (1M channels)** | 2.53 s (Voyager-1 H5) | **0.06 s (45x faster)** |
| **Processing speed (65K channels)** | 0.08–0.64 s | 0.42–0.54 s (comparable) |
| **ML classification** | None | CNN + Transformer (9 classes) |
| **RFI handling** | Manual SNR threshold | Learned rejection + rule-based + persistence scoring |
| **Out-of-distribution detection** | None | Ensemble OOD (MSP + Energy + Spectral) |
| **Signal taxonomy** | Hit / no hit | 9 classes with confidence scores |
| **Streaming mode** | Batch only | Multi-day continuous with auto-training |
| **Self-correcting thresholds** | No | Adjusts SNR based on candidate rates |
| **ON-OFF cadence** | External script (turboSETI `find_event`) | Built-in ABACAD pattern analysis |
| **Optical cross-reference** | None | AstroLens integration |
| **Catalog matching** | Basic (manual) | SIMBAD, NVSS, FIRST, ATNF Pulsar (24h cache) |
| **Web interface** | None | FastAPI + Jinja2 + WebSocket live stream |
| **Desktop application** | None | PyQt5 with waterfall viewer, sky map, gallery |
| **REST API** | None | Full REST API with Swagger docs |
| **Daily reports** | None | Auto-generated HTML with charts |
| **Database** | DAT files (text) | Async SQLite with structured queries |
| **File format support** | .fil, .h5 | .fil, .h5 (with blimpy fallback) |

---

## Architectural Differences

### turboSETI

```
.h5/.fil  →  FindDoppler  →  .dat file (text hits)  →  find_event (cadence)
              Python loop     frequency, drift, SNR     manual review
              single-thread
```

turboSETI is a focused tool that does one thing well: exhaustive Doppler drift-rate search. It reads filterbank data, searches for narrowband signals at all trial drift rates using the Taylor tree algorithm, and outputs a `.dat` text file listing every detection above the SNR threshold. Cadence analysis (ON-OFF rejection) is handled by a separate script (`find_event_pipeline`).

### MitraSETI

```
.h5/.fil  →  Rust Core  →  Python ML  →  Catalog  →  DB + API + UI
              parallel      2-stage       SIMBAD     SQLite
              de-Doppler    classifier    NVSS       FastAPI
              RFI filter    OOD detect    FIRST      PyQt5
                                          Pulsar     WebSocket
```

MitraSETI is an integrated platform that processes, classifies, cross-references, stores, and visualizes signals in a single pipeline. The Rust core handles compute-intensive stages, Python handles ML and orchestration, and the results flow into a database accessible through REST API, desktop app, and web interface.

### Key Architectural Distinctions

| Aspect | turboSETI | MitraSETI |
|--------|-----------|-----------|
| **Design philosophy** | Unix tool (do one thing) | Integrated platform |
| **Output format** | Text `.dat` files | Structured database + API |
| **Intelligence** | None (raw detections) | Two-stage ML classification |
| **Extensibility** | Script-based workflows | Plugin architecture (catalogs, classifiers) |
| **State management** | Stateless (per-file) | Persistent state (JSON + SQLite) |
| **Deployment** | Script | Docker, web server, desktop app |

---

## When to Use Each Tool

### Use turboSETI When

- **You need the Taylor tree algorithm.** turboSETI's O(N log N) de-Doppler search is algorithmically superior for very large files. MitraSETI's brute-force search compensates with parallelism but will benefit from Taylor tree in v0.2.0.

- **You're running existing BL pipelines.** If your workflow is built around turboSETI's `.dat` output format and `find_event` cadence scripts, switching tools requires migration effort.

- **You need maximum compatibility.** turboSETI is the standard tool used by Breakthrough Listen and has years of validation on real observations.

- **You only need raw detections.** If downstream analysis handles classification and filtering, turboSETI's unprocessed hit list is sufficient.

### Use MitraSETI When

- **You want classified candidates, not raw hits.** MitraSETI's two-stage classifier reduces thousands of raw detections to a handful of actionable candidates with confidence scores.

- **You need to process high-channel-count data fast.** The 45x speedup on million-channel observations dramatically reduces processing time for large datasets.

- **You're running multi-day observation campaigns.** Streaming mode with auto-training, self-correcting thresholds, and daily reports is designed for unattended operation.

- **You want anomaly detection.** The OOD ensemble detector flags signals that don't fit any known category — something no other SETI tool offers.

- **You need catalog cross-reference.** Automatic matching against SIMBAD, NVSS, FIRST, and ATNF Pulsar catalogs plus AstroLens optical anomalies.

- **You want a UI.** Desktop app with waterfall viewer, signal gallery, and sky map. Web interface with live WebSocket updates.

---

## Complementary Usage

turboSETI and MitraSETI are not mutually exclusive. They can be used together in a complementary workflow:

### Validation Pipeline

```
1. Process data with both tools independently
2. Compare raw detections:
   - turboSETI hits not found by MitraSETI → check RFI filter thresholds
   - MitraSETI candidates not in turboSETI → check drift rate range
3. Use agreement as a confidence booster:
   - Signal found by both tools → higher confidence
   - Signal found only by MitraSETI with ML classification → novel detection
```

### Hybrid Workflow

```
1. Run turboSETI for exhaustive, validated de-Doppler search
2. Feed turboSETI .dat hits into MitraSETI for ML classification
3. Use MitraSETI's catalog cross-reference and OOD detection
4. Visualize results in MitraSETI's web/desktop UI
```

### Large Survey Processing

```
1. Use MitraSETI's streaming mode for initial triage (fast, auto-filtering)
2. Re-process interesting targets with turboSETI for independent confirmation
3. Use MitraSETI's API to aggregate and query results across the survey
```

---

## Migration Guide from turboSETI

### Step 1: Install MitraSETI

```bash
git clone https://github.com/SamanTabworlds/MitraSETI.git
cd MitraSETI

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install maturin
maturin develop --release
pip install -e .
```

### Step 2: Verify Installation

```bash
python -c "import mitraseti_core; print('Rust core loaded')"
python -c "from inference.signal_classifier import SignalClassifier; print('ML layer ready')"
```

### Step 3: Process Your First File

```bash
# turboSETI equivalent:
#   from turbo_seti.find_doppler.find_doppler import FindDoppler
#   fd = FindDoppler('observation.h5', max_drift=4, snr=10)
#   fd.search()

# MitraSETI:
python pipeline.py observation.h5
```

### Step 4: Map turboSETI Concepts

| turboSETI Concept | MitraSETI Equivalent |
|-------------------|---------------------|
| `FindDoppler` class | `MitraSETIPipeline` class |
| `max_drift` parameter | `SearchParams.max_drift_rate` |
| `snr` parameter | `SearchParams.min_snr` |
| `.dat` output file | JSON output + SQLite database |
| `find_event_pipeline` | Built-in ON-OFF cadence analysis |
| Manual hit review | ML classification + OOD detection |
| No catalog matching | Automatic SIMBAD/NVSS/FIRST/Pulsar |
| No visualization | Desktop + Web UI |

### Step 5: Adapt Your Workflow

**If you parse `.dat` files:**

Replace `.dat` parsing with MitraSETI JSON output or database queries:

```python
# Before (turboSETI):
# Parse .dat file line by line

# After (MitraSETI):
import json
with open("results.json") as f:
    results = json.load(f)
    for candidate in results["candidates"]:
        print(f"{candidate['frequency_hz']} Hz, SNR {candidate['snr']}")
```

Or use the API:

```bash
curl "http://localhost:8000/signals?min_snr=20&is_candidate=true"
```

**If you use `find_event_pipeline`:**

MitraSETI's streaming mode handles ON-OFF cadence analysis automatically. No separate script needed.

**If you batch-process many files:**

```bash
# Process all files in a directory
python pipeline.py data/*.fil data/*.h5

# Or use streaming mode for continuous processing
python scripts/streaming_observation.py --days 7
```

---

## Performance Summary

| Metric | turboSETI | MitraSETI | Notes |
|--------|-----------|-----------|-------|
| **Speed (1M channels)** | 2.53 s | 0.06 s | 45x faster |
| **Speed (65K channels)** | 0.08–0.64 s | 0.42–0.54 s | Comparable |
| **Output** | Raw hits | Classified candidates | More actionable |
| **Human effort per file** | Review all hits | Review candidates only | Significant reduction |
| **Scalability** | Single-core limited | Multi-core parallel | Scales with hardware |
| **Unattended operation** | No | Yes (streaming mode) | Multi-day campaigns |

The intelligence layer is the key difference. turboSETI finds signals. MitraSETI *understands* them — automatically classifying, scoring, cross-referencing, and prioritizing so researchers focus on what matters.
