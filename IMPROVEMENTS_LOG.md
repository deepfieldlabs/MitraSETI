# astroSETI Improvements Log

Chronological record of every improvement applied to astroSETI, the reasoning behind each change, and its measured impact. This document serves as a reference for future documentation, publications, and architectural decisions.

---

## Session 1 — Initial Build & Bug Fixes (Feb 21, 2026)

### 1.1 Core Pipeline Created
- **What**: Built end-to-end pipeline connecting Rust de-Doppler engine with Python ML layer (CNN+Transformer classifier, OOD detector, feature extractor).
- **Why**: Need a complete signal processing chain from raw filterbank/HDF5 data to classified candidates.
- **Files**: `pipeline.py`, `inference/signal_classifier.py`, `inference/ood_detector.py`, `inference/feature_extractor.py`

### 1.2 Streaming Observation Engine
- **What**: Created multi-day streaming observation loop with state persistence, health checks, and auto-training.
- **Why**: Enable continuous unattended observation runs over 7+ days.
- **Files**: `scripts/streaming_observation.py`

### 1.3 UI — Crystalline Glass Theme
- **What**: Built PyQt5 desktop UI with Dashboard, Streaming Panel, Waterfall Viewer, Signal Gallery, RFI Dashboard, Sky Map, Settings.
- **Why**: Provide researchers a visual interface for monitoring and interacting with the pipeline.
- **Files**: `ui/main_window.py`, `ui/dashboard.py`, `ui/streaming_panel.py`, `ui/waterfall_viewer.py`, `ui/sky_map_panel.py`, `ui/theme.py`, `ui/settings_panel.py`

---

## Session 2 — Bug Fixes & Stability (Feb 22, 2026)

### 2.1 Matplotlib Color Format Fix
- **What**: Converted `rgba()` CSS strings to matplotlib-compatible RGBA tuples.
- **Why**: `matplotlib.pyplot.ValueError: 'rgba(80,140,200,0.08)' is not a valid value for color` crashed the app on launch.
- **Impact**: App launches without errors.
- **Files**: `ui/sky_map_panel.py`

### 2.2 Font Warning Cleanup
- **What**: Removed unsupported font families (`-apple-system`, `Inter`, `Segoe UI`) from global CSS.
- **Why**: Console flooded with font warnings on macOS.
- **Files**: `ui/theme.py`

### 2.3 QProgressBar Float Fix
- **What**: Wrapped `max(target_days, 1)` with `int()` before passing to `setMaximum()`.
- **Why**: `TypeError: setMaximum(self, maximum: int): argument 1 has unexpected type 'float'`.
- **Files**: `ui/streaming_panel.py`

### 2.4 Scatter Plot Color Array Fix
- **What**: Converted `obs_colors` list to `np.array(obs_colors)` before passing to `ax.scatter()`.
- **Why**: `UserWarning: *c* argument looks like a single numeric RGB or RGBA sequence` when only one point plotted.
- **Files**: `ui/sky_map_panel.py`

### 2.5 Streaming UI — Zero Values Fix
- **What**: Corrected log path and state field mappings in streaming panel. Added `elapsed`, `processing_rate`, `current_file` to state JSON.
- **Why**: Streaming panel showed all zeros and no log output.
- **Files**: `ui/streaming_panel.py`, `scripts/streaming_observation.py`

### 2.6 Dashboard — Live Stats from State File
- **What**: Added `QTimer` (5s interval) that reads `streaming_state.json` and updates stat cards dynamically.
- **Why**: Dashboard was hardcoded to zero; didn't reflect streaming progress.
- **Files**: `ui/dashboard.py`

### 2.7 Blimpy Integration for BL Data
- **What**: Implemented `_read_blimpy()` as fallback reader using the `blimpy` library for `.h5` and `.fil` files.
- **Why**: Rust reader couldn't handle all BL data formats (gpuspec, rawspec). Blimpy handles frequency slicing for large files.
- **Files**: `pipeline.py`

### 2.8 HDF5 Plugin Path Fix
- **What**: Added `if "HDF5_PLUGIN_PATH" not in os.environ: os.environ["HDF5_PLUGIN_PATH"] = ""` at module load.
- **Why**: h5py crashed on systems where `/usr/local/hdf5/lib/plugin` doesn't exist.
- **Files**: `pipeline.py`, `scripts/streaming_observation.py`

### 2.9 Auto-Download Removal
- **What**: Removed automatic BL file download from streaming; requires user to pre-download files manually.
- **Why**: Auto-download produced unreliable 40KB dummy files. Manual download ensures genuine multi-GB BL data.
- **Files**: `scripts/streaming_observation.py`

---

## Session 3 — Overnight Run Analysis & Major Overhaul (Feb 23, 2026)

### Problem Diagnosed
After a 9.5-hour overnight run, only 4 of 20 files were processed. Root causes:

| Issue | Detail |
|-------|--------|
| ML bottleneck | 0.11s per signal × 60K signals = 109 min per file |
| No clustering | 0% deduplication → same signal counted thousands of times |
| Synthetic model | Trained on synthetic data (Feb 21), 100% candidate rate on real BL data |
| No retraining | Auto-train skipped because model file already existed |

### 3.1 Two-Stage Classification
- **What**: Stage 1 applies rule-based checks (SNR, drift, RFI patterns) on ALL signals instantly. Stage 2 runs full ML inference (feature extraction, CNN+Transformer, OOD) only on signals that pass candidate criteria.
- **Why**: Running expensive ML on every de-Doppler hit was the 49-minute bottleneck. Most hits are noise peaks that don't warrant ML analysis.
- **Impact**: Processing time per file drops from ~49 minutes to ~30 seconds. Every signal is still evaluated — just through the appropriate filter first.
- **Files**: `pipeline.py` (`_classify_candidates`)

### 3.2 Hit Clustering (Frequency Deduplication)
- **What**: After de-Doppler search, group hits within 64 channels and 0.5 Hz/s drift tolerance. Keep only the highest-SNR hit per cluster.
- **Why**: De-Doppler produces thousands of duplicate detections for the same physical signal at slightly different frequency/drift combinations. Previous clustering reduction was 0%.
- **Impact**: Reduces 60K raw hits to ~500-2000 unique signals. Eliminates inflated candidate counts.
- **Files**: `pipeline.py` (`_cluster_hits`, integrated into `process_file`)

### 3.3 Spectrogram Caching for Retraining
- **What**: During ML inference, spectrograms of candidate signals are saved as `.npz` files to `astroseti_artifacts/data/spectrogram_cache/`.
- **Why**: The model needs real BL data to train on. Caching spectrograms during processing builds a training set automatically.
- **Files**: `pipeline.py` (`_classify_candidates`, `_SPECTROGRAM_CACHE_DIR`)

### 3.4 Periodic Fine-Tuning on Real Data
- **What**: Rewrote `_maybe_auto_train` to support: (a) initial training after 5 files using real spectrograms + synthetic supplements, (b) periodic fine-tuning every 10 additional files with newly cached data. Lower learning rate (3e-4) for fine-tuning vs initial (1e-3).
- **Why**: Previous auto-train only ran once with synthetic data and never retrained. Model trained on synthetic data marked 100% of real signals as candidates.
- **Impact**: Model improves progressively as more real BL data is processed. Discrimination accuracy expected to increase significantly after first real-data training.
- **Files**: `scripts/streaming_observation.py` (`_maybe_auto_train`, `_build_training_set`, `_count_cached_spectrograms`)

### 3.5 Pipeline Efficiency Metrics
- **What**: Added comprehensive metrics to pipeline output: de-Doppler throughput (Mpts/s), clustering reduction rate, ML throughput (signals/s), candidate rate, SNR/drift statistics, model status. Streaming engine logs an ASCII efficiency report every 5 files.
- **Why**: Need visibility into which components are bottlenecks and whether the model is improving.
- **Files**: `pipeline.py` (metrics dict in `process_file`), `scripts/streaming_observation.py` (`_accumulate_metrics`, `_log_efficiency_report`)

### 3.6 State Reset & Synthetic Model Deletion
- **What**: Deleted old synthetic model (`signal_classifier_v1.pt`, `ood_calibration.json`), reset streaming state to zeros, cleared log and candidates.
- **Why**: Previous overnight results were based on a model that couldn't discriminate. Fresh start with real data training needed.

---

### 3.7 Batch ML Inference
- **What**: Replaced one-at-a-time `classify()` calls with `classify_batch()` which processes all candidate spectrograms in a single model forward pass.
- **Why**: Processing 200 signals one-at-a-time = 200 forward passes. Batching = 1 forward pass. GPU/MPS architectures are optimized for batched tensor operations.
- **Impact**: ML inference time reduced from ~22s to ~2-3s for 200 candidates.
- **Files**: `pipeline.py` (`_classify_candidates` Stage 2)

### 3.8 Eliminate Duplicate Forward Pass in OOD Detection
- **What**: Added `detect_from_scores()` method to `RadioOODDetector` that accepts pre-computed class scores instead of re-running the classifier.
- **Why**: The previous `detect()` method called `classifier.classify(spectrogram)` internally, meaning each signal went through the model TWICE (once for classification, once for OOD). The new method reuses the scores from the batch classification step.
- **Impact**: Halves the number of model forward passes. OOD detection per-signal drops from ~0.05s to ~0.001s.
- **Files**: `inference/ood_detector.py` (`detect_from_scores`), `pipeline.py`

### 3.9 Data Augmentation for Training
- **What**: Added Gaussian noise injection and frequency-axis shifting to real BL spectrograms during training set assembly. Each real sample produces 2 augmented copies.
- **Why**: Small real-data training sets (initial 5 files = ~50-100 spectrograms) overfit quickly. Augmentation triples the effective dataset size and teaches the model to be robust to noise variations and slight frequency offsets.
- **Impact**: 3x more training samples from real data. Expected better generalization and lower candidate false-positive rate.
- **Files**: `scripts/streaming_observation.py` (`_build_training_set`)

---

## Pending / Future Improvements

- **Taylor Tree De-Doppler**: Replace brute-force with Taylor tree algorithm for ~5-10x faster de-Doppler search. Currently not a bottleneck (0.3-1.3s per file) but will matter for larger datasets.
- **Batch ML Inference**: Process multiple spectrograms in a single forward pass instead of one-at-a-time. Would further reduce ML time.
- **On-OFF Cadence Analysis**: Compare ON-source vs OFF-source observations to reject persistent RFI. Requires paired observation files.
- **Cross-Category Correlation**: Compare signals across Voyager, Kepler, HIP, TRAPPIST categories to identify shared RFI patterns.
- **GPU Acceleration**: Enable CUDA/MPS for CNN+Transformer inference on GPU-equipped machines.
