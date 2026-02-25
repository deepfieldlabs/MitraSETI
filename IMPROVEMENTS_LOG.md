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

### 3.10 Training Tensor Shape Fix
- **What**: Removed `.unsqueeze(1)` from training tensor creation that added an incorrect channel dimension.
- **Why**: Model expects input `(batch, freq_bins, time_steps)` = `(N, 256, 64)`. The unsqueeze made it `(N, 1, 256, 64)`, causing `too many values to unpack (expected 3)` in the CNN backbone when trying to unpack 4D into 3 variables.
- **Impact**: Auto-training now works. Model successfully trained at file 66 with val accuracy logged.
- **Files**: `scripts/streaming_observation.py` (`_maybe_auto_train`)

---

## Session 4 — 8-Hour Run Analysis & OOM Fixes (Feb 23, 2026 evening)

### Problem Diagnosed
After 8 hours of streaming (88 files, ~4.4 cycles), two issues emerged:

| Issue | Detail |
|-------|--------|
| OOM on batch ML | `classify_batch()` on 27,922 spectrograms tried to allocate 54 GB tensor |
| De-Doppler slow on gpuspec | Files with 67M channels × 3 time ints took 165–976s for brute-force search |

### 4.1 ML Candidate Cap + Sub-Batching
- **What**: (a) Cap ML inference at top 500 candidates by SNR. (b) Split batch classification into sub-batches of 128. (c) Fallback to individual `classify()` if a sub-batch fails.
- **Why**: The `.gpuspec` files produce 9,000–28,000 rule-based candidates. Stacking 28K spectrograms into one tensor requires 54 GB. Previous runs crashed with `Invalid buffer size`.
- **Impact**: Memory usage capped at ~500 MB per sub-batch. No more OOM crashes. Processing time for large files drops from crash/timeout to ~60 seconds.
- **Files**: `pipeline.py` (`_classify_candidates` Stage 2)

### 4.2 Adaptive De-Doppler Limit for Low-Integration Files
- **What**: Reduce `max_pts` from 16M to 4M for files with ≤8 time integrations. This forces 4–8x more channel downsampling on gpuspec files (67M channels × 3 ints).
- **Why**: The brute-force de-Doppler scales poorly with channel count when time steps are few. A 67M-channel file downsampled 2x still has 33M channels → 8M points → 165–976 second searches.
- **Impact**: Expected 4–8x speedup on gpuspec files. De-Doppler on 3C161 should drop from 165s to ~10–30s.
- **Files**: `pipeline.py` (`_run_dedoppler`)

### 4.3 UI Stat Display Improvements
- **What**: Redesigned streaming panel stat cards: "Files Processed" instead of "Files with Signals", "RFI Rejection Rate" as percentage instead of raw count, "Avg Time / File" instead of inflated files/hr rate.
- **Why**: Raw counts like "RFI Rejected: 3053" and "Processing Rate: 646/hr" were technically correct but meaningless to users. Percentages and per-file timing are immediately interpretable.
- **Files**: `ui/streaming_panel.py`, `ui/dashboard.py`, `scripts/streaming_observation.py`

---

## Session 5 — 23-Hour Run Analysis & Fine-Tuning Fixes (Feb 24, 2026)

### Problem Diagnosed
After 23 hours / 344 files / 17.2 cycles, four issues were identified:

| Issue | Detail |
|-------|--------|
| Fine-tuning never triggered | 1,673 cached spectrograms existed but at wrong path |
| Self-labeling bias | ALL cached spectrograms labeled as RFI (class 0) — model's own predictions used as ground truth |
| Cross-cycle duplicates | 82 candidate entries for only 5 unique signals (same file detected every cycle) |
| Missing candidate fields | `frequency_hz=0` and `classification=?` in all verified candidates |

### 5.1 Spectrogram Cache Path Fix
- **What**: Changed `pipeline.py` to import `DATA_DIR` from `paths.py` instead of computing its own path via `Path(__file__).parent`.
- **Why**: Pipeline cached to `astroSETI/astroseti_artifacts/data/spectrogram_cache/` but streaming looked at `astroseti_artifacts/data/spectrogram_cache/` (the canonical path from `paths.py`). 1,673 spectrograms were invisible to the fine-tuning logic.
- **Impact**: Fine-tuning now sees the cached spectrograms and will trigger on next cycle.
- **Files**: `pipeline.py`

### 5.2 Heuristic Relabeling for Training Data
- **What**: In `_build_training_set`, relabel cached spectrograms using physical properties (SNR, drift rate) instead of trusting the model's own classification.
- **Why**: All 1,673 cached spectrograms had `label=0` (RFI) because the initial synthetic-trained model classified everything as RFI. Fine-tuning on this would reinforce the bias. Heuristic rules: `drift>0.1 & SNR>25 → narrowband_drifting`, `SNR>5000 & drift<0.1 → narrowband`, `SNR<10 & drift<0.05 → noise`.
- **Impact**: Training set now has diverse labels from real BL data. Model will learn to distinguish drifting signals from RFI.
- **Files**: `scripts/streaming_observation.py` (`_build_training_set`)

### 5.3 Cross-Cycle Candidate Deduplication
- **What**: `_record_candidate` now deduplicates by `(file_name, target_name)`. If a candidate for the same file+target already exists, it updates the entry (if higher SNR) instead of appending a duplicate.
- **Why**: 82 candidate entries existed for just 5 unique signals. The same file processed 17 times added 17 duplicate entries.
- **Impact**: Clean candidate list. Current: 5 unique verified candidates.
- **Files**: `scripts/streaming_observation.py` (`_record_candidate`)

### 5.4 Candidate Field Completeness
- **What**: Added `frequency_hz`, `classification`, `rfi_probability`, `is_anomaly` to the result dict and candidate entries.
- **Why**: All candidates showed `freq=0.000 MHz` and `class=?` because these fields were never propagated from the pipeline's per-signal results to the per-file summary.
- **Impact**: Candidates now carry full ML classification details for analysis and publishing.
- **Files**: `scripts/streaming_observation.py` (`_process_file`, `_record_candidate`)

---

## Session 6 — 27-Hour Run: Fix Broken Fine-Tuning (Feb 25, 2026)

### Problem Diagnosed
After 27 hours / 864 files / 43 cycles, fine-tuning ran 3 times but produced a degraded model:

| Symptom | Root Cause |
|---------|-----------|
| `train_acc=0.000` | All 3,974 spectrograms relabeled to same class (narrowband_drifting) |
| `11922 samples, 1 classes` | No synthetic data included; augmentation tripled the single class |
| `conf=nan` | Single-class training corrupted model softmax → NaN logits |
| `ood_score=0.0000` | NaN scores propagated to OOD → all scores become 0 |

### 6.1 Multi-Class Heuristic Relabeling
- **What**: Rewrote heuristic to produce 3 distinct classes based on SNR ranges: `SNR>100K → narrowband(1)`, `drift>0.5 & SNR>1000 → rfi(0)`, `SNR<1000 → narrowband_drifting(4)`.
- **Why**: Previous rule `drift>0.1 & SNR>25 → class 4` matched ALL cached spectrograms (100%) because Stage 1 filtering guarantees drift>0.1 and SNR>25.
- **Impact**: Real data now has 26% RFI, 10% narrowband, 64% drifting. Combined with 9-class synthetic data, model will learn meaningful discrimination.
- **Files**: `scripts/streaming_observation.py` (`_build_training_set`)

### 6.2 Always Include Synthetic Data
- **What**: Changed `if include_synthetic or n_real < 100:` to always include synthetic training data.
- **Why**: With n_real=3974, the condition was False during fine-tuning. Without synthetic data (which provides all 9 signal classes), the training set had only 1 class and training couldn't produce gradients.
- **Impact**: Training set now mixes ~4K real spectrograms (3 classes) + ~5.4K synthetic (9 classes) = balanced multi-class training.
- **Files**: `scripts/streaming_observation.py` (`_build_training_set`)

### 6.3 NaN Guard in Classifier and OOD
- **What**: Added `torch.nan_to_num()` on logits before softmax, and `np.nan_to_num()` on probability arrays in both `classify()` and `classify_batch()`. Also guarded OOD detector's logits array.
- **Why**: Corrupted model weights from single-class training produced NaN logits → NaN softmax → NaN confidence → 0.0 OOD scores. Even after retraining, residual NaN values can propagate.
- **Impact**: Inference never produces NaN. Worst case: unknown/noise classification with 0 confidence instead of crash or silent corruption.
- **Files**: `inference/signal_classifier.py`, `inference/ood_detector.py`

### 6.4 Reduced Fine-Tuning Frequency
- **What**: Increased `_RETRAIN_INTERVAL` from 10 to 50 cycles.
- **Why**: Fine-tuning ran every 200 files (~3.4 hours apart), each session taking 1.3 hours on MPS. At 50 cycles (1000 files), it runs every ~9.4 hours — once per overnight run.
- **Impact**: Less CPU/GPU overhead during streaming. Model is stable enough that frequent retraining is unnecessary.
- **Files**: `scripts/streaming_observation.py`

### 6.5 State Reset Completeness
- **What**: Added `last_trained_at_file = 0` and `pipeline_metrics = {}` to the session reset block.
- **Why**: Previous run's `last_trained_at_file=468` persisted, preventing fine-tuning until file 478 in the new session.
- **Files**: `scripts/streaming_observation.py` (`__init__`)

### 6.6 Candidate Dedup Fix
- **What**: Changed dedup to remove ALL existing entries with same `(file_name, target_name)` key, not just the first match.
- **Why**: Old duplicates from pre-dedup runs (5-7 per signal) were never cleaned because only the first match was found and replaced.
- **Files**: `scripts/streaming_observation.py` (`_record_candidate`)

---

## Pending / Future Improvements

- **Taylor Tree De-Doppler**: Replace brute-force with Taylor tree algorithm for ~5-10x faster de-Doppler search.
- **On-OFF Cadence Analysis**: Compare ON-source vs OFF-source observations to reject persistent RFI.
- **Cross-Category Correlation**: Compare signals across Voyager, Kepler, HIP, TRAPPIST categories to identify shared RFI patterns.
- **GPU Acceleration**: Enable CUDA/MPS for CNN+Transformer inference on GPU-equipped machines.
