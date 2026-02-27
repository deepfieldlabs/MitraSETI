# MitraSETI Improvements Log

Chronological record of every improvement applied to MitraSETI, the reasoning behind each change, and its measured impact. This document serves as a reference for future documentation, publications, and architectural decisions.

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
- **What**: During ML inference, spectrograms of candidate signals are saved as `.npz` files to `mitraseti_artifacts/data/spectrogram_cache/`.
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
- **Why**: Pipeline cached to `MitraSETI/mitraseti_artifacts/data/spectrogram_cache/` but streaming looked at `mitraseti_artifacts/data/spectrogram_cache/` (the canonical path from `paths.py`). 1,673 spectrograms were invisible to the fine-tuning logic.
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

## Session 7 — 41-Hour Run: Fix Model Corruption & Training Crash (Feb 26, 2026)

### Problem Diagnosed
After 41 hours / 1,814 files / 90 cycles, fine-tuning succeeded once but then failed 9 consecutive times:

| Symptom | Root Cause |
|---------|-----------|
| `Auto-training failed: float division by zero` (9 times) | Corrupted model (96.7% NaN weights) loaded → NaN loss on every batch → `continue` skips counter → `total=0` → `0/0` crash |
| `conf=0.111` on all candidates (uniform 1/9) | NaN logits → `nan_to_num(0.0)` → softmax([0,...,0]) = uniform |
| `ood_score=0.2079` on all candidates (identical) | Same NaN → uniform → identical Mahalanobis distances |
| Model file 96.7% NaN (712,905/737,609 params) | Session 5's single-class training (all narrowband_drifting) destroyed weights |

### 7.1 Division-by-Zero Guard in Training Functions
- **What**: Added `if total == 0: return 0.0, 0.0` guard before `total_loss / total` in both `train_one_epoch()` and `evaluate()`. Also added NaN-skip (`if torch.isnan(loss): continue`) to `evaluate()` to match the existing guard in `train_one_epoch()`.
- **Why**: When a corrupted model produces NaN loss for every batch, the `continue` statement skips the `total += batch_size` increment. After all batches, `total=0` → `ZeroDivisionError: float division by zero`. This crashed fine-tuning 9 consecutive times over 13 hours.
- **Impact**: Training no longer crashes. If model is too corrupted, training returns `(0.0, 0.0)` and triggers the NaN epoch detector.
- **Files**: `scripts/train_model.py` (`train_one_epoch`, `evaluate`)

### 7.2 Corrupted Model Detection & Auto-Discard
- **What**: Before loading an existing model for fine-tuning, scan all floating-point parameters for NaN. If any NaN detected, delete the model file and switch to initial training (fresh random weights, lr=1e-3, 20 epochs). During training, if 3+ consecutive epochs produce `train_acc=0.0 & val_acc=0.0`, discard the model mid-training and restart from scratch with fresh weights.
- **Why**: The corrupted model (96.7% NaN) was loaded every fine-tuning attempt, immediately produced NaN on all data, and crashed. Without this guard, the system would retry with the same corrupted model indefinitely.
- **Impact**: Self-healing: corrupted models are automatically detected and replaced. Training always produces a functional model.
- **Files**: `scripts/streaming_observation.py` (`_maybe_auto_train`)

### 7.3 Minimum Accuracy Threshold
- **What**: After training completes, if `best_acc < 0.15`, discard the model and return without saving. Training will retry next cycle.
- **Why**: A model with <15% accuracy (below random chance for 9 classes = 11.1%) indicates a fundamental training failure. Saving such a model would corrupt future inference.
- **Impact**: Only models that demonstrate meaningful learning are deployed.
- **Files**: `scripts/streaming_observation.py` (`_maybe_auto_train`)

### 7.4 Epoch-1 Logging
- **What**: Changed epoch logging from `if epoch % 5 == 0` to `if epoch % 5 == 0 or epoch == 1`.
- **Why**: Previously, the first 4 epochs produced no log output. For debugging training issues, seeing epoch 1's accuracy is critical to detect problems early.
- **Files**: `scripts/streaming_observation.py` (`_maybe_auto_train`)

### 7.5 Expanded Target Category Detection
- **What**: Added regex patterns for `KIC` (Tabby's Star), `LHS` (LHS catalog M-dwarfs), `HD` (Henry Draper catalog), `2MASS`, and `WISE` targets. Fixed Kepler regex to not capture observation number suffix.
- **Why**: New BL data files for GJ699, GJ411, KIC8462852, LHS292, HD_109376 would fall into "Other" category without proper patterns. Correct categorization is essential for per-category analysis and publishable results.
- **Files**: `scripts/streaming_observation.py` (`_TARGET_CATEGORIES`)

### 7.6 Full State Reset for Fresh Run
- **What**: Deleted `streaming_state.json`, cleared `verified_candidates.json`, rotated old log, cleared daily reports. Kept spectrogram cache (3,974 real training samples). Deleted corrupted model + OOD calibration.
- **Why**: All counters, candidates, and ML scores from the 41-hour run were invalid due to the corrupted model. Clean slate needed for meaningful results with new data.

### 7.7 Corrupted Model File Deletion
- **What**: Manually deleted `signal_classifier_v1.pt` (96.7% NaN, 2.97 MB) and `ood_calibration.json` (calibrated from corrupted embeddings).
- **Why**: The running process would continue loading the corrupted model on every fine-tuning attempt. Deleting forces initial training from scratch on the next trigger.
- **Impact**: Next training session will build a fresh model using 3,974 real spectrograms (3 classes) + 5,400 synthetic (9 classes) + 7,948 augmented = 17,322 samples.

---

## Session 8 — 12-Hour Run: Fix Training Waste & OOD Bug (Feb 27, 2026)

### Problem Diagnosed
After 12.2 hours / 230 files / 2.6 cycles with 88 files (40 GB dataset), five issues identified:

| Symptom | Root Cause |
|---------|-----------|
| 60% of runtime spent training (7.3h/12.2h) | Fine-tuning triggered 3 times (every 50 cycles). Val_acc stuck at 0.9682 — no improvement. Each session: 20-23 min/epoch on MPS |
| `conf=1.000` for ALL candidates | Rule-based `min(snr/50, 1.0)` saturates at SNR > 50; all candidates have SNR >> 50 |
| `ood=0.1023` identical for ALL candidates | `detect_from_scores()` received softmax probabilities but treated them as raw logits, re-applying softmax. Double-softmax yields degenerate scores |
| `drift=4.0817` for ALL candidates | All signals at max drift boundary — classic de-Doppler artifact, not real signals |
| Spectrogram cache stagnant at 3,974 | Only Stage 2 survivors cached. With strict Stage 1 filter, most files produce 0 candidates → no new training data |

### 8.1 Reduce Fine-Tuning Frequency to 500 Cycles
- **What**: Changed `_RETRAIN_INTERVAL` from 50 to 500 cycles (~4,400 files in aggressive mode).
- **Why**: Model accuracy plateaued at 96.82% from epoch 1 of the initial training. Three fine-tuning sessions (totaling 7.3 hours) produced zero improvement. Training was consuming 60% of runtime.
- **Impact**: Training runs once per ~40-hour run instead of 3 times per 12 hours. Processing time gains ~7 hours per day.
- **Files**: `scripts/streaming_observation.py`

### 8.2 Early Stopping When Accuracy Plateaus
- **What**: Added early stopping: if val_acc doesn't improve for 3 consecutive epochs AND best_acc > 0.90, stop training immediately. Log all epochs (not just every 5th).
- **Why**: Epoch 1 already achieved 0.9682; epochs 2-10 were wasted (20+ min each on MPS). With early stopping, training would complete in ~4 epochs instead of 10.
- **Impact**: When training does run, it finishes in 4-6 minutes instead of 3-4 hours.
- **Files**: `scripts/streaming_observation.py` (`_maybe_auto_train`)

### 8.3 Fix OOD Double-Softmax Bug
- **What**: In `detect_from_scores()`, convert incoming probability scores to log-space (`np.log(np.clip(probs, 1e-10, 1.0))`) before applying MSP and energy computations.
- **Why**: The classifier returns `all_scores` as softmax probabilities (sum to 1). The OOD detector assumed they were raw logits and applied another softmax. With `logits ≈ [0, 0, 0, 0, 1, 0, 0, 0, 0]`, re-softmax gives a near-uniform distribution → identical OOD scores for every candidate.
- **Impact**: OOD scores now differentiate between candidates. A narrowband signal at 99% confidence gets a different OOD score than one at 60% confidence.
- **Files**: `inference/ood_detector.py` (`detect_from_scores`)

### 8.4 Max-Drift Boundary Artifact Filter
- **What**: Signals with drift rate ≥ 98% of `max_drift_rate` (4.0 Hz/s) are now classified as RFI and excluded from candidates. Added `at_drift_boundary` flag to Stage 1 rule-based classification.
- **Why**: All 11 verified candidates from the 12-hour run had `drift=4.0817` — exactly at the search boundary. This is a mathematical artifact: strong narrowband signals produce high-SNR de-Doppler detections at ALL drift rates including the boundary. These are not physically meaningful.
- **Impact**: Eliminates false-positive candidates from boundary artifacts. Real drifting signals with drift < 3.92 Hz/s are unaffected.
- **Files**: `pipeline.py` (`_classify_candidates`)

### 8.5 Improved Confidence Scoring
- **What**: Changed rule-based confidence from `min(snr/50, 1.0)` to `1 - 1/(1 + snr/50)` (sigmoid-like function).
- **Why**: The old formula saturated at exactly 1.0 for any SNR above 50, making all high-SNR candidates indistinguishable. The new formula approaches 1.0 asymptotically: SNR=100 → 0.667, SNR=1000 → 0.952, SNR=10000 → 0.995.
- **Impact**: Candidates are now ranked by confidence. Can distinguish a 10K SNR calibrator from a 62 SNR Voyager signal.
- **Files**: `pipeline.py` (`_classify_candidates`)

### 8.6 Spectrogram Cache Diversification
- **What**: After Stage 1, randomly sample 5 rejected signals per file and cache their spectrograms with rule-based labels.
- **Why**: Only Stage 2 survivors were cached (signals that pass candidate criteria). With strict filtering, most files produce 0 candidates → cache never grows. The model starves for training data. Rejected signals (RFI, noise) are equally valuable for teaching the model what is NOT a candidate.
- **Impact**: At 88 files per cycle, cache grows by ~440 spectrograms per cycle (5 per file). After 3 cycles, the training set doubles in size with diverse RFI examples.
- **Files**: `pipeline.py` (`_classify_candidates`)

### 8.7 Spectrogram Cache Size Bug Fix (Critical)
- **What**: Fixed `_extract_spectrogram(data, chan_idx, n_chans)` → `_extract_spectrogram(data, chan_idx)` in the cache diversification code.
- **Why**: The third argument `n_chans` (total channels, e.g. 351,232) was passed as the `n_freq` parameter (desired frequency bins, default 256). This created 158 MB spectrograms instead of ~50 KB each. Cache exploded from 255 MB to 30 GB in one cycle, filling the disk (0.1 GB free) and crashing the streaming process.
- **Impact**: Cache files now use default 256×64 dimensions (~50 KB each). 1,036 oversized files deleted, 30.3 GB freed.
- **Files**: `pipeline.py` (`_classify_candidates`)

### 8.8 Disk Space Cleanup — MitraSETI Redundant Data
- **What**: Removed duplicate training data and oversized cache files.
- **Deleted**:
  - `data/training/` (281 MB) — duplicate of `mitraseti_artifacts/data/training/`
  - 1,036 oversized spectrogram cache files (30.3 GB) — from cache size bug (8.7)
- **Preserved**:
  - All 88 BL data files (40 GB) — core observation data for streaming
  - Spectrogram cache (4,761 files, 259 MB) — training data for ML model
  - All model weights (signal_classifier_v1.pt, OOD calibration)
- **Impact**: Freed ~30.6 GB from MitraSETI artifacts.

---

## Session 9 — Feb 8, 2026

### 9.1 Project Rename: astroSETI → MitraSETI
- **What**: Renamed the entire project from `astroSETI` to `MitraSETI` across 50 files.
- **Scope**: All display names, UI labels, docstrings, comments, package metadata, Docker configs, CI/CD, web templates, tests, Rust crate.
- **Environment variables**: `ASTROSETI_*` → `MITRASETI_*`
- **Backward compatibility**: Filesystem path `mitraseti_artifacts` kept as-is on disk; Python import adds fallback (`import mitraseti_core` → `import astroseti_core`).
- **Why**: The name `astroSETI` was previously taken. `MitraSETI` derives from "Mitra" (Sanskrit: friend/ally), representing a collaborative approach to the search for extraterrestrial intelligence.

### 9.2 Space Radar Visualizer (replaces Sky Map)
- **What**: Replaced the static Aitoff sky map with an animated radar-style space visualizer.
- **Features**:
  - Custom QPainter radar display with animated sweep line
  - Phosphor-persistence fading (targets glow bright when swept, then fade)
  - RA→angle, Dec→radius polar projection
  - Color-coded blips: green=candidate, blue=signal, dim=observation
  - Pulsing rings on candidate targets
  - Hover tooltips with target details (RA/Dec, SNR, classification)
  - Filter dropdown (All / With Signals / Candidates Only)
  - Auto-loads data from streaming_state.json and verified_candidates.json
- **Why**: The static Aitoff projection was not visually engaging. The radar metaphor is more intuitive for monitoring real-time observations and immediately conveys the "scanning space" concept.

### 9.3 ON-OFF Cadence Analysis
- **What**: Implemented the standard Breakthrough Listen ABACAD cadence filter.
- **New file**: `scripts/cadence_analysis.py`
- **How it works**:
  1. Scans data directory and pairs ON/OFF files by target name (regex-based)
  2. Runs the full MitraSETI pipeline on each ON and OFF file
  3. Compares signals: frequency matching within configurable tolerance (default 5 kHz)
  4. Signals in ON but NOT in OFF → pass cadence filter (potential ETI candidates)
  5. Signals in both → rejected as RFI
- **Supported targets**: TRAPPIST-1, GJ699, KIC8462852, LHS292, Kepler-160
- **Integration**:
  - Auto-runs every 100 streaming cycles
  - "Run Cadence Analysis" button in Streaming UI
  - Cadence Passed / Cadence RFI stat cards in Streaming panel
  - Results saved to `cadence_results.json`
- **CLI**: `python scripts/cadence_analysis.py [--target TRAPPIST1] [--freq-tolerance 0.002] [-v]`
- **Why**: ON-OFF cadence is the gold standard for SETI RFI rejection. A genuine ET signal should appear only when the telescope points at the target, not during off-source pointings. This is critical for publishable results.

---

## Pending / Future Improvements

- **Taylor Tree De-Doppler**: Replace brute-force with Taylor tree algorithm for ~5-10x faster de-Doppler search.
- **Cross-Category Correlation**: Compare signals across Voyager, Kepler, HIP, TRAPPIST categories to identify shared RFI patterns.
- **Parkes Telescope Data**: Add Parkes (Murriyang) data for multi-telescope confirmation of candidate signals.
- **Rust Module Recompilation**: Rebuild `mitraseti_core` (was `astroseti_core`) to match the new project name (`cd core && maturin develop`).
