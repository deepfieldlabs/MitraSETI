# Streaming Mode

MitraSETI's streaming observation engine runs multi-day campaigns unattended, continuously processing filterbank files as they arrive. It includes auto-training, self-correcting thresholds, ON-OFF cadence analysis, daily HTML reports, and full state persistence.

---

## Getting Started

### Starting a Streaming Session

```bash
# 7-day continuous observation (default mode: normal)
python scripts/streaming_observation.py --days 7

# 3-day aggressive mode (lower SNR thresholds, more candidates)
python scripts/streaming_observation.py --days 3 --mode aggressive

# Turbo mode for rapid scanning
python scripts/streaming_observation.py --days 1 --mode turbo
```

### Other Commands

```bash
# Generate a report from existing data without processing
python scripts/streaming_observation.py --report-only

# Reset state and start fresh
python scripts/streaming_observation.py --reset
```

### Observation Modes

| Mode | Interval | SNR Threshold | Use Case |
|------|----------|---------------|----------|
| `normal` | 30 s | ≥ 10.0 | Standard observation — balanced sensitivity and false positive rate |
| `aggressive` | 10 s | ≥ 6.0 | Higher sensitivity — more candidates, more false positives |
| `turbo` | 2 s | ≥ 3.0 | Maximum sensitivity — rapid scanning, high false positive rate |

---

## Configuration

### File Discovery

The streaming engine watches `FILTERBANK_DIR` (configured in `paths.py`, default: `mitraseti_artifacts/data/filterbank/`) for `.fil` and `.h5` files. It maintains a set of already-processed files to avoid duplicate processing.

### Directory Structure

```
mitraseti_artifacts/
├── data/
│   ├── filterbank/              # Place observation files here
│   ├── streaming_state.json     # Persistent state
│   └── catalog_cache/           # Catalog query cache (24h TTL)
├── models/
│   ├── signal_classifier_v1.pt  # ML model weights
│   └── ood_calibration.json     # OOD detector calibration
├── streaming_reports/
│   └── daily/                   # Daily HTML reports
└── logs/
    └── streaming.log            # Processing log
```

---

## Auto-Training

The streaming engine trains and fine-tunes the ML model automatically based on accumulated data.

### Training Triggers

| Trigger | Action |
|---------|--------|
| 5 files processed (first time) | Train model from scratch using cached spectrograms + synthetic data |
| Every 500 processing cycles | Fine-tune existing model on new accumulated data |

### Training Process

1. **Data collection:** During normal processing, spectrograms extracted for ML inference are cached to disk.

2. **Synthetic augmentation:** The training pipeline generates synthetic examples (drifting narrowband, broadband, pulsed, chirp, noise) and applies data augmentation (Gaussian noise injection, frequency shift).

3. **Model training:** CNN+Transformer trained with Adam optimizer (lr=1e-3), early stopping (patience 5 epochs), and cross-entropy loss.

4. **OOD recalibration:** After training, all training data is passed through the new model to compute reference feature statistics. The OOD spectral distance threshold is set at the 95th percentile.

5. **Hot swap:** The new model weights replace the old ones. The pipeline automatically picks up the updated model on the next processing cycle.

---

## Self-Correcting Thresholds

The streaming engine dynamically adjusts its sensitivity based on the candidate rate — the fraction of detected signals classified as candidates.

### Adjustment Logic

```
candidate_rate = total_candidates / total_signals

if candidate_rate > 15%:
    # Too many candidates → likely too sensitive → raise SNR threshold
    snr_threshold += adjustment

if candidate_rate < 0.5%:
    # Too few candidates → likely too strict → lower SNR threshold
    snr_threshold -= adjustment
```

### Mode Escalation

If the engine detects no candidates after an extended period, it automatically escalates to a more sensitive mode:

| Condition | Action |
|-----------|--------|
| No candidates after 2 days | Switch `normal` → `aggressive` |
| No candidates after 3 days | Switch `aggressive` → `turbo` |

### RFI Flood Detection

If >80% of detected signals are classified as RFI, the engine logs a warning. This can indicate a new interference source that should be added to the known-band database.

---

## ON-OFF Cadence Analysis

MitraSETI implements the standard SETI ON-OFF cadence rejection pattern used in Breakthrough Listen observations.

### ABACAD Pattern

In this cadence, the telescope alternates between the target source (A) and reference positions (B, C, D):

```
A (ON)  → B (OFF) → A (ON)  → C (OFF) → A (ON)  → D (OFF)
target    ref #1     target    ref #2     target    ref #3
```

A genuine ET signal should appear in all ON (A) observations and be absent from all OFF (B, C, D) observations. Any signal present in both ON and OFF pointings is local RFI.

### Implementation

1. **Pair discovery:** The engine identifies source/reference observation pairs based on filename patterns and timestamps.

2. **Signal matching:** Signals detected in ON observations are cross-checked against OFF observations using frequency and drift rate matching.

3. **Rejection:** Signals present in any OFF observation are flagged as RFI and excluded from the candidate list.

4. **Statistics:** The engine tracks `cadence_passed` (signals unique to ON) and `cadence_rfi_rejected` (signals found in OFF).

---

## Daily HTML Reports

At the end of each day (or when `--report-only` is used), the streaming engine generates an HTML report with charts and statistics.

### Report Contents

- **Summary statistics:** Files processed, signals detected, candidates found, RFI rejected
- **Candidate rankings:** Top candidates sorted by SNR with classification details
- **Processing performance:** Throughput (files/hour), average processing time
- **Threshold history:** How SNR thresholds changed over time
- **Mode history:** When and why mode switches occurred
- **Error log:** Any processing failures or warnings
- **Charts:** Time-series plots of key metrics

### Report Location

Reports are saved to `mitraseti_artifacts/streaming_reports/daily/` with timestamped filenames.

---

## State Persistence

The streaming engine saves its complete state to JSON after every processing cycle. This allows resuming after restarts, crashes, or system reboots.

### State File: `streaming_state.json`

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
  "processed_files": ["obs_001.fil", "obs_002.fil"],
  "best_candidates": [
    {
      "frequency_hz": 8419921066.0,
      "drift_rate": 0.3928,
      "snr": 245.7,
      "classification": "narrowband_drifting",
      "confidence": 0.982
    }
  ],
  "daily_snapshots": [
    {
      "date": "2025-01-15",
      "files_processed": 48,
      "signals": 3421,
      "candidates": 2,
      "rfi_rejected": 3312
    }
  ]
}
```

### Resume Behavior

On startup, the engine checks for an existing state file:

- **State found:** Resumes from where it left off. Already-processed files are skipped. Thresholds, mode, and counters are restored.
- **No state / `--reset`:** Starts fresh with default configuration.

---

## Health Monitoring

The streaming engine continuously monitors system health:

| Check | Threshold | Action |
|-------|-----------|--------|
| API status | Port 9000 reachable | Warning if unreachable |
| Disk space | < 2 GB free | Warning logged |
| Log size | > 10 MB | Log rotation |
| Model files | Exist on disk | Warning if missing |
| Error rate | Consecutive failures | Warning + possible pause |

---

## Monitoring a Running Session

### Log File

The streaming log is written to `mitraseti_artifacts/logs/streaming.log`:

```
2025-01-15 08:00:00 | INFO  | Streaming started (mode=normal, days=7)
2025-01-15 08:00:32 | INFO  | Processed obs_001.fil: 847 signals, 1 candidate, 0.06s
2025-01-15 08:01:04 | INFO  | Processed obs_002.fil: 234 signals, 0 candidates, 0.43s
2025-01-15 08:30:00 | INFO  | Health check: OK (disk=142GB, model=loaded)
2025-01-16 00:00:00 | INFO  | Daily report generated: streaming_reports/daily/2025-01-15.html
```

### Real-Time via WebSocket

If the web server is running, connect to `ws://localhost:8000/ws/live` for real-time signal updates during processing.

### Target Categorization

The engine categorizes observations by scientific target based on filename patterns (Voyager, Kepler, TRAPPIST, HIP, GJ, etc.) and tracks per-category statistics for analysis.
