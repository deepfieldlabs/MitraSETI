# ML Model Architecture

MitraSETI uses a hybrid CNN + Transformer signal classifier to automatically categorize radio signals into 9 classes, combined with an ensemble out-of-distribution detector to flag anomalous signals that don't fit any known category.

---

## Architecture Overview

```
Input spectrogram (256 freq × 64 time)
    │
    ▼
┌──────────────────────────────────┐
│     SpectralCNNBackbone          │
│                                  │
│  Conv1d 1→32, kernel=7, ReLU    │
│  BatchNorm1d(32)                 │
│  Conv1d 32→64, kernel=5, ReLU   │
│  BatchNorm1d(64)                 │
│  Conv1d 64→128, kernel=3, ReLU  │
│  BatchNorm1d(128)                │
│  AdaptiveAvgPool1d              │
│                                  │
│  Output: (batch, time, 128)      │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│     PositionalEncoding           │
│  Sinusoidal temporal encoding    │
│  Added to CNN features           │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│     TransformerEncoder           │
│                                  │
│  2 layers, 4 attention heads     │
│  d_model = 128                   │
│  GELU activation                 │
│  Dropout 0.1                     │
│                                  │
│  Captures temporal dependencies  │
│  across time steps               │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│     Classification Head (MLP)    │
│                                  │
│  Linear(128 → 128), ReLU        │
│  Dropout 0.1                     │
│  Linear(128 → 9)                │
│                                  │
│  Output: 9-class logits          │
└──────────────┬───────────────────┘
               │
               ├──→ Softmax → class probabilities
               ├──→ argmax → predicted class
               └──→ feature projection → 128-dim vector
                    (for OOD detection)
```

---

## SpectralCNNBackbone

The backbone processes each time step's frequency spectrum independently through 1D convolutions, extracting hierarchical spectral features:

| Layer | In Channels | Out Channels | Kernel Size | Purpose |
|-------|-------------|--------------|-------------|---------|
| Conv1d + BN + ReLU | 1 | 32 | 7 | Capture broad spectral features |
| Conv1d + BN + ReLU | 32 | 64 | 5 | Intermediate spectral patterns |
| Conv1d + BN + ReLU | 64 | 128 | 3 | Fine-grained spectral detail |
| AdaptiveAvgPool1d | 128 | 128 | — | Reduce to fixed-size embedding |

The decreasing kernel sizes (7 → 5 → 3) progressively narrow the receptive field, moving from broad spectral context to fine-grained feature detection. BatchNorm after each convolution stabilizes training across the wide dynamic range of radio data.

Output shape: `(batch_size, time_steps, 128)` — one 128-dimensional feature vector per time step.

---

## Positional Encoding

Standard sinusoidal positional encoding adds temporal information to the CNN features before they enter the Transformer. Without this, the Transformer's self-attention mechanism would be permutation-invariant and unable to model signal drift over time.

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where `pos` is the time step index and `i` is the dimension index within the 128-dim embedding.

---

## Transformer Encoder

The Transformer captures temporal dependencies across time steps — critical for detecting signals that drift in frequency over the observation duration.

| Parameter | Value |
|-----------|-------|
| Layers | 2 |
| Attention heads | 4 |
| Model dimension | 128 |
| Feedforward dimension | 512 |
| Activation | GELU |
| Dropout | 0.1 |

With 4 attention heads of 32 dimensions each, the model learns multiple types of temporal relationships simultaneously. Two layers provides sufficient depth for the temporal patterns in SETI data without overfitting on limited training data.

---

## Classification Head

A two-layer MLP maps the Transformer output to 9 class logits:

```
mean(transformer_output, dim=time)  →  Linear(128, 128)  →  ReLU  →  Dropout(0.1)  →  Linear(128, 9)
```

Mean-pooling across time steps before the head provides a fixed-size representation regardless of observation duration.

Additionally, a **feature projection** layer produces a 128-dimensional feature vector used by the OOD detector for spectral distance computation.

---

## 9 Signal Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | `NARROWBAND_DRIFTING` | Narrowband signal with non-zero drift rate — the primary ET signature. A transmitter on a rotating/orbiting body produces a frequency drift due to changing relative velocity. |
| 1 | `NARROWBAND_STATIONARY` | Narrowband signal with zero or near-zero drift rate — likely local RFI or a geostationary satellite. |
| 2 | `BROADBAND` | Wideband emission spanning many channels — typically natural astrophysical emission or broadband RFI. |
| 3 | `PULSED` | Periodic pulsed signal — could be a pulsar or a radar transmitter. |
| 4 | `CHIRP` | Frequency-swept signal — linear or nonlinear frequency change over time, common in radar systems. |
| 5 | `RFI_TERRESTRIAL` | Terrestrial radio frequency interference — cell towers, WiFi, power lines, vehicles. |
| 6 | `RFI_SATELLITE` | Satellite downlink interference — GPS, Iridium, communication satellites. |
| 7 | `NOISE` | Background noise with no detectable signal — statistical fluctuation. |
| 8 | `CANDIDATE_ET` | Passes all filters and does not match known RFI patterns — requires human review. This class is the rarest and most significant. |

---

## Out-of-Distribution (OOD) Detection

Signals that don't match any of the 9 training classes should be flagged as anomalous rather than forced into the nearest category. MitraSETI uses an ensemble of three complementary methods.

### Method 1: Maximum Softmax Probability (MSP)

The simplest OOD indicator. If the model is uncertain about its prediction (max softmax probability is low), the input may be outside the training distribution.

```
msp_score = 1 - max(softmax(logits))
```

High `msp_score` → likely OOD.

### Method 2: Energy Score

Based on the free energy of the logit vector. In-distribution samples tend to produce higher energy (more confident logit distributions).

```
energy_score = -T × log(Σ exp(logit_i / T))
```

Where `T` is a temperature parameter. Low energy → likely OOD.

### Method 3: Spectral Distance

Measures the distance between the signal's 128-dim feature vector and the reference feature distribution computed from in-distribution training data.

```
spectral_distance = z_score(||feature - mean_ref|| / std_ref)
```

The reference distribution statistics (mean, standard deviation) are computed during calibration. The threshold is set at the 95th percentile of reference distances.

### Ensemble Decision

A signal is flagged as anomalous when **≥2 of the 3 methods** agree that it is out-of-distribution. This voting scheme reduces false positives from any single method.

```
OODResult:
  ood_score:     float   # Combined score (0.0 = in-distribution, 1.0 = anomalous)
  is_anomaly:    bool    # True if ≥2 methods flag OOD
  threshold:     float   # Calibrated decision threshold
  method_scores: dict    # Per-method scores (msp, energy, spectral)
```

---

## Training Pipeline

### Auto-Training from Streaming Data

The streaming observation engine triggers model training automatically:

| Trigger | Action |
|---------|--------|
| After 5 processed files | Initial model training from scratch |
| Every 500 processing cycles | Fine-tune existing model on accumulated data |

### Training Data Sources

1. **Cached spectrograms:** Real spectrograms saved during ML inference in streaming mode
2. **Synthetic data:** Generated with controlled signal injection (drifting narrowband, broadband, pulsed, chirp)
3. **Data augmentation:** Gaussian noise injection + frequency shift

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Early stopping | Patience 5 epochs |
| Loss | CrossEntropyLoss |

### OOD Calibration

After training, the OOD detector is calibrated by:

1. Running all training data through the classifier
2. Computing reference feature statistics (mean, std)
3. Setting spectral distance threshold at the 95th percentile
4. Saving calibration data to `models/ood_calibration.json`

---

## Model Files

| File | Size | Description |
|------|------|-------------|
| `models/signal_classifier_v1.pt` | ~3 MB | CNN+Transformer weights |
| `models/ood_calibration.json` | ~5 MB | OOD detector calibration thresholds and reference statistics |

---

## Device Support

Inference automatically selects the best available compute device:

| Priority | Device | Backend |
|----------|--------|---------|
| 1 | NVIDIA GPU | CUDA |
| 2 | Apple Silicon | MPS |
| 3 | CPU | Default |

Device detection is cached after the first call to avoid repeated hardware probing. Batch inference groups signals into sub-batches of 128 for efficient GPU utilization.

---

## Feature Extractor

Alongside ML classification, a physics-based feature extractor (`inference/feature_extractor.py`) computes interpretable signal properties:

| Feature | Method | Unit |
|---------|--------|------|
| SNR | Peak-to-MAD ratio | dimensionless |
| Drift rate | Hough transform projection | Hz/s |
| Bandwidth | 50% peak power width | Hz |
| Central frequency | Peak frequency bin | Hz |
| Duration | Time extent above threshold | seconds |
| Spectral index | Log-log linear regression of power vs frequency | dimensionless |
| Polarization ratio | Cross-pol power ratio | 0–1 |
| Modulation index | Amplitude variation over time | dimensionless |
| Kurtosis | Spectral peakedness (4th moment) | dimensionless |
| Skewness | Spectral asymmetry (3rd moment) | dimensionless |

These features are stored alongside ML predictions for downstream analysis and are available through the API.
