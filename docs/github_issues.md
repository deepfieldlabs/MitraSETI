# Curated GitHub Issues for MitraSETI

These issues are selected from the IMPROVEMENTS_LOG and represent real development activity. Copy-paste each section into GitHub when creating issues after pushing.

---

## Issue 1: Implement Taylor tree de-Doppler algorithm

**Labels:** enhancement, performance

**Title:** [FEATURE] Implement Taylor tree de-Doppler algorithm

**Body:**
```
Currently the de-Doppler engine uses brute-force drift search which is O(N*M) for N frequency channels and M drift rates. The Taylor tree algorithm reduces this to O(N*log(N)), providing significant speedup for high-resolution data.

**Impact**: 10-100x faster de-Doppler on files with 1M+ channels
**Reference**: Taylor (1974), "A sensitive method for detecting dispersed signals"
```

---

## Issue 2: GPU-accelerated de-Doppler via CUDA/Metal

**Labels:** enhancement, performance

**Title:** [FEATURE] GPU-accelerated de-Doppler via CUDA/Metal

**Body:**
```
Move the de-Doppler search to GPU for massive parallelism. Metal compute shaders for Apple Silicon, CUDA for NVIDIA GPUs.

**Impact**: 10-1000x speedup on large filterbank files
**Consideration**: PyO3 + cupy or wgpu-rs for cross-platform GPU
```

---

## Issue 3: CLI tool for headless server operation

**Labels:** enhancement, v0.2.0

**Title:** [FEATURE] CLI tool for headless server operation

**Body:**
```
Add a `mitraseti` CLI command that allows running the full pipeline from the command line without any UI. Useful for EC2, HPC, and Docker deployments.

Commands:
- `mitraseti process <file.fil>` — process a single file
- `mitraseti stream --dir <path>` — start streaming mode
- `mitraseti benchmark --realdata` — run benchmarks
- `mitraseti train --data-dir <path>` — train/fine-tune the model
```

---

## Issue 4: Real-time SDR input support

**Labels:** enhancement, v0.3.0

**Title:** [FEATURE] Real-time SDR input support

**Body:**
```
Add support for real-time signal input from Software Defined Radios (SDR) via SoapySDR/pyrtlsdr. This would enable MitraSETI to be used with consumer radio hardware for citizen science SETI.
```

---

## Issue 5: Pre-trained model zoo

**Labels:** enhancement, v0.2.0

**Title:** [FEATURE] Pre-trained model zoo

**Body:**
```
Publish multiple pre-trained models optimized for different observation scenarios:
- General purpose (current model)
- L-band optimized (1-2 GHz)
- S-band optimized (2-4 GHz)
- High-RFI environment
```

---

## Issue 6: Waterfall viewer performance on files > 500MB

**Labels:** bug, performance

**Title:** [BUG] Waterfall viewer performance on files > 500MB

**Body:**
```
Large filterbank files (>500MB) cause the waterfall viewer to be slow on initial load. The current approach subsamples via h5py slicing but rendering can still be sluggish.

**Workaround**: The viewer now uses block-average downsampling to max 2048x1024, but further optimization is needed for multi-GB files.
```

---

## Issue 7: Cloud deployment guide (AWS/GCP)

**Labels:** documentation, enhancement

**Title:** [FEATURE] Cloud deployment guide (AWS/GCP)

**Body:**
```
Write a deployment guide for running MitraSETI on cloud infrastructure:
- AWS: EC2 GPU instances, S3 for data, SageMaker for training
- GCP: Compute Engine, Cloud Storage, Vertex AI
- Terraform/CDK templates for infrastructure
```

---

## Issue 8: REST API authentication and rate limiting

**Labels:** enhancement, security

**Title:** [FEATURE] REST API authentication and rate limiting

**Body:**
```
The FastAPI backend currently has no authentication. For production deployment, add:
- API key authentication
- JWT tokens for web UI sessions
- Rate limiting per client
- CORS configuration for production domains
```
