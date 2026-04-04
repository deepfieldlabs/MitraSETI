#!/usr/bin/env python3
"""
GPU vs CPU Taylor Tree Benchmark — MitraSETI v0.3.0

Compares three backends:
  1. Rust + rayon (CPU)        — the v0.2.0 production implementation
  2. CuPy (GPU)                — new in v0.3.0
  3. NumPy (CPU fallback)      — pure Python reference

Measures wall-clock time, throughput (Mpoints/s), and speedup across
data sizes from 1K to 256K channels.

Output:
  mitraseti_artifacts/benchmarks/gpu_benchmark_<timestamp>.json
  mitraseti_artifacts/benchmarks/gpu_comparison.png

Usage:
  python scripts/gpu_benchmark.py
  python scripts/gpu_benchmark.py --max-chans 131072 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

if "HDF5_PLUGIN_PATH" not in os.environ:
    os.environ["HDF5_PLUGIN_PATH"] = ""

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import ARTIFACTS_DIR

logger = logging.getLogger("mitraseti.gpu_benchmark")

BENCH_DIR = ARTIFACTS_DIR / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)


def make_header(n_chans: int, n_times: int) -> Dict[str, Any]:
    """Create a synthetic header for benchmarking."""
    return {
        "fch1": 1500.0,
        "foff": -0.00028,
        "tsamp": 18.253611,
        "tstart": 59000.0,
        "source_name": "BENCHMARK",
        "nchans": n_chans,
    }


def benchmark_rust(
    data: np.ndarray,
    header: Dict[str, Any],
    n_times: int,
    n_chans: int,
    repeats: int,
) -> Dict[str, Any]:
    """Benchmark the Rust/rayon Taylor tree."""
    try:
        import mitraseti_core as _core
    except ImportError:
        try:
            import astroseti_core as _core
        except ImportError:
            return {"backend": "rust", "available": False}

    params = _core.SearchParams(max_drift_rate=4.0, min_snr=10.0, use_taylor_tree=True)
    engine = _core.DedopplerEngine(params)

    rust_header = _core.FilterbankHeader(
        nchans=n_chans,
        nifs=1,
        nbits=32,
        tsamp=header["tsamp"],
        fch1=header["fch1"],
        foff=header["foff"],
        tstart=header["tstart"],
        source_name="BENCHMARK",
        ra=0.0,
        dec=0.0,
    )

    flat = data.astype(np.float32).ravel().tolist()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        engine.search(flat, n_times, n_chans, rust_header)
        times.append(time.perf_counter() - t0)

    total_points = n_times * n_chans
    mean_s = float(np.mean(times))

    return {
        "backend": "rust_rayon",
        "available": True,
        "times_s": [round(t, 6) for t in times],
        "mean_s": round(mean_s, 6),
        "std_s": round(float(np.std(times)), 6),
        "throughput_mpts_s": round(total_points / mean_s / 1e6, 2),
    }


def benchmark_gpu(
    data: np.ndarray,
    header: Dict[str, Any],
    repeats: int,
) -> Dict[str, Any]:
    """Benchmark the CuPy GPU Taylor tree."""
    from core_gpu.taylor_tree_gpu import get_gpu_info, gpu_taylor_tree_search, is_gpu_available

    if not is_gpu_available():
        return {"backend": "cupy_gpu", "available": False}

    n_times, n_chans = data.shape
    total_points = n_times * n_chans

    # Warm-up run
    gpu_taylor_tree_search(data, header, max_drift_rate=4.0, min_snr=10.0)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        gpu_taylor_tree_search(data, header, max_drift_rate=4.0, min_snr=10.0)
        times.append(time.perf_counter() - t0)

    mean_s = float(np.mean(times))

    return {
        "backend": "cupy_gpu",
        "available": True,
        "gpu_info": get_gpu_info(),
        "times_s": [round(t, 6) for t in times],
        "mean_s": round(mean_s, 6),
        "std_s": round(float(np.std(times)), 6),
        "throughput_mpts_s": round(total_points / mean_s / 1e6, 2),
    }


def benchmark_numpy(
    data: np.ndarray,
    header: Dict[str, Any],
    repeats: int,
) -> Dict[str, Any]:
    """Benchmark the NumPy CPU fallback."""
    from core_gpu.taylor_tree_gpu import gpu_taylor_tree_search

    n_times, n_chans = data.shape
    total_points = n_times * n_chans

    # Force NumPy by temporarily disabling GPU
    import core_gpu.taylor_tree_gpu as mod

    orig = mod._GPU_AVAILABLE
    mod._GPU_AVAILABLE = False

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        gpu_taylor_tree_search(data, header, max_drift_rate=4.0, min_snr=10.0)
        times.append(time.perf_counter() - t0)

    mod._GPU_AVAILABLE = orig

    mean_s = float(np.mean(times))

    return {
        "backend": "numpy_cpu",
        "available": True,
        "times_s": [round(t, 6) for t in times],
        "mean_s": round(mean_s, 6),
        "std_s": round(float(np.std(times)), 6),
        "throughput_mpts_s": round(total_points / mean_s / 1e6, 2),
    }


def run_benchmark_suite(
    channel_sizes: List[int],
    n_times: int = 16,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Run the full benchmark suite across all backends and data sizes."""

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_times": n_times,
        "repeats": repeats,
        "channel_sizes": channel_sizes,
        "rust": [],
        "gpu": [],
        "numpy": [],
        "speedups": [],
    }

    for n_chans in channel_sizes:
        print(f"\n{'=' * 60}")
        print(f"  Channels: {n_chans:,}  |  Total points: {n_times * n_chans:,}")
        print(f"{'=' * 60}")

        # Synthetic data with an injected drifting signal
        np.random.seed(42)
        data = np.random.randn(n_times, n_chans).astype(np.float32)
        sig_ch = n_chans // 3
        for t in range(n_times):
            ch = sig_ch + t
            if 0 <= ch < n_chans:
                data[t, ch] += 20.0

        header = make_header(n_chans, n_times)

        # Rust benchmark
        print("  [1/3] Rust/rayon ...", end=" ", flush=True)
        rust_result = benchmark_rust(data, header, n_times, n_chans, repeats)
        if rust_result.get("available"):
            print(f"{rust_result['mean_s'] * 1000:.1f} ms")
        else:
            print("not available")
        results["rust"].append({**rust_result, "n_chans": n_chans})

        # GPU benchmark
        print("  [2/3] CuPy/GPU ...", end=" ", flush=True)
        gpu_result = benchmark_gpu(data, header, repeats)
        if gpu_result.get("available"):
            print(f"{gpu_result['mean_s'] * 1000:.1f} ms")
        else:
            print("not available")
        results["gpu"].append({**gpu_result, "n_chans": n_chans})

        # NumPy benchmark
        print("  [3/3] NumPy/CPU ...", end=" ", flush=True)
        numpy_result = benchmark_numpy(data, header, repeats)
        print(f"{numpy_result['mean_s'] * 1000:.1f} ms")
        results["numpy"].append({**numpy_result, "n_chans": n_chans})

        # Compute speedups
        speedup = {"n_chans": n_chans}
        rust_ms = rust_result.get("mean_s", 0) * 1000 if rust_result.get("available") else None
        gpu_ms = gpu_result.get("mean_s", 0) * 1000 if gpu_result.get("available") else None
        numpy_ms = numpy_result.get("mean_s", 0) * 1000

        speedup["rust_ms"] = round(rust_ms, 2) if rust_ms else None
        speedup["gpu_ms"] = round(gpu_ms, 2) if gpu_ms else None
        speedup["numpy_ms"] = round(numpy_ms, 2)

        if rust_ms and gpu_ms:
            speedup["gpu_vs_rust"] = round(rust_ms / gpu_ms, 1)
        if rust_ms:
            speedup["rust_vs_numpy"] = round(numpy_ms / rust_ms, 1)

        results["speedups"].append(speedup)

        print("  Speedups: ", end="")
        if rust_ms and gpu_ms:
            print(f"GPU vs Rust: {speedup.get('gpu_vs_rust', 'N/A')}x  |  ", end="")
        if rust_ms:
            print(f"Rust vs NumPy: {speedup.get('rust_vs_numpy', 'N/A')}x")
        else:
            print()

    return results


def main():
    parser = argparse.ArgumentParser(description="GPU vs CPU Taylor Tree Benchmark")
    parser.add_argument("--max-chans", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-times", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    sizes = []
    s = 1024
    while s <= args.max_chans:
        sizes.append(s)
        s *= 2

    print("=" * 60)
    print("  MitraSETI GPU vs CPU Benchmark")
    print(f"  Channels: {sizes}")
    print(f"  Time steps: {args.n_times}")
    print(f"  Repeats: {args.repeats}")
    print("=" * 60)

    results = run_benchmark_suite(sizes, args.n_times, args.repeats)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = BENCH_DIR / f"gpu_benchmark_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print(
        f"{'Channels':>10} {'Rust (ms)':>12} {'GPU (ms)':>12} {'NumPy (ms)':>12} {'GPU/Rust':>10}"
    )
    print(f"{'-' * 70}")
    for sp in results["speedups"]:
        rust = f"{sp['rust_ms']:.1f}" if sp.get("rust_ms") else "N/A"
        gpu = f"{sp['gpu_ms']:.1f}" if sp.get("gpu_ms") else "N/A"
        numpy = f"{sp['numpy_ms']:.1f}"
        ratio = f"{sp['gpu_vs_rust']:.1f}x" if sp.get("gpu_vs_rust") else "N/A"
        print(f"{sp['n_chans']:>10,} {rust:>12} {gpu:>12} {numpy:>12} {ratio:>10}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
