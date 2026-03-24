#!/usr/bin/env python3
"""
Taylor Tree vs Brute-Force Speed Benchmark — MitraSETI

Rigorous wall-clock comparison of the Taylor tree O(N·F·log₂N)
de-Doppler algorithm against the brute-force O(D·N·F) approach.

Measures throughput (Mpoints/s), speedup factor, and scaling behaviour
across different data sizes, producing a publication-ready figure.

Output:
  mitraseti_artifacts/benchmarks/speed_comparison.png
  mitraseti_artifacts/benchmarks/speed_results_<timestamp>.json

Usage:
  python scripts/speed_benchmark.py
  python scripts/speed_benchmark.py --repeats 5 --max-chans 262144
"""

from __future__ import annotations

import argparse
import contextlib
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

logger = logging.getLogger("mitraseti.benchmark")

BENCH_DIR = ARTIFACTS_DIR / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)


def create_synthetic_header(n_chans: int, n_times: int):
    """Create a synthetic filterbank header for benchmarking."""
    try:
        import mitraseti_core as _core
    except ImportError:
        import astroseti_core as _core

    return _core.FilterbankHeader(
        nchans=n_chans,
        nifs=1,
        nbits=32,
        tsamp=18.253611008,
        fch1=1420.5,
        foff=-2.7939677238464355e-06,
        tstart=59000.0,
        source_name="SYNTHETIC_BENCH",
        ra=0.0,
        dec=0.0,
    )


def run_single_benchmark(
    n_chans: int,
    n_times: int,
    use_taylor: bool,
    max_drift_rate: float = 4.0,
    min_snr: float = 10.0,
) -> Dict[str, Any]:
    """Run a single de-Doppler benchmark at the given data size."""
    try:
        import mitraseti_core as _core
    except ImportError:
        import astroseti_core as _core

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_times, n_chans)).astype(np.float32)

    mid_ch = n_chans // 2
    for t in range(n_times):
        drift_ch = int(3 * t / max(n_times - 1, 1))
        ch = mid_ch + drift_ch
        if 0 <= ch < n_chans:
            data[t, ch] += 30.0

    header = create_synthetic_header(n_chans, n_times)
    params = _core.SearchParams(
        max_drift_rate=max_drift_rate,
        min_snr=min_snr,
        use_taylor_tree=use_taylor,
    )
    engine = _core.DedopplerEngine(params)

    data_flat = data.ravel().tolist()

    # Warm-up run
    with contextlib.suppress(Exception):
        engine.search(data_flat, n_times, n_chans, header)

    # Timed run
    t0 = time.perf_counter()
    result = engine.search(data_flat, n_times, n_chans, header)
    elapsed = time.perf_counter() - t0

    total_points = n_times * n_chans
    throughput = total_points / elapsed / 1e6

    return {
        "n_chans": n_chans,
        "n_times": n_times,
        "total_points": total_points,
        "algorithm": "taylor_tree" if use_taylor else "brute_force",
        "elapsed_s": round(elapsed, 6),
        "throughput_mpts_s": round(throughput, 2),
        "n_candidates": len(result.candidates),
    }


def run_benchmark_suite(
    channel_sizes: List[int],
    n_times: int = 16,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Run the full benchmark suite across multiple data sizes."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_times": n_times,
        "repeats": repeats,
        "channel_sizes": channel_sizes,
        "taylor_tree": [],
        "brute_force": [],
    }

    for n_chans in channel_sizes:
        for algo_name, use_taylor in [("taylor_tree", True), ("brute_force", False)]:
            timings = []
            throughputs = []

            for _rep in range(repeats):
                r = run_single_benchmark(n_chans, n_times, use_taylor)
                timings.append(r["elapsed_s"])
                throughputs.append(r["throughput_mpts_s"])

            entry = {
                "n_chans": n_chans,
                "total_points": n_chans * n_times,
                "times_s": timings,
                "mean_s": round(np.mean(timings), 6),
                "std_s": round(np.std(timings), 6),
                "throughput_mpts_s": round(np.mean(throughputs), 2),
            }
            results[algo_name].append(entry)

            logger.info(
                f"  {algo_name:12s}  {n_chans:>8,} chans × {n_times} ints  "
                f"{np.mean(timings)*1000:8.1f} ms  "
                f"{np.mean(throughputs):8.1f} Mpts/s"
            )

    # Compute speedup factors
    speedups = []
    for tt, bf in zip(results["taylor_tree"], results["brute_force"], strict=False):
        speedup = bf["mean_s"] / tt["mean_s"] if tt["mean_s"] > 0 else float("inf")
        speedups.append({
            "n_chans": tt["n_chans"],
            "speedup_x": round(speedup, 1),
            "taylor_ms": round(tt["mean_s"] * 1000, 1),
            "brute_ms": round(bf["mean_s"] * 1000, 1),
        })
    results["speedups"] = speedups

    return results


def plot_benchmark(results: Dict[str, Any], output_path: Path) -> None:
    """Generate a publication-ready speed comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0a0e18")

    tt = results["taylor_tree"]
    bf = results["brute_force"]
    speedups = results["speedups"]

    chans_tt = [e["n_chans"] for e in tt]
    chans_bf = [e["n_chans"] for e in bf]
    time_tt = [e["mean_s"] * 1000 for e in tt]
    time_bf = [e["mean_s"] * 1000 for e in bf]
    tput_tt = [e["throughput_mpts_s"] for e in tt]
    tput_bf = [e["throughput_mpts_s"] for e in bf]

    # Left: Time vs data size
    ax1.set_facecolor("#080c14")
    ax1.plot(chans_tt, time_tt, "o-", color="#00d4ff", linewidth=2, markersize=6,
             label="Taylor Tree (this work)")
    ax1.plot(chans_bf, time_bf, "s--", color="#ff3366", linewidth=2, markersize=6,
             label="Brute-Force")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Frequency Channels", color="#8ca5c8", fontsize=11)
    ax1.set_ylabel("Time (ms)", color="#8ca5c8", fontsize=11)
    ax1.set_title("De-Doppler Search Time", color="#e0e8f0", fontsize=14, fontweight=300)
    ax1.legend(facecolor="#0f192d", edgecolor="#1a3a5c", labelcolor="#8ca5c8", fontsize=10)
    ax1.grid(True, alpha=0.1, color="#4da6ff")
    ax1.tick_params(colors="#8ca5c8")
    for spine in ax1.spines.values():
        spine.set_color("#1a3a5c")

    # Annotate speedup on the left plot
    for s in speedups:
        idx_tt = chans_tt.index(s["n_chans"])
        ax1.annotate(
            f"{s['speedup_x']}×",
            xy=(s["n_chans"], time_tt[idx_tt]),
            xytext=(0, -18), textcoords="offset points",
            color="#00ff88", fontsize=9, fontweight=600, ha="center",
        )

    # Right: Throughput bar chart
    ax2.set_facecolor("#080c14")
    x = np.arange(len(chans_tt))
    w = 0.35
    ax2.bar(x - w/2, tput_tt, w, color="#00d4ff", alpha=0.8, label="Taylor Tree")
    ax2.bar(x + w/2, tput_bf, w, color="#ff3366", alpha=0.8, label="Brute-Force")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c//1024}K" if c >= 1024 else str(c) for c in chans_tt],
                        fontsize=9)
    ax2.set_xlabel("Channels", color="#8ca5c8", fontsize=11)
    ax2.set_ylabel("Throughput (Mpoints/s)", color="#8ca5c8", fontsize=11)
    ax2.set_title("Processing Throughput", color="#e0e8f0", fontsize=14, fontweight=300)
    ax2.legend(facecolor="#0f192d", edgecolor="#1a3a5c", labelcolor="#8ca5c8", fontsize=10)
    ax2.grid(True, alpha=0.1, color="#4da6ff", axis="y")
    ax2.tick_params(colors="#8ca5c8")
    for spine in ax2.spines.values():
        spine.set_color("#1a3a5c")

    fig.suptitle(
        "MitraSETI — Taylor Tree vs Brute-Force De-Doppler Performance",
        color="#4da6ff", fontsize=16, fontweight=300, y=0.98,
    )
    fig.text(
        0.5, 0.01,
        f"N_time = {results['n_times']} · {results['repeats']} repeats · "
        f"{results['timestamp'][:10]}",
        ha="center", color="#8ca5c8", fontsize=9, alpha=0.6,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Benchmark plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MitraSETI Speed Benchmark")
    parser.add_argument("--repeats", type=int, default=3, help="Repetitions per size (default: 3)")
    parser.add_argument("--n-times", type=int, default=16, help="Time integrations (default: 16)")
    parser.add_argument("--max-chans", type=int, default=131072,
                        help="Max channels to test (default: 131072 = 128K)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sizes = []
    ch = 1024
    while ch <= args.max_chans:
        sizes.append(ch)
        ch *= 2

    logger.info(f"Speed benchmark: {len(sizes)} sizes, {args.repeats} repeats")
    logger.info(f"  Sizes: {[f'{s//1024}K' for s in sizes]}")

    results = run_benchmark_suite(sizes, n_times=args.n_times, repeats=args.repeats)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = BENCH_DIR / f"speed_results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    plot_path = BENCH_DIR / "speed_comparison.png"
    try:
        plot_benchmark(results, plot_path)
    except Exception as e:
        logger.warning(f"Failed to generate plot: {e}")

    print(f"\n{'='*60}")
    print("SPEED BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Channels':>10s}  {'Taylor(ms)':>12s}  {'Brute(ms)':>12s}  {'Speedup':>8s}")
    print(f"{'-'*10:>10s}  {'-'*12:>12s}  {'-'*12:>12s}  {'-'*8:>8s}")
    for s in results["speedups"]:
        print(f"{s['n_chans']:>10,}  {s['taylor_ms']:>12.1f}  {s['brute_ms']:>12.1f}  {s['speedup_x']:>7.1f}×")

    max_speedup = max(s["speedup_x"] for s in results["speedups"])
    print(f"\n  Peak speedup: {max_speedup}× at {max(sizes):,} channels")
    print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
