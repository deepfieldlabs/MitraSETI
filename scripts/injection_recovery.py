#!/usr/bin/env python3
"""
Signal Injection & Recovery Test — MitraSETI

Standard SETI benchmark: inject synthetic drifting narrowband signals into
REAL Breakthrough Listen data at controlled SNR levels, then measure the
pipeline's recovery rate.  Produces detection efficiency (completeness)
curves — the primary metric for validating any de-Doppler search tool.

This script:
  1. Loads real .fil / .h5 observation files
  2. Injects synthetic signals at a grid of (SNR, drift_rate) values
  3. Runs the pipeline in both Taylor tree and brute-force modes
  4. Measures recovery fraction at each parameter point
  5. Generates a publication-ready completeness plot

Output:
  - mitraseti_artifacts/injection_recovery/results_<timestamp>.json
  - mitraseti_artifacts/injection_recovery/completeness_curve.png

Usage:
  python scripts/injection_recovery.py
  python scripts/injection_recovery.py --snr-min 3 --snr-max 50 --snr-steps 12
  python scripts/injection_recovery.py --n-injections 50 --files 3
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
from typing import Any, Dict, List, Optional, Tuple

if "HDF5_PLUGIN_PATH" not in os.environ:
    os.environ["HDF5_PLUGIN_PATH"] = ""

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import ARTIFACTS_DIR, FILTERBANK_DIR

logger = logging.getLogger("mitraseti.injection")

RESULTS_DIR = ARTIFACTS_DIR / "injection_recovery"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def inject_signal(
    data: np.ndarray,
    n_times: int,
    n_chans: int,
    freq_channel: int,
    drift_channels: float,
    target_snr: float,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Inject a synthetic narrowband drifting signal into a spectrogram.

    The signal has constant power in one channel at each time step,
    drifting linearly from freq_channel to freq_channel + drift_channels.

    Args:
        data: Original spectrogram (n_times, n_chans), modified in-place.
        freq_channel: Starting channel for the injected signal.
        drift_channels: Total channel drift over the observation.
        target_snr: Desired signal-to-noise ratio (per-channel MAD-based).

    Returns:
        (modified_data, injection_metadata)
    """
    col_medians = np.median(data, axis=0)
    col_mads = np.median(np.abs(data - col_medians[np.newaxis, :]), axis=0)
    col_sigmas = 1.4826 * col_mads
    col_sigmas[col_sigmas < 1e-10] = np.median(col_sigmas[col_sigmas > 0]) or 1.0

    channels_hit = []
    for t in range(n_times):
        ch = freq_channel + int(round(drift_channels * t / max(n_times - 1, 1)))
        if 0 <= ch < n_chans:
            amplitude = target_snr * col_sigmas[ch]
            data[t, ch] += amplitude
            channels_hit.append(ch)

    metadata = {
        "freq_channel": freq_channel,
        "drift_channels": drift_channels,
        "target_snr": target_snr,
        "channels_hit": [int(channels_hit[0]), int(channels_hit[-1])] if channels_hit else [],
        "seed": seed,
    }
    return data, metadata


def check_recovery(
    candidates: List[Dict],
    injection: Dict[str, Any],
    header: Dict[str, Any],
    freq_tol_channels: int = 5,
    drift_tol_hz_s: float = 0.5,
) -> bool:
    """Check if any detected candidate matches the injected signal."""
    fch1 = header.get("fch1", 1420.0)
    foff = abs(header.get("foff", 0.00028))
    tsamp = header.get("tsamp", 18.0)
    n_times = header.get("n_times", 16)

    inj_ch = injection["freq_channel"]
    inj_drift_ch = injection["drift_channels"]
    obs_length = tsamp * n_times
    channel_bw_hz = foff * 1e6
    inj_drift_hz_s = (inj_drift_ch * channel_bw_hz / obs_length) if obs_length > 0 else 0

    for c in candidates:
        freq_hz = c.get("frequency_hz", 0)
        cand_ch = int((fch1 * 1e6 - freq_hz) / (channel_bw_hz)) if channel_bw_hz > 0 else 0
        if foff < 0:
            cand_ch = int((freq_hz - fch1 * 1e6) / channel_bw_hz)

        cand_drift = c.get("drift_rate", 0)

        if (abs(cand_ch - inj_ch) <= freq_tol_channels and
                abs(cand_drift - inj_drift_hz_s) <= drift_tol_hz_s):
            return True

    return False


def run_injection_recovery(
    snr_values: List[float],
    drift_values: List[float],
    n_injections: int = 20,
    max_files: int = 5,
    min_snr_threshold: float = 5.0,
) -> Dict[str, Any]:
    """Run the full injection/recovery experiment.

    Returns a results dict with completeness at each (snr, drift) point
    for both Taylor tree and brute-force algorithms.
    """
    try:
        import mitraseti_core as _core
    except ImportError:
        try:
            import astroseti_core as _core
        except ImportError:
            logger.error("Rust core not available — cannot run injection test")
            return {}

    fil_files = sorted(FILTERBANK_DIR.glob("*.fil"))
    h5_files = sorted(FILTERBANK_DIR.glob("*.h5"))
    all_files = fil_files + h5_files

    if not all_files:
        logger.error(f"No filterbank files found in {FILTERBANK_DIR}")
        return {}

    files_to_use = all_files[:max_files]
    logger.info(f"Using {len(files_to_use)} files for injection/recovery")

    from pipeline import MitraSETIPipeline
    pipe = MitraSETIPipeline()

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_injections_per_point": n_injections,
        "files_used": [f.name for f in files_to_use],
        "snr_values": snr_values,
        "drift_values": drift_values,
        "taylor_tree": {},
        "brute_force": {},
    }

    rng = np.random.default_rng(42)

    for algo_name, use_taylor in [("taylor_tree", True), ("brute_force", False)]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Algorithm: {algo_name}")
        logger.info(f"{'='*60}")

        params = _core.SearchParams(
            max_drift_rate=4.0,
            min_snr=min_snr_threshold,
            use_taylor_tree=use_taylor,
        )
        pipe._dedoppler = _core.DedopplerEngine(params)

        algo_results = {}

        for drift_ch in drift_values:
            drift_key = f"drift_{drift_ch:.1f}"
            algo_results[drift_key] = {}

            for target_snr in snr_values:
                snr_key = f"snr_{target_snr:.1f}"
                recovered = 0
                total = 0
                times = []

                for inj_idx in range(n_injections):
                    file_idx = inj_idx % len(files_to_use)
                    filepath = files_to_use[file_idx]

                    try:
                        file_info = pipe._read_file(str(filepath))
                    except Exception as e:
                        logger.warning(f"Failed to read {filepath.name}: {e}")
                        continue

                    data = file_info["data"].copy()
                    n_t = file_info["n_times"]
                    n_c = file_info["n_chans"]
                    header = file_info["header"]

                    margin = int(abs(drift_ch)) + 50
                    max_start = n_c - margin - 1
                    min_start = margin
                    if max_start <= min_start:
                        continue

                    freq_ch = rng.integers(min_start, max_start)

                    data, injection = inject_signal(
                        data, n_t, n_c, freq_ch, drift_ch, target_snr,
                        seed=inj_idx,
                    )

                    file_info_copy = {
                        "header": header,
                        "data": data,
                        "n_times": n_t,
                        "n_chans": n_c,
                    }

                    t0 = time.perf_counter()
                    try:
                        raw_cands = pipe._run_dedoppler(file_info_copy)
                    except Exception as e:
                        logger.warning(f"De-Doppler failed: {e}")
                        continue
                    elapsed = time.perf_counter() - t0
                    times.append(elapsed)

                    if check_recovery(raw_cands, injection, header):
                        recovered += 1
                    total += 1

                completeness = recovered / total if total > 0 else 0.0
                avg_time = np.mean(times) if times else 0.0

                algo_results[drift_key][snr_key] = {
                    "recovered": recovered,
                    "total": total,
                    "completeness": completeness,
                    "avg_time_s": round(avg_time, 4),
                }

                logger.info(
                    f"  drift={drift_ch:+5.1f}ch, SNR={target_snr:5.1f}: "
                    f"{recovered}/{total} = {completeness:.1%} "
                    f"({avg_time*1000:.1f} ms avg)"
                )

        results[algo_name] = algo_results

    return results


def plot_completeness(results: Dict[str, Any], output_path: Path) -> None:
    """Generate a publication-ready completeness curve plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor("#0a0e18")

    snr_values = results["snr_values"]
    drift_values = results["drift_values"]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(drift_values)))

    for ax_idx, (algo, algo_label) in enumerate([
        ("taylor_tree", "Taylor Tree (this work)"),
        ("brute_force", "Brute-Force"),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor("#080c14")
        ax.set_title(algo_label, color="#e0e8f0", fontsize=14, fontweight=300, pad=12)
        ax.set_xlabel("Injected SNR", color="#8ca5c8", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Recovery Fraction", color="#8ca5c8", fontsize=11)

        algo_data = results.get(algo, {})
        for i, drift_ch in enumerate(drift_values):
            drift_key = f"drift_{drift_ch:.1f}"
            drift_data = algo_data.get(drift_key, {})

            completeness_vals = []
            for snr in snr_values:
                snr_key = f"snr_{snr:.1f}"
                entry = drift_data.get(snr_key, {})
                completeness_vals.append(entry.get("completeness", 0.0))

            label = f"drift = {drift_ch:+.0f} ch"
            ax.plot(
                snr_values, completeness_vals,
                marker="o", markersize=5, linewidth=2,
                color=colors[i], label=label,
            )

        ax.axhline(y=0.9, color="#ff3366", linestyle="--", alpha=0.4, linewidth=1)
        ax.text(
            snr_values[-1], 0.92, "90% threshold",
            color="#ff3366", alpha=0.5, fontsize=9, ha="right",
        )

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(min(snr_values) - 1, max(snr_values) + 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(
            loc="lower right", fontsize=9,
            facecolor="#0f192d", edgecolor="#1a3a5c",
            labelcolor="#8ca5c8",
        )
        ax.tick_params(colors="#8ca5c8")
        ax.grid(True, alpha=0.1, color="#4da6ff")
        for spine in ax.spines.values():
            spine.set_color("#1a3a5c")

    fig.suptitle(
        "MitraSETI Signal Injection & Recovery — Detection Efficiency",
        color="#4da6ff", fontsize=16, fontweight=300, y=0.98,
    )
    fig.text(
        0.5, 0.01,
        f"Injections per point: {results['n_injections_per_point']} · "
        f"Files: {len(results['files_used'])} · "
        f"{results['timestamp'][:10]}",
        ha="center", color="#8ca5c8", fontsize=9, alpha=0.6,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Completeness plot saved to {output_path}")


def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary table to stdout."""
    snr_values = results.get("snr_values", [])

    print(f"\n{'='*70}")
    print("SIGNAL INJECTION & RECOVERY — SUMMARY")
    print(f"{'='*70}")
    print(f"Files used:           {len(results.get('files_used', []))}")
    print(f"Injections per point: {results.get('n_injections_per_point', 0)}")
    print(f"SNR range:            {min(snr_values):.0f} – {max(snr_values):.0f}")
    print(f"Timestamp:            {results.get('timestamp', '')}")

    for algo_name, algo_label in [("taylor_tree", "TAYLOR TREE"), ("brute_force", "BRUTE-FORCE")]:
        algo_data = results.get(algo_name, {})
        if not algo_data:
            continue

        print(f"\n--- {algo_label} ---")
        print(f"{'Drift':>8s}", end="")
        for snr in snr_values:
            print(f"  SNR={snr:4.0f}", end="")
        print()

        for drift_key, drift_data in sorted(algo_data.items()):
            drift_label = drift_key.replace("drift_", "").replace(".0", "")
            print(f"{drift_label:>8s}ch", end="")
            for snr in snr_values:
                snr_key = f"snr_{snr:.1f}"
                entry = drift_data.get(snr_key, {})
                comp = entry.get("completeness", 0)
                print(f"  {comp:7.0%}", end="")
            print()

    # 90% completeness SNR
    print(f"\n--- 90% completeness SNR ---")
    for algo_name, algo_label in [("taylor_tree", "Taylor"), ("brute_force", "Brute")]:
        algo_data = results.get(algo_name, {})
        for drift_key, drift_data in sorted(algo_data.items()):
            for i, snr in enumerate(snr_values):
                snr_key = f"snr_{snr:.1f}"
                if drift_data.get(snr_key, {}).get("completeness", 0) >= 0.9:
                    print(f"  {algo_label:8s} {drift_key}: SNR >= {snr:.0f}")
                    break
            else:
                print(f"  {algo_label:8s} {drift_key}: not reached")


def main():
    parser = argparse.ArgumentParser(
        description="MitraSETI Signal Injection & Recovery Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--snr-min", type=float, default=5.0, help="Minimum SNR (default: 5)")
    parser.add_argument("--snr-max", type=float, default=50.0, help="Maximum SNR (default: 50)")
    parser.add_argument("--snr-steps", type=int, default=8, help="Number of SNR steps (default: 8)")
    parser.add_argument("--drift-values", type=str, default="-4,-2,0,2,4",
                        help="Comma-separated drift values in channels (default: -4,-2,0,2,4)")
    parser.add_argument("--n-injections", type=int, default=20,
                        help="Injections per (SNR, drift) point (default: 20)")
    parser.add_argument("--files", type=int, default=5, help="Max files to use (default: 5)")
    parser.add_argument("--min-snr", type=float, default=5.0,
                        help="Pipeline min_snr threshold (default: 5.0)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    snr_values = list(np.linspace(args.snr_min, args.snr_max, args.snr_steps))
    drift_values = [float(x) for x in args.drift_values.split(",")]

    logger.info("Starting signal injection & recovery test")
    logger.info(f"  SNR range: {args.snr_min} to {args.snr_max} ({args.snr_steps} steps)")
    logger.info(f"  Drift values: {drift_values}")
    logger.info(f"  Injections per point: {args.n_injections}")
    logger.info(f"  Max files: {args.files}")

    results = run_injection_recovery(
        snr_values=snr_values,
        drift_values=drift_values,
        n_injections=args.n_injections,
        max_files=args.files,
        min_snr_threshold=args.min_snr,
    )

    if not results:
        logger.error("No results produced")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    plot_path = RESULTS_DIR / "completeness_curve.png"
    try:
        plot_completeness(results, plot_path)
    except Exception as e:
        logger.warning(f"Failed to generate plot: {e}")

    print_summary(results)


if __name__ == "__main__":
    main()
