#!/usr/bin/env python3
"""
False Positive Rate & ROC Curve Analysis — MitraSETI

Measures the pipeline's false positive rate by running it on pure
Gaussian noise data (where there should be zero detections), then
combines with injection/recovery results to produce a proper ROC curve.

This is essential for publication: it quantifies both sensitivity (TPR)
and specificity (1 - FPR) of the de-Doppler + ML pipeline.

Output:
  mitraseti_artifacts/roc/fpr_results_<timestamp>.json
  mitraseti_artifacts/roc/roc_curve.png

Usage:
  python scripts/fpr_roc_analysis.py
  python scripts/fpr_roc_analysis.py --n-noise-trials 50 --snr-steps 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

if "HDF5_PLUGIN_PATH" not in os.environ:
    os.environ["HDF5_PLUGIN_PATH"] = ""

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import ARTIFACTS_DIR

logger = logging.getLogger("mitraseti.fpr")

ROC_DIR = ARTIFACTS_DIR / "roc"
ROC_DIR.mkdir(parents=True, exist_ok=True)


def measure_false_positive_rate(
    snr_thresholds: List[float],
    n_trials: int = 30,
    n_chans: int = 8192,
    n_times: int = 16,
) -> Dict[str, Any]:
    """Measure FPR at each SNR threshold on pure Gaussian noise.

    For each threshold, runs the de-Doppler search n_trials times on
    random noise data and counts how many false detections occur.
    """
    try:
        import mitraseti_core as _core
    except ImportError:
        try:
            import astroseti_core as _core
        except ImportError:
            logger.error("Rust core not available")
            return {}

    rng = np.random.default_rng(42)
    results = {"snr_thresholds": snr_thresholds, "n_trials": n_trials, "fpr": []}

    header = _core.FilterbankHeader(
        nchans=n_chans,
        nifs=1,
        nbits=32,
        tsamp=18.253611008,
        fch1=1420.5,
        foff=-2.7939677238464355e-06,
        tstart=59000.0,
        source_name="NOISE_TEST",
        ra=0.0,
        dec=0.0,
    )

    for snr_thresh in snr_thresholds:
        params = _core.SearchParams(
            max_drift_rate=4.0,
            min_snr=snr_thresh,
            use_taylor_tree=True,
        )
        engine = _core.DedopplerEngine(params)

        total_false_positives = 0
        total_channels_searched = 0

        for _trial in range(n_trials):
            noise = rng.standard_normal((n_times, n_chans)).astype(np.float32)
            data_flat = noise.ravel().tolist()

            result = engine.search(data_flat, n_times, n_chans, header)
            n_fp = len(result.candidates)
            total_false_positives += n_fp
            total_channels_searched += n_chans

        fpr = total_false_positives / (n_trials * n_chans)
        results["fpr"].append(
            {
                "snr_threshold": snr_thresh,
                "total_false_positives": total_false_positives,
                "total_channels": n_trials * n_chans,
                "fpr": round(fpr, 8),
                "fpr_per_channel": round(total_false_positives / total_channels_searched, 8),
            }
        )

        logger.info(
            f"  SNR>={snr_thresh:5.1f}: "
            f"{total_false_positives:4d} FP in {n_trials} noise trials "
            f"(FPR={fpr:.6f})"
        )

    return results


def load_injection_recovery() -> Dict[str, Any]:
    """Load the most recent injection/recovery results for TPR."""
    inj_dir = ARTIFACTS_DIR / "injection_recovery"
    files = sorted(inj_dir.glob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if files:
        with open(files[0]) as f:
            return json.load(f)
    return {}


def compute_roc(fpr_data: Dict, tpr_data: Dict) -> Dict[str, Any]:
    """Combine FPR and TPR data into ROC points."""
    roc_points = []

    fpr_by_snr = {e["snr_threshold"]: e["fpr"] for e in fpr_data.get("fpr", [])}

    tt_data = tpr_data.get("taylor_tree", {})
    snr_values = tpr_data.get("snr_values", [])

    for drift_key, drift_data in tt_data.items():
        for snr in snr_values:
            snr_key = f"snr_{snr:.1f}"
            entry = drift_data.get(snr_key, {})
            tpr = entry.get("completeness", 0)

            closest_snr = min(fpr_by_snr.keys(), key=lambda x: abs(x - snr), default=snr)
            fpr = fpr_by_snr.get(closest_snr, 0)

            roc_points.append(
                {
                    "snr_threshold": snr,
                    "drift": drift_key,
                    "tpr": tpr,
                    "fpr": fpr,
                }
            )

    return {"roc_points": roc_points}


def plot_roc_and_fpr(fpr_data: Dict, tpr_data: Dict, output_path: Path) -> None:
    """Generate ROC curve and FPR plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0a0e18")

    # Left: FPR vs SNR threshold
    ax1.set_facecolor("#080c14")
    fpr_entries = fpr_data.get("fpr", [])
    if fpr_entries:
        snr_vals = [e["snr_threshold"] for e in fpr_entries]
        fpr_vals = [e["fpr"] for e in fpr_entries]
        ax1.semilogy(snr_vals, fpr_vals, "o-", color="#ff3366", linewidth=2, markersize=6)
        ax1.fill_between(snr_vals, fpr_vals, alpha=0.1, color="#ff3366")

    ax1.set_xlabel("SNR Threshold", color="#8ca5c8", fontsize=11)
    ax1.set_ylabel("False Positive Rate", color="#8ca5c8", fontsize=11)
    ax1.set_title("FPR vs Detection Threshold", color="#e0e8f0", fontsize=14, fontweight=300)
    ax1.grid(True, alpha=0.1, color="#4da6ff")
    ax1.tick_params(colors="#8ca5c8")
    for spine in ax1.spines.values():
        spine.set_color("#1a3a5c")

    # Right: ROC curve (TPR vs FPR)
    ax2.set_facecolor("#080c14")

    tt_data = tpr_data.get("taylor_tree", {})
    snr_values = tpr_data.get("snr_values", [])
    fpr_by_snr = {e["snr_threshold"]: e["fpr"] for e in fpr_entries}
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(tt_data)))

    for i, (drift_key, drift_data) in enumerate(sorted(tt_data.items())):
        tpr_vals = []
        fpr_roc = []
        for snr in snr_values:
            snr_key = f"snr_{snr:.1f}"
            entry = drift_data.get(snr_key, {})
            tpr = entry.get("completeness", 0)
            closest = min(fpr_by_snr.keys(), key=lambda x: abs(x - snr), default=0)
            fpr = fpr_by_snr.get(closest, 0)
            tpr_vals.append(tpr)
            fpr_roc.append(fpr)

        label = drift_key.replace("drift_", "drift=").replace(".0", "")
        ax2.plot(
            fpr_roc, tpr_vals, "o-", color=colors[i], linewidth=2, markersize=5, label=f"{label} ch"
        )

    ax2.plot([0, 1], [0, 1], "--", color="#8ca5c8", alpha=0.3, linewidth=1)
    ax2.set_xlabel("False Positive Rate", color="#8ca5c8", fontsize=11)
    ax2.set_ylabel("True Positive Rate", color="#8ca5c8", fontsize=11)
    ax2.set_title("ROC Curve", color="#e0e8f0", fontsize=14, fontweight=300)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(facecolor="#0f192d", edgecolor="#1a3a5c", labelcolor="#8ca5c8", fontsize=9)
    ax2.grid(True, alpha=0.1, color="#4da6ff")
    ax2.tick_params(colors="#8ca5c8")
    for spine in ax2.spines.values():
        spine.set_color("#1a3a5c")

    fig.suptitle(
        "MitraSETI — False Positive Rate & ROC Analysis",
        color="#4da6ff",
        fontsize=16,
        fontweight=300,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MitraSETI False Positive Rate & ROC Analysis",
    )
    parser.add_argument(
        "--n-noise-trials",
        type=int,
        default=30,
        help="Number of pure-noise trials per threshold (default: 30)",
    )
    parser.add_argument(
        "--snr-steps", type=int, default=8, help="Number of SNR thresholds to test (default: 8)"
    )
    parser.add_argument("--snr-min", type=float, default=5.0, help="Min SNR (default: 5)")
    parser.add_argument("--snr-max", type=float, default=50.0, help="Max SNR (default: 50)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    snr_thresholds = list(np.linspace(args.snr_min, args.snr_max, args.snr_steps))

    logger.info("Measuring false positive rate on pure noise")
    fpr_data = measure_false_positive_rate(
        snr_thresholds,
        n_trials=args.n_noise_trials,
    )

    tpr_data = load_injection_recovery()
    has_tpr = bool(tpr_data.get("taylor_tree"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = ROC_DIR / f"fpr_results_{ts}.json"
    output = {
        "timestamp": datetime.now().isoformat(),
        "fpr": fpr_data,
        "has_injection_recovery": has_tpr,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    plot_path = ROC_DIR / "roc_curve.png"
    try:
        plot_roc_and_fpr(fpr_data, tpr_data, plot_path)
    except Exception as e:
        logger.warning(f"Failed to generate plot: {e}")

    print(f"\n{'=' * 60}")
    print("FALSE POSITIVE RATE ANALYSIS")
    print(f"{'=' * 60}")
    print(f"{'SNR Threshold':>15s}  {'False Positives':>16s}  {'FPR':>10s}")
    for e in fpr_data.get("fpr", []):
        print(f"{e['snr_threshold']:>15.1f}  {e['total_false_positives']:>16d}  {e['fpr']:>10.6f}")

    if has_tpr:
        print("\n  ROC curve generated using injection/recovery TPR data")
    else:
        print("\n  Run injection_recovery.py first to generate ROC curve")
    print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
