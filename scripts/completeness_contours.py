#!/usr/bin/env python3
"""
2D Completeness Contour Maps — MitraSETI

Generates the standard SETI survey publication figure: a 2D heatmap
showing detection probability as a function of drift rate and SNR.
This is required for any SETI survey paper (cf. COSMIC 2025, BL Galaxy Survey).

Runs injection-recovery across a fine grid of (drift_rate, SNR) values
and plots iso-completeness contours.

Output:
  mitraseti_artifacts/completeness/completeness_<timestamp>.json
  mitraseti_artifacts/completeness/completeness_contour.png

Usage:
  python scripts/completeness_contours.py
  python scripts/completeness_contours.py --snr-steps 10 --drift-steps 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

if "HDF5_PLUGIN_PATH" not in os.environ:
    os.environ["HDF5_PLUGIN_PATH"] = ""

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import ARTIFACTS_DIR

logger = logging.getLogger("mitraseti.completeness")

COMP_DIR = ARTIFACTS_DIR / "completeness"
COMP_DIR.mkdir(parents=True, exist_ok=True)


def run_grid_injection(
    snr_range: tuple = (5.0, 100.0),
    drift_range: tuple = (-4.0, 4.0),
    snr_steps: int = 8,
    drift_steps: int = 9,
    n_injections: int = 20,
    n_chans: int = 8192,
    n_times: int = 16,
) -> Dict[str, Any]:
    """Run injection-recovery across a 2D grid."""
    try:
        import mitraseti_core as _core
    except ImportError:
        try:
            import astroseti_core as _core
        except ImportError:
            logger.error("Rust core not available")
            return {}

    snr_vals = np.linspace(snr_range[0], snr_range[1], snr_steps)
    drift_vals = np.linspace(drift_range[0], drift_range[1], drift_steps)

    header = _core.FilterbankHeader(
        nchans=n_chans, nifs=1, nbits=32, tsamp=18.253611008,
        fch1=1420.5, foff=-2.7939677238464355e-06,
        tstart=59000.0, source_name="COMPLETENESS_TEST",
        ra=0.0, dec=0.0,
    )

    grid = np.zeros((drift_steps, snr_steps))
    rng = np.random.default_rng(42)

    total = snr_steps * drift_steps
    done = 0

    for di, drift in enumerate(drift_vals):
        for si, snr_target in enumerate(snr_vals):
            recovered = 0

            # Dedoppler normalizes per-channel by median, so inject
            # the signal in ALL time steps of the drifting channel,
            # causing the integrated power along that drift path to
            # exceed the noise floor.
            for _ in range(n_injections):
                noise = rng.standard_normal((n_times, n_chans)).astype(np.float32)
                noise = np.abs(noise) + 1.0  # ensure positive (power-like)

                inject_ch = rng.integers(n_chans // 4, 3 * n_chans // 4)
                total_drift_ch = drift
                for t in range(n_times):
                    ch = int(inject_ch + t * total_drift_ch / max(n_times - 1, 1))
                    if 0 <= ch < n_chans:
                        noise[t, ch] += snr_target * noise[:, ch].std()

                params = _core.SearchParams(
                    max_drift_rate=max(abs(drift) + 1.0, 4.0),
                    min_snr=5.0,
                    use_taylor_tree=True,
                )
                engine = _core.DedopplerEngine(params)
                result = engine.search(noise.ravel().tolist(), n_times, n_chans, header)

                if len(result.candidates) > 0:
                    foff_val = abs(header.foff)
                    inject_freq = header.fch1 + inject_ch * header.foff
                    for c in result.candidates:
                        c_freq_mhz = getattr(c, "frequency_hz", 0) / 1e6
                        freq_dist = abs(c_freq_mhz - inject_freq) / foff_val
                        if freq_dist < max(abs(total_drift_ch) + 10, 20):
                            recovered += 1
                            break

            completeness = recovered / n_injections
            grid[di, si] = completeness

            done += 1
            if done % 10 == 0:
                logger.info(f"  Progress: {done}/{total} ({100*done/total:.0f}%)")

    return {
        "snr_values": snr_vals.tolist(),
        "drift_values": drift_vals.tolist(),
        "grid": grid.tolist(),
        "n_injections": n_injections,
        "n_chans": n_chans,
        "n_times": n_times,
    }


def plot_completeness(data: Dict, output_path: Path) -> None:
    """Generate publication-ready 2D completeness contour map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.array(data["grid"])
    snr_vals = np.array(data["snr_values"])
    drift_vals = np.array(data["drift_values"])

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0a0e18")
    ax.set_facecolor("#080c14")

    # Main heatmap
    im = ax.imshow(
        grid, aspect="auto", origin="lower", cmap="inferno",
        extent=[snr_vals[0], snr_vals[-1], drift_vals[0], drift_vals[-1]],
        vmin=0, vmax=1, interpolation="bilinear",
    )

    # Contour lines at key completeness levels
    X, Y = np.meshgrid(snr_vals, drift_vals)
    contours = ax.contour(
        X, Y, grid, levels=[0.5, 0.8, 0.9, 0.95],
        colors=["#ff3366", "#ff9f43", "#00d4ff", "#00ff88"],
        linewidths=1.5, linestyles="--",
    )
    ax.clabel(contours, fmt="%.0f%%", fontsize=9, colors="white")

    cbar = fig.colorbar(im, ax=ax, label="Detection Completeness", shrink=0.85)
    cbar.ax.yaxis.label.set_color("#8ca5c8")
    cbar.ax.tick_params(colors="#8ca5c8")

    ax.set_xlabel("Injected SNR", color="#8ca5c8", fontsize=13)
    ax.set_ylabel("Drift Rate (channels/observation)", color="#8ca5c8", fontsize=13)
    ax.set_title(
        "MitraSETI Detection Completeness\n(Taylor Tree De-Doppler)",
        color="#e0e8f0", fontsize=15, fontweight=300,
    )
    ax.tick_params(colors="#8ca5c8")
    for spine in ax.spines.values():
        spine.set_color("#1a3a5c")

    # Annotation
    ax.text(
        0.02, 0.98,
        f"Injections per cell: {data['n_injections']}\n"
        f"Grid: {len(drift_vals)}×{len(snr_vals)}\n"
        f"Data: {data['n_chans']} ch × {data['n_times']} t",
        transform=ax.transAxes, color="#8ca5c8", fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f192d", edgecolor="#1a3a5c"),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Completeness contour saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MitraSETI Completeness Contour Maps")
    parser.add_argument("--snr-steps", type=int, default=8)
    parser.add_argument("--drift-steps", type=int, default=9)
    parser.add_argument("--n-injections", type=int, default=20)
    parser.add_argument("--snr-min", type=float, default=5.0)
    parser.add_argument("--snr-max", type=float, default=100.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Running 2D completeness grid")
    data = run_grid_injection(
        snr_range=(args.snr_min, args.snr_max),
        snr_steps=args.snr_steps,
        drift_steps=args.drift_steps,
        n_injections=args.n_injections,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = COMP_DIR / f"completeness_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    plot_completeness(data, COMP_DIR / "completeness_contour.png")

    grid = np.array(data["grid"])
    print(f"\n{'='*60}")
    print("COMPLETENESS SUMMARY")
    print(f"{'='*60}")
    print(f"  Grid: {grid.shape[0]} drift × {grid.shape[1]} SNR")
    print(f"  Mean completeness: {grid.mean():.1%}")
    print(f"  Min completeness:  {grid.min():.1%}")
    print(f"  Max completeness:  {grid.max():.1%}")
    print(f"  >90% cells:        {(grid >= 0.9).sum()}/{grid.size}")
    print(f"  Plot: {COMP_DIR / 'completeness_contour.png'}")


if __name__ == "__main__":
    main()
