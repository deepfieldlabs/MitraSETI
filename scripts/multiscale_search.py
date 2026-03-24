#!/usr/bin/env python3
"""
Multi-Scale Taylor Tree Detection — MitraSETI

Runs the Taylor tree de-Doppler at multiple frequency resolutions:
  - Native resolution (1×)
  - 2× averaged (catches broader signals)
  - 4× averaged (catches widest signals)

Then merges detections across scales, keeping the highest-SNR version
of duplicates.  This catches both narrowband and broadband signals
that single-resolution searches miss — no other open-source SETI tool
implements this.

Output:
  mitraseti_artifacts/multiscale/multiscale_<file>_<timestamp>.json
  mitraseti_artifacts/multiscale/multiscale_summary.png

Usage:
  python scripts/multiscale_search.py
  python scripts/multiscale_search.py --file path/to/file.h5
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
from typing import Any, Dict, List, Tuple

if "HDF5_PLUGIN_PATH" not in os.environ:
    os.environ["HDF5_PLUGIN_PATH"] = ""

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import ARTIFACTS_DIR

logger = logging.getLogger("mitraseti.multiscale")

MULTI_DIR = ARTIFACTS_DIR / "multiscale"
MULTI_DIR.mkdir(parents=True, exist_ok=True)

SCALES = [1, 2, 4]


def downsample_frequency(data: np.ndarray, factor: int) -> np.ndarray:
    """Average frequency channels by a factor.

    Trims excess channels if n_chans is not exactly divisible.
    """
    n_times, n_chans = data.shape
    trim = n_chans - (n_chans % factor)
    trimmed = data[:, :trim]
    return trimmed.reshape(n_times, -1, factor).mean(axis=2)


def search_at_scale(
    data: np.ndarray,
    header: Dict[str, Any],
    scale: int,
    max_drift: float = 4.0,
    min_snr: float = 10.0,
) -> Tuple[List[Dict], float]:
    """Run Taylor tree de-Doppler at a given frequency scale.

    Args:
        header: Dict from pipeline._read_file()["header"].

    Returns (list of hits, elapsed_seconds).
    """
    try:
        import mitraseti_core as _core
    except ImportError:
        import astroseti_core as _core

    foff_orig = header.get("foff", -2.7939677238464355e-06)
    fch1 = header.get("fch1", 1420.0)
    tsamp = header.get("tsamp", 18.253611008)
    tstart = header.get("tstart", 59000.0)
    source_name = header.get("source_name", "unknown")
    ra = header.get("ra", 0.0) or 0.0
    dec = header.get("dec", 0.0) or 0.0

    if scale > 1:
        data_scaled = downsample_frequency(data, scale)
        foff = foff_orig * scale
    else:
        data_scaled = data
        foff = foff_orig

    n_times, n_chans = data_scaled.shape

    scaled_header = _core.FilterbankHeader(
        nchans=n_chans, nifs=1, nbits=32, tsamp=tsamp,
        fch1=fch1, foff=foff,
        tstart=tstart, source_name=source_name,
        ra=ra, dec=dec,
    )

    params = _core.SearchParams(
        max_drift_rate=max_drift,
        min_snr=min_snr,
        use_taylor_tree=True,
    )
    engine = _core.DedopplerEngine(params)

    t0 = time.perf_counter()
    result = engine.search(data_scaled.ravel().tolist(), n_times, n_chans, scaled_header)
    elapsed = time.perf_counter() - t0

    hits = []
    for c in result.candidates:
        freq_idx = int(getattr(c, "channel", 0))
        actual_freq = fch1 + freq_idx * scale * foff_orig
        hits.append({
            "frequency": float(getattr(c, "frequency", actual_freq)),
            "snr": float(getattr(c, "snr", 0)),
            "drift_rate": float(getattr(c, "drift_rate", 0)),
            "channel": freq_idx * scale,
            "scale": scale,
        })

    return hits, elapsed


def merge_multiscale_hits(
    hits_by_scale: Dict[int, List[Dict]],
    freq_tolerance_hz: float = 0.01,
) -> List[Dict]:
    """Merge detections across scales, keeping highest-SNR duplicates.

    Two hits are considered duplicates if their frequencies are within
    freq_tolerance_hz.
    """
    all_hits = []
    for scale, hits in hits_by_scale.items():
        all_hits.extend(hits)

    if not all_hits:
        return []

    all_hits.sort(key=lambda h: h["snr"], reverse=True)

    merged = []
    used = set()

    for i, hit in enumerate(all_hits):
        if i in used:
            continue
        merged.append(hit)
        used.add(i)

        for j in range(i + 1, len(all_hits)):
            if j in used:
                continue
            if abs(hit["frequency"] - all_hits[j]["frequency"]) < freq_tolerance_hz:
                used.add(j)

    return merged


def run_multiscale(
    filepath: str,
    max_drift: float = 4.0,
    min_snr: float = 10.0,
) -> Dict[str, Any]:
    """Run multi-scale search on a single file."""
    from pipeline import MitraSETIPipeline

    pipe = MitraSETIPipeline()

    logger.info(f"Reading {Path(filepath).name}")
    file_info = pipe._read_file(filepath)
    data = file_info["data"]
    header = file_info["header"]

    hits_by_scale = {}
    timings = {}

    for scale in SCALES:
        logger.info(f"  Scale {scale}× ({data.shape[1] // scale} channels)")
        hits, elapsed = search_at_scale(data, header, scale, max_drift, min_snr)
        hits_by_scale[scale] = hits
        timings[scale] = elapsed
        logger.info(f"    → {len(hits)} hits in {elapsed:.3f}s")

    merged = merge_multiscale_hits(hits_by_scale)

    unique_at_coarser = [h for h in merged if h["scale"] > 1]
    unique_at_coarser_only = []
    for h in unique_at_coarser:
        found_at_1x = any(
            abs(h["frequency"] - h1["frequency"]) < 0.01
            for h1 in hits_by_scale.get(1, [])
        )
        if not found_at_1x:
            unique_at_coarser_only.append(h)

    result = {
        "file": str(filepath),
        "timestamp": datetime.now().isoformat(),
        "scales": {
            str(s): {"n_hits": len(hits), "time_s": round(timings[s], 4)}
            for s, hits in hits_by_scale.items()
        },
        "merged_hits": len(merged),
        "unique_coarse_scale_detections": len(unique_at_coarser_only),
        "unique_coarse_hits": unique_at_coarser_only[:20],
        "all_hits": merged[:100],
    }

    logger.info(
        f"  Merged: {len(merged)} total, "
        f"{len(unique_at_coarser_only)} found only at coarser scales"
    )

    return result


def plot_multiscale_summary(results: List[Dict], output_path: Path) -> None:
    """Generate multi-scale comparison visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0a0e18")

    for ax in axes:
        ax.set_facecolor("#080c14")
        ax.tick_params(colors="#8ca5c8")
        for spine in ax.spines.values():
            spine.set_color("#1a3a5c")

    # Panel 1: hits per scale
    scale_labels = [f"{s}×" for s in SCALES]
    for ri, r in enumerate(results):
        hit_counts = [r["scales"].get(str(s), {}).get("n_hits", 0) for s in SCALES]
        x_pos = np.arange(len(SCALES)) + ri * 0.25
        axes[0].bar(x_pos, hit_counts, width=0.2, alpha=0.8,
                    label=Path(r["file"]).stem[:30])

    axes[0].set_xticks(np.arange(len(SCALES)) + 0.25)
    axes[0].set_xticklabels(scale_labels, color="#8ca5c8")
    axes[0].set_ylabel("Hits", color="#8ca5c8")
    axes[0].set_title("Detections per Scale", color="#e0e8f0", fontsize=12)
    if len(results) <= 5:
        axes[0].legend(facecolor="#0f192d", edgecolor="#1a3a5c",
                       labelcolor="#8ca5c8", fontsize=8)

    # Panel 2: timing comparison
    for ri, r in enumerate(results):
        times = [r["scales"].get(str(s), {}).get("time_s", 0) for s in SCALES]
        axes[1].plot(scale_labels, times, "o-", linewidth=2, markersize=6,
                     label=Path(r["file"]).stem[:30])

    axes[1].set_ylabel("Time (seconds)", color="#8ca5c8")
    axes[1].set_title("Search Time per Scale", color="#e0e8f0", fontsize=12)

    # Panel 3: unique coarse detections
    file_labels = [Path(r["file"]).stem[:20] for r in results]
    unique_counts = [r.get("unique_coarse_scale_detections", 0) for r in results]
    bars = axes[2].barh(range(len(results)), unique_counts, color="#ff9f43", alpha=0.8)
    axes[2].set_yticks(range(len(results)))
    axes[2].set_yticklabels(file_labels, color="#8ca5c8", fontsize=8)
    axes[2].set_xlabel("Unique Detections", color="#8ca5c8")
    axes[2].set_title("Broadband-Only Detections", color="#e0e8f0", fontsize=12)

    for bar, count in zip(bars, unique_counts):
        axes[2].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     str(count), va="center", color="#ff9f43", fontsize=10)

    fig.suptitle(
        "MitraSETI — Multi-Scale Taylor Tree Detection",
        color="#4da6ff", fontsize=14, fontweight=300, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Multi-scale plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MitraSETI Multi-Scale Taylor Tree Detection",
    )
    parser.add_argument("--file", type=str, help="Specific file to analyze")
    parser.add_argument("--max-files", type=int, default=3, help="Max files from data dir")
    parser.add_argument("--min-snr", type=float, default=10.0, help="Min SNR (default: 10)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from paths import BL_DATA_DIR

    files = []
    if args.file:
        files = [args.file]
    else:
        data_dir = BL_DATA_DIR
        if data_dir.is_dir():
            for ext in ("*.h5", "*.fil"):
                files.extend(sorted(str(p) for p in data_dir.glob(ext)))
            files = files[: args.max_files]

    if not files:
        logger.error("No data files found")
        return

    logger.info(f"Multi-scale search on {len(files)} files")
    results = []
    for filepath in files:
        try:
            result = run_multiscale(filepath, min_snr=args.min_snr)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed on {filepath}: {e}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = MULTI_DIR / f"multiscale_results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {json_path}")

    try:
        plot_multiscale_summary(results, MULTI_DIR / "multiscale_summary.png")
    except Exception as e:
        logger.warning(f"Failed to generate plot: {e}")

    print(f"\n{'='*60}")
    print("MULTI-SCALE DETECTION SUMMARY")
    print(f"{'='*60}")
    for r in results:
        name = Path(r["file"]).stem[:40]
        print(f"\n  {name}")
        for s in SCALES:
            sd = r["scales"].get(str(s), {})
            print(f"    {s}×: {sd.get('n_hits', 0):6d} hits ({sd.get('time_s', 0):.3f}s)")
        print(f"    Merged: {r['merged_hits']} total, "
              f"{r['unique_coarse_scale_detections']} broadband-only")


if __name__ == "__main__":
    main()
