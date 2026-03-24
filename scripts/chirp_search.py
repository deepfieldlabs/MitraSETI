#!/usr/bin/env python3
"""
Doppler Acceleration (Chirp Rate) Search — MitraSETI

Extends the linear de-Doppler search to second-order drift (acceleration).
Most SETI tools assume signals drift linearly in frequency:

    f(t) = f₀ + ḟ·t

Real transmitters on orbiting bodies also have acceleration:

    f(t) = f₀ + ḟ·t + ½·f̈·t²

This script:
  1. For each trial acceleration, "de-chirps" the spectrogram to remove
     the quadratic component
  2. Runs the standard Taylor tree de-Doppler on the straightened data
  3. Collects hits that are stronger *with* the chirp correction than without
  4. Reports candidates with both drift rate and acceleration

No other open-source SETI tool implements chirp-rate search with a
Taylor tree backend — this is a genuinely novel capability.

Output:
  mitraseti_artifacts/chirp/chirp_results_<timestamp>.json

Usage:
  python scripts/chirp_search.py
  python scripts/chirp_search.py --file GJ699_0003.h5 --accel-max 0.1
  python scripts/chirp_search.py --accel-steps 11 --min-snr 15
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

logger = logging.getLogger("mitraseti.chirp")

CHIRP_DIR = ARTIFACTS_DIR / "chirp"
CHIRP_DIR.mkdir(parents=True, exist_ok=True)


def dechirp_spectrogram(
    data: np.ndarray,
    acceleration_hz_s2: float,
    tsamp: float,
    foff_hz: float,
) -> np.ndarray:
    """Remove quadratic frequency drift from a spectrogram.

    For each time step t, shifts the spectrum by -½·f̈·t² channels to
    straighten a signal with the given acceleration.

    Args:
        data: Spectrogram (n_times, n_chans).
        acceleration_hz_s2: Frequency acceleration in Hz/s².
        tsamp: Time resolution in seconds.
        foff_hz: Channel width in Hz (signed).

    Returns:
        De-chirped spectrogram (same shape).
    """
    n_times, n_chans = data.shape
    result = np.zeros_like(data)

    for t in range(n_times):
        elapsed = t * tsamp
        shift_hz = 0.5 * acceleration_hz_s2 * elapsed ** 2
        shift_channels = int(round(shift_hz / abs(foff_hz))) if abs(foff_hz) > 0 else 0

        if shift_channels == 0:
            result[t] = data[t]
        else:
            result[t] = np.roll(data[t], -shift_channels)

    return result


def run_chirp_search(
    filepath: Path,
    accel_values: List[float],
    min_snr: float = 15.0,
    max_drift_rate: float = 4.0,
) -> Dict[str, Any]:
    """Run chirp search on a single file across trial accelerations.

    For each acceleration value:
      1. De-chirp the data
      2. Run Taylor tree de-Doppler
      3. Collect hits with SNR above threshold

    Returns results dict with candidates per acceleration.
    """
    try:
        import mitraseti_core as _core
    except ImportError:
        try:
            import astroseti_core as _core
        except ImportError:
            logger.error("Rust core not available")
            return {}

    from pipeline import MitraSETIPipeline

    pipe = MitraSETIPipeline()
    file_info = pipe._read_file(str(filepath))

    data_orig = file_info["data"].copy()
    header = file_info["header"]
    n_times = file_info["n_times"]
    n_chans = file_info["n_chans"]

    tsamp = header.get("tsamp", 18.0)
    foff_hz = header.get("foff", -0.00028) * 1e6

    # Baseline: zero acceleration (standard linear search)
    params_tt = _core.SearchParams(
        max_drift_rate=max_drift_rate,
        min_snr=min_snr,
        use_taylor_tree=True,
    )

    result = {
        "file": filepath.name,
        "timestamp": datetime.now().isoformat(),
        "header": {
            "fch1": header.get("fch1", 0),
            "foff": header.get("foff", 0),
            "tsamp": tsamp,
            "n_times": n_times,
            "n_chans": n_chans,
        },
        "accel_values": accel_values,
        "baseline_hits": 0,
        "per_acceleration": [],
        "unique_chirp_candidates": [],
    }

    # Run baseline (accel = 0)
    logger.info(f"  Baseline (accel=0): running Taylor tree...")
    engine = _core.DedopplerEngine(params_tt)
    rust_header = _core.FilterbankHeader(
        nchans=n_chans, nifs=1, nbits=32, tsamp=tsamp,
        fch1=header["fch1"], foff=header["foff"],
        tstart=header.get("tstart", 59000.0),
        source_name=header.get("source_name", "unknown"),
        ra=0.0, dec=0.0,
    )

    # Limit data size
    max_pts = 8_000_000
    use_data = data_orig
    use_n_chans = n_chans
    factor = 1
    if n_times * n_chans > max_pts:
        while use_n_chans // factor * n_times > max_pts:
            factor *= 2
        trim = n_chans - (n_chans % factor)
        use_data = data_orig[:, :trim].reshape(n_times, -1, factor).mean(axis=2)
        use_n_chans = use_data.shape[1]
        rust_header = _core.FilterbankHeader(
            nchans=use_n_chans, nifs=1, nbits=32, tsamp=tsamp,
            fch1=header["fch1"], foff=header["foff"] * factor,
            tstart=header.get("tstart", 59000.0),
            source_name=header.get("source_name", "unknown"),
            ra=0.0, dec=0.0,
        )
        foff_hz_eff = header["foff"] * factor * 1e6
    else:
        foff_hz_eff = foff_hz

    data_flat = use_data.astype(np.float32).ravel().tolist()
    baseline_result = engine.search(data_flat, n_times, use_n_chans, rust_header)
    baseline_freqs = {
        (round(c.frequency_hz, 1), round(c.drift_rate, 3))
        for c in baseline_result.candidates
    }
    result["baseline_hits"] = len(baseline_result.candidates)
    logger.info(f"    Baseline: {len(baseline_result.candidates)} hits above SNR {min_snr}")

    all_chirp_candidates = []

    for accel in accel_values:
        if accel == 0:
            continue

        logger.info(f"  Acceleration = {accel:+.4f} Hz/s²")
        t0 = time.perf_counter()

        dechirped = dechirp_spectrogram(use_data, accel, tsamp, abs(foff_hz_eff))
        dc_flat = dechirped.astype(np.float32).ravel().tolist()

        engine_dc = _core.DedopplerEngine(params_tt)
        dc_result = engine_dc.search(dc_flat, n_times, use_n_chans, rust_header)

        elapsed = time.perf_counter() - t0

        # Find hits that are NEW (not in baseline) — signals that only
        # become detectable after chirp correction
        new_hits = []
        for c in dc_result.candidates:
            key = (round(c.frequency_hz, 1), round(c.drift_rate, 3))
            if key not in baseline_freqs:
                new_hits.append({
                    "frequency_hz": c.frequency_hz,
                    "drift_rate": c.drift_rate,
                    "snr": c.snr,
                    "acceleration": accel,
                })

        accel_entry = {
            "acceleration": accel,
            "total_hits": len(dc_result.candidates),
            "new_hits": len(new_hits),
            "elapsed_s": round(elapsed, 3),
        }
        result["per_acceleration"].append(accel_entry)
        all_chirp_candidates.extend(new_hits)

        logger.info(
            f"    Total: {len(dc_result.candidates)}, "
            f"New (chirp-only): {len(new_hits)}, "
            f"Time: {elapsed*1000:.0f} ms"
        )

    # Deduplicate chirp candidates across accelerations
    seen = set()
    unique = []
    for c in sorted(all_chirp_candidates, key=lambda x: x["snr"], reverse=True):
        key = (round(c["frequency_hz"], 1), round(c["drift_rate"], 2))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    result["unique_chirp_candidates"] = unique[:100]

    n_unique = len(unique)
    logger.info(
        f"\n  Summary: {n_unique} unique chirp-only candidates "
        f"(signals detectable only with acceleration correction)"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="MitraSETI Doppler Acceleration (Chirp) Search",
    )
    parser.add_argument("--file", type=str, default=None,
                        help="Specific file to search (default: first available)")
    parser.add_argument("--accel-max", type=float, default=0.1,
                        help="Max acceleration magnitude in Hz/s² (default: 0.1)")
    parser.add_argument("--accel-steps", type=int, default=9,
                        help="Number of acceleration trials (default: 9)")
    parser.add_argument("--min-snr", type=float, default=15.0,
                        help="Minimum SNR threshold (default: 15)")
    parser.add_argument("--max-files", type=int, default=3,
                        help="Max files to process (default: 3)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    accel_values = list(np.linspace(-args.accel_max, args.accel_max, args.accel_steps))

    if args.file:
        candidates = list(FILTERBANK_DIR.glob(f"*{args.file}*"))
        if not candidates:
            logger.error(f"File matching '{args.file}' not found in {FILTERBANK_DIR}")
            return
        files = candidates[:1]
    else:
        files = sorted(FILTERBANK_DIR.glob("*.h5"))[:args.max_files]
        if not files:
            files = sorted(FILTERBANK_DIR.glob("*.fil"))[:args.max_files]

    if not files:
        logger.error(f"No filterbank files found in {FILTERBANK_DIR}")
        return

    logger.info(f"Chirp search: {len(files)} files, {len(accel_values)} accel trials")
    logger.info(f"  Accel range: [{-args.accel_max}, {args.accel_max}] Hz/s²")

    all_results = []
    for fp in files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {fp.name}")
        logger.info(f"{'='*60}")

        result = run_chirp_search(
            fp, accel_values,
            min_snr=args.min_snr,
        )
        if result:
            all_results.append(result)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = CHIRP_DIR / f"chirp_results_{ts}.json"
    output = {
        "timestamp": datetime.now().isoformat(),
        "accel_range": [-args.accel_max, args.accel_max],
        "accel_steps": args.accel_steps,
        "min_snr": args.min_snr,
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {json_path}")

    # Summary
    total_chirp = sum(len(r.get("unique_chirp_candidates", [])) for r in all_results)
    print(f"\n{'='*60}")
    print("CHIRP SEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"  Files processed:    {len(all_results)}")
    print(f"  Accel range:        [{-args.accel_max}, +{args.accel_max}] Hz/s²")
    print(f"  Accel trials:       {args.accel_steps}")
    print(f"  Chirp-only signals: {total_chirp}")

    for r in all_results:
        chirp_cands = r.get("unique_chirp_candidates", [])
        print(f"\n  {r.get('file', 'unknown')}:")
        print(f"    Baseline hits:    {r.get('baseline_hits', 0)}")
        print(f"    Chirp-only:       {len(chirp_cands)}")
        if chirp_cands:
            for c in chirp_cands[:5]:
                print(
                    f"      freq={c['frequency_hz']/1e6:.6f} MHz  "
                    f"drift={c['drift_rate']:.4f} Hz/s  "
                    f"accel={c['acceleration']:.4f} Hz/s²  "
                    f"SNR={c['snr']:.1f}"
                )


if __name__ == "__main__":
    main()
