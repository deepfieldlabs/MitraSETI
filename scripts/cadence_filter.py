#!/usr/bin/env python3
"""
ON/OFF Cadence Filter — MitraSETI

The standard Breakthrough Listen verification technique: a signal is
considered a genuine candidate only if it appears in ALL ON-target
observations but in NONE of the OFF-target (reference) observations.

This eliminates terrestrial RFI which is direction-independent and
therefore present in both ON and OFF scans.

Cadence patterns supported:
  - ABAB:   ON → OFF → ON → OFF
  - ABABAB: ON → OFF → ON → OFF → ON → OFF  (standard BL 6-scan)
  - Custom groupings via filename conventions

Usage:
  python scripts/cadence_filter.py
  python scripts/cadence_filter.py --target TRAPPIST1
  python scripts/cadence_filter.py --target GJ699 --freq-tol 10 --drift-tol 0.5
  python scripts/cadence_filter.py --list-targets

Output:
  mitraseti_artifacts/cadence/cadence_results_<target>_<timestamp>.json
  mitraseti_artifacts/cadence/cadence_summary.html
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if "HDF5_PLUGIN_PATH" not in os.environ:
    os.environ["HDF5_PLUGIN_PATH"] = ""

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import ARTIFACTS_DIR, FILTERBANK_DIR

logger = logging.getLogger("mitraseti.cadence")

CADENCE_DIR = ARTIFACTS_DIR / "cadence"
CADENCE_DIR.mkdir(parents=True, exist_ok=True)


# ── File grouping ─────────────────────────────────────────────────────────────

_BL_FILENAME_RE = re.compile(
    r"""
    (?:spliced_)?                     # optional spliced_ prefix
    (?:blc[\d_]+_)?                   # optional beam prefix
    (?:2bit_)?                        # optional 2bit
    guppi_                            # guppi marker
    (\d+)_                            # MJD
    (\d+)_                            # scan start seconds
    (?:DIAG_)?                        # optional DIAG_ prefix
    ([A-Za-z0-9+_.-]+?)               # target name (non-greedy)
    (?:_(OFF))?                       # optional _OFF marker
    _(\d{4})                          # scan number
    \.gpuspec                         # extension
    """,
    re.VERBOSE,
)

_SIMPLE_ON_OFF_RE = re.compile(r"^(.+?)_(ON|OFF)_?(\d+)?\.(?:fil|h5)$", re.IGNORECASE)


def parse_filename(filepath: Path) -> Optional[Dict[str, Any]]:
    """Extract target, scan number, and ON/OFF status from a BL filename."""
    name = filepath.name

    m = _BL_FILENAME_RE.search(name)
    if m:
        mjd, scan_start, target, off_marker, scan_num = m.groups()
        return {
            "target": target.replace("_", "").upper(),
            "is_off": off_marker is not None,
            "scan_num": int(scan_num),
            "mjd": int(mjd),
            "scan_start": int(scan_start),
            "filepath": filepath,
        }

    m = _SIMPLE_ON_OFF_RE.match(name)
    if m:
        target, on_off, idx = m.groups()
        return {
            "target": target.replace("-", "").replace("_", "").upper(),
            "is_off": on_off.upper() == "OFF",
            "scan_num": int(idx) if idx else 0,
            "mjd": 0,
            "scan_start": 0,
            "filepath": filepath,
        }

    return None


def discover_cadence_groups(data_dir: Path) -> Dict[str, Dict[str, List[Dict]]]:
    """Scan the data directory and group files into ON/OFF cadences per target.

    Returns:
        {target_name: {"on": [file_info, ...], "off": [file_info, ...]}}
    """
    groups: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: {"on": [], "off": []})

    for ext in ("*.fil", "*.h5"):
        for fp in sorted(data_dir.glob(ext)):
            info = parse_filename(fp)
            if info is None:
                continue
            key = "off" if info["is_off"] else "on"
            groups[info["target"]][key].append(info)

    valid = {}
    for target, scans in groups.items():
        if scans["on"] and scans["off"]:
            for key in ("on", "off"):
                scans[key].sort(key=lambda x: (x["mjd"], x["scan_start"]))
            valid[target] = scans

    return valid


# ── Signal matching ───────────────────────────────────────────────────────────


def signals_match(
    sig_a: Dict[str, Any],
    sig_b: Dict[str, Any],
    freq_tol_hz: float = 50.0,
    drift_tol_hz_s: float = 0.5,
) -> bool:
    """Check if two signals are the same physical emitter."""
    freq_diff = abs(sig_a["frequency_hz"] - sig_b["frequency_hz"])
    drift_diff = abs(sig_a["drift_rate"] - sig_b["drift_rate"])
    return freq_diff <= freq_tol_hz and drift_diff <= drift_tol_hz_s


def cross_match_signals(
    on_signals: List[Dict[str, Any]],
    off_signals: List[Dict[str, Any]],
    freq_tol_hz: float = 50.0,
    drift_tol_hz_s: float = 0.5,
) -> Tuple[List[Dict], List[Dict]]:
    """Separate ON signals into those also present in OFF (RFI) and those not.

    Returns:
        (passing_signals, rfi_signals)
        passing_signals: present in ON but NOT in any OFF scan
        rfi_signals: present in both ON and OFF (confirmed RFI)
    """
    passing = []
    rfi = []

    for sig in on_signals:
        in_off = any(
            signals_match(sig, off_sig, freq_tol_hz, drift_tol_hz_s) for off_sig in off_signals
        )
        if in_off:
            rfi.append(sig)
        else:
            passing.append(sig)

    return passing, rfi


def multi_on_consensus(
    on_scan_signals: List[List[Dict[str, Any]]],
    min_on_detections: int = 2,
    freq_tol_hz: float = 50.0,
    drift_tol_hz_s: float = 0.5,
) -> List[Dict[str, Any]]:
    """Find signals present in at least min_on_detections ON scans.

    Returns signals from the first ON scan that have matches in enough others.
    """
    if not on_scan_signals:
        return []

    if len(on_scan_signals) == 1:
        return on_scan_signals[0]

    reference = on_scan_signals[0]
    consensus = []

    for sig in reference:
        count = 1
        for other_scan in on_scan_signals[1:]:
            if any(signals_match(sig, other, freq_tol_hz, drift_tol_hz_s) for other in other_scan):
                count += 1
        if count >= min_on_detections:
            sig["on_detections"] = count
            sig["on_scans_total"] = len(on_scan_signals)
            consensus.append(sig)

    return consensus


# ── Pipeline runner ───────────────────────────────────────────────────────────


def run_dedoppler_on_file(filepath: Path, min_snr: float = 10.0) -> List[Dict[str, Any]]:
    """Run the de-Doppler search on a single file and return hit list."""
    from pipeline import MitraSETIPipeline

    pipe = MitraSETIPipeline()

    try:
        result = pipe.process_file(str(filepath))
    except Exception as e:
        logger.warning(f"Failed to process {filepath.name}: {e}")
        return []

    candidates = result.get("candidates", [])
    return [c for c in candidates if c.get("snr", 0) >= min_snr]


def run_cadence_filter(
    target: str,
    cadence_group: Dict[str, List[Dict]],
    freq_tol_hz: float = 50.0,
    drift_tol_hz_s: float = 0.5,
    min_snr: float = 10.0,
    min_on_detections: int = 2,
) -> Dict[str, Any]:
    """Run the full ON/OFF cadence filter for one target.

    Steps:
      1. Run de-Doppler on each ON scan
      2. Run de-Doppler on each OFF scan
      3. Find signals present in multiple ON scans (consensus)
      4. Remove signals also present in OFF scans
      5. Report survivors
    """
    on_files = cadence_group["on"]
    off_files = cadence_group["off"]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"CADENCE FILTER: {target}")
    logger.info(f"  ON scans:  {len(on_files)}")
    logger.info(f"  OFF scans: {len(off_files)}")
    logger.info(f"{'=' * 60}")

    # Process ON scans
    on_scan_signals = []
    on_timings = []
    for i, fi in enumerate(on_files):
        fp = fi["filepath"]
        logger.info(f"  Processing ON scan {i + 1}/{len(on_files)}: {fp.name}")
        t0 = time.perf_counter()
        signals = run_dedoppler_on_file(fp, min_snr=min_snr)
        elapsed = time.perf_counter() - t0
        on_timings.append(elapsed)
        on_scan_signals.append(signals)
        logger.info(f"    → {len(signals)} signals above SNR {min_snr} ({elapsed:.1f}s)")

    # Process OFF scans
    all_off_signals = []
    off_timings = []
    for i, fi in enumerate(off_files):
        fp = fi["filepath"]
        logger.info(f"  Processing OFF scan {i + 1}/{len(off_files)}: {fp.name}")
        t0 = time.perf_counter()
        signals = run_dedoppler_on_file(fp, min_snr=min_snr)
        elapsed = time.perf_counter() - t0
        off_timings.append(elapsed)
        all_off_signals.extend(signals)
        logger.info(f"    → {len(signals)} signals ({elapsed:.1f}s)")

    # Multi-ON consensus
    total_on = sum(len(s) for s in on_scan_signals)
    effective_min = min(min_on_detections, len(on_scan_signals))
    consensus = multi_on_consensus(
        on_scan_signals,
        min_on_detections=effective_min,
        freq_tol_hz=freq_tol_hz,
        drift_tol_hz_s=drift_tol_hz_s,
    )
    logger.info(
        f"  Multi-ON consensus: {total_on} total → {len(consensus)} in ≥{effective_min} scans"
    )

    # ON/OFF cross-match
    passing, rfi_matched = cross_match_signals(
        consensus,
        all_off_signals,
        freq_tol_hz=freq_tol_hz,
        drift_tol_hz_s=drift_tol_hz_s,
    )
    logger.info(
        f"  OFF rejection: {len(consensus)} → {len(passing)} surviving "
        f"({len(rfi_matched)} matched to OFF = RFI)"
    )

    # Sort by SNR descending
    passing.sort(key=lambda x: x.get("snr", 0), reverse=True)

    result = {
        "target": target,
        "timestamp": datetime.now().isoformat(),
        "cadence": {
            "on_scans": len(on_files),
            "off_scans": len(off_files),
            "on_files": [str(f["filepath"].name) for f in on_files],
            "off_files": [str(f["filepath"].name) for f in off_files],
        },
        "statistics": {
            "total_on_signals": total_on,
            "on_per_scan": [len(s) for s in on_scan_signals],
            "total_off_signals": len(all_off_signals),
            "multi_on_consensus": len(consensus),
            "min_on_detections": effective_min,
            "off_matched_rfi": len(rfi_matched),
            "cadence_survivors": len(passing),
            "rejection_rate": 1 - len(passing) / max(total_on, 1),
        },
        "timing": {
            "on_scan_times": [round(t, 2) for t in on_timings],
            "off_scan_times": [round(t, 2) for t in off_timings],
            "total_s": round(sum(on_timings) + sum(off_timings), 2),
        },
        "candidates": _sanitize_candidates(passing[:100]),
        "top_rfi": _sanitize_candidates(
            sorted(rfi_matched, key=lambda x: x.get("snr", 0), reverse=True)[:20]
        ),
    }

    if passing:
        logger.info(f"\n  *** {len(passing)} CANDIDATES SURVIVED CADENCE FILTER ***")
        for c in passing[:10]:
            logger.info(
                f"    freq={c.get('frequency_hz', 0) / 1e6:.6f} MHz  "
                f"drift={c.get('drift_rate', 0):.4f} Hz/s  "
                f"SNR={c.get('snr', 0):.1f}  "
                f"ON={c.get('on_detections', '?')}/{c.get('on_scans_total', '?')}"
            )
    else:
        logger.info("\n  No candidates survived cadence filter (all matched to OFF → RFI)")

    return result


def _sanitize_candidates(candidates: List[Dict]) -> List[Dict]:
    """Remove non-serializable fields from candidates for JSON output."""
    clean = []
    for c in candidates:
        entry = {}
        for k, v in c.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                entry[k] = v
            elif isinstance(v, dict):
                entry[k] = {
                    sk: sv
                    for sk, sv in v.items()
                    if isinstance(sv, (str, int, float, bool, type(None)))
                }
            elif isinstance(v, np.floating):
                entry[k] = float(v)
            elif isinstance(v, np.integer):
                entry[k] = int(v)
        clean.append(entry)
    return clean


# ── Report generation ─────────────────────────────────────────────────────────


def generate_html_report(all_results: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate an HTML summary of cadence filter results for all targets."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MitraSETI — Cadence Filter Results</title>
<style>
  body {{ font-family: 'Inter', sans-serif; background: #080c14; color: #e0e8f0; margin: 0; padding: 32px; }}
  h1 {{ color: #4da6ff; font-weight: 300; letter-spacing: 2px; }}
  h2 {{ color: #00d4ff; font-weight: 400; margin-top: 32px; }}
  .card {{ background: rgba(15,25,45,0.85); border: 1px solid rgba(100,180,255,0.15); border-radius: 16px; padding: 24px; margin: 16px 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; padding: 10px; color: rgba(0,212,255,0.7); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid rgba(100,180,255,0.1); }}
  td {{ padding: 10px; font-size: 13px; border-bottom: 1px solid rgba(100,180,255,0.06); }}
  .stat {{ display: inline-block; padding: 12px 20px; background: rgba(15,25,45,0.7); border: 1px solid rgba(100,180,255,0.15); border-radius: 12px; margin: 4px; text-align: center; }}
  .stat .val {{ font-size: 28px; color: #00d4ff; font-weight: 300; }}
  .stat .lbl {{ font-size: 10px; color: rgba(140,165,200,0.5); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}
  .pass {{ color: #00ff88; }} .fail {{ color: #ff3366; }}
  .timestamp {{ color: rgba(140,165,200,0.4); font-size: 12px; }}
</style>
</head>
<body>
<h1>MitraSETI — ON/OFF Cadence Filter</h1>
<p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
"""
    total_survivors = sum(r["statistics"]["cadence_survivors"] for r in all_results)
    total_signals = sum(r["statistics"]["total_on_signals"] for r in all_results)

    html += f"""
<div style="display: flex; gap: 8px; flex-wrap: wrap; margin: 24px 0;">
  <div class="stat"><div class="val">{len(all_results)}</div><div class="lbl">Targets</div></div>
  <div class="stat"><div class="val">{total_signals:,}</div><div class="lbl">Total ON Signals</div></div>
  <div class="stat"><div class="val {"pass" if total_survivors else "fail"}">{total_survivors}</div><div class="lbl">Cadence Survivors</div></div>
  <div class="stat"><div class="val">{(1 - total_survivors / max(total_signals, 1)) * 100:.1f}%</div><div class="lbl">RFI Rejection</div></div>
</div>
"""

    for r in all_results:
        stats = r["statistics"]
        cadence = r["cadence"]
        timing = r["timing"]
        candidates = r.get("candidates", [])

        html += f"""
<h2>{r["target"]}</h2>
<div class="card">
  <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px;">
    <div class="stat"><div class="val">{cadence["on_scans"]}</div><div class="lbl">ON Scans</div></div>
    <div class="stat"><div class="val">{cadence["off_scans"]}</div><div class="lbl">OFF Scans</div></div>
    <div class="stat"><div class="val">{stats["total_on_signals"]:,}</div><div class="lbl">ON Signals</div></div>
    <div class="stat"><div class="val">{stats["multi_on_consensus"]}</div><div class="lbl">Multi-ON</div></div>
    <div class="stat"><div class="val">{stats["off_matched_rfi"]}</div><div class="lbl">OFF Matched</div></div>
    <div class="stat"><div class="val {"pass" if stats["cadence_survivors"] else "fail"}">{stats["cadence_survivors"]}</div><div class="lbl">Survivors</div></div>
    <div class="stat"><div class="val">{timing["total_s"]:.1f}s</div><div class="lbl">Time</div></div>
  </div>
"""
        if candidates:
            html += """<table><thead><tr>
  <th>Freq (MHz)</th><th>Drift (Hz/s)</th><th>SNR</th>
  <th>ON Detect</th><th>Classification</th></tr></thead><tbody>"""
            for c in candidates[:20]:
                freq = (
                    c.get("frequency_hz", 0) / 1e6
                    if c.get("frequency_hz", 0) > 1e6
                    else c.get("frequency_mhz", 0)
                )
                html += f"""<tr>
  <td>{freq:.6f}</td>
  <td>{c.get("drift_rate", 0):.4f}</td>
  <td style="color: #00d4ff;">{c.get("snr", 0):.1f}</td>
  <td>{c.get("on_detections", "?")}/{c.get("on_scans_total", "?")}</td>
  <td>{c.get("classification", "N/A")}</td>
</tr>"""
            html += "</tbody></table>"
        else:
            html += '<p style="color: rgba(140,165,200,0.5);">No signals survived the cadence filter — all matched to OFF scans (confirmed RFI).</p>'

        html += "</div>"

    html += """
<div style="text-align: center; margin-top: 40px; color: rgba(140,165,200,0.3); font-size: 11px; letter-spacing: 1px;">
  MitraSETI — Intelligent SETI Signal Analysis
</div>
</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    logger.info(f"HTML report saved to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="MitraSETI ON/OFF Cadence Filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target", type=str, default=None, help="Filter specific target (e.g. TRAPPIST1, GJ699)"
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List available targets with ON/OFF pairs and exit",
    )
    parser.add_argument(
        "--freq-tol",
        type=float,
        default=50.0,
        help="Frequency matching tolerance in Hz (default: 50)",
    )
    parser.add_argument(
        "--drift-tol",
        type=float,
        default=0.5,
        help="Drift rate matching tolerance in Hz/s (default: 0.5)",
    )
    parser.add_argument(
        "--min-snr", type=float, default=10.0, help="Minimum SNR threshold (default: 10)"
    )
    parser.add_argument(
        "--min-on", type=int, default=2, help="Minimum ON detections for consensus (default: 2)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    groups = discover_cadence_groups(FILTERBANK_DIR)

    if args.list_targets:
        print(f"\nAvailable targets with ON/OFF cadence pairs in {FILTERBANK_DIR}:\n")
        for target, scans in sorted(groups.items()):
            print(
                f"  {target:20s}  ON: {len(scans['on']):2d} scans  "
                f"OFF: {len(scans['off']):2d} scans"
            )
            for fi in scans["on"][:3]:
                print(f"    ON:  {fi['filepath'].name}")
            for fi in scans["off"][:2]:
                print(f"    OFF: {fi['filepath'].name}")
            print()
        return

    if not groups:
        logger.error("No targets with ON/OFF pairs found. Check data directory.")
        return

    targets_to_process = {}
    if args.target:
        key = args.target.replace("-", "").replace("_", "").upper()
        if key in groups:
            targets_to_process[key] = groups[key]
        else:
            logger.error(f"Target '{args.target}' not found. Available: {list(groups.keys())}")
            return
    else:
        targets_to_process = groups

    logger.info(f"Cadence filter: {len(targets_to_process)} targets")

    all_results = []
    for target, cadence_group in targets_to_process.items():
        result = run_cadence_filter(
            target=target,
            cadence_group=cadence_group,
            freq_tol_hz=args.freq_tol,
            drift_tol_hz_s=args.drift_tol,
            min_snr=args.min_snr,
            min_on_detections=args.min_on,
        )
        all_results.append(result)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = CADENCE_DIR / f"cadence_{target}_{ts}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {json_path}")

    html_path = CADENCE_DIR / "cadence_summary.html"
    generate_html_report(all_results, html_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print("CADENCE FILTER SUMMARY")
    print(f"{'=' * 60}")
    total_survivors = 0
    for r in all_results:
        s = r["statistics"]
        survivors = s["cadence_survivors"]
        total_survivors += survivors
        status = f"*** {survivors} CANDIDATES ***" if survivors else "all RFI"
        print(
            f"  {r['target']:15s}  "
            f"ON={s['total_on_signals']:5d}  "
            f"consensus={s['multi_on_consensus']:4d}  "
            f"OFF_match={s['off_matched_rfi']:4d}  "
            f"survivors={survivors:3d}  "
            f"({status})"
        )

    print(f"\n  Total cadence-verified candidates: {total_survivors}")
    print(f"  Report: {html_path}")


if __name__ == "__main__":
    main()
