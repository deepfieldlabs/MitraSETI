#!/usr/bin/env python3
"""
ON-OFF Cadence Analysis for MitraSETI

Implements the standard Breakthrough Listen ABACAD cadence filter:
  A = ON-source (target)
  B/C/D = OFF-source (reference pointing)

A genuine signal should appear in all A observations but none of B/C/D.
Signals present in both ON and OFF are RFI and rejected.

Usage:
    python scripts/cadence_analysis.py
    python scripts/cadence_analysis.py --target TRAPPIST1
    python scripts/cadence_analysis.py --freq-tolerance 0.002
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import DATA_DIR, FILTERBANK_DIR, CANDIDATES_DIR

logger = logging.getLogger("cadence_analysis")

_FREQ_TOLERANCE_MHZ = 0.005
_DRIFT_TOLERANCE_HZ_S = 0.5
_SNR_FLOOR = 8.0


@dataclass
class Signal:
    """A detected signal with frequency and drift."""
    frequency_mhz: float
    drift_rate: float
    snr: float
    classification: str = ""
    ood_score: float = 0.0
    is_candidate: bool = False
    source_file: str = ""


@dataclass
class CadenceGroup:
    """A group of ON/OFF observations for one target."""
    target: str
    on_files: List[Path] = field(default_factory=list)
    off_files: List[Path] = field(default_factory=list)
    on_signals: Dict[str, List[Signal]] = field(default_factory=dict)
    off_signals: Dict[str, List[Signal]] = field(default_factory=dict)
    passed: List[Signal] = field(default_factory=list)
    rejected_rfi: int = 0


@dataclass
class CadenceResult:
    """Complete results of cadence analysis."""
    groups: List[CadenceGroup] = field(default_factory=list)
    total_on_signals: int = 0
    total_off_signals: int = 0
    cadence_passed: int = 0
    rfi_rejected: int = 0
    timestamp: str = ""


def discover_on_off_pairs(data_dir: Optional[Path] = None) -> List[CadenceGroup]:
    """Scan the data directory and pair ON/OFF files by target name.

    BL naming convention:
      ON:  blcNN_guppi_MJDNN_NNNNN_TARGETNAME_NNNN.ext
      OFF: blcNN_guppi_MJDNN_NNNNN_TARGETNAME_OFF_NNNN.ext
    """
    if data_dir is None:
        data_dir = FILTERBANK_DIR

    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return []

    extensions = {".fil", ".h5"}
    all_files = sorted(
        f for f in data_dir.iterdir()
        if f.suffix.lower() in extensions and f.stat().st_size > 1_000_000
    )

    target_pattern = re.compile(
        r"(?:DIAG_)?([A-Za-z0-9+_-]+?)(?:_OFF)?_\d{4}\.",
        re.IGNORECASE,
    )
    off_pattern = re.compile(r"_OFF_\d{4}\.", re.IGNORECASE)
    on_pattern_simple = re.compile(r"_ON_\d+\.", re.IGNORECASE)
    off_pattern_simple = re.compile(r"_OFF_\d+\.", re.IGNORECASE)

    groups_map: Dict[str, CadenceGroup] = {}

    for f in all_files:
        name = f.name

        is_off = bool(off_pattern.search(name)) or bool(off_pattern_simple.search(name))

        m = target_pattern.search(name)
        if not m:
            if on_pattern_simple.search(name):
                target = name.split("_ON_")[0].upper()
            elif off_pattern_simple.search(name):
                target = name.split("_OFF_")[0].upper()
            else:
                continue
        else:
            target = m.group(1).upper().replace("DIAG_", "")

        target = target.rstrip("_")

        if target not in groups_map:
            groups_map[target] = CadenceGroup(target=target)

        if is_off:
            groups_map[target].off_files.append(f)
        else:
            groups_map[target].on_files.append(f)

    valid = [g for g in groups_map.values() if g.on_files and g.off_files]
    valid.sort(key=lambda g: g.target)

    logger.info(
        f"Found {len(valid)} targets with ON-OFF pairs "
        f"(out of {len(groups_map)} total targets)"
    )
    for g in valid:
        logger.info(f"  {g.target}: {len(g.on_files)} ON, {len(g.off_files)} OFF")

    return valid


def _extract_signals(result: dict, filepath: Path) -> List[Signal]:
    """Extract Signal objects from pipeline result."""
    signals = []
    for c in result.get("candidates", []):
        freq_hz = c.get("frequency_hz", 0)
        freq_mhz = freq_hz / 1e6 if freq_hz > 1e4 else freq_hz
        sig = Signal(
            frequency_mhz=freq_mhz,
            drift_rate=c.get("drift_rate", 0.0),
            snr=c.get("snr", 0.0),
            classification=c.get("classification", ""),
            ood_score=c.get("ood_score", 0.0),
            is_candidate=c.get("is_candidate", False),
            source_file=filepath.name,
        )
        if sig.snr >= _SNR_FLOOR:
            signals.append(sig)
    return signals


def _signals_match(a: Signal, b: Signal,
                   freq_tol: float = _FREQ_TOLERANCE_MHZ,
                   drift_tol: float = _DRIFT_TOLERANCE_HZ_S) -> bool:
    """Check if two signals are the same (within tolerance)."""
    freq_match = abs(a.frequency_mhz - b.frequency_mhz) < freq_tol
    drift_match = abs(a.drift_rate - b.drift_rate) < drift_tol
    return freq_match and drift_match


def run_cadence_filter(
    groups: List[CadenceGroup],
    pipeline=None,
    freq_tolerance: float = _FREQ_TOLERANCE_MHZ,
) -> CadenceResult:
    """Run ON-OFF cadence analysis on all groups.

    For each target group:
    1. Process all ON files → collect signals
    2. Process all OFF files → collect signals
    3. For each ON signal, check if a matching signal appears in any OFF file
    4. Signals in ON but not in OFF → pass cadence filter (potential ETI)
    5. Signals in both → rejected as RFI
    """
    if pipeline is None:
        from pipeline import MitraSETIPipeline
        pipeline = MitraSETIPipeline()

    result = CadenceResult(timestamp=datetime.now().isoformat())

    for group in groups:
        logger.info(f"\n{'='*60}")
        logger.info(f"Cadence analysis: {group.target}")
        logger.info(f"  ON files: {len(group.on_files)}")
        logger.info(f"  OFF files: {len(group.off_files)}")

        for f in group.on_files:
            try:
                logger.info(f"  Processing ON: {f.name}")
                res = pipeline.process_file(str(f))
                sigs = _extract_signals(res, f)
                group.on_signals[f.name] = sigs
                logger.info(f"    → {len(sigs)} signals (SNR ≥ {_SNR_FLOOR})")
            except Exception as e:
                logger.error(f"    Failed: {e}")
                group.on_signals[f.name] = []

        for f in group.off_files:
            try:
                logger.info(f"  Processing OFF: {f.name}")
                res = pipeline.process_file(str(f))
                sigs = _extract_signals(res, f)
                group.off_signals[f.name] = sigs
                logger.info(f"    → {len(sigs)} signals (SNR ≥ {_SNR_FLOOR})")
            except Exception as e:
                logger.error(f"    Failed: {e}")
                group.off_signals[f.name] = []

        all_on_sigs: List[Signal] = []
        for sigs in group.on_signals.values():
            all_on_sigs.extend(sigs)

        all_off_sigs: List[Signal] = []
        for sigs in group.off_signals.values():
            all_off_sigs.extend(sigs)

        result.total_on_signals += len(all_on_sigs)
        result.total_off_signals += len(all_off_sigs)

        for on_sig in all_on_sigs:
            in_off = any(
                _signals_match(on_sig, off_sig, freq_tolerance)
                for off_sig in all_off_sigs
            )
            if in_off:
                group.rejected_rfi += 1
            else:
                group.passed.append(on_sig)

        result.cadence_passed += len(group.passed)
        result.rfi_rejected += group.rejected_rfi

        logger.info(f"  Result: {len(group.passed)} passed cadence, "
                     f"{group.rejected_rfi} rejected as RFI")

        if group.passed:
            logger.info("  Cadence-passing signals:")
            for s in sorted(group.passed, key=lambda x: -x.snr)[:10]:
                logger.info(
                    f"    freq={s.frequency_mhz:.6f} MHz  "
                    f"drift={s.drift_rate:.4f} Hz/s  "
                    f"SNR={s.snr:.1f}  class={s.classification}  "
                    f"file={s.source_file}"
                )

    result.groups = groups
    return result


def save_cadence_results(result: CadenceResult, output_dir: Optional[Path] = None):
    """Save cadence analysis results to JSON."""
    if output_dir is None:
        output_dir = CANDIDATES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": result.timestamp,
        "summary": {
            "total_targets": len(result.groups),
            "total_on_signals": result.total_on_signals,
            "total_off_signals": result.total_off_signals,
            "cadence_passed": result.cadence_passed,
            "rfi_rejected": result.rfi_rejected,
        },
        "targets": [],
    }

    for g in result.groups:
        target_data = {
            "target": g.target,
            "on_files": [f.name for f in g.on_files],
            "off_files": [f.name for f in g.off_files],
            "on_signal_count": sum(len(s) for s in g.on_signals.values()),
            "off_signal_count": sum(len(s) for s in g.off_signals.values()),
            "cadence_passed": len(g.passed),
            "rfi_rejected": g.rejected_rfi,
            "passed_signals": [
                {
                    "frequency_mhz": s.frequency_mhz,
                    "drift_rate": s.drift_rate,
                    "snr": s.snr,
                    "classification": s.classification,
                    "ood_score": s.ood_score,
                    "source_file": s.source_file,
                }
                for s in g.passed
            ],
        }
        output["targets"].append(target_data)

    out_path = output_dir / "cadence_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    return out_path


def print_summary(result: CadenceResult):
    """Print a formatted summary of cadence results."""
    print("\n" + "=" * 70)
    print("  MitraSETI ON-OFF CADENCE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Targets analyzed:  {len(result.groups)}")
    print(f"  Total ON signals:  {result.total_on_signals}")
    print(f"  Total OFF signals: {result.total_off_signals}")
    print(f"  Cadence PASSED:    {result.cadence_passed}")
    print(f"  RFI rejected:      {result.rfi_rejected}")
    print("-" * 70)

    for g in result.groups:
        status = "✓ HITS" if g.passed else "· clean"
        print(f"\n  {g.target} [{status}]")
        print(f"    ON files:  {len(g.on_files)}  ({sum(len(s) for s in g.on_signals.values())} signals)")
        print(f"    OFF files: {len(g.off_files)}  ({sum(len(s) for s in g.off_signals.values())} signals)")
        print(f"    Passed:    {len(g.passed)}  |  RFI: {g.rejected_rfi}")

        if g.passed:
            print("    Top cadence-passing signals:")
            for s in sorted(g.passed, key=lambda x: -x.snr)[:5]:
                print(
                    f"      {s.frequency_mhz:.6f} MHz | "
                    f"drift {s.drift_rate:+.4f} Hz/s | "
                    f"SNR {s.snr:.1f} | {s.classification}"
                )

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MitraSETI ON-OFF Cadence Analysis"
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Analyze only this target (e.g., TRAPPIST1, GJ699)",
    )
    parser.add_argument(
        "--freq-tolerance", type=float, default=_FREQ_TOLERANCE_MHZ,
        help=f"Frequency matching tolerance in MHz (default: {_FREQ_TOLERANCE_MHZ})",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory path",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    data_dir = Path(args.data_dir) if args.data_dir else None
    groups = discover_on_off_pairs(data_dir)

    if args.target:
        target_upper = args.target.upper()
        groups = [g for g in groups if target_upper in g.target]
        if not groups:
            print(f"No ON-OFF pairs found for target '{args.target}'")
            sys.exit(1)

    if not groups:
        print("No ON-OFF pairs found in data directory.")
        print(f"Searched: {data_dir or FILTERBANK_DIR}")
        sys.exit(1)

    print(f"\nFound {len(groups)} targets with ON-OFF pairs:")
    for g in groups:
        print(f"  {g.target}: {len(g.on_files)} ON, {len(g.off_files)} OFF")

    result = run_cadence_filter(groups, freq_tolerance=args.freq_tolerance)
    out_path = save_cadence_results(result)
    print_summary(result)
    print(f"\nDetailed results: {out_path}")


if __name__ == "__main__":
    main()
