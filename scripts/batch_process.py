#!/usr/bin/env python3
"""
MitraSETI Batch Processor

Process a directory of filterbank (.fil / .h5) files through the full
MitraSETI pipeline with parallel processing and detailed reporting.

Features:
- Parallel processing via multiprocessing (one file per core)
- Progress bar with tqdm
- Results saved to JSON summary file
- CSV export of all detected signals
- Benchmark mode: time each file, report total processing speed

Usage:
    python scripts/batch_process.py --input-dir /path/to/filterbank/files
    python scripts/batch_process.py --input-dir ./data --workers 8 --min-snr 5
    python scripts/batch_process.py --input-dir ./data --benchmark

Output:
    batch_results.json     – Full results for every file
    batch_signals.csv      – CSV of all detected signals
    batch_summary.json     – Summary statistics
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import (
    DATA_DIR,
    ARTIFACTS_DIR,
    FILTERBANK_DIR,
    MODELS_DIR,
    CANDIDATES_DIR,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileResult:
    """Result of processing a single filterbank file."""
    file_path: str
    file_name: str
    file_size_mb: float
    status: str  # "processed", "error", "below_snr", "skipped"
    signal_type: str = ""
    confidence: float = 0.0
    rfi_probability: float = 0.0
    is_rfi: bool = False
    is_candidate: bool = False
    snr: float = 0.0
    drift_rate: float = 0.0
    bandwidth: float = 0.0
    central_freq: float = 0.0
    ood_score: float = 0.0
    is_ood_anomaly: bool = False
    all_scores: Dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0
    error_message: str = ""


@dataclass
class BatchSummary:
    """Summary statistics for a batch run."""
    total_files: int = 0
    processed: int = 0
    errors: int = 0
    skipped: int = 0
    signals_found: int = 0
    candidates_found: int = 0
    rfi_rejected: int = 0
    ood_anomalies: int = 0
    total_size_mb: float = 0.0
    total_elapsed_s: float = 0.0
    throughput_files_per_min: float = 0.0
    throughput_mb_per_s: float = 0.0
    classification_distribution: Dict[str, int] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""
    workers: int = 1
    min_snr: float = 10.0
    max_drift: float = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# File processing (designed for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _load_spectrogram(filepath: Path) -> Optional[np.ndarray]:
    """Load a filterbank / HDF5 file and return 2D spectrogram."""
    try:
        suffix = filepath.suffix.lower()

        if suffix == ".h5":
            try:
                import h5py
                with h5py.File(str(filepath), "r") as f:
                    if "data" in f:
                        data = f["data"][:]
                    elif "filterbank" in f:
                        data = f["filterbank"]["data"][:]
                    else:
                        key = list(f.keys())[0]
                        data = f[key][:]
                data = np.squeeze(data)
                if data.ndim == 3:
                    data = data[0]
                return data.astype(np.float32)
            except ImportError:
                return None

        elif suffix == ".fil":
            try:
                import blimpy
                wf = blimpy.Waterfall(str(filepath), load_data=True)
                data = np.squeeze(wf.data)
                if data.ndim == 3:
                    data = data[0]
                return data.astype(np.float32)
            except ImportError:
                return _read_raw_filterbank(filepath)

        return None
    except Exception:
        return None


def _read_raw_filterbank(filepath: Path) -> Optional[np.ndarray]:
    """Minimal raw filterbank reader."""
    try:
        with open(filepath, "rb") as f:
            raw = f.read()

        header_end = raw.find(b"HEADER_END")
        if header_end < 0:
            return None
        data_start = header_end + len(b"HEADER_END")

        nchans = 256
        header = raw[:data_start]
        idx = header.find(b"nchans")
        if idx >= 0:
            import struct
            nchans = struct.unpack("i", header[idx + 10 : idx + 14])[0]

        data = np.frombuffer(raw[data_start:], dtype=np.float32)
        if len(data) < nchans:
            return None

        ntime = len(data) // nchans
        data = data[: ntime * nchans].reshape(ntime, nchans).T
        return data.astype(np.float32)
    except Exception:
        return None


def process_single_file(
    filepath_str: str,
    min_snr: float = 10.0,
    max_drift: float = 4.0,
) -> dict:
    """
    Process a single file through the MitraSETI pipeline.

    Designed to be called via multiprocessing.Pool.map().
    Must import everything inside to work across process boundaries.
    """
    filepath = Path(filepath_str)
    start_time = time.time()

    result = FileResult(
        file_path=str(filepath),
        file_name=filepath.name,
        file_size_mb=round(filepath.stat().st_size / (1024 * 1024), 2),
        status="error",
    )

    try:
        # Add project root to path (for subprocess)
        proj_root = str(Path(__file__).parent.parent)
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)

        # Load spectrogram
        spectrogram = _load_spectrogram(filepath)
        if spectrogram is None:
            result.status = "skipped"
            result.error_message = "Could not load file"
            result.elapsed_s = time.time() - start_time
            return asdict(result)

        # Feature extraction
        from inference.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()
        features = extractor.extract(spectrogram)

        # SNR filter
        if features.snr < min_snr:
            result.status = "below_snr"
            result.snr = features.snr
            result.elapsed_s = time.time() - start_time
            return asdict(result)

        # Drift rate filter
        if abs(features.drift_rate) > max_drift:
            result.status = "above_max_drift"
            result.snr = features.snr
            result.drift_rate = features.drift_rate
            result.elapsed_s = time.time() - start_time
            return asdict(result)

        # Classification
        from inference.signal_classifier import SignalClassifier, RFI_CLASSES
        classifier = SignalClassifier()
        classification = classifier.classify(spectrogram)

        # OOD detection
        from inference.ood_detector import RadioOODDetector
        ood = RadioOODDetector()
        ood_result = ood.detect(spectrogram, classification.feature_vector)

        # Populate result
        result.status = "processed"
        result.signal_type = classification.signal_type.name.lower()
        result.confidence = round(classification.confidence, 4)
        result.rfi_probability = round(classification.rfi_probability, 4)
        result.is_rfi = SignalClassifier.is_rfi(classification)
        result.is_candidate = SignalClassifier.is_candidate(classification)
        result.snr = round(features.snr, 2)
        result.drift_rate = round(features.drift_rate, 4)
        result.bandwidth = round(features.bandwidth, 2)
        result.central_freq = round(features.central_freq, 2)
        result.ood_score = round(ood_result.ood_score, 4) if ood_result else 0.0
        result.is_ood_anomaly = ood_result.is_anomaly if ood_result else False
        result.all_scores = classification.all_scores
        result.elapsed_s = round(time.time() - start_time, 3)

        return asdict(result)

    except Exception as e:
        result.error_message = str(e)[:200]
        result.elapsed_s = round(time.time() - start_time, 3)
        return asdict(result)


# ─────────────────────────────────────────────────────────────────────────────
# Batch Processor
# ─────────────────────────────────────────────────────────────────────────────

class BatchProcessor:
    """
    Process a directory of filterbank files through the MitraSETI pipeline.

    Supports parallel processing, progress tracking, and detailed output.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        workers: int = 0,
        min_snr: float = 10.0,
        max_drift: float = 4.0,
        benchmark: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else DATA_DIR / "batch_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.workers = workers if workers > 0 else max(1, multiprocessing.cpu_count() - 1)
        self.min_snr = min_snr
        self.max_drift = max_drift
        self.benchmark = benchmark

    def discover_files(self) -> List[Path]:
        """Find all filterbank / HDF5 files in the input directory."""
        files: List[Path] = []
        for ext in ["*.fil", "*.h5"]:
            files.extend(self.input_dir.glob(ext))
            files.extend(self.input_dir.glob(f"**/{ext}"))

        # Deduplicate and sort by size (smallest first for better load balancing)
        seen = set()
        unique = []
        for f in files:
            resolved = f.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(f)

        unique.sort(key=lambda f: f.stat().st_size)
        return unique

    def run(self) -> BatchSummary:
        """Run the batch processor."""
        files = self.discover_files()
        if not files:
            logger.error(f"No .fil or .h5 files found in {self.input_dir}")
            return BatchSummary()

        logger.info(f"Found {len(files)} filterbank files in {self.input_dir}")
        logger.info(f"Using {self.workers} worker(s)")
        logger.info(f"Min SNR: {self.min_snr}, Max drift: {self.max_drift} Hz/s")
        if self.benchmark:
            logger.info("Benchmark mode: timing each file individually")

        summary = BatchSummary(
            total_files=len(files),
            started_at=datetime.now().isoformat(),
            workers=self.workers,
            min_snr=self.min_snr,
            max_drift=self.max_drift,
        )

        total_size = sum(f.stat().st_size for f in files)
        summary.total_size_mb = round(total_size / (1024 * 1024), 2)

        # Process files
        file_paths = [str(f) for f in files]
        process_fn = partial(
            process_single_file,
            min_snr=self.min_snr,
            max_drift=self.max_drift,
        )

        batch_start = time.time()
        results: List[dict] = []

        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        if self.workers > 1:
            with multiprocessing.Pool(processes=self.workers) as pool:
                if has_tqdm:
                    iterator = tqdm(
                        pool.imap_unordered(process_fn, file_paths),
                        total=len(file_paths),
                        desc="Processing",
                        unit="file",
                        ncols=80,
                    )
                else:
                    iterator = pool.imap_unordered(process_fn, file_paths)

                for result in iterator:
                    results.append(result)
        else:
            if has_tqdm:
                iterator = tqdm(file_paths, desc="Processing", unit="file", ncols=80)
            else:
                iterator = file_paths

            for fp in iterator:
                result = process_fn(fp)
                results.append(result)

        batch_elapsed = time.time() - batch_start
        summary.total_elapsed_s = round(batch_elapsed, 2)
        summary.completed_at = datetime.now().isoformat()

        # Aggregate results
        for r in results:
            status = r.get("status", "error")
            if status == "processed":
                summary.processed += 1
                summary.signals_found += 1

                sig_type = r.get("signal_type", "unknown")
                summary.classification_distribution[sig_type] = (
                    summary.classification_distribution.get(sig_type, 0) + 1
                )

                if r.get("is_rfi"):
                    summary.rfi_rejected += 1
                if r.get("is_candidate"):
                    summary.candidates_found += 1
                if r.get("is_ood_anomaly"):
                    summary.ood_anomalies += 1

            elif status == "error":
                summary.errors += 1
            else:
                summary.skipped += 1

        if batch_elapsed > 0:
            summary.throughput_files_per_min = round(
                len(files) / (batch_elapsed / 60), 2
            )
            summary.throughput_mb_per_s = round(
                summary.total_size_mb / batch_elapsed, 2
            )

        # Save results
        self._save_results(results, summary)
        self._save_csv(results)
        self._print_summary(summary, results)

        return summary

    def _save_results(self, results: List[dict], summary: BatchSummary):
        """Save full results and summary to JSON."""
        results_path = self.output_dir / "batch_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")

        summary_path = self.output_dir / "batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(summary), f, indent=2, default=str)
        logger.info(f"Summary saved to {summary_path}")

    def _save_csv(self, results: List[dict]):
        """Export all detected signals to CSV."""
        csv_path = self.output_dir / "batch_signals.csv"

        fieldnames = [
            "file_name", "status", "signal_type", "confidence",
            "rfi_probability", "is_rfi", "is_candidate", "snr",
            "drift_rate", "bandwidth", "central_freq", "ood_score",
            "is_ood_anomaly", "file_size_mb", "elapsed_s",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                if r.get("status") == "processed":
                    writer.writerow(r)

        logger.info(f"CSV export saved to {csv_path}")

    def _print_summary(self, summary: BatchSummary, results: List[dict]):
        """Print summary to console."""
        print("\n" + "=" * 60)
        print("  BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"  Total files:     {summary.total_files}")
        print(f"  Processed:       {summary.processed}")
        print(f"  Errors:          {summary.errors}")
        print(f"  Skipped:         {summary.skipped}")
        print(f"  Total size:      {summary.total_size_mb:.1f} MB")
        print(f"  Total time:      {summary.total_elapsed_s:.1f}s")
        print(f"  Throughput:      {summary.throughput_files_per_min:.1f} files/min")
        print(f"                   {summary.throughput_mb_per_s:.2f} MB/s")
        print(f"  Workers:         {summary.workers}")
        print("")
        print(f"  Signals found:   {summary.signals_found}")
        print(f"  Candidates:      {summary.candidates_found}")
        print(f"  RFI rejected:    {summary.rfi_rejected}")
        print(f"  OOD anomalies:   {summary.ood_anomalies}")

        if summary.classification_distribution:
            print(f"\n  Classification Distribution:")
            for sig_type, count in sorted(
                summary.classification_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                pct = count / summary.signals_found * 100 if summary.signals_found > 0 else 0
                print(f"    {sig_type:30s} {count:5d} ({pct:5.1f}%)")

        if self.benchmark:
            print(f"\n  Benchmark Results:")
            processed = [r for r in results if r.get("status") == "processed"]
            if processed:
                times = [r["elapsed_s"] for r in processed]
                sizes = [r["file_size_mb"] for r in processed]
                print(f"    Avg time/file:  {np.mean(times):.3f}s")
                print(f"    Min time/file:  {np.min(times):.3f}s")
                print(f"    Max time/file:  {np.max(times):.3f}s")
                print(f"    Median time:    {np.median(times):.3f}s")
                print(f"    Avg file size:  {np.mean(sizes):.1f} MB")

                # Per-file benchmark table
                print(f"\n  {'File':<40} {'Size MB':>8} {'Time (s)':>10} {'MB/s':>8}")
                print(f"  {'-'*40} {'-'*8} {'-'*10} {'-'*8}")
                for r in sorted(processed, key=lambda x: x["elapsed_s"], reverse=True):
                    speed = r["file_size_mb"] / r["elapsed_s"] if r["elapsed_s"] > 0 else 0
                    print(
                        f"  {r['file_name']:<40} "
                        f"{r['file_size_mb']:>8.1f} "
                        f"{r['elapsed_s']:>10.3f} "
                        f"{speed:>8.2f}"
                    )

        print(f"\n  Output: {self.output_dir}")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MitraSETI Batch Processor – process filterbank files in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/batch_process.py --input-dir ./data/filterbank
    python scripts/batch_process.py --input-dir ./data --workers 8 --min-snr 5
    python scripts/batch_process.py --input-dir ./data --benchmark
    python scripts/batch_process.py --input-dir ./data --output-dir ./results
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(FILTERBANK_DIR),
        help=f"Directory containing .fil / .h5 files (default: {FILTERBANK_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: astroseti_artifacts/data/batch_output)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=10.0,
        help="Minimum signal-to-noise ratio threshold (default: 10.0)",
    )
    parser.add_argument(
        "--max-drift",
        type=float,
        default=4.0,
        help="Maximum drift rate in Hz/s (default: 4.0)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmark mode: time each file individually",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    processor = BatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        min_snr=args.min_snr,
        max_drift=args.max_drift,
        benchmark=args.benchmark,
    )
    processor.run()


if __name__ == "__main__":
    main()
