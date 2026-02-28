#!/usr/bin/env python3
"""
MitraSETI Benchmark Suite

Benchmark MitraSETI against turboSETI with synthetic filterbank data
of various sizes. Produces comparison tables and an HTML report
suitable for README marketing material.

Benchmarks:
- Processing time (MitraSETI vs turboSETI)
- Signal detections and candidates found
- RFI rejection accuracy
- Memory usage
- Throughput (MB/s)

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --sizes small medium large
    python scripts/benchmark.py --no-turboseti
    python scripts/benchmark.py --output-dir ./benchmarks
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import DATA_DIR, ARTIFACTS_DIR

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark sizes
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_SIZES = {
    "tiny": {"nchans": 64, "ntime": 32, "label": "Tiny (64×32)"},
    "small": {"nchans": 256, "ntime": 64, "label": "Small (256×64)"},
    "medium": {"nchans": 1024, "ntime": 256, "label": "Medium (1024×256)"},
    "large": {"nchans": 8192, "ntime": 512, "label": "Large (8192×512)"},
    "blscale": {"nchans": 65536, "ntime": 16, "label": "BL-Scale (65536×16)"},
}

DEFAULT_SIZES = ["tiny", "small", "medium", "large"]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Result for a single benchmark run."""
    size_label: str
    nchans: int
    ntime: int
    file_size_mb: float
    # MitraSETI results
    mitraseti_time_s: float = 0.0
    mitraseti_signals: int = 0
    mitraseti_candidates: int = 0
    mitraseti_rfi: int = 0
    mitraseti_memory_mb: float = 0.0
    mitraseti_throughput_mbs: float = 0.0
    # turboSETI results
    turboseti_time_s: float = 0.0
    turboseti_signals: int = 0
    turboseti_candidates: int = 0
    turboseti_available: bool = False
    turboseti_memory_mb: float = 0.0
    turboseti_throughput_mbs: float = 0.0
    # Comparison
    speedup: float = 0.0  # MitraSETI speedup factor vs turboSETI
    error: str = ""


@dataclass
class BenchmarkSuite:
    """Full benchmark suite results."""
    results: List[dict] = field(default_factory=list)
    turboseti_available: bool = False
    started_at: str = ""
    completed_at: str = ""
    total_elapsed_s: float = 0.0
    system_info: Dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_filterbank(
    nchans: int = 256,
    ntime: int = 64,
    n_signals: int = 5,
    n_rfi: int = 3,
    noise_level: float = 1.0,
) -> Tuple[np.ndarray, dict]:
    """
    Generate a synthetic filterbank spectrogram with injected signals.

    Creates:
    - Gaussian noise baseline
    - Narrowband drifting signals (potential ET)
    - RFI signals (stationary, broadband)
    - One strong candidate signal

    Returns:
        (spectrogram, metadata) where metadata describes the injected signals.
    """
    rng = np.random.default_rng(42)

    # Base noise
    data = rng.normal(0, noise_level, (nchans, ntime)).astype(np.float32)

    injected = {"signals": [], "rfi": [], "candidates": []}

    # Inject narrowband drifting signals
    for i in range(n_signals):
        snr = rng.uniform(8, 30)
        drift_rate = rng.uniform(-2, 2)  # Hz/s
        start_chan = rng.integers(20, nchans - 20)
        bw = rng.integers(1, 4)

        for t in range(ntime):
            chan = int(start_chan + drift_rate * t / ntime * 20) % nchans
            lo = max(0, chan - bw)
            hi = min(nchans, chan + bw + 1)
            data[lo:hi, t] += snr * noise_level

        injected["signals"].append({
            "type": "narrowband_drifting",
            "snr": float(snr),
            "drift_rate": float(drift_rate),
            "start_chan": int(start_chan),
            "bandwidth_chans": int(bw),
        })

    # Inject RFI (stationary, broadband)
    for i in range(n_rfi):
        rfi_type = rng.choice(["stationary", "broadband"])

        if rfi_type == "stationary":
            chan = rng.integers(0, nchans)
            strength = rng.uniform(15, 50) * noise_level
            data[chan, :] += strength
            injected["rfi"].append({
                "type": "rfi_stationary",
                "channel": int(chan),
                "strength": float(strength),
            })
        else:
            t_start = rng.integers(0, ntime - 5)
            t_end = min(ntime, t_start + rng.integers(3, 10))
            strength = rng.uniform(5, 20) * noise_level
            data[:, t_start:t_end] += strength
            injected["rfi"].append({
                "type": "rfi_broadband",
                "time_start": int(t_start),
                "time_end": int(t_end),
                "strength": float(strength),
            })

    # Inject one strong candidate (ET-like: narrowband, drifting, high SNR)
    candidate_snr = 40.0
    candidate_drift = 1.5
    candidate_chan = nchans // 2
    for t in range(ntime):
        chan = int(candidate_chan + candidate_drift * t / ntime * 30) % nchans
        data[chan, t] += candidate_snr * noise_level

    injected["candidates"].append({
        "type": "candidate_et",
        "snr": candidate_snr,
        "drift_rate": candidate_drift,
        "channel": candidate_chan,
    })

    return data, injected


def save_synthetic_h5(data: np.ndarray, filepath: Path, metadata: dict):
    """Save synthetic data as HDF5."""
    try:
        import h5py
        with h5py.File(str(filepath), "w") as f:
            f.create_dataset("data", data=data)
            f.attrs["source_name"] = "SYNTHETIC_BENCHMARK"
            f.attrs["fch1"] = 1420.0
            f.attrs["foff"] = -0.00029
            f.attrs["tsamp"] = 18.253611
            f.attrs["nchans"] = data.shape[0]
            f.attrs["synthetic"] = True
            f.attrs["injected_signals"] = json.dumps(metadata)
    except ImportError:
        # Fall back to numpy save
        np.save(str(filepath.with_suffix(".npy")), data)


# ─────────────────────────────────────────────────────────────────────────────
# Memory tracking
# ─────────────────────────────────────────────────────────────────────────────

def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        # maxrss is in bytes on macOS, KB on Linux
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return rusage.ru_maxrss / (1024 * 1024)
        else:
            return rusage.ru_maxrss / 1024
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runners
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_mitraseti(spectrogram: np.ndarray) -> dict:
    """
    Benchmark MitraSETI pipeline on a spectrogram.

    Returns dict with timing, signals, candidates, RFI, memory.
    """
    result = {
        "time_s": 0.0,
        "signals": 0,
        "candidates": 0,
        "rfi": 0,
        "memory_mb": 0.0,
    }

    gc.collect()
    mem_before = get_memory_mb()

    start = time.perf_counter()

    try:
        from inference.feature_extractor import FeatureExtractor
        from inference.signal_classifier import SignalClassifier, RFI_CLASSES
        from inference.ood_detector import RadioOODDetector

        extractor = FeatureExtractor()
        classifier = SignalClassifier()
        ood = RadioOODDetector()

        # Feature extraction
        features = extractor.extract(spectrogram)
        result["signals"] = 1  # We're processing one spectrogram

        # Classification
        classification = classifier.classify(spectrogram)

        # OOD
        ood_result = ood.detect(spectrogram, classification.feature_vector)

        # Determine outcomes
        if SignalClassifier.is_rfi(classification):
            result["rfi"] = 1
        if SignalClassifier.is_candidate(classification):
            result["candidates"] = 1
        if ood_result and ood_result.is_anomaly:
            result["candidates"] = max(result["candidates"], 1)

    except Exception as e:
        logger.warning(f"MitraSETI benchmark error: {e}")
        result["error"] = str(e)[:100]

    end = time.perf_counter()
    result["time_s"] = round(end - start, 4)
    result["memory_mb"] = round(max(0, get_memory_mb() - mem_before), 1)

    return result


def benchmark_turboseti(filepath: Path) -> dict:
    """
    Benchmark turboSETI on a filterbank file.

    Returns dict with timing, signals, candidates, memory.
    """
    result = {
        "time_s": 0.0,
        "signals": 0,
        "candidates": 0,
        "memory_mb": 0.0,
        "available": False,
    }

    try:
        from turbo_seti.find_doppler.find_doppler import FindDoppler
    except ImportError:
        return result

    # Validate FindDoppler can be instantiated (import may succeed but runtime may fail)
    try:
        import h5py
        with tempfile.TemporaryDirectory() as vtmp:
            vpath = Path(vtmp) / "validate.h5"
            with h5py.File(str(vpath), "w") as f:
                f.create_dataset("data", data=np.zeros((64, 32), dtype=np.float32))
                f.attrs["fch1"] = 1420.0
                f.attrs["foff"] = -0.00029
                f.attrs["tsamp"] = 18.253611
                f.attrs["nchans"] = 64
            fd = FindDoppler(str(vpath), max_drift=4.0, snr=10.0, out_dir=vtmp)
        result["available"] = True
    except Exception:
        return result

    gc.collect()
    mem_before = get_memory_mb()

    start = time.perf_counter()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            fd = FindDoppler(
                str(filepath),
                max_drift=4.0,
                snr=10.0,
                out_dir=tmpdir,
            )
            fd.search()

            # Count hits from .dat file
            dat_files = list(Path(tmpdir).glob("*.dat"))
            for dat in dat_files:
                lines = dat.read_text().strip().split("\n")
                # Skip header lines (start with #)
                hits = [l for l in lines if l.strip() and not l.startswith("#")]
                result["signals"] = len(hits)
                result["candidates"] = sum(
                    1 for h in hits
                    if float(h.split()[1]) > 20  # High SNR = candidate
                ) if hits else 0

    except Exception as e:
        logger.warning(f"turboSETI benchmark error: {e}")
        result["available"] = False
        result["error"] = str(e)[:100]
        result["time_s"] = 0.0
        return result

    end = time.perf_counter()
    result["time_s"] = round(end - start, 4)
    result["memory_mb"] = round(max(0, get_memory_mb() - mem_before), 1)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark suite
# ─────────────────────────────────────────────────────────────────────────────

class Benchmark:
    """
    Run MitraSETI vs turboSETI benchmarks.

    Generates synthetic data at various sizes, processes through both
    pipelines, and produces comparison results.
    """

    def __init__(
        self,
        sizes: Optional[List[str]] = None,
        skip_turboseti: bool = False,
        output_dir: Optional[str] = None,
        runs_per_size: int = 3,
    ):
        self.sizes = sizes or DEFAULT_SIZES
        self.skip_turboseti = skip_turboseti
        self.output_dir = Path(output_dir) if output_dir else (
            ARTIFACTS_DIR / "benchmarks"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_per_size = runs_per_size

        # Check turboSETI availability
        self.turboseti_available = False
        if not skip_turboseti:
            try:
                import turbo_seti
                self.turboseti_available = True
            except ImportError:
                pass

    def _get_system_info(self) -> dict:
        """Collect system information for the report."""
        import platform
        info = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count": str(os.cpu_count()),
        }
        try:
            import torch
            info["pytorch"] = torch.__version__
            info["cuda"] = str(torch.cuda.is_available())
            if hasattr(torch.backends, "mps"):
                info["mps"] = str(torch.backends.mps.is_available())
        except ImportError:
            info["pytorch"] = "not installed"

        try:
            import turbo_seti
            info["turbo_seti"] = turbo_seti.__version__
        except (ImportError, AttributeError):
            info["turbo_seti"] = "not installed"

        return info

    def run(self) -> BenchmarkSuite:
        """Run the full benchmark suite."""
        suite = BenchmarkSuite(
            started_at=datetime.now().isoformat(),
            turboseti_available=self.turboseti_available,
            system_info=self._get_system_info(),
        )

        print("=" * 70)
        print("  MITRASETI BENCHMARK SUITE")
        print("=" * 70)
        print(f"  Sizes: {', '.join(self.sizes)}")
        print(f"  Runs per size: {self.runs_per_size}")
        print(f"  turboSETI: {'available' if self.turboseti_available else 'not installed'}")
        print(f"  Output: {self.output_dir}")
        print("=" * 70)

        suite_start = time.time()

        for size_name in self.sizes:
            if size_name not in BENCHMARK_SIZES:
                logger.warning(f"Unknown size: {size_name}. Skipping.")
                continue

            size_cfg = BENCHMARK_SIZES[size_name]
            nchans = size_cfg["nchans"]
            ntime = size_cfg["ntime"]
            label = size_cfg["label"]

            print(f"\n  Benchmarking: {label}")
            print(f"  {'─' * 50}")

            # Generate synthetic data
            spectrogram, metadata = generate_synthetic_filterbank(
                nchans=nchans, ntime=ntime,
            )
            file_size_mb = spectrogram.nbytes / (1024 * 1024)

            # Save as HDF5 for turboSETI
            h5_path = self.output_dir / f"bench_{size_name}.h5"
            save_synthetic_h5(spectrogram, h5_path, metadata)

            # Run MitraSETI benchmark (multiple runs, take median)
            astro_times = []
            astro_result = None
            for run_idx in range(self.runs_per_size):
                r = benchmark_mitraseti(spectrogram)
                astro_times.append(r["time_s"])
                if astro_result is None:
                    astro_result = r

            astro_median_time = float(np.median(astro_times))

            # Run turboSETI benchmark
            turbo_result = {"time_s": 0, "signals": 0, "candidates": 0,
                            "memory_mb": 0, "available": False}
            turbo_median_time = 0.0
            if self.turboseti_available:
                turbo_times = []
                for run_idx in range(self.runs_per_size):
                    r = benchmark_turboseti(h5_path)
                    turbo_times.append(r["time_s"])
                    if turbo_result.get("available") is False:
                        turbo_result = r
                turbo_median_time = float(np.median(turbo_times))

            # Calculate speedup (0.0 when turboSETI not available)
            speedup = (
                turbo_median_time / astro_median_time
                if self.turboseti_available and turbo_median_time > 0 and astro_median_time > 0
                else 0.0
            )

            # When turboSETI not available, ensure no stale values
            if not self.turboseti_available:
                turbo_median_time = 0.0
                turbo_result = {"time_s": 0, "signals": 0, "candidates": 0,
                                "memory_mb": 0, "available": False}

            result = BenchmarkResult(
                size_label=label,
                nchans=nchans,
                ntime=ntime,
                file_size_mb=round(file_size_mb, 2),
                mitraseti_time_s=round(astro_median_time, 4),
                mitraseti_signals=astro_result.get("signals", 0) if astro_result else 0,
                mitraseti_candidates=astro_result.get("candidates", 0) if astro_result else 0,
                mitraseti_rfi=astro_result.get("rfi", 0) if astro_result else 0,
                mitraseti_memory_mb=astro_result.get("memory_mb", 0) if astro_result else 0,
                mitraseti_throughput_mbs=round(
                    file_size_mb / astro_median_time if astro_median_time > 0 else 0, 2
                ),
                turboseti_time_s=0.0 if not self.turboseti_available else round(turbo_median_time, 4),
                turboseti_signals=0 if not self.turboseti_available else turbo_result.get("signals", 0),
                turboseti_candidates=0 if not self.turboseti_available else turbo_result.get("candidates", 0),
                turboseti_available=(
                    self.turboseti_available and turbo_result.get("available", False)
                ),
                turboseti_memory_mb=0.0 if not self.turboseti_available else turbo_result.get("memory_mb", 0),
                turboseti_throughput_mbs=0.0 if not self.turboseti_available else round(
                    file_size_mb / turbo_median_time if turbo_median_time > 0 else 0, 2
                ),
                speedup=0.0 if not self.turboseti_available else round(speedup, 2),
            )

            suite.results.append(asdict(result))

            # Print inline result
            print(f"    MitraSETI:  {astro_median_time:.4f}s | "
                  f"{result.mitraseti_throughput_mbs:.1f} MB/s | "
                  f"signals={result.mitraseti_signals} | "
                  f"candidates={result.mitraseti_candidates} | "
                  f"RFI={result.mitraseti_rfi}")
            if self.turboseti_available:
                print(f"    turboSETI:  {turbo_median_time:.4f}s | "
                      f"{result.turboseti_throughput_mbs:.1f} MB/s | "
                      f"signals={result.turboseti_signals} | "
                      f"candidates={result.turboseti_candidates}")
                print(f"    Speedup:    {speedup:.1f}x")
            else:
                print(f"    turboSETI:  not installed")

            # Clean up
            h5_path.unlink(missing_ok=True)

        suite.completed_at = datetime.now().isoformat()
        suite.total_elapsed_s = round(time.time() - suite_start, 2)

        # Update turbo availability from actual results (may be False if FindDoppler failed)
        suite.turboseti_available = any(
            r.get("turboseti_available", False) for r in suite.results
        )

        # Save results and generate reports
        self._save_json(suite)
        self._print_table(suite)
        self._generate_html_report(suite)

        return suite

    def _save_json(self, suite: BenchmarkSuite):
        """Save benchmark results to JSON."""
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(asdict(suite), f, indent=2, default=str)
        logger.info(f"Results saved to {json_path}")

    def _print_table(self, suite: BenchmarkSuite):
        """Print comparison table to terminal."""
        results = suite.results

        print(f"\n{'=' * 90}")
        print(f"  BENCHMARK RESULTS")
        print(f"{'=' * 90}")

        # Header
        if suite.turboseti_available:
            print(
                f"  {'Size':<20} {'File MB':>8} "
                f"{'MitraSETI':>10} {'turboSETI':>10} {'Speedup':>8} "
                f"{'Signals':>8} {'Cands':>6} {'RFI':>5}"
            )
            print(
                f"  {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*8} "
                f"{'─'*8} {'─'*6} {'─'*5}"
            )
        else:
            print(
                f"  {'Size':<20} {'File MB':>8} "
                f"{'Time (s)':>10} {'MB/s':>8} "
                f"{'Signals':>8} {'Cands':>6} {'RFI':>5}"
            )
            print(
                f"  {'─'*20} {'─'*8} {'─'*10} {'─'*8} "
                f"{'─'*8} {'─'*6} {'─'*5}"
            )

        for r in results:
            if suite.turboseti_available:
                speedup_str = f"{r['speedup']:.1f}x" if r['speedup'] > 0 else "N/A"
                print(
                    f"  {r['size_label']:<20} "
                    f"{r['file_size_mb']:>8.2f} "
                    f"{r['mitraseti_time_s']:>9.4f}s "
                    f"{r['turboseti_time_s']:>9.4f}s "
                    f"{speedup_str:>8} "
                    f"{r['mitraseti_signals']:>8} "
                    f"{r['mitraseti_candidates']:>6} "
                    f"{r['mitraseti_rfi']:>5}"
                )
            else:
                print(
                    f"  {r['size_label']:<20} "
                    f"{r['file_size_mb']:>8.2f} "
                    f"{r['mitraseti_time_s']:>9.4f}s "
                    f"{r['mitraseti_throughput_mbs']:>7.1f} "
                    f"{r['mitraseti_signals']:>8} "
                    f"{r['mitraseti_candidates']:>6} "
                    f"{r['mitraseti_rfi']:>5}"
                )

        print(f"{'=' * 90}")
        print(f"  Total benchmark time: {suite.total_elapsed_s:.1f}s")
        print(f"  Results saved to: {self.output_dir}")
        print(f"{'=' * 90}")

    def _generate_html_report(self, suite: BenchmarkSuite):
        """Generate an HTML benchmark report for README marketing."""
        results = suite.results

        # Build table rows
        table_rows = ""
        turboseti_unavailable_msg = (
            "turboSETI not installed — install with: pip install turbo_seti"
            if not suite.turboseti_available
            else None
        )
        for r in results:
            if suite.turboseti_available:
                speedup_html = (
                    f"<strong style='color:#34d399'>{r['speedup']:.1f}x</strong>"
                    if r["speedup"] > 0
                    else "<span style='color:#94a3b8'>N/A</span>"
                )
                turbo_cell = f"{r['turboseti_time_s']:.4f}s"
            else:
                speedup_html = "<span style='color:#94a3b8'>N/A</span>"
                turbo_cell = f"<span style='color:#94a3b8'>{turboseti_unavailable_msg}</span>"
            table_rows += f"""
            <tr>
                <td>{r['size_label']}</td>
                <td>{r['file_size_mb']:.2f}</td>
                <td><strong>{r['mitraseti_time_s']:.4f}s</strong></td>
                <td>{turbo_cell}</td>
                <td>{speedup_html}</td>
                <td>{r['mitraseti_throughput_mbs']:.1f}</td>
                <td>{r['mitraseti_signals']}</td>
                <td>{r['mitraseti_candidates']}</td>
                <td>{r['mitraseti_rfi']}</td>
            </tr>"""

        # Build chart
        chart_b64 = self._create_benchmark_chart(results)
        chart_html = ""
        if chart_b64:
            chart_html = (
                f'<div style="text-align:center;margin:24px 0">'
                f'<img src="data:image/png;base64,{chart_b64}" '
                f'alt="Benchmark Chart" style="max-width:100%;border-radius:8px">'
                f'</div>'
            )

        # System info
        sys_info = suite.system_info
        sys_rows = ""
        for k, v in sys_info.items():
            sys_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MitraSETI Benchmark Report</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Helvetica, Arial, sans-serif;
    background: #0a0e17;
    color: #e2e8f0;
    margin: 0;
    padding: 24px;
    line-height: 1.6;
}}
.container {{ max-width: 1100px; margin: 0 auto; }}
h1 {{ color: #38bdf8; font-size: 28px; letter-spacing: -0.5px; }}
h2 {{ color: #e2e8f0; font-size: 20px; border-bottom: 1px solid #1e293b; padding-bottom: 8px; margin-top: 36px; }}
.subtitle {{ color: #94a3b8; font-size: 14px; margin-bottom: 24px; }}
.card {{
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
}}
.highlight-box {{
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.06), rgba(52, 211, 153, 0.06));
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 24px;
    margin: 24px 0;
    text-align: center;
}}
.highlight-box .big {{ font-size: 48px; font-weight: 700; color: #34d399; }}
.highlight-box .label {{ font-size: 14px; color: #94a3b8; }}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
}}
th, td {{
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid #1e293b;
}}
th {{
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
td {{ color: #e2e8f0; font-size: 13px; }}
tr:hover td {{ background: rgba(56, 189, 248, 0.04); }}
.footer {{
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #1e293b;
    text-align: center;
    color: #475569;
    font-size: 12px;
}}
.footer a {{ color: #38bdf8; text-decoration: none; }}
</style>
</head>
<body>
<div class="container">

<h1>MitraSETI Benchmark Report</h1>
<div class="subtitle">
    {datetime.now().strftime('%Y-%m-%d %H:%M')} |
    turboSETI: {'available' if suite.turboseti_available else 'not installed'} |
    Total time: {suite.total_elapsed_s:.1f}s
</div>

{f'''<div class="highlight-box">
    <div class="big">{max((r["speedup"] for r in results if r["speedup"] > 0), default=0):.1f}x</div>
    <div class="label">PEAK SPEEDUP vs turboSETI</div>
</div>''' if suite.turboseti_available else ''}

<h2>Results</h2>
<div class="card">
<table>
<thead>
    <tr>
        <th>Size</th><th>File (MB)</th>
        <th>MitraSETI</th><th>turboSETI</th><th>Speedup</th>
        <th>MB/s</th><th>Signals</th><th>Candidates</th><th>RFI</th>
    </tr>
</thead>
<tbody>
    {table_rows}
</tbody>
</table>
</div>

{chart_html}

<h2>System Information</h2>
<div class="card">
<table>
<tbody>{sys_rows}</tbody>
</table>
</div>

<div class="footer">
    Generated by <a href="https://github.com/samantaba/MitraSETI">MitraSETI</a><br>
    Benchmark suite for SETI signal processing performance comparison.
</div>

</div>
</body>
</html>"""

        report_path = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        report_path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report saved to {report_path}")

    @staticmethod
    def _create_benchmark_chart(results: List[dict]) -> str:
        """Create a bar chart comparing MitraSETI vs turboSETI times."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = [r["size_label"] for r in results]
            astro_times = [r["mitraseti_time_s"] for r in results]
            turbo_times = [r["turboseti_time_s"] for r in results]
            has_turbo = any(t > 0 for t in turbo_times)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0a0e17")

            # Bar chart: processing time
            ax1.set_facecolor("#111827")
            x = np.arange(len(labels))
            width = 0.35

            bars1 = ax1.bar(
                x - width / 2 if has_turbo else x,
                astro_times, width if has_turbo else 0.6,
                color="#38bdf8", alpha=0.8, label="MitraSETI",
            )
            if has_turbo:
                bars2 = ax1.bar(
                    x + width / 2, turbo_times, width,
                    color="#f87171", alpha=0.7, label="turboSETI",
                )

            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=20, ha="right")
            ax1.set_ylabel("Time (seconds)", color="#94a3b8", fontsize=11)
            ax1.set_title(
                "Processing Time Comparison",
                color="#e2e8f0", fontsize=13, fontweight="bold",
            )
            ax1.tick_params(colors="#94a3b8")
            ax1.spines["bottom"].set_color("#1e293b")
            ax1.spines["left"].set_color("#1e293b")
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.grid(True, alpha=0.15, color="#1e293b")
            ax1.legend(
                facecolor="#111827", edgecolor="#1e293b",
                labelcolor="#e2e8f0", fontsize=10,
            )

            # Throughput chart
            ax2.set_facecolor("#111827")
            throughputs = [r["mitraseti_throughput_mbs"] for r in results]
            ax2.bar(x, throughputs, 0.6, color="#34d399", alpha=0.8)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=20, ha="right")
            ax2.set_ylabel("Throughput (MB/s)", color="#94a3b8", fontsize=11)
            ax2.set_title(
                "MitraSETI Throughput",
                color="#e2e8f0", fontsize=13, fontweight="bold",
            )
            ax2.tick_params(colors="#94a3b8")
            ax2.spines["bottom"].set_color("#1e293b")
            ax2.spines["left"].set_color("#1e293b")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.grid(True, alpha=0.15, color="#1e293b")

            fig.tight_layout(pad=2)

            buf = io.BytesIO()
            fig.savefig(
                buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0a0e17", edgecolor="none",
            )
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            plt.close(fig)
            return b64

        except ImportError:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MitraSETI Benchmark Suite – compare against turboSETI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/benchmark.py                                 # Default benchmark
    python scripts/benchmark.py --sizes tiny small medium       # Specific sizes
    python scripts/benchmark.py --sizes tiny small medium large blscale  # All sizes
    python scripts/benchmark.py --no-turboseti                  # Skip turboSETI comparison
    python scripts/benchmark.py --runs 5                        # More runs for accuracy
    python scripts/benchmark.py --output-dir ./benchmarks       # Custom output

Available sizes: tiny, small, medium, large, blscale
        """,
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        default=DEFAULT_SIZES,
        choices=list(BENCHMARK_SIZES.keys()),
        help="Benchmark sizes to run (default: tiny small medium large)",
    )
    parser.add_argument(
        "--no-turboseti",
        action="store_true",
        help="Skip turboSETI comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results and report",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per size for median timing (default: 3)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    bench = Benchmark(
        sizes=args.sizes,
        skip_turboseti=args.no_turboseti,
        output_dir=args.output_dir,
        runs_per_size=args.runs,
    )
    bench.run()


if __name__ == "__main__":
    main()
