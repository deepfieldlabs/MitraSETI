#!/usr/bin/env python3
"""
astroSETI Streaming Observation Engine

Multi-day continuous observation runner with self-correcting intelligence,
daily reporting, and publishing-ready summaries.

Adapted from AstroLens streaming_discovery.py for the radio SETI domain.

This wraps the full astroSETI pipeline with:
- Continuous processing of filterbank (.fil / .h5) files
- Daily HTML report generation (charts, candidate rankings)
- Self-correcting strategy (sensitivity, file source rebalancing)
- Health monitoring (API, disk space, model status)
- State persistence to JSON (save/resume across restarts)

Usage:
    python scripts/streaming_observation.py --days 7
    python scripts/streaming_observation.py --days 3 --mode aggressive
    python scripts/streaming_observation.py --report-only
    python scripts/streaming_observation.py --reset

Reports are saved to: astroseti_artifacts/streaming_reports/
"""

from __future__ import annotations

import argparse
import atexit
import glob as globmod
import json
import logging
import os
import signal
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import (
    DATA_DIR,
    ARTIFACTS_DIR,
    FILTERBANK_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    CANDIDATES_DIR,
    CANDIDATES_FILE,
    STREAMING_STATE,
    DISCOVERY_STATE,
    ASTROLENS_CANDIDATES_FILE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Streaming-specific paths
# ─────────────────────────────────────────────────────────────────────────────

STREAMING_DIR = ARTIFACTS_DIR / "streaming_reports"
STREAMING_DIR.mkdir(parents=True, exist_ok=True)

DAILY_REPORTS_DIR = STREAMING_DIR / "daily"
DAILY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE = DATA_DIR / "streaming_observation.log"

logger = logging.getLogger("astroseti.streaming")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    logger.addHandler(_sh)
    logger.propagate = False


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DailySnapshot:
    """Snapshot of metrics at end of each observation day."""
    day: int = 0
    date: str = ""
    cycles_completed: int = 0
    files_processed: int = 0
    signals_found: int = 0
    candidates_found: int = 0
    rfi_rejected: int = 0
    ood_anomalies: int = 0
    highest_snr: float = 0.0
    highest_ood_score: float = 0.0
    sensitivity_start: float = 0.0
    sensitivity_end: float = 0.0
    processing_speed_mhz_s: float = 0.0
    top_candidates: List[dict] = field(default_factory=list)
    source_effectiveness: Dict[str, dict] = field(default_factory=dict)
    # Self-correction actions taken
    corrections_applied: List[str] = field(default_factory=list)
    # Rate metrics
    candidate_rate: float = 0.0  # candidates per 100 signals
    files_per_hour: float = 0.0
    # Health & errors
    errors_today: int = 0
    health_issues: List[str] = field(default_factory=list)
    # Classification distribution
    classification_counts: Dict[str, int] = field(default_factory=dict)
    # AstroLens cross-reference hits
    astrolens_crossref_hits: int = 0


@dataclass
class StreamingState:
    """Persistent state for multi-day streaming observation."""
    started_at: str = ""
    target_days: float = 7
    current_day: int = 0
    total_runtime_hours: float = 0.0
    daily_snapshots: List[dict] = field(default_factory=list)
    # Cumulative
    total_files_processed: int = 0
    total_signals: int = 0
    total_candidates: int = 0
    total_rfi_rejected: int = 0
    total_ood_anomalies: int = 0
    total_corrections: int = 0
    # Strategy
    current_mode: str = "normal"  # normal, aggressive, turbo
    mode_history: List[dict] = field(default_factory=list)
    # Sensitivity (minimum SNR threshold)
    current_min_snr: float = 10.0
    current_max_drift: float = 4.0  # Hz/s
    # Best overall
    best_candidates: List[dict] = field(default_factory=list)
    # Completion
    completed: bool = False
    completed_at: str = ""
    # Health monitoring
    total_errors: int = 0
    consecutive_errors: int = 0
    error_log: List[dict] = field(default_factory=list)
    last_health_check: str = ""
    # AstroLens cross-reference
    astrolens_crossref_total: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Observer
# ─────────────────────────────────────────────────────────────────────────────

class StreamingObserver:
    """
    Multi-day streaming observation orchestrator.

    Wraps the astroSETI pipeline (Rust core -> classifier -> OOD -> catalog)
    with intelligence layers:
    1. Daily assessment and report generation
    2. Self-correcting sensitivity (threshold, source, mode adjustments)
    3. Health monitoring (API, disk, model status)
    4. Publishing-ready final summary
    """

    # Cycle interval per mode (seconds between processing attempts)
    MODE_INTERVALS = {
        "normal": 30,
        "aggressive": 10,
        "turbo": 2,
    }

    # SNR thresholds per mode
    MODE_SNR = {
        "normal": 10.0,
        "aggressive": 6.0,
        "turbo": 3.0,
    }

    def __init__(
        self,
        target_days: float = 7,
        mode: str = "normal",
        daily_report_hour: int = 0,
    ):
        self.target_days = target_days
        self.mode = mode
        self.daily_report_hour = daily_report_hour
        self.running = True

        # Load or create streaming state
        self.state = self._load_state()
        if not self.state.started_at:
            self.state.started_at = datetime.now().isoformat()

        # Always update from CLI args
        if target_days > 0:
            self.state.target_days = target_days
        self.state.current_mode = mode or self.state.current_mode or "normal"
        self.state.current_min_snr = self.MODE_SNR.get(
            self.state.current_mode, 10.0
        )

        # Track daily metrics for delta calculation
        self._day_start_metrics: Dict = {}
        self._last_report_date: Optional[datetime] = None

        # Unified pipeline
        self._pipeline = None

        # File source tracking
        self._processed_files: set = set()

        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        atexit.register(self._save_state)

    # ── Pipeline (lazy-loaded) ───────────────────────────────────────────

    @property
    def pipeline(self):
        if self._pipeline is None:
            from pipeline import AstroSETIPipeline
            model_path = MODELS_DIR / "signal_classifier_v1.pt"
            ood_cal_path = MODELS_DIR / "ood_calibration.json"
            self._pipeline = AstroSETIPipeline(
                model_path=str(model_path) if model_path.exists() else None,
                ood_calibration_path=str(ood_cal_path) if ood_cal_path.exists() else None,
            )
        return self._pipeline

    # ── State persistence ─────────────────────────────────────────────────

    def _load_state(self) -> StreamingState:
        if STREAMING_STATE.exists():
            try:
                with open(STREAMING_STATE) as f:
                    data = json.load(f)
                valid_fields = {f.name for f in fields(StreamingState)}
                filtered = {k: v for k, v in data.items() if k in valid_fields}
                return StreamingState(**filtered)
            except Exception as e:
                logger.warning(f"Failed to load streaming state: {e}")
        return StreamingState()

    def _save_state(self):
        try:
            data = self._to_native(asdict(self.state))
            # Write runtime fields for UI consumption
            if self.state.started_at:
                try:
                    started = datetime.fromisoformat(self.state.started_at)
                    delta = datetime.now() - started
                    hours, remainder = divmod(int(delta.total_seconds()), 3600)
                    mins, secs = divmod(remainder, 60)
                    data["elapsed"] = f"{hours}:{mins:02d}:{secs:02d}"
                except Exception:
                    pass
            fps = self.state.total_files_processed
            rt = self.state.total_runtime_hours or 0.001
            data["processing_rate"] = f"{fps / max(rt, 0.001):.1f}/hr"
            data["current_file"] = getattr(self, "_current_file_name", "—")
            with open(STREAMING_STATE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save streaming state: {e}")

    @staticmethod
    def _to_native(obj):
        """Convert numpy/non-JSON types to Python native types."""
        if isinstance(obj, dict):
            return {k: StreamingObserver._to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [StreamingObserver._to_native(item) for item in obj]
        elif hasattr(obj, "item"):
            return obj.item()
        return obj

    # ── Handle shutdown ───────────────────────────────────────────────────

    def _handle_shutdown(self, signum=None, frame=None):
        logger.info("\nShutdown signal received. Generating final report...")
        self.running = False
        self._take_daily_snapshot()
        self._generate_daily_report()
        self._generate_final_summary()
        self._save_state()
        self._print_star_reminder()

    # ── File discovery ────────────────────────────────────────────────────

    def _discover_files(self) -> List[Path]:
        """
        Find filterbank / HDF5 files to process.

        Searches the configured FILTERBANK_DIR for .fil and .h5 files
        that haven't been processed yet in this session.
        """
        patterns = ["**/*.fil", "**/*.h5"]
        seen: set = set()
        files: List[Path] = []
        for pattern in patterns:
            for f in FILTERBANK_DIR.glob(pattern):
                resolved = f.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(f)

        # Remove already-processed files
        new_files = [
            f for f in files if str(f) not in self._processed_files
        ]

        if not new_files and files:
            # All files processed; allow re-processing in next cycle
            logger.info(
                f"All {len(files)} files processed. Resetting for next cycle."
            )
            self._processed_files.clear()
            new_files = list(files)

        return sorted(new_files, key=lambda f: f.stat().st_mtime)

    # ── Single file processing via unified pipeline ─────────────────────

    def _process_file(self, filepath: Path) -> Optional[dict]:
        """
        Process a single filterbank file through AstroSETIPipeline.

        Returns:
            Result dictionary or None on failure.
        """
        start_time = time.time()
        file_size_mb = filepath.stat().st_size / (1024 * 1024)

        try:
            pipe_result = self.pipeline.process_file(str(filepath))
        except Exception as e:
            logger.error(f"Pipeline failed for {filepath.name}: {e}")
            return None

        summary = pipe_result.get("summary", {})
        if summary.get("status") == "error":
            logger.warning(f"Pipeline error for {filepath.name}: {summary.get('error')}")
            return None

        candidates = pipe_result.get("candidates", [])
        timing = pipe_result.get("timing", {})

        n_rfi = summary.get("rfi_count", 0)
        n_candidates = summary.get("candidate_count", 0)
        n_anomalies = summary.get("anomaly_count", 0)
        total_hits = summary.get("total_hits_filtered", len(candidates))

        best_snr = max((c.get("snr", 0) for c in candidates), default=0.0)
        best_ood = max((c.get("ood_score", 0) for c in candidates), default=0.0)
        best_drift = max((abs(c.get("drift_rate", 0)) for c in candidates), default=0.0)

        if total_hits == 0 or best_snr < self.state.current_min_snr:
            return {
                "file": str(filepath),
                "file_name": filepath.name,
                "status": "below_snr_threshold",
                "snr": best_snr,
                "min_snr": self.state.current_min_snr,
                "elapsed_s": time.time() - start_time,
            }

        elapsed = time.time() - start_time

        result = {
            "file": str(filepath),
            "file_name": filepath.name,
            "file_size_mb": round(file_size_mb, 2),
            "status": "processed",
            "signal_type": candidates[0].get("classification", "unknown") if candidates else "unknown",
            "confidence": candidates[0].get("confidence", 0.0) if candidates else 0.0,
            "rfi_probability": candidates[0].get("rfi_probability", 0.0) if candidates else 0.0,
            "is_rfi": n_rfi > 0 and n_candidates == 0,
            "is_candidate": n_candidates > 0,
            "snr": round(best_snr, 2),
            "drift_rate": round(best_drift, 4),
            "ood_score": round(best_ood, 4),
            "is_ood_anomaly": n_anomalies > 0,
            "total_hits": total_hits,
            "rfi_count": n_rfi,
            "candidate_count": n_candidates,
            "anomaly_count": n_anomalies,
            "candidates": candidates,
            "timing": timing,
            "elapsed_s": round(elapsed, 3),
            "processed_at": datetime.now().isoformat(),
        }

        return result

    # ── Cycle runner ──────────────────────────────────────────────────────

    def _run_cycle(self) -> dict:
        """
        Run one observation cycle: pick a file, process it, log results.

        Returns:
            Cycle summary dict.
        """
        cycle_summary = {
            "signals_found": 0,
            "candidates": 0,
            "rfi_rejected": 0,
            "ood_anomalies": 0,
            "files_processed": 0,
            "classification_counts": {},
        }

        files = self._discover_files()
        if not files:
            logger.debug("No files available for processing")
            return cycle_summary

        # Pick next file (or batch in turbo mode)
        batch_size = {"normal": 1, "aggressive": 2, "turbo": 5}.get(
            self.state.current_mode, 1
        )
        batch = files[:batch_size]

        for filepath in batch:
            self._current_file_name = filepath.name
            result = self._process_file(filepath)
            self._processed_files.add(str(filepath))

            if result is None:
                continue

            cycle_summary["files_processed"] += 1

            if result.get("status") == "below_snr_threshold":
                continue

            total_hits = result.get("total_hits", 1)
            cycle_summary["signals_found"] += total_hits

            sig_type = result.get("signal_type", "unknown")
            cycle_summary["classification_counts"][sig_type] = (
                cycle_summary["classification_counts"].get(sig_type, 0) + 1
            )

            rfi_count = result.get("rfi_count", 1 if result.get("is_rfi") else 0)
            cycle_summary["rfi_rejected"] += rfi_count
            if rfi_count > 0:
                logger.debug(
                    f"  RFI rejected: {filepath.name} "
                    f"({sig_type}, count={rfi_count})"
                )

            cand_count = result.get("candidate_count", 1 if result.get("is_candidate") else 0)
            cycle_summary["candidates"] += cand_count
            if cand_count > 0:
                self._record_candidate(result)
                logger.info(
                    f"  CANDIDATE: {filepath.name} | "
                    f"type={sig_type} | "
                    f"SNR={result.get('snr', 0):.1f} | "
                    f"drift={result.get('drift_rate', 0):.4f} Hz/s | "
                    f"conf={result.get('confidence', 0):.2f}"
                )

            anom_count = result.get("anomaly_count", 1 if result.get("is_ood_anomaly") else 0)
            cycle_summary["ood_anomalies"] += anom_count
            if anom_count > 0:
                logger.info(
                    f"  OOD ANOMALY: {filepath.name} | "
                    f"ood_score={result.get('ood_score', 0):.4f} | "
                    f"type={sig_type}"
                )

        return cycle_summary

    def _record_candidate(self, result: dict):
        """Append a candidate to the persistent candidates file."""
        try:
            candidates = []
            if CANDIDATES_FILE.exists():
                with open(CANDIDATES_FILE) as f:
                    candidates = json.load(f)

            candidates.append(result)

            with open(CANDIDATES_FILE, "w") as f:
                json.dump(candidates, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to record candidate: {e}")

    # ── Metrics capture ───────────────────────────────────────────────────

    def _capture_metrics(self) -> dict:
        """Read current cumulative metrics from state."""
        return {
            "files_processed": self.state.total_files_processed,
            "signals": self.state.total_signals,
            "candidates": self.state.total_candidates,
            "rfi_rejected": self.state.total_rfi_rejected,
            "ood_anomalies": self.state.total_ood_anomalies,
            "min_snr": self.state.current_min_snr,
            "max_drift": self.state.current_max_drift,
        }

    # ── Daily snapshot & self-correction ──────────────────────────────────

    def _take_daily_snapshot(self):
        """Capture end-of-day metrics and calculate deltas."""
        day_num = self.state.current_day + 1
        start = self._day_start_metrics

        delta_files = self.state.total_files_processed - start.get("files_processed", 0)
        delta_signals = self.state.total_signals - start.get("signals", 0)
        delta_candidates = self.state.total_candidates - start.get("candidates", 0)
        delta_rfi = self.state.total_rfi_rejected - start.get("rfi_rejected", 0)
        delta_ood = self.state.total_ood_anomalies - start.get("ood_anomalies", 0)

        candidate_rate = (
            (delta_candidates / delta_signals * 100)
            if delta_signals > 0
            else 0.0
        )

        hours_elapsed = max(1, (datetime.now() - datetime.fromisoformat(
            self.state.started_at
        )).total_seconds() / 3600)
        files_per_hour = self.state.total_files_processed / hours_elapsed

        # Get best candidates
        best = self._load_candidates()
        sorted_candidates = sorted(
            best, key=lambda c: c.get("ood_score", 0), reverse=True
        )[:10]

        # Self-correction
        corrections = self._self_correct(day_num, delta_signals, delta_candidates, delta_rfi)

        # Health check
        health_issues = self._health_check()

        snapshot = DailySnapshot(
            day=day_num,
            date=datetime.now().strftime("%Y-%m-%d"),
            files_processed=delta_files,
            signals_found=delta_signals,
            candidates_found=delta_candidates,
            rfi_rejected=delta_rfi,
            ood_anomalies=delta_ood,
            highest_ood_score=max(
                (c.get("ood_score", 0) for c in sorted_candidates), default=0.0
            ),
            sensitivity_start=start.get("min_snr", 10.0),
            sensitivity_end=self.state.current_min_snr,
            top_candidates=sorted_candidates,
            corrections_applied=corrections,
            candidate_rate=candidate_rate,
            files_per_hour=files_per_hour,
            errors_today=self.state.consecutive_errors,
            health_issues=health_issues,
        )

        self.state.daily_snapshots.append(asdict(snapshot))
        self.state.current_day = day_num
        self.state.total_corrections += len(corrections)
        self.state.best_candidates = sorted_candidates[:20]
        self._save_state()
        return snapshot

    def _load_candidates(self) -> List[dict]:
        """Load candidates from persistent file."""
        if CANDIDATES_FILE.exists():
            try:
                with open(CANDIDATES_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _self_correct(
        self,
        day_num: int,
        delta_signals: int,
        delta_candidates: int,
        delta_rfi: int,
    ) -> List[str]:
        """
        Self-correcting intelligence layer.

        Analyzes daily performance and makes strategic adjustments:
        1. Sensitivity correction: adjust min SNR threshold
        2. RFI flood detection: tighten if too much RFI
        3. Mode escalation: switch to aggressive/turbo if needed
        4. Source rebalancing: note file source effectiveness
        """
        corrections: List[str] = []

        if delta_signals < 10:
            return corrections  # Not enough data to self-correct

        # 1. Sensitivity correction
        candidate_pct = delta_candidates / delta_signals * 100
        SNR_FLOOR = 3.0
        SNR_CEILING = 25.0

        if candidate_pct > 15:
            # Too many candidates -> likely false positives, raise SNR
            old = self.state.current_min_snr
            new = min(old * 1.3, SNR_CEILING)
            self.state.current_min_snr = new
            corrections.append(
                f"SNR threshold raised {old:.1f} -> {new:.1f} "
                f"(candidate rate {candidate_pct:.1f}% too high)"
            )
            logger.info(f"  [SELF-CORRECT] {corrections[-1]}")

        elif candidate_pct < 0.5 and delta_signals > 50:
            # Too few candidates -> lower SNR threshold
            old = self.state.current_min_snr
            new = max(old * 0.85, SNR_FLOOR)
            if new < old:
                self.state.current_min_snr = new
                corrections.append(
                    f"SNR threshold lowered {old:.1f} -> {new:.1f} "
                    f"(candidate rate {candidate_pct:.1f}% too low)"
                )
                logger.info(f"  [SELF-CORRECT] {corrections[-1]}")

        # 2. RFI flood detection
        rfi_pct = delta_rfi / delta_signals * 100 if delta_signals > 0 else 0
        if rfi_pct > 80:
            corrections.append(
                f"RFI flood detected ({rfi_pct:.0f}% of signals). "
                f"Consider different observation frequency or time."
            )
            logger.warning(f"  [SELF-CORRECT] {corrections[-1]}")

        # 3. Mode escalation
        if day_num >= 2 and self.state.total_candidates == 0:
            if self.state.current_mode == "normal":
                self.state.current_mode = "aggressive"
                self.state.current_min_snr = self.MODE_SNR["aggressive"]
                corrections.append(
                    "Escalated to AGGRESSIVE mode (0 candidates after 2 days)"
                )
                logger.info(f"  [SELF-CORRECT] {corrections[-1]}")
                self.state.mode_history.append({
                    "day": day_num,
                    "from": "normal",
                    "to": "aggressive",
                    "reason": "Zero candidates after 2 days",
                })
            elif day_num >= 3 and self.state.current_mode == "aggressive":
                self.state.current_mode = "turbo"
                self.state.current_min_snr = self.MODE_SNR["turbo"]
                corrections.append(
                    "Escalated to TURBO mode (0 candidates after 3 days)"
                )
                logger.info(f"  [SELF-CORRECT] {corrections[-1]}")
                self.state.mode_history.append({
                    "day": day_num,
                    "from": "aggressive",
                    "to": "turbo",
                    "reason": "Zero candidates after 3 days",
                })

        return corrections

    # ── Health monitoring ─────────────────────────────────────────────────

    def _health_check(self) -> List[str]:
        """Run health checks and return list of issues found."""
        issues: List[str] = []

        # 1. Check API
        try:
            import httpx
            resp = httpx.get("http://localhost:9000/health", timeout=5)
            if resp.status_code != 200:
                issues.append(f"API returned status {resp.status_code}")
        except Exception:
            issues.append("API not responding on port 9000")

        # 2. Disk space
        try:
            usage = shutil.disk_usage(str(ARTIFACTS_DIR))
            free_gb = usage.free / (1024 ** 3)
            if free_gb < 2:
                issues.append(f"LOW DISK SPACE: {free_gb:.1f}GB free")
            elif free_gb < 5:
                issues.append(f"Disk space warning: {free_gb:.1f}GB free")
        except Exception:
            pass

        # 3. Model files exist
        model_dir = MODELS_DIR
        if not any(model_dir.glob("*.pt")) and not any(model_dir.glob("*.pth")):
            issues.append("No model weights found (using untrained models)")

        # 4. Filterbank directory has files
        fil_count = len(list(FILTERBANK_DIR.glob("*.fil")))
        h5_count = len(list(FILTERBANK_DIR.glob("*.h5")))
        if fil_count == 0 and h5_count == 0:
            issues.append(
                f"No filterbank files in {FILTERBANK_DIR}. "
                f"Run download_bl_data.py or add files manually."
            )

        # 5. Error rate
        if self.state.consecutive_errors > 5:
            issues.append(
                f"High error rate: {self.state.consecutive_errors} "
                f"consecutive errors"
            )

        self.state.last_health_check = datetime.now().isoformat()
        return issues

    def _track_error(self, error_msg: str, recoverable: bool = True):
        """Track an error for reporting."""
        self.state.total_errors += 1
        self.state.consecutive_errors += 1
        entry = {
            "time": datetime.now().isoformat(),
            "message": str(error_msg)[:200],
            "recoverable": recoverable,
        }
        self.state.error_log.append(entry)
        if len(self.state.error_log) > 50:
            self.state.error_log = self.state.error_log[-50:]

    def _clear_error_streak(self):
        """Reset consecutive error counter after successful cycle."""
        self.state.consecutive_errors = 0

    # ── Report generation ─────────────────────────────────────────────────

    def _generate_daily_report(self):
        """Generate HTML daily report with charts."""
        try:
            from scripts.streaming_report import generate_daily_report
            report_path = generate_daily_report(
                day_number=self.state.current_day,
                snapshot=self.state.daily_snapshots[-1] if self.state.daily_snapshots else {},
                artifacts_dir=DAILY_REPORTS_DIR,
                streaming_state=asdict(self.state),
            )
            logger.info(f"  Daily report saved: {report_path}")
        except Exception as e:
            logger.warning(f"  Failed to generate daily report: {e}")

    def _generate_final_summary(self):
        """Generate the final publishing-ready summary."""
        try:
            from scripts.streaming_report import generate_final_summary
            summary_path = generate_final_summary(
                state=asdict(self.state),
                artifacts_dir=STREAMING_DIR,
            )
            logger.info(f"\n  Final summary saved: {summary_path}")
        except Exception as e:
            logger.warning(f"  Failed to generate final summary: {e}")

    # ── Star reminder ─────────────────────────────────────────────────────

    @staticmethod
    def _print_star_reminder():
        print("\n" + "=" * 60)
        print("  If astroSETI helped your research, please star the repo:")
        print("  https://github.com/samantaba/astroSETI")
        print("  It takes 2 seconds and helps others discover the tool.")
        print("=" * 60)

    # ── Main run loop ─────────────────────────────────────────────────────

    def run(self):
        """Run multi-day streaming observation."""
        logger.info("=" * 60)
        logger.info("  ASTROSETI STREAMING OBSERVATION ENGINE")
        logger.info("=" * 60)
        logger.info(f"  Target duration: {self.target_days} days")
        logger.info(f"  Mode: {self.state.current_mode.upper()}")
        logger.info(f"  Min SNR: {self.state.current_min_snr}")
        logger.info(f"  Max drift: {self.state.current_max_drift} Hz/s")
        logger.info(f"  Reports: {DAILY_REPORTS_DIR}")
        logger.info(f"  State: {STREAMING_STATE}")
        logger.info(f"  Filterbank dir: {FILTERBANK_DIR}")
        logger.info(f"  Press Ctrl+C to stop gracefully")
        logger.info("=" * 60)

        # Capture starting metrics
        self._day_start_metrics = self._capture_metrics()
        self._last_report_date = datetime.now().date()

        start_time = datetime.now()
        target_end = start_time + timedelta(days=self.target_days)

        logger.info(f"\n  Started: {start_time.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"  Target end: {target_end.strftime('%Y-%m-%d %H:%M')}")
        logger.info("")

        # Auto-download real Breakthrough Listen data if no filterbank files exist
        fil_count = len(list(FILTERBANK_DIR.glob("*.fil")))
        h5_count = len(list(FILTERBANK_DIR.glob("*.h5")))
        if fil_count == 0 and h5_count == 0:
            logger.info("\n  No filterbank files found. Downloading real Breakthrough Listen data...")
            try:
                from scripts.download_bl_data import BLDataDownloader
                downloader = BLDataDownloader(count=5)
                downloaded = downloader.run()
                logger.info(f"  Downloaded {len(downloaded)} BL files to {FILTERBANK_DIR}")
            except Exception as e:
                logger.warning(f"  BL download failed: {e}")
                logger.info("  Falling back to synthetic data generation...")
                try:
                    from scripts.generate_training_data import main as gen_main
                    gen_main()
                    logger.info("  Synthetic filterbank files generated")
                except Exception as e2:
                    logger.error(f"  Synthetic generation also failed: {e2}")
                    logger.error("  No data available. Add .fil or .h5 files to:")
                    logger.error(f"    {FILTERBANK_DIR}")
        else:
            logger.info(f"  Found {fil_count} .fil and {h5_count} .h5 files in {FILTERBANK_DIR}")

        # Initial health check
        logger.info("\n  Running initial health check...")
        initial_issues = self._health_check()
        if initial_issues:
            for issue in initial_issues:
                logger.warning(f"  [HEALTH] {issue}")
        else:
            logger.info("  All health checks passed")

        # Initialize pipeline
        logger.info("\n  Initializing pipeline...")
        try:
            _ = self.pipeline
            logger.info(
                f"  Pipeline initialized (Rust: {self.pipeline._rust_available}, "
                f"Model: {self.pipeline._model_loaded})"
            )
        except Exception as e:
            logger.warning(f"  Pipeline initialization warning: {e}")

        # Main streaming loop
        cycle_count = 0
        last_health_check = datetime.now()
        health_check_interval = 300  # Every 5 minutes
        cycle_interval = self.MODE_INTERVALS.get(self.state.current_mode, 30)

        while self.running and datetime.now() < target_end:
            try:
                # Run one observation cycle
                cycle_result = self._run_cycle()
                cycle_count += 1
                self._clear_error_streak()

                # Update cumulative stats
                self.state.total_files_processed += cycle_result["files_processed"]
                self.state.total_signals += cycle_result["signals_found"]
                self.state.total_candidates += cycle_result["candidates"]
                self.state.total_rfi_rejected += cycle_result["rfi_rejected"]
                self.state.total_ood_anomalies += cycle_result["ood_anomalies"]

                now = datetime.now()

                # Periodic health check
                if (now - last_health_check).total_seconds() > health_check_interval:
                    issues = self._health_check()
                    if issues:
                        for issue in issues:
                            logger.warning(f"  [HEALTH] {issue}")
                    last_health_check = now

                # Daily report
                if (
                    now.date() != self._last_report_date
                    and now.hour >= self.daily_report_hour
                ):
                    logger.info("\n" + "=" * 60)
                    logger.info(f"  DAILY ASSESSMENT - Day {self.state.current_day + 1}")
                    logger.info("=" * 60)

                    snapshot = self._take_daily_snapshot()
                    self._generate_daily_report()

                    self._day_start_metrics = self._capture_metrics()
                    self._last_report_date = now.date()

                    logger.info(
                        f"  Day {snapshot.day} complete: "
                        f"{snapshot.files_processed} files, "
                        f"{snapshot.signals_found} signals, "
                        f"{snapshot.candidates_found} candidates, "
                        f"{snapshot.rfi_rejected} RFI rejected"
                    )
                    if snapshot.corrections_applied:
                        for c in snapshot.corrections_applied:
                            logger.info(f"    - {c}")

                # Save state every cycle so UI stays responsive
                self.state.total_runtime_hours = (
                    (now - datetime.fromisoformat(self.state.started_at))
                    .total_seconds() / 3600
                )
                self._save_state()

                # Progress update every 10 cycles
                if cycle_count % 10 == 0:
                    elapsed = (now - start_time).total_seconds() / 3600
                    remaining = (target_end - now).total_seconds() / 3600
                    logger.info(
                        f"\n  [PROGRESS] {elapsed:.1f}h elapsed, "
                        f"{remaining:.1f}h remaining | "
                        f"{self.state.total_files_processed} files, "
                        f"{self.state.total_signals} signals, "
                        f"{self.state.total_candidates} candidates | "
                        f"mode={self.state.current_mode} | "
                        f"SNR>={self.state.current_min_snr:.1f} | "
                        f"errors={self.state.total_errors}"
                    )

                # Wait between cycles
                if self.running:
                    time.sleep(cycle_interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self._track_error(str(e))
                logger.error(f"Streaming cycle error: {e}")
                logger.debug(traceback.format_exc())

                if self.state.consecutive_errors >= 5:
                    logger.error(
                        f"  {self.state.consecutive_errors} consecutive errors! "
                        f"Consider restarting if this persists."
                    )

                time.sleep(60)

        # Final wrap-up
        self.state.completed = datetime.now() >= target_end
        self.state.completed_at = datetime.now().isoformat()
        self.state.total_runtime_hours = (
            (datetime.now() - datetime.fromisoformat(self.state.started_at))
            .total_seconds() / 3600
        )

        self._take_daily_snapshot()
        self._generate_daily_report()
        self._generate_final_summary()
        self._save_state()

        self._print_final_summary()
        self._print_star_reminder()

    def _print_final_summary(self):
        """Print final summary to console."""
        print("\n" + "=" * 60)
        print("  STREAMING OBSERVATION COMPLETE")
        print("=" * 60)
        print(f"  Duration: {self.state.total_runtime_hours:.1f} hours")
        print(f"  Days: {self.state.current_day}")
        print(f"  Total files processed: {self.state.total_files_processed}")
        print(f"  Total signals: {self.state.total_signals}")
        print(f"  Total candidates: {self.state.total_candidates}")
        print(f"  Total RFI rejected: {self.state.total_rfi_rejected}")
        print(f"  Total OOD anomalies: {self.state.total_ood_anomalies}")
        print(f"  Self-corrections: {self.state.total_corrections}")
        print(f"  Mode: {self.state.current_mode}")
        print(f"  Final SNR threshold: {self.state.current_min_snr:.1f}")

        if self.state.best_candidates:
            print(f"\n  Top candidates:")
            for i, c in enumerate(self.state.best_candidates[:5], 1):
                print(
                    f"    {i}. OOD={c.get('ood_score', 0):.4f} | "
                    f"SNR={c.get('snr', 0):.1f} | "
                    f"{c.get('signal_type', '?')} | "
                    f"drift={c.get('drift_rate', 0):.4f} Hz/s"
                )

        print(f"\n  Reports: {STREAMING_DIR}")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="astroSETI Streaming Observation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/streaming_observation.py --days 7
    python scripts/streaming_observation.py --days 3 --mode aggressive
    python scripts/streaming_observation.py --days 1 --mode turbo
    python scripts/streaming_observation.py --report-only
    python scripts/streaming_observation.py --reset
        """,
    )

    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to run (default: 7)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Number of hours to run (alternative to --days)",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "aggressive", "turbo"],
        default="normal",
        help="Observation mode: normal (default), aggressive, or turbo",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing data without running observation",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset streaming state and start fresh",
    )

    args = parser.parse_args()

    if args.reset:
        if STREAMING_STATE.exists():
            STREAMING_STATE.unlink()
            logger.info("Streaming state reset")

        if DISCOVERY_STATE.exists():
            try:
                with open(DISCOVERY_STATE) as f:
                    ds = json.load(f)
                ds["total_signals"] = 0
                ds["total_candidates"] = 0
                ds["total_rfi"] = 0
                with open(DISCOVERY_STATE, "w") as f:
                    json.dump(ds, f, indent=2)
                logger.info("Discovery state reset")
            except Exception as e:
                logger.warning(f"Failed to reset discovery state: {e}")

        if CANDIDATES_FILE.exists():
            archive = CANDIDATES_DIR / (
                f"candidates_archive_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            )
            shutil.copy2(CANDIDATES_FILE, archive)
            with open(CANDIDATES_FILE, "w") as f:
                json.dump([], f)
            logger.info(f"Candidates archived to {archive.name}")

    # Resolve duration: --hours takes precedence, then --days, then default 7 days
    if args.hours is not None:
        target_days = args.hours / 24.0
    elif args.days is not None:
        target_days = args.days
    else:
        target_days = 7

    if args.report_only:
        engine = StreamingObserver(target_days=0)
        engine._take_daily_snapshot()
        engine._generate_daily_report()
        engine._generate_final_summary()
        engine._save_state()
        logger.info("Reports generated from existing data.")
        return

    engine = StreamingObserver(
        target_days=target_days,
        mode=args.mode,
    )
    engine.run()


if __name__ == "__main__":
    main()
