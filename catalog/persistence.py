"""
Cross-Epoch Signal Persistence Tracking — MitraSETI

Tracks whether the same signal reappears across different observation
epochs of the same target.  Persistent signals are orders of magnitude
more interesting than one-off detections.

Stores history in a lightweight JSON file alongside streaming state.
No other open-source SETI tool implements this.

Usage:
    from catalog.persistence import PersistenceTracker
    tracker = PersistenceTracker()
    tracker.record(source_name, candidates, epoch_id)
    persistent = tracker.get_persistent(source_name, min_epochs=2)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PersistenceTracker:
    """Track signal persistence across observation epochs."""

    def __init__(self, state_path: Optional[Path] = None):
        if state_path is None:
            from paths import DATA_DIR
            state_path = DATA_DIR / "persistence_state.json"

        self._path = Path(state_path)
        self._state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return {"sources": {}, "version": 1}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

    def record(
        self,
        source_name: str,
        candidates: List[Dict[str, Any]],
        epoch_id: Optional[str] = None,
        freq_tolerance_mhz: float = 0.001,
        drift_tolerance: float = 0.5,
    ) -> Dict[str, Any]:
        """Record candidates from an observation epoch.

        Matches new candidates against previously seen signals for this
        source.  Returns summary of new vs persistent signals.
        """
        if epoch_id is None:
            epoch_id = datetime.now().isoformat()

        source_key = source_name.strip().upper()

        if source_key not in self._state["sources"]:
            self._state["sources"][source_key] = {
                "signal_history": [],
                "epochs": [],
            }

        src = self._state["sources"][source_key]
        src["epochs"].append({"epoch_id": epoch_id, "n_candidates": len(candidates)})

        n_new = 0
        n_matched = 0

        for cand in candidates:
            freq_mhz = cand.get("frequency_hz", 0) / 1e6
            drift = cand.get("drift_rate", 0)
            snr = cand.get("snr", 0)

            matched = False
            for signal in src["signal_history"]:
                freq_close = abs(signal["frequency_mhz"] - freq_mhz) < freq_tolerance_mhz
                drift_close = abs(signal["drift_rate"] - drift) < drift_tolerance
                if freq_close and drift_close:
                    signal["epoch_count"] += 1
                    signal["last_seen"] = epoch_id
                    signal["max_snr"] = max(signal["max_snr"], snr)
                    signal["snr_history"].append(round(snr, 2))
                    if len(signal["snr_history"]) > 50:
                        signal["snr_history"] = signal["snr_history"][-50:]
                    matched = True
                    n_matched += 1
                    break

            if not matched:
                src["signal_history"].append({
                    "frequency_mhz": round(freq_mhz, 6),
                    "drift_rate": round(drift, 4),
                    "first_seen": epoch_id,
                    "last_seen": epoch_id,
                    "epoch_count": 1,
                    "max_snr": round(snr, 2),
                    "snr_history": [round(snr, 2)],
                    "classification": cand.get("classification", "unknown"),
                })
                n_new += 1

        self._save()

        return {
            "source": source_name,
            "epoch_id": epoch_id,
            "total_candidates": len(candidates),
            "new_signals": n_new,
            "matched_existing": n_matched,
            "total_tracked": len(src["signal_history"]),
            "total_epochs": len(src["epochs"]),
        }

    def get_persistent(
        self, source_name: str, min_epochs: int = 2
    ) -> List[Dict[str, Any]]:
        """Get signals that persist across multiple epochs."""
        source_key = source_name.strip().upper()
        src = self._state.get("sources", {}).get(source_key, {})
        history = src.get("signal_history", [])

        return [
            s for s in history
            if s.get("epoch_count", 0) >= min_epochs
        ]

    def get_all_sources(self) -> Dict[str, Dict[str, Any]]:
        """Summary of all tracked sources."""
        result = {}
        for source_key, src in self._state.get("sources", {}).items():
            n_persistent = sum(
                1 for s in src.get("signal_history", [])
                if s.get("epoch_count", 0) >= 2
            )
            result[source_key] = {
                "total_epochs": len(src.get("epochs", [])),
                "total_signals": len(src.get("signal_history", [])),
                "persistent_signals": n_persistent,
            }
        return result

    def clear(self, source_name: Optional[str] = None) -> None:
        """Clear tracking history."""
        if source_name:
            self._state["sources"].pop(source_name.strip().upper(), None)
        else:
            self._state["sources"] = {}
        self._save()
