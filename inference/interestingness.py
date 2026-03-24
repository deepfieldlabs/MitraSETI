"""
Candidate Interestingness Score — MitraSETI

Composite scoring function that combines multiple signal properties
into a single "interestingness" metric for ranking candidates.
Higher scores indicate signals more worthy of follow-up observation.

Components:
  - SNR significance (log-scaled)
  - Drift rate meaningfulness (non-zero, non-boundary)
  - RFI rejection confidence (1 - rfi_probability)
  - OOD anomaly score (novel signals score higher)
  - Classification confidence
  - Cadence survival bonus (if available)

Usage:
    from inference.interestingness import compute_interestingness
    score = compute_interestingness(candidate_dict)
"""

from __future__ import annotations

import math
from typing import Any, Dict


def compute_interestingness(
    candidate: Dict[str, Any],
    max_drift_rate: float = 4.0,
    weights: Dict[str, float] | None = None,
) -> float:
    """Compute the interestingness score for a signal candidate.

    Args:
        candidate: Dict with keys from pipeline output (snr, drift_rate,
                   rfi_probability, ood_score, confidence, is_candidate, etc.)
        max_drift_rate: Pipeline's max drift rate setting.
        weights: Optional custom weights for each component.

    Returns:
        Score in [0, 100].  Higher = more interesting.
    """
    w = weights or {
        "snr": 0.25,
        "drift": 0.20,
        "rfi_clean": 0.20,
        "ood": 0.15,
        "confidence": 0.10,
        "cadence": 0.10,
    }

    scores = {}

    # 1. SNR significance (log-scaled, saturates around 100)
    snr = candidate.get("snr", 0)
    scores["snr"] = min(1.0, math.log1p(max(snr, 0)) / math.log1p(100))

    # 2. Drift rate meaningfulness
    drift = abs(candidate.get("drift_rate", 0))
    at_boundary = drift >= max_drift_rate * 0.98
    if at_boundary or drift < 0.001:
        scores["drift"] = 0.0
    else:
        # Sweet spot: 0.05 - 2.0 Hz/s (typical ET drift range)
        if 0.05 <= drift <= 2.0:
            scores["drift"] = 1.0
        elif drift < 0.05:
            scores["drift"] = drift / 0.05
        else:
            scores["drift"] = max(0, 1.0 - (drift - 2.0) / (max_drift_rate - 2.0))

    # 3. RFI cleanliness (1 - rfi_probability)
    rfi_prob = candidate.get("rfi_probability", 0.5)
    scores["rfi_clean"] = max(0, 1.0 - rfi_prob)

    # 4. OOD anomaly (novel signals are more interesting)
    ood = candidate.get("ood_score", 0)
    scores["ood"] = min(1.0, ood * 2)

    # 5. Classification confidence
    conf = candidate.get("confidence", 0.5)
    scores["confidence"] = conf

    # 6. Cadence survival bonus
    on_detections = candidate.get("on_detections", 0)
    on_total = candidate.get("on_scans_total", 0)
    if on_total > 0:
        scores["cadence"] = on_detections / on_total
    elif candidate.get("is_candidate", False):
        scores["cadence"] = 0.5
    else:
        scores["cadence"] = 0.0

    # Weighted combination
    total = sum(w.get(k, 0) * v for k, v in scores.items())
    total_weight = sum(w.get(k, 0) for k in scores)
    normalized = (total / total_weight) * 100 if total_weight > 0 else 0

    return round(min(100, max(0, normalized)), 2)


def rank_candidates(
    candidates: list[Dict[str, Any]],
    max_drift_rate: float = 4.0,
) -> list[Dict[str, Any]]:
    """Score and rank a list of candidates by interestingness.

    Adds 'interestingness_score' to each candidate and returns
    the list sorted by score descending.
    """
    for c in candidates:
        c["interestingness_score"] = compute_interestingness(c, max_drift_rate)

    return sorted(candidates, key=lambda c: c["interestingness_score"], reverse=True)
