"""
Proper Astropy SkyCoord Cross-Matching — MitraSETI ↔ AstroLens

Replaces filename-regex coordinate matching with proper
astropy.coordinates.match_coordinates_sky() KD-tree matching.
This makes the optical-radio cross-match scientifically rigorous.

Usage:
    from catalog.astropy_crossmatch import crossmatch_radio_optical
    matches = crossmatch_radio_optical(
        radio_candidates, optical_candidates, max_sep_arcsec=120
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def crossmatch_radio_optical(
    radio_candidates: List[Dict[str, Any]],
    optical_candidates: List[Dict[str, Any]],
    max_sep_arcsec: float = 120.0,
    radio_ra_key: str = "ra",
    radio_dec_key: str = "dec",
    optical_ra_key: str = "ra_deg",
    optical_dec_key: str = "dec_deg",
) -> Dict[str, Any]:
    """Cross-match radio and optical catalogs using astropy SkyCoord.

    Args:
        radio_candidates: MitraSETI results with RA/Dec.
        optical_candidates: AstroLens results with RA/Dec.
        max_sep_arcsec: Maximum angular separation for a match.

    Returns:
        Dict with matched pairs, unmatched radio, unmatched optical,
        and statistics.
    """
    from astropy.coordinates import SkyCoord, match_coordinates_sky
    import astropy.units as u

    # Filter to candidates with valid coordinates
    radio_with_coords = []
    for i, c in enumerate(radio_candidates):
        ra = _safe_float(c.get(radio_ra_key))
        dec = _safe_float(c.get(radio_dec_key))
        if ra != 0 or dec != 0:
            radio_with_coords.append((i, ra, dec))

    optical_with_coords = []
    for i, c in enumerate(optical_candidates):
        ra = _safe_float(c.get(optical_ra_key))
        dec = _safe_float(c.get(optical_dec_key))
        if ra != 0 or dec != 0:
            optical_with_coords.append((i, ra, dec))

    if not radio_with_coords or not optical_with_coords:
        return {
            "matches": [],
            "n_matches": 0,
            "unmatched_radio": len(radio_candidates),
            "unmatched_optical": len(optical_candidates),
            "n_radio_with_coords": len(radio_with_coords),
            "n_optical_with_coords": len(optical_with_coords),
            "max_sep_arcsec": max_sep_arcsec,
            "mean_separation_arcsec": 0,
        }

    # Build SkyCoord arrays
    radio_coords = SkyCoord(
        ra=[r[1] for r in radio_with_coords] * u.deg,
        dec=[r[2] for r in radio_with_coords] * u.deg,
        frame="icrs",
    )

    optical_coords = SkyCoord(
        ra=[o[1] for o in optical_with_coords] * u.deg,
        dec=[o[2] for o in optical_with_coords] * u.deg,
        frame="icrs",
    )

    # KD-tree cross-match
    idx, sep2d, _ = match_coordinates_sky(radio_coords, optical_coords)

    matches = []
    matched_radio_indices = set()
    matched_optical_indices = set()

    for ri in range(len(radio_with_coords)):
        sep_arcsec = sep2d[ri].arcsec
        if sep_arcsec <= max_sep_arcsec:
            oi = int(idx[ri])
            r_orig_idx = radio_with_coords[ri][0]
            o_orig_idx = optical_with_coords[oi][0]

            matches.append({
                "radio_idx": r_orig_idx,
                "optical_idx": o_orig_idx,
                "separation_arcsec": round(float(sep_arcsec), 2),
                "radio_ra": radio_with_coords[ri][1],
                "radio_dec": radio_with_coords[ri][2],
                "optical_ra": optical_with_coords[oi][1],
                "optical_dec": optical_with_coords[oi][2],
                "radio_snr": radio_candidates[r_orig_idx].get("snr", 0),
                "optical_ood": optical_candidates[o_orig_idx].get("ood_score", 0),
                "radio_source": radio_candidates[r_orig_idx].get("source_name", ""),
                "optical_class": optical_candidates[o_orig_idx].get("classification", ""),
            })
            matched_radio_indices.add(r_orig_idx)
            matched_optical_indices.add(o_orig_idx)

    return {
        "matches": matches,
        "n_matches": len(matches),
        "unmatched_radio": len(radio_candidates) - len(matched_radio_indices),
        "unmatched_optical": len(optical_candidates) - len(matched_optical_indices),
        "n_radio_with_coords": len(radio_with_coords),
        "n_optical_with_coords": len(optical_with_coords),
        "max_sep_arcsec": max_sep_arcsec,
        "mean_separation_arcsec": (
            round(float(np.mean([m["separation_arcsec"] for m in matches])), 2)
            if matches else 0
        ),
    }


def load_astrolens_candidates(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load AstroLens skymap export JSON."""
    if path is None:
        from paths import ASTROLENS_SKYMAP_FILE
        path = ASTROLENS_SKYMAP_FILE

    if not Path(path).exists():
        logger.warning(f"AstroLens skymap not found: {path}")
        return []

    with open(path) as f:
        data = json.load(f)
        return data if isinstance(data, list) else data.get("candidates", [])


def load_radio_candidates(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load MitraSETI streaming state candidates."""
    if path is None:
        from paths import STREAMING_STATE
        path = STREAMING_STATE

    if not Path(path).exists():
        return []

    with open(path) as f:
        state = json.load(f)
        return state.get("candidates", [])
