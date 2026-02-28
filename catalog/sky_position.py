"""
Sky Position Utilities and AstroLens Cross-Reference.

Provides coordinate conversions (equatorial ↔ galactic), angular-separation
calculations (Vincenty formula), and a function that checks whether a radio
signal position has a matching optical anomaly in the AstroLens
``anomaly_candidates.json`` file.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# AstroLens artefact path (sibling project)
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_ASTROLENS_ARTIFACTS = Path(
    _PROJECT_ROOT.parent / "astrolens_artifacts" / "data" / "anomaly_candidates.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SkyPosition:
    """
    A sky position with both equatorial and galactic coordinates.

    Provide *ra*/*dec* (J2000 decimal degrees) and galactic coordinates
    are computed automatically, or supply them explicitly.
    """

    ra: float  # Right ascension  (degrees, J2000)
    dec: float  # Declination      (degrees, J2000)
    gal_l: float = 0.0  # Galactic longitude (degrees)
    gal_b: float = 0.0  # Galactic latitude  (degrees)

    def __post_init__(self) -> None:
        """Auto-convert equatorial → galactic if not supplied."""
        if self.gal_l == 0.0 and self.gal_b == 0.0:
            self.gal_l, self.gal_b = self._equatorial_to_galactic(self.ra, self.dec)

    # ── Coordinate conversion ────────────────────────────────────────────

    @staticmethod
    def _equatorial_to_galactic(ra_deg: float, dec_deg: float) -> tuple[float, float]:
        """
        Convert J2000 equatorial (ra, dec) to galactic (l, b).

        Uses the IAU 1958 galactic pole and origin:
            αGP = 192.85948°  δGP = 27.12825°  lΩ = 32.93192°
        """
        ra = math.radians(ra_deg)
        dec = math.radians(dec_deg)

        # North galactic pole (J2000)
        ra_gp = math.radians(192.85948)
        dec_gp = math.radians(27.12825)
        l_omega = math.radians(32.93192)

        sin_b = math.sin(dec) * math.sin(dec_gp) + math.cos(dec) * math.cos(dec_gp) * math.cos(
            ra - ra_gp
        )
        b = math.asin(max(-1.0, min(1.0, sin_b)))

        num = math.sin(ra - ra_gp) * math.cos(dec)
        den = math.cos(dec) * math.sin(dec_gp) * math.cos(ra - ra_gp) - math.sin(dec) * math.cos(
            dec_gp
        )
        lon = l_omega - math.atan2(num, den)
        lon_deg = math.degrees(lon) % 360.0

        return round(lon_deg, 6), round(math.degrees(b), 6)

    @staticmethod
    def _galactic_to_equatorial(l_deg: float, b_deg: float) -> tuple[float, float]:
        """Convert galactic (l, b) to J2000 equatorial (ra, dec)."""
        l_rad = math.radians(l_deg)
        b_rad = math.radians(b_deg)

        ra_gp = math.radians(192.85948)
        dec_gp = math.radians(27.12825)
        l_omega = math.radians(32.93192)

        sin_dec = math.sin(b_rad) * math.sin(dec_gp) + math.cos(b_rad) * math.cos(
            dec_gp
        ) * math.sin(l_rad - l_omega)
        dec = math.asin(max(-1.0, min(1.0, sin_dec)))

        num = math.cos(b_rad) * math.cos(l_rad - l_omega)
        den = math.sin(b_rad) * math.cos(dec_gp) - math.cos(b_rad) * math.sin(dec_gp) * math.sin(
            l_rad - l_omega
        )
        ra = ra_gp + math.atan2(num, den)
        ra_deg = math.degrees(ra) % 360.0

        return round(ra_deg, 6), round(math.degrees(dec), 6)

    @classmethod
    def from_galactic(cls, l_deg: float, b_deg: float) -> SkyPosition:
        """Create a SkyPosition from galactic coordinates."""
        ra, dec = cls._galactic_to_equatorial(l_deg, b_deg)
        return cls(ra=ra, dec=dec, gal_l=l_deg, gal_b=b_deg)


@dataclass
class CrossMatchResult:
    """Result of cross-matching a radio position with AstroLens optical anomalies."""

    astrolens_candidate: Dict[str, Any]
    angular_sep: float  # degrees
    has_optical_anomaly: bool


# ─────────────────────────────────────────────────────────────────────────────
# Angular separation (Vincenty formula)
# ─────────────────────────────────────────────────────────────────────────────


def angular_separation(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
) -> float:
    """
    Vincenty angular separation between two sky positions.

    Parameters are in *decimal degrees (J2000)*.
    Returns separation in **degrees**.
    """
    ra1r, dec1r = math.radians(ra1), math.radians(dec1)
    ra2r, dec2r = math.radians(ra2), math.radians(dec2)
    dra = ra2r - ra1r

    numerator = math.sqrt(
        (math.cos(dec2r) * math.sin(dra)) ** 2
        + (math.cos(dec1r) * math.sin(dec2r) - math.sin(dec1r) * math.cos(dec2r) * math.cos(dra))
        ** 2
    )
    denominator = math.sin(dec1r) * math.sin(dec2r) + math.cos(dec1r) * math.cos(dec2r) * math.cos(
        dra
    )
    return math.degrees(math.atan2(numerator, denominator))


# ─────────────────────────────────────────────────────────────────────────────
# AstroLens cross-reference
# ─────────────────────────────────────────────────────────────────────────────


def _load_astrolens_candidates(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the AstroLens anomaly_candidates.json file."""
    fpath = path or _ASTROLENS_ARTIFACTS
    if not fpath.exists():
        logger.info(
            "AstroLens candidates file not found at %s – cross-ref will return empty.",
            fpath,
        )
        return []
    try:
        with open(fpath) as f:
            data = json.load(f)
        # Support both a bare list and a {"candidates": [...]} wrapper
        if isinstance(data, list):
            return data
        return data.get("candidates", data.get("anomalies", []))
    except Exception as exc:
        logger.warning("Failed to load AstroLens candidates: %s", exc)
        return []


def astrolens_crossref(
    ra: float,
    dec: float,
    radius_arcmin: float = 2.0,
    candidates_path: Optional[Path] = None,
) -> List[CrossMatchResult]:
    """
    Check AstroLens anomaly_candidates.json for optical anomalies near a
    radio-signal sky position.

    Parameters:
        ra:  Right ascension in degrees (J2000).
        dec: Declination in degrees (J2000).
        radius_arcmin: Search radius in arc-minutes.
        candidates_path: Override path to the candidates JSON file.

    Returns:
        A list of :class:`CrossMatchResult`, sorted by angular separation.
    """
    candidates = _load_astrolens_candidates(candidates_path)
    if not candidates:
        return []

    radius_deg = radius_arcmin / 60.0
    matches: List[CrossMatchResult] = []

    for candidate in candidates:
        # Try multiple key conventions used by AstroLens exports
        c_ra = candidate.get("ra") or candidate.get("RA")
        c_dec = candidate.get("dec") or candidate.get("DEC") or candidate.get("Dec")
        if c_ra is None or c_dec is None:
            continue

        try:
            c_ra = float(c_ra)
            c_dec = float(c_dec)
        except (ValueError, TypeError):
            continue

        sep = angular_separation(ra, dec, c_ra, c_dec)
        if sep <= radius_deg:
            matches.append(
                CrossMatchResult(
                    astrolens_candidate=candidate,
                    angular_sep=round(sep, 6),
                    has_optical_anomaly=candidate.get("is_anomaly", True),
                )
            )

    matches.sort(key=lambda m: m.angular_sep)
    return matches
