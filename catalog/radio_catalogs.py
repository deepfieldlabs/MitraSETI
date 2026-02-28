"""
Radio Astronomy Catalog Queries for MitraSETI.

Query SIMBAD, NVSS, FIRST, and pulsar catalogs to determine whether a
detected radio signal matches a known astronomical source.  All remote
queries go through HTTP (with local caching) so that astroquery is
optional.

Catalog coverage:
    - SIMBAD   – general astronomical objects near a sky position
    - NVSS     – NRAO VLA Sky Survey (1.4 GHz continuum)
    - FIRST    – Faint Images of the Radio Sky at Twenty-cm
    - ATNF     – Australia Telescope National Facility Pulsar Catalogue
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Local cache directory (inside artifacts)
# ─────────────────────────────────────────────────────────────────────────────

_ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "mitraseti_artifacts"
_CACHE_DIR = _ARTIFACTS_DIR / "data" / "catalog_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CatalogResult:
    """A single match from a radio-astronomy catalog."""

    source_name: str
    catalog: str  # SIMBAD, NVSS, FIRST, ATNF
    ra: float  # degrees (J2000)
    dec: float  # degrees (J2000)
    distance_arcmin: float  # angular separation from query position
    flux_density: Optional[float] = None  # mJy (when available)
    spectral_type: Optional[str] = None
    notes: str = ""
    raw_data: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _angular_separation_deg(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
) -> float:
    """Vincenty angular separation in *degrees*."""
    ra1r, dec1r = math.radians(ra1), math.radians(dec1)
    ra2r, dec2r = math.radians(ra2), math.radians(dec2)
    dra = ra2r - ra1r

    num = math.sqrt(
        (math.cos(dec2r) * math.sin(dra)) ** 2
        + (math.cos(dec1r) * math.sin(dec2r) - math.sin(dec1r) * math.cos(dec2r) * math.cos(dra))
        ** 2
    )
    den = math.sin(dec1r) * math.sin(dec2r) + math.cos(dec1r) * math.cos(dec2r) * math.cos(dra)
    return math.degrees(math.atan2(num, den))


def _cache_key(catalog: str, ra: float, dec: float, radius: float) -> str:
    """Deterministic cache filename."""
    return f"{catalog}_{ra:.4f}_{dec:.4f}_{radius:.2f}.json"


def _read_cache(key: str, max_age_hours: float = 24.0) -> Optional[List[Dict]]:
    """Return cached JSON list or *None* if stale / missing."""
    path = _CACHE_DIR / key
    if not path.exists():
        return None
    try:
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        if age_hours > max_age_hours:
            return None
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(key: str, data: List[Dict]) -> None:
    """Persist a JSON-serialisable list to disk."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_DIR / key, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.debug("Cache write failed for %s: %s", key, exc)


# ─────────────────────────────────────────────────────────────────────────────
# RadioCatalogQuery
# ─────────────────────────────────────────────────────────────────────────────


class RadioCatalogQuery:
    """
    Query radio-astronomy catalogs for known sources near a sky position.

    All methods accept *ra* / *dec* in **decimal degrees (J2000)** and
    return lists of :class:`CatalogResult`.
    """

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        cache_hours: float = 24.0,
    ) -> None:
        self.timeout = timeout_seconds
        self.cache_hours = cache_hours

    # ── SIMBAD ───────────────────────────────────────────────────────────

    def query_simbad(
        self,
        ra: float,
        dec: float,
        radius_arcmin: float = 5.0,
    ) -> List[CatalogResult]:
        """Return known radio sources near *ra*/*dec* from SIMBAD (TAP/ADQL)."""
        cache_key = _cache_key("simbad", ra, dec, radius_arcmin)
        cached = _read_cache(cache_key, self.cache_hours)
        if cached is not None:
            return [CatalogResult(**r) for r in cached]

        radius_deg = radius_arcmin / 60.0
        adql = (
            f"SELECT TOP 30 main_id, otype_txt, ra, dec, nbref, flux "
            f"FROM basic "
            f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1 "
            f"ORDER BY nbref DESC"
        )

        tap_url = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync"
        try:
            resp = httpx.post(
                tap_url,
                data={
                    "request": "doQuery",
                    "lang": "adql",
                    "format": "json",
                    "query": adql,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("data", [])
        except Exception as exc:
            logger.warning("SIMBAD query failed: %s", exc)
            return []

        results: List[CatalogResult] = []
        for row in rows:
            try:
                if isinstance(row, dict):
                    name = row.get("main_id", "Unknown")
                    otype = row.get("otype_txt", "Unknown")
                    obj_ra = float(row.get("ra", ra))
                    obj_dec = float(row.get("dec", dec))
                    flux = row.get("flux")
                else:
                    name = str(row[0]) if len(row) > 0 else "Unknown"
                    otype = str(row[1]) if len(row) > 1 else "Unknown"
                    obj_ra = float(row[2]) if len(row) > 2 else ra
                    obj_dec = float(row[3]) if len(row) > 3 else dec
                    flux = row[5] if len(row) > 5 else None

                sep = _angular_separation_deg(ra, dec, obj_ra, obj_dec) * 60.0
                results.append(
                    CatalogResult(
                        source_name=name,
                        catalog="SIMBAD",
                        ra=obj_ra,
                        dec=obj_dec,
                        distance_arcmin=round(sep, 4),
                        flux_density=float(flux) if flux is not None else None,
                        spectral_type=otype,
                        notes=f"https://simbad.u-strasbg.fr/simbad/sim-id?Ident={quote(str(name))}",
                    )
                )
            except (ValueError, IndexError, TypeError) as exc:
                logger.debug("Skipping SIMBAD row: %s", exc)

        _write_cache(cache_key, [asdict(r) for r in results])
        return results

    # ── NVSS (NRAO VLA Sky Survey, 1.4 GHz) ─────────────────────────────

    def query_nvss(
        self,
        ra: float,
        dec: float,
        radius_arcmin: float = 5.0,
    ) -> List[CatalogResult]:
        """Query NVSS via VizieR TAP (catalog VIII/65/nvss)."""
        cache_key = _cache_key("nvss", ra, dec, radius_arcmin)
        cached = _read_cache(cache_key, self.cache_hours)
        if cached is not None:
            return [CatalogResult(**r) for r in cached]

        radius_deg = radius_arcmin / 60.0
        adql = (
            f"SELECT TOP 30 NVSS, RAJ2000, DEJ2000, S1_4, e_S1_4 "
            f'FROM "VIII/65/nvss" '
            f"WHERE 1=CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))"
        )

        tap_url = "https://tapvizier.u-strasbg.fr/TAPVizieR/tap/sync"
        try:
            resp = httpx.post(
                tap_url,
                data={
                    "request": "doQuery",
                    "lang": "adql",
                    "format": "json",
                    "query": adql,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("data", [])
        except Exception as exc:
            logger.warning("NVSS/VizieR query failed: %s", exc)
            return []

        results: List[CatalogResult] = []
        for row in rows:
            try:
                if isinstance(row, dict):
                    name = row.get("NVSS", "NVSS source")
                    obj_ra = float(row.get("RAJ2000", ra))
                    obj_dec = float(row.get("DEJ2000", dec))
                    flux = row.get("S1_4")
                else:
                    name = str(row[0]) if len(row) > 0 else "NVSS source"
                    obj_ra = float(row[1]) if len(row) > 1 else ra
                    obj_dec = float(row[2]) if len(row) > 2 else dec
                    flux = row[3] if len(row) > 3 else None

                sep = _angular_separation_deg(ra, dec, obj_ra, obj_dec) * 60.0
                results.append(
                    CatalogResult(
                        source_name=name,
                        catalog="NVSS",
                        ra=obj_ra,
                        dec=obj_dec,
                        distance_arcmin=round(sep, 4),
                        flux_density=float(flux) if flux is not None else None,
                        spectral_type="radio_continuum",
                        notes="1.4 GHz NRAO VLA Sky Survey",
                    )
                )
            except (ValueError, IndexError, TypeError) as exc:
                logger.debug("Skipping NVSS row: %s", exc)

        _write_cache(cache_key, [asdict(r) for r in results])
        return results

    # ── FIRST (Faint Images of the Radio Sky at Twenty-cm) ───────────────

    def query_first(
        self,
        ra: float,
        dec: float,
        radius_arcmin: float = 5.0,
    ) -> List[CatalogResult]:
        """Query FIRST via VizieR TAP (catalog VIII/92/first14)."""
        cache_key = _cache_key("first", ra, dec, radius_arcmin)
        cached = _read_cache(cache_key, self.cache_hours)
        if cached is not None:
            return [CatalogResult(**r) for r in cached]

        radius_deg = radius_arcmin / 60.0
        adql = (
            f"SELECT TOP 30 FIRST, RAJ2000, DEJ2000, Fint, Fpeak "
            f'FROM "VIII/92/first14" '
            f"WHERE 1=CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))"
        )

        tap_url = "https://tapvizier.u-strasbg.fr/TAPVizieR/tap/sync"
        try:
            resp = httpx.post(
                tap_url,
                data={
                    "request": "doQuery",
                    "lang": "adql",
                    "format": "json",
                    "query": adql,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("data", [])
        except Exception as exc:
            logger.warning("FIRST/VizieR query failed: %s", exc)
            return []

        results: List[CatalogResult] = []
        for row in rows:
            try:
                if isinstance(row, dict):
                    name = row.get("FIRST", "FIRST source")
                    obj_ra = float(row.get("RAJ2000", ra))
                    obj_dec = float(row.get("DEJ2000", dec))
                    flux = row.get("Fint") or row.get("Fpeak")
                else:
                    name = str(row[0]) if len(row) > 0 else "FIRST source"
                    obj_ra = float(row[1]) if len(row) > 1 else ra
                    obj_dec = float(row[2]) if len(row) > 2 else dec
                    flux = row[3] if len(row) > 3 else None

                sep = _angular_separation_deg(ra, dec, obj_ra, obj_dec) * 60.0
                results.append(
                    CatalogResult(
                        source_name=name,
                        catalog="FIRST",
                        ra=obj_ra,
                        dec=obj_dec,
                        distance_arcmin=round(sep, 4),
                        flux_density=float(flux) if flux is not None else None,
                        spectral_type="radio_continuum",
                        notes="1.4 GHz Faint Images of the Radio Sky at Twenty-cm",
                    )
                )
            except (ValueError, IndexError, TypeError) as exc:
                logger.debug("Skipping FIRST row: %s", exc)

        _write_cache(cache_key, [asdict(r) for r in results])
        return results

    # ── ATNF Pulsar Catalogue ────────────────────────────────────────────

    def query_pulsar_catalog(
        self,
        ra: float,
        dec: float,
        radius_arcmin: float = 5.0,
    ) -> List[CatalogResult]:
        """
        Query ATNF Pulsar Catalogue via its web API.

        The ATNF catalogue is the definitive list of published pulsars.
        """
        cache_key = _cache_key("atnf", ra, dec, radius_arcmin)
        cached = _read_cache(cache_key, self.cache_hours)
        if cached is not None:
            return [CatalogResult(**r) for r in cached]

        # ATNF web interface – request pulsars within a circle
        params = {
            "type": "expert",
            "startUserDefined": "true",
            "c1_val": f"{ra:.6f}",
            "c2_val": f"{dec:.6f}",
            "radius": f"{radius_arcmin:.2f}",
            "rajCol": "on",
            "decjCol": "on",
            "p0Col": "on",
            "dmCol": "on",
            "s1400Col": "on",
            "nameCol": "on",
            "submit_format": "json",
        }
        url = "https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php"

        try:
            resp = httpx.get(url, params=params, timeout=self.timeout, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("ATNF Pulsar Catalogue query failed: %s", exc)
            return []

        results: List[CatalogResult] = []
        pulsars = data if isinstance(data, list) else data.get("pulsars", [])
        for psr in pulsars:
            try:
                name = psr.get("PSRJ") or psr.get("NAME") or "Unknown"
                psr_ra = self._hms_to_deg(psr.get("RAJ", ""))
                psr_dec = self._dms_to_deg(psr.get("DECJ", ""))
                if psr_ra is None or psr_dec is None:
                    continue

                sep = _angular_separation_deg(ra, dec, psr_ra, psr_dec) * 60.0
                flux = psr.get("S1400")
                period = psr.get("P0", "")

                results.append(
                    CatalogResult(
                        source_name=f"PSR {name}",
                        catalog="ATNF",
                        ra=psr_ra,
                        dec=psr_dec,
                        distance_arcmin=round(sep, 4),
                        flux_density=float(flux) if flux else None,
                        spectral_type="pulsar",
                        notes=f"P0={period}s" if period else "",
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping ATNF row: %s", exc)

        _write_cache(cache_key, [asdict(r) for r in results])
        return results

    # ── Composite check ──────────────────────────────────────────────────

    def is_known_source(
        self,
        ra: float,
        dec: float,
        freq_mhz: float,
        radius_arcmin: float = 5.0,
    ) -> Tuple[bool, str]:
        """
        Check all catalogs to decide whether a signal is from a known source.

        Returns:
            (is_known, source_description)
            e.g.  (True, "PSR J1921+2153 (pulsar, 0.32 arcmin)")
                  (False, "No known radio source within 5.0 arcmin")
        """
        all_results: List[CatalogResult] = []

        # Query all catalogs in order of specificity
        all_results.extend(self.query_pulsar_catalog(ra, dec, radius_arcmin))
        all_results.extend(self.query_simbad(ra, dec, radius_arcmin))

        # NVSS and FIRST are 1.4 GHz surveys – always useful context
        all_results.extend(self.query_nvss(ra, dec, radius_arcmin))
        all_results.extend(self.query_first(ra, dec, radius_arcmin))

        if not all_results:
            return False, f"No known radio source within {radius_arcmin} arcmin"

        # Sort by distance; the closest match wins
        all_results.sort(key=lambda r: r.distance_arcmin)
        best = all_results[0]

        description = (
            f"{best.source_name} ({best.spectral_type or best.catalog}, "
            f"{best.distance_arcmin:.2f} arcmin)"
        )
        return True, description

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _hms_to_deg(hms: str) -> Optional[float]:
        """Convert 'HH:MM:SS.ss' to decimal degrees."""
        if not hms:
            return None
        try:
            parts = hms.strip().split(":")
            h = float(parts[0])
            m = float(parts[1]) if len(parts) > 1 else 0.0
            s = float(parts[2]) if len(parts) > 2 else 0.0
            return (h + m / 60.0 + s / 3600.0) * 15.0  # 1 h = 15°
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _dms_to_deg(dms: str) -> Optional[float]:
        """Convert '±DD:MM:SS.ss' to decimal degrees."""
        if not dms:
            return None
        try:
            sign = -1.0 if dms.strip().startswith("-") else 1.0
            cleaned = dms.strip().lstrip("+-")
            parts = cleaned.split(":")
            d = float(parts[0])
            m = float(parts[1]) if len(parts) > 1 else 0.0
            s = float(parts[2]) if len(parts) > 2 else 0.0
            return sign * (d + m / 60.0 + s / 3600.0)
        except (ValueError, IndexError):
            return None
