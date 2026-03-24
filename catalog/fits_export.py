"""
FITS Catalog Export — MitraSETI

Exports pipeline results as FITS binary tables using astropy,
the standard interchange format for astronomical catalogs.
Compatible with TOPCAT, DS9, Aladin, and all VO tools.

Usage:
    from catalog.fits_export import export_candidates_fits, export_skymap_fits
    export_candidates_fits(candidates, header_info, output_path)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def export_candidates_fits(
    candidates: List[Dict[str, Any]],
    file_info: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Export candidate signals as a FITS binary table.

    Columns: frequency_hz, drift_rate, snr, classification,
    confidence, rfi_probability, ood_score, is_candidate,
    interestingness_score, ra_deg, dec_deg, known_rfi, etc.
    """
    from astropy.io import fits
    from astropy.table import Table

    if output_path is None:
        from paths import CANDIDATES_DIR
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = CANDIDATES_DIR / f"candidates_{ts}.fits"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not candidates:
        t = Table()
        t.write(str(output_path), format="fits", overwrite=True)
        logger.info(f"Empty FITS catalog written to {output_path}")
        return output_path

    # Build typed columns
    col_data = {
        "frequency_hz": np.array([c.get("frequency_hz", 0) for c in candidates], dtype=np.float64),
        "drift_rate": np.array([c.get("drift_rate", 0) for c in candidates], dtype=np.float32),
        "snr": np.array([c.get("snr", 0) for c in candidates], dtype=np.float32),
        "classification": [c.get("classification", "unknown") for c in candidates],
        "confidence": np.array([c.get("confidence", 0) for c in candidates], dtype=np.float32),
        "rfi_probability": np.array(
            [c.get("rfi_probability", 0) for c in candidates], dtype=np.float32
        ),
        "ood_score": np.array([c.get("ood_score", 0) for c in candidates], dtype=np.float32),
        "is_candidate": np.array(
            [c.get("is_candidate", False) for c in candidates], dtype=np.bool_
        ),
        "interestingness": np.array(
            [c.get("interestingness_score", 0) for c in candidates], dtype=np.float32
        ),
        "known_rfi": np.array(
            [c.get("known_rfi", False) for c in candidates], dtype=np.bool_
        ),
        "known_rfi_source": [c.get("known_rfi_source", "") for c in candidates],
    }

    t = Table(col_data)

    # Add FITS header metadata
    t.meta["EXTNAME"] = "MITRASETI_CANDIDATES"
    t.meta["PIPELINE"] = "MitraSETI v0.2.0"
    t.meta["DATE"] = datetime.now().isoformat()
    t.meta["AUTHOR"] = "Saman Tabatabaeian"

    if file_info:
        t.meta["SRCFILE"] = str(file_info.get("filepath", ""))[:68]
        t.meta["SOURCE"] = str(file_info.get("source_name", ""))[:68]
        t.meta["FCH1"] = file_info.get("fch1_mhz", 0.0)
        t.meta["FOFF"] = file_info.get("foff_mhz", 0.0)
        t.meta["TSAMP"] = file_info.get("tsamp_s", 0.0)
        t.meta["NCHANS"] = file_info.get("n_chans", 0)
        t.meta["NTIMES"] = file_info.get("n_times", 0)

    t.write(str(output_path), format="fits", overwrite=True)
    logger.info(f"FITS catalog: {len(candidates)} candidates → {output_path}")
    return output_path


def export_skymap_fits(
    radio_candidates: List[Dict[str, Any]],
    optical_candidates: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Export unified sky map as a multi-extension FITS file.

    Extension 1: RADIO_CANDIDATES (MitraSETI)
    Extension 2: OPTICAL_ANOMALIES (AstroLens, if provided)
    Extension 3: CROSS_MATCHES (spatial associations)
    """
    from astropy.io import fits
    from astropy.table import Table

    if output_path is None:
        from paths import CANDIDATES_DIR
        output_path = CANDIDATES_DIR / "unified_skymap.fits"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hdu_list = [fits.PrimaryHDU()]

    # Extension 1: Radio candidates
    if radio_candidates:
        radio_table = Table({
            "ra_deg": np.array([c.get("ra", 0) for c in radio_candidates], dtype=np.float64),
            "dec_deg": np.array([c.get("dec", 0) for c in radio_candidates], dtype=np.float64),
            "frequency_mhz": np.array(
                [c.get("frequency_hz", 0) / 1e6 for c in radio_candidates], dtype=np.float64
            ),
            "snr": np.array([c.get("snr", 0) for c in radio_candidates], dtype=np.float32),
            "drift_rate": np.array(
                [c.get("drift_rate", 0) for c in radio_candidates], dtype=np.float32
            ),
            "source_name": [c.get("source_name", "") for c in radio_candidates],
        })
        radio_table.meta["EXTNAME"] = "RADIO_CANDIDATES"
        hdu_list.append(fits.table_to_hdu(radio_table))

    # Extension 2: Optical anomalies
    if optical_candidates:
        optical_table = Table({
            "ra_deg": np.array([c.get("ra_deg", 0) for c in optical_candidates], dtype=np.float64),
            "dec_deg": np.array(
                [c.get("dec_deg", 0) for c in optical_candidates], dtype=np.float64
            ),
            "ood_score": np.array(
                [c.get("ood_score", 0) for c in optical_candidates], dtype=np.float32
            ),
            "classification": [c.get("classification", "") for c in optical_candidates],
            "source": [c.get("source", "astrolens") for c in optical_candidates],
        })
        optical_table.meta["EXTNAME"] = "OPTICAL_ANOMALIES"
        hdu_list.append(fits.table_to_hdu(optical_table))

    hdul = fits.HDUList(hdu_list)
    hdul.writeto(str(output_path), overwrite=True)
    logger.info(f"Unified skymap FITS → {output_path}")
    return output_path
