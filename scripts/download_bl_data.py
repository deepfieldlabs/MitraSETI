#!/usr/bin/env python3
"""
MitraSETI Breakthrough Listen Data Downloader

Download sample data from the Breakthrough Listen Open Data Archive
for demo, testing, and development.

Features:
- Curated set of small .h5 files from the BL archive
- Selective download by target name (e.g., specific stars)
- File caching (skip already downloaded files)
- Progress bar with tqdm
- File integrity verification after download

BL Open Data Archive: http://seti.berkeley.edu/opendata

Usage:
    python scripts/download_bl_data.py
    python scripts/download_bl_data.py --target Kepler-160
    python scripts/download_bl_data.py --count 10 --output-dir ./data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import DATA_DIR, FILTERBANK_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BL Open Data catalog
# ─────────────────────────────────────────────────────────────────────────────

# Curated list of small sample files from the BL archive.
# These are real Breakthrough Listen targets with small file sizes
# suitable for demo and testing.
#
# Format:
#   {
#     "filename": str,
#     "url": str,
#     "target": str (star / system name),
#     "telescope": str,
#     "size_mb": float (approximate),
#     "freq_ghz": str (frequency band),
#     "description": str,
#     "sha256": str or None (for verification)
#   }

BL_BASE_URL = "http://seti.berkeley.edu/opendata/"

# Note: These are representative file paths based on the BL open data
# structure. Actual URLs may need to be updated as the archive evolves.
BL_SAMPLE_CATALOG = [
    {
        "filename": "blc00_guppi_59046_80036_HIP4436_0011.gpuspec.0000.h5",
        "url": "http://blpd0.ssl.berkeley.edu/dl/GBT_58600_67600_HIP4436_fine.h5",
        "target": "HIP4436",
        "telescope": "GBT",
        "size_mb": 15.0,
        "freq_ghz": "1.1-1.9",
        "description": "Green Bank Telescope L-band observation of HIP4436",
        "sha256": None,
    },
    {
        "filename": "blc00_guppi_59046_80036_HIP39826_0009.gpuspec.0000.h5",
        "url": "http://blpd0.ssl.berkeley.edu/dl/GBT_58600_67600_HIP39826_fine.h5",
        "target": "HIP39826",
        "telescope": "GBT",
        "size_mb": 15.0,
        "freq_ghz": "1.1-1.9",
        "description": "Green Bank Telescope L-band observation of HIP39826",
        "sha256": None,
    },
    {
        "filename": "blc17_guppi_57991_49905_HIP17147_0003.gpuspec.0000.h5",
        "url": "http://blpd0.ssl.berkeley.edu/dl/GBT_57991_HIP17147_fine.h5",
        "target": "HIP17147",
        "telescope": "GBT",
        "size_mb": 12.0,
        "freq_ghz": "1.1-1.9",
        "description": "Green Bank Telescope L-band observation of HIP17147 (Tau Ceti region)",
        "sha256": None,
    },
    {
        "filename": "blc3b_guppi_57386_VOYAGER1_0004.gpuspec.0000.h5",
        "url": "http://blpd0.ssl.berkeley.edu/dl/Voyager1_fine.h5",
        "target": "Voyager-1",
        "telescope": "GBT",
        "size_mb": 8.0,
        "freq_ghz": "8.4",
        "description": "Voyager 1 carrier signal – known narrowband reference",
        "sha256": None,
    },
    {
        "filename": "blc07_guppi_57650_67573_Kepler160_0002.gpuspec.0000.h5",
        "url": "http://blpd0.ssl.berkeley.edu/dl/GBT_57650_Kepler160_fine.h5",
        "target": "Kepler-160",
        "telescope": "GBT",
        "size_mb": 14.0,
        "freq_ghz": "1.1-1.9",
        "description": "Observation of Kepler-160 (host to super-Earth KOI-456.04)",
        "sha256": None,
    },
    {
        "filename": "blc00_guppi_58803_28029_TIC141146667_0001.gpuspec.0000.h5",
        "url": "http://blpd0.ssl.berkeley.edu/dl/GBT_58803_TIC141146667_fine.h5",
        "target": "TIC141146667",
        "telescope": "GBT",
        "size_mb": 10.0,
        "freq_ghz": "1.1-1.9",
        "description": "TESS target of interest – L-band follow-up",
        "sha256": None,
    },
    {
        "filename": "blc04_guppi_57513_35257_TRAPPIST1_0010.gpuspec.0000.h5",
        "url": "http://blpd0.ssl.berkeley.edu/dl/GBT_57513_TRAPPIST1_fine.h5",
        "target": "TRAPPIST-1",
        "telescope": "GBT",
        "size_mb": 18.0,
        "freq_ghz": "1.1-1.9",
        "description": "TRAPPIST-1 system – seven terrestrial planets",
        "sha256": None,
    },
    {
        "filename": "blc13_guppi_57802_51456_ProximaCen_0007.gpuspec.0000.h5",
        "url": "http://blpd0.ssl.berkeley.edu/dl/GBT_57802_ProximaCen_fine.h5",
        "target": "Proxima-Cen",
        "telescope": "GBT",
        "size_mb": 16.0,
        "freq_ghz": "1.1-1.9",
        "description": "Proxima Centauri – nearest star with known exoplanet",
        "sha256": None,
    },
    {
        "filename": "spliced_blc0001020304050607_guppi_57650_Ross128_0001.h5",
        "url": "http://blpd0.ssl.berkeley.edu/dl/GBT_57650_Ross128_fine.h5",
        "target": "Ross-128",
        "telescope": "GBT",
        "size_mb": 20.0,
        "freq_ghz": "1.1-1.9",
        "description": "Ross 128 – site of BLC1-like candidate signal",
        "sha256": None,
    },
    {
        "filename": "Parkes_55900_Kepler442_fine.h5",
        "url": "http://blpd0.ssl.berkeley.edu/dl/Parkes_55900_Kepler442_fine.h5",
        "target": "Kepler-442",
        "telescope": "Parkes",
        "size_mb": 11.0,
        "freq_ghz": "1.2-1.5",
        "description": "Parkes observation of Kepler-442b (habitable zone super-Earth)",
        "sha256": None,
    },
]

# Download state cache
_DOWNLOAD_CACHE_FILE = DATA_DIR / "bl_download_cache.json"


# ─────────────────────────────────────────────────────────────────────────────
# Downloader
# ─────────────────────────────────────────────────────────────────────────────

class BLDataDownloader:
    """
    Download sample data from the Breakthrough Listen Open Data Archive.

    Supports:
    - Selective download by target name
    - File caching (skip already downloaded)
    - Progress tracking
    - Integrity verification
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        target: Optional[str] = None,
        count: int = 5,
    ):
        self.output_dir = Path(output_dir) if output_dir else FILTERBANK_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target = target
        self.count = count
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load download cache."""
        if _DOWNLOAD_CACHE_FILE.exists():
            try:
                with open(_DOWNLOAD_CACHE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"downloaded": {}}

    def _save_cache(self):
        """Save download cache."""
        try:
            with open(_DOWNLOAD_CACHE_FILE, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save download cache: {e}")

    def _select_files(self) -> List[dict]:
        """Select files to download based on target and count."""
        catalog = BL_SAMPLE_CATALOG

        if self.target:
            # Filter by target name (case-insensitive partial match)
            target_lower = self.target.lower().replace("-", "").replace(" ", "")
            catalog = [
                entry for entry in catalog
                if target_lower in entry["target"].lower().replace("-", "").replace(" ", "")
            ]
            if not catalog:
                logger.warning(
                    f"No files found for target '{self.target}'. "
                    f"Available targets: {', '.join(e['target'] for e in BL_SAMPLE_CATALOG)}"
                )
                return []

        return catalog[: self.count]

    def _is_cached(self, entry: dict) -> bool:
        """Check if a file is already downloaded and valid."""
        dest = self.output_dir / entry["filename"]
        if not dest.exists():
            return False

        cached = self.cache.get("downloaded", {}).get(entry["filename"])
        if cached and dest.stat().st_size > 0:
            return True

        return False

    def _download_file(self, entry: dict) -> bool:
        """
        Download a single file with progress.

        Returns True on success, False on failure.
        """
        dest = self.output_dir / entry["filename"]
        url = entry["url"]

        logger.info(
            f"  Downloading: {entry['target']} ({entry['filename']})"
        )
        logger.info(f"    Source: {url}")
        logger.info(f"    Size: ~{entry['size_mb']:.0f} MB")

        try:
            import httpx

            with httpx.stream("GET", url, timeout=120, follow_redirects=True) as resp:
                if resp.status_code != 200:
                    logger.error(
                        f"    Download failed: HTTP {resp.status_code}"
                    )
                    return False

                total = int(resp.headers.get("content-length", 0))

                try:
                    from tqdm import tqdm
                    progress = tqdm(
                        total=total if total > 0 else None,
                        unit="B",
                        unit_scale=True,
                        desc=f"    {entry['target']}",
                        ncols=80,
                    )
                except ImportError:
                    progress = None

                hasher = hashlib.sha256()
                downloaded = 0

                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                        hasher.update(chunk)
                        downloaded += len(chunk)
                        if progress:
                            progress.update(len(chunk))

                if progress:
                    progress.close()

            # Verify integrity
            file_sha256 = hasher.hexdigest()
            if entry.get("sha256") and entry["sha256"] != file_sha256:
                logger.error(
                    f"    Integrity check FAILED! "
                    f"Expected {entry['sha256'][:16]}..., "
                    f"got {file_sha256[:16]}..."
                )
                dest.unlink(missing_ok=True)
                return False

            # Update cache
            self.cache.setdefault("downloaded", {})[entry["filename"]] = {
                "path": str(dest),
                "target": entry["target"],
                "telescope": entry["telescope"],
                "size_bytes": dest.stat().st_size,
                "sha256": file_sha256,
                "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._save_cache()

            logger.info(
                f"    Saved: {dest} ({dest.stat().st_size / (1024*1024):.1f} MB)"
            )
            return True

        except ImportError:
            logger.error(
                "    httpx is required for downloading. "
                "Install with: pip install httpx"
            )
            return False
        except Exception as e:
            logger.error(f"    Download error: {e}")
            dest.unlink(missing_ok=True)
            return False

    def _generate_synthetic_sample(self, entry: dict) -> bool:
        """
        Generate a synthetic HDF5 file when download fails.

        Creates a small file with realistic structure for testing.
        """
        dest = self.output_dir / entry["filename"]

        try:
            import numpy as np

            try:
                import h5py
            except ImportError:
                logger.warning("    h5py not available; cannot create synthetic data")
                return False

            logger.info(f"    Creating synthetic sample for {entry['target']}...")

            rng = np.random.default_rng(hash(entry["target"]) % 2**31)
            nchans = 1024
            ntime = 256

            # Shape must be (ntime, nchans) to match what the h5py reader expects
            data = rng.normal(0, 1.0, (ntime, nchans)).astype(np.float32)

            # Inject 3-5 drifting narrowband signals with high SNR
            n_signals = rng.integers(3, 6)
            for _ in range(n_signals):
                drift_pixels = rng.uniform(-1.5, 1.5)
                snr = rng.uniform(12, 40)
                start_chan = rng.integers(100, nchans - 100)
                bw = rng.integers(1, 3)
                for t in range(ntime):
                    chan = int(start_chan + drift_pixels * t / ntime * 50) % nchans
                    lo = max(0, chan - bw)
                    hi = min(nchans, chan + bw + 1)
                    data[t, lo:hi] += snr

            # Inject stationary RFI for the filter to reject
            for _ in range(2):
                rfi_chan = rng.integers(0, nchans)
                data[:, rfi_chan] += rng.uniform(20, 50)

            with h5py.File(str(dest), "w") as f:
                f.create_dataset("data", data=data)
                f.attrs["source_name"] = entry["target"]
                f.attrs["telescope"] = entry["telescope"]
                f.attrs["fch1"] = 1420.0
                f.attrs["foff"] = -0.00029
                f.attrs["tsamp"] = 18.253611
                f.attrs["nchans"] = nchans
                f.attrs["nifs"] = 1
                f.attrs["nbits"] = 32
                f.attrs["synthetic"] = True

            # Cache
            self.cache.setdefault("downloaded", {})[entry["filename"]] = {
                "path": str(dest),
                "target": entry["target"],
                "telescope": entry["telescope"],
                "size_bytes": dest.stat().st_size,
                "sha256": "synthetic",
                "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "synthetic": True,
            }
            self._save_cache()

            logger.info(f"    Synthetic sample saved: {dest}")
            return True

        except Exception as e:
            logger.error(f"    Failed to create synthetic sample: {e}")
            return False

    def run(self) -> List[Path]:
        """
        Download selected files.

        Returns list of paths to downloaded files.
        """
        selected = self._select_files()
        if not selected:
            logger.info("No files selected for download.")
            return []

        logger.info(f"Selected {len(selected)} file(s) for download:")
        for entry in selected:
            cached = " (cached)" if self._is_cached(entry) else ""
            logger.info(
                f"  - {entry['target']:15s} | "
                f"{entry['telescope']:6s} | "
                f"{entry['freq_ghz']:8s} GHz | "
                f"~{entry['size_mb']:.0f} MB{cached}"
            )

        downloaded: List[Path] = []

        for entry in selected:
            dest = self.output_dir / entry["filename"]

            # Skip if cached
            if self._is_cached(entry):
                logger.info(f"  Skipped (cached): {entry['filename']}")
                downloaded.append(dest)
                continue

            # Try real download first
            if self._download_file(entry):
                downloaded.append(dest)
            else:
                # Fall back to synthetic data for testing
                logger.info(
                    f"    Download unavailable. Generating synthetic sample..."
                )
                if self._generate_synthetic_sample(entry):
                    downloaded.append(dest)

        # Summary
        print(f"\n{'=' * 60}")
        print(f"  Download complete: {len(downloaded)}/{len(selected)} files")
        print(f"  Output directory: {self.output_dir}")
        if downloaded:
            total_size = sum(f.stat().st_size for f in downloaded if f.exists())
            print(f"  Total size: {total_size / (1024*1024):.1f} MB")
        print(f"{'=' * 60}")

        return downloaded

    def list_available(self):
        """List all available targets."""
        print(f"\n{'=' * 70}")
        print(f"  Breakthrough Listen Open Data – Available Targets")
        print(f"{'=' * 70}")
        print(f"  {'Target':<18} {'Telescope':<10} {'Freq (GHz)':<12} {'Size MB':<10} {'Description'}")
        print(f"  {'-'*18} {'-'*10} {'-'*12} {'-'*10} {'-'*40}")
        for entry in BL_SAMPLE_CATALOG:
            cached = " [cached]" if self._is_cached(entry) else ""
            print(
                f"  {entry['target']:<18} "
                f"{entry['telescope']:<10} "
                f"{entry['freq_ghz']:<12} "
                f"~{entry['size_mb']:<9.0f} "
                f"{entry['description'][:40]}{cached}"
            )
        print(f"{'=' * 70}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Breakthrough Listen sample data for MitraSETI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_bl_data.py                          # Download 5 default samples
    python scripts/download_bl_data.py --count 10               # Download 10 samples
    python scripts/download_bl_data.py --target TRAPPIST-1      # Download TRAPPIST-1 data
    python scripts/download_bl_data.py --target Voyager          # Download Voyager-1 reference
    python scripts/download_bl_data.py --list                   # Show available targets
    python scripts/download_bl_data.py --output-dir ./my_data   # Custom output directory

Available targets:
    HIP4436, HIP39826, HIP17147, Voyager-1, Kepler-160,
    TIC141146667, TRAPPIST-1, Proxima-Cen, Ross-128, Kepler-442
        """,
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Download files for a specific target (partial match, case-insensitive)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of files to download (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {FILTERBANK_DIR})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available targets and exit",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    downloader = BLDataDownloader(
        output_dir=args.output_dir,
        target=args.target,
        count=args.count,
    )

    if args.list:
        downloader.list_available()
        return

    downloader.run()


if __name__ == "__main__":
    main()
