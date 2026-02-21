"""
astroSETI End-to-End Processing Pipeline

Connects the Rust core (de-Doppler search, filterbank reading, RFI filtering)
with the Python ML layer (signal classification, feature extraction, OOD detection)
and the database/catalog backend.

Pipeline stages:
    1. Read filterbank/HDF5 file
    2. Run de-Doppler search (Rust DedopplerEngine)
    3. Apply RFI filter (Rust RFIFilter)
    4. Extract physical features (Python FeatureExtractor)
    5. Classify signals (Python SignalClassifier, if model loaded)
    6. Run OOD detection (Python RadioOODDetector, if calibrated)
    7. Return comprehensive results dict

Usage:
    from pipeline import AstroSETIPipeline

    pipe = AstroSETIPipeline(model_path="models/signal_classifier_v1.pt")
    results = pipe.process_file("data/filterbank/observation.fil")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AstroSETIPipeline:
    """End-to-end processing pipeline for radio SETI observations.

    Orchestrates Rust-accelerated de-Doppler search with Python ML inference.
    Falls back gracefully when components are unavailable (no model weights,
    no GPU, Rust core not compiled, etc.).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        db_path: Optional[str] = None,
        ood_calibration_path: Optional[str] = None,
    ):
        """Initialize the pipeline.

        Args:
            model_path: Path to trained classifier weights (.pt file).
                        If None or missing, classification is skipped.
            db_path: Path to SQLite database. If None, DB storage is skipped.
            ood_calibration_path: Path to OOD calibration JSON. If None,
                                  OOD detection runs uncalibrated.
        """
        self._rust_available = False
        self._model_loaded = False
        self._ood_calibrated = False
        self._db = None

        self._init_rust_core()
        self._init_ml(model_path, ood_calibration_path)
        self._init_db(db_path)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_rust_core(self) -> None:
        """Load the compiled Rust extension module."""
        try:
            import astroseti_core
            self._core = astroseti_core
            params = astroseti_core.SearchParams(max_drift_rate=4.0, min_snr=5.0)
            self._dedoppler = astroseti_core.DedopplerEngine(params)
            self._rfi_filter = astroseti_core.RFIFilter()
            self._rust_available = True
            logger.info("Rust core loaded successfully")
        except ImportError:
            self._core = None
            self._dedoppler = None
            self._rfi_filter = None
            logger.warning(
                "Rust core (astroseti_core) not available — "
                "de-Doppler search and .fil reading will be disabled"
            )

    def _init_ml(
        self,
        model_path: Optional[str],
        ood_calibration_path: Optional[str],
    ) -> None:
        """Initialize ML components (classifier, feature extractor, OOD detector)."""
        from inference.feature_extractor import FeatureExtractor
        from inference.signal_classifier import SignalClassifier, SIGNAL_LABELS
        from inference.ood_detector import RadioOODDetector

        self._feature_extractor = FeatureExtractor()
        self._signal_labels = SIGNAL_LABELS

        resolved_model = None
        if model_path and Path(model_path).exists():
            resolved_model = model_path
        self._classifier = SignalClassifier(model_path=resolved_model)
        self._model_loaded = resolved_model is not None
        if not self._model_loaded:
            logger.info("No model weights found — classification will use untrained model")

        self._ood_detector = RadioOODDetector()
        if ood_calibration_path and Path(ood_calibration_path).exists():
            try:
                with open(ood_calibration_path, "r") as f:
                    cal = json.load(f)
                if "spectral_threshold" in cal:
                    self._ood_detector.spectral_threshold = cal["spectral_threshold"]
                self._ood_calibrated = True
                logger.info(f"OOD calibration loaded from {ood_calibration_path}")
            except Exception as e:
                logger.warning(f"Failed to load OOD calibration: {e}")

    def _init_db(self, db_path: Optional[str]) -> None:
        """Initialize database connection (lazy — only stores path)."""
        self._db_path = db_path
        if db_path:
            logger.info(f"Database path set to {db_path}")

    # ------------------------------------------------------------------
    # File reading
    # ------------------------------------------------------------------

    def _read_fil_rust(self, filepath: str) -> Dict[str, Any]:
        """Read a .fil file using the Rust FilterbankReader.

        Returns:
            Dict with keys: header (dict), data (np.ndarray of shape (n_time, n_chans)),
            n_times (int), n_chans (int).
        """
        reader = self._core.FilterbankReader()
        header, data_flat, n_times, n_chans = reader.read(filepath)

        data = np.array(data_flat, dtype=np.float32).reshape(n_times, n_chans)

        header_dict = {
            "fch1": header.fch1,
            "foff": header.foff,
            "tsamp": header.tsamp,
            "nchans": int(n_chans),
            "n_times": int(n_times),
            "source_name": getattr(header, "source_name", "unknown"),
            "nbits": getattr(header, "nbits", 32),
        }

        return {
            "header": header_dict,
            "data": data,
            "n_times": n_times,
            "n_chans": n_chans,
        }

    def _read_h5_python(self, filepath: str) -> Dict[str, Any]:
        """Read an HDF5 file using h5py (fallback for Rust without HDF5 support).

        Supports Breakthrough Listen HDF5 format (.h5) where data is stored
        under the 'data' dataset and header attributes are on the root group.

        Returns:
            Dict with same structure as _read_fil_rust: header, data, n_times, n_chans.
        """
        import h5py

        with h5py.File(filepath, "r") as f:
            data = f["data"][:]
            attrs = dict(f.attrs)

            if data.ndim == 3:
                data = data[:, 0, :]

            header_dict = {
                "fch1": float(attrs.get("fch1", 0.0)),
                "foff": float(attrs.get("foff", 0.0)),
                "tsamp": float(attrs.get("tsamp", 0.0)),
                "nchans": int(attrs.get("nchans", data.shape[-1])),
                "n_times": int(data.shape[0]),
                "source_name": attrs.get("source_name", b"unknown"),
                "nbits": int(attrs.get("nbits", 32)),
            }

            if isinstance(header_dict["source_name"], bytes):
                header_dict["source_name"] = header_dict["source_name"].decode("utf-8", errors="replace")

        return {
            "header": header_dict,
            "data": data.astype(np.float32),
            "n_times": int(data.shape[0]),
            "n_chans": int(data.shape[-1]),
        }

    def _read_file(self, filepath: str) -> Dict[str, Any]:
        """Read a filterbank or HDF5 file, choosing the appropriate reader."""
        ext = Path(filepath).suffix.lower()

        if ext == ".h5":
            return self._read_h5_python(filepath)
        elif ext == ".fil":
            if self._rust_available:
                return self._read_fil_rust(filepath)
            else:
                raise RuntimeError(
                    f"Cannot read .fil file without Rust core: {filepath}. "
                    "Build astroseti_core or convert to .h5."
                )
        else:
            raise ValueError(f"Unsupported file format: {ext} (expected .fil or .h5)")

    # ------------------------------------------------------------------
    # Processing stages
    # ------------------------------------------------------------------

    def _run_dedoppler(
        self, file_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run de-Doppler search using the Rust engine.

        Returns a list of candidate dicts with freq, drift_rate, snr.
        """
        if not self._rust_available:
            logger.warning("Rust core unavailable — skipping de-Doppler search")
            return []

        header_dict = file_info["header"]
        data = file_info["data"]

        h = header_dict
        rust_header = self._core.FilterbankHeader(
            nchans=h["nchans"], nifs=h.get("nifs", 1), nbits=h.get("nbits", 32),
            tsamp=h["tsamp"], fch1=h["fch1"], foff=h["foff"],
            tstart=h.get("tstart", 59000.0),
            source_name=h.get("source_name", "unknown"),
            ra=h.get("ra", 0.0), dec=h.get("dec", 0.0),
        )

        data_flat = data.astype(np.float32).ravel().tolist()
        search_result = self._dedoppler.search(
            data_flat, file_info["n_times"], file_info["n_chans"], rust_header,
        )

        self._last_header = rust_header

        candidates = []
        for c in search_result.candidates:
            candidates.append({
                "frequency_hz": c.frequency_hz,
                "frequency_mhz": c.frequency_hz / 1e6 if c.frequency_hz > 1e6 else c.frequency_hz,
                "drift_rate": c.drift_rate,
                "snr": c.snr,
            })

        return candidates

    def _run_rfi_filter(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply RFI filtering using the Rust engine."""
        if not self._rust_available or not candidates:
            return candidates

        rust_candidates = []
        for c in candidates:
            sc = self._core.SignalCandidate(
                frequency_hz=c["frequency_hz"], drift_rate=c["drift_rate"],
                snr=c["snr"], start_time=0.0, end_time=0.0, bandwidth=0.0,
            )
            rust_candidates.append(sc)

        header = getattr(self, "_last_header", None)
        if header is None:
            return candidates

        filtered = self._rfi_filter.filter(rust_candidates, header)

        result = []
        for c in filtered:
            result.append({
                "frequency_hz": c.frequency_hz,
                "drift_rate": c.drift_rate,
                "snr": c.snr,
            })

        return result

    def _extract_spectrogram(
        self, data: np.ndarray, freq_idx: int, n_freq: int = 256, n_time: int = 64
    ) -> np.ndarray:
        """Extract a spectrogram around a candidate frequency.

        Args:
            data: Full observation data (n_time_full, n_chans).
            freq_idx: Center channel index.
            n_freq: Desired frequency bins.
            n_time: Desired time steps.

        Returns:
            2D array (n_freq, n_time).
        """
        n_time_full, n_chans = data.shape
        half_f = n_freq // 2
        f_start = max(0, freq_idx - half_f)
        f_end = min(n_chans, f_start + n_freq)
        f_start = max(0, f_end - n_freq)

        snippet = data[:, f_start:f_end].T

        if snippet.shape != (n_freq, n_time):
            from scipy.ndimage import zoom
            zoom_f = (n_freq / snippet.shape[0], n_time / snippet.shape[1])
            snippet = zoom(snippet, zoom_f, order=1)

        return snippet.astype(np.float32)

    def _classify_candidates(
        self, candidates: List[Dict[str, Any]], data: np.ndarray, header: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run ML classification and feature extraction on each candidate."""
        if not candidates:
            return candidates

        n_chans = header.get("nchans", data.shape[1])
        fch1 = header.get("fch1", 1420.0)
        foff = abs(header.get("foff", 0.00028))

        for cand in candidates:
            freq_hz = cand["frequency_hz"]
            if foff > 0:
                freq_idx = int((fch1 - freq_hz / 1e6) / foff) if fch1 > freq_hz / 1e6 else n_chans // 2
            else:
                freq_idx = n_chans // 2
            freq_idx = max(0, min(freq_idx, n_chans - 1))

            spectrogram = self._extract_spectrogram(data, freq_idx)

            features = self._feature_extractor.extract(spectrogram, header)
            cand["features"] = asdict(features)

            if self._model_loaded:
                cls_result = self._classifier.classify(spectrogram)
                cand["classification"] = cls_result.signal_type.name.lower()
                cand["confidence"] = cls_result.confidence
                cand["rfi_probability"] = cls_result.rfi_probability
                cand["all_scores"] = cls_result.all_scores
                cand["is_candidate"] = self._classifier.is_candidate(cls_result)

                ood_result = self._ood_detector.detect(spectrogram, self._classifier)
                cand["ood_score"] = ood_result.ood_score
                cand["is_anomaly"] = ood_result.is_anomaly
                cand["ood_method_scores"] = ood_result.method_scores
            else:
                drift = abs(cand.get("drift_rate", 0))
                snr_val = cand.get("snr", 0)
                has_drift = drift > 0.01
                high_snr = snr_val >= 10.0
                rfi_like = drift < 0.001 and snr_val > 50
                cand["classification"] = (
                    "rfi_stationary" if rfi_like
                    else "narrowband_drifting" if has_drift
                    else "narrowband_stationary"
                )
                cand["confidence"] = min(snr_val / 50.0, 1.0)
                cand["rfi_probability"] = 0.9 if rfi_like else (0.3 if not has_drift else 0.1)
                cand["all_scores"] = {}
                cand["is_candidate"] = has_drift and high_snr and not rfi_like
                cand["ood_score"] = 0.0
                cand["is_anomaly"] = snr_val > 30 and has_drift
                cand["ood_method_scores"] = {}

        return candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(self, filepath: str) -> Dict[str, Any]:
        """Process a single filterbank/HDF5 file through the full pipeline.

        Args:
            filepath: Path to a .fil or .h5 file.

        Returns:
            Dict with keys:
                file_info: source filename, format, header summary
                candidates: list of candidate dicts (freq, drift_rate, snr,
                            classification, ood_score, features, etc.)
                timing: per-stage wall-clock times in seconds
                summary: aggregate statistics (total hits, RFI count, candidates)
        """
        timings: Dict[str, float] = {}
        filepath = str(Path(filepath).resolve())

        # Stage 1: Read file
        t0 = time.perf_counter()
        try:
            file_info = self._read_file(filepath)
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return {
                "file_info": {"filepath": filepath, "error": str(e)},
                "candidates": [],
                "timing": {},
                "summary": {"status": "error", "error": str(e)},
            }
        timings["read_file"] = time.perf_counter() - t0

        header = file_info["header"]
        data = file_info["data"]

        # Stage 2: De-Doppler search
        t0 = time.perf_counter()
        raw_candidates = self._run_dedoppler(file_info)
        timings["dedoppler_search"] = time.perf_counter() - t0

        # Stage 3: RFI filter
        t0 = time.perf_counter()
        filtered_candidates = self._run_rfi_filter(raw_candidates)
        timings["rfi_filter"] = time.perf_counter() - t0

        # Stage 4–6: Feature extraction, classification, OOD detection
        t0 = time.perf_counter()
        classified = self._classify_candidates(filtered_candidates, data, header)
        timings["ml_inference"] = time.perf_counter() - t0

        timings["total"] = sum(timings.values())

        n_rfi = sum(
            1 for c in classified
            if c.get("classification", "").startswith("rfi_")
        )
        n_candidates = sum(1 for c in classified if c.get("is_candidate", False))
        n_anomalies = sum(1 for c in classified if c.get("is_anomaly", False))

        summary = {
            "status": "success",
            "total_hits_raw": len(raw_candidates),
            "total_hits_filtered": len(filtered_candidates),
            "rfi_count": n_rfi,
            "candidate_count": n_candidates,
            "anomaly_count": n_anomalies,
            "rust_core_used": self._rust_available,
            "model_loaded": self._model_loaded,
            "ood_calibrated": self._ood_calibrated,
        }

        result = {
            "file_info": {
                "filepath": filepath,
                "format": Path(filepath).suffix.lower(),
                "source_name": header.get("source_name", "unknown"),
                "n_times": file_info["n_times"],
                "n_chans": file_info["n_chans"],
                "fch1_mhz": header.get("fch1", 0.0),
                "foff_mhz": header.get("foff", 0.0),
                "tsamp_s": header.get("tsamp", 0.0),
            },
            "candidates": classified,
            "timing": timings,
            "summary": summary,
        }

        logger.info(
            f"Processed {Path(filepath).name}: "
            f"{len(raw_candidates)} raw → {len(filtered_candidates)} filtered → "
            f"{n_candidates} candidates ({timings['total']:.2f}s)"
        )

        return result

    def process_batch(self, filepaths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files sequentially.

        Args:
            filepaths: List of paths to .fil or .h5 files.

        Returns:
            List of result dicts, one per file.
        """
        results = []
        for i, fp in enumerate(filepaths):
            logger.info(f"Processing file {i + 1}/{len(filepaths)}: {fp}")
            result = self.process_file(fp)
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run astroSETI processing pipeline")
    parser.add_argument("files", nargs="+", help="Filterbank (.fil) or HDF5 (.h5) files")
    parser.add_argument("--model", type=str, default="models/signal_classifier_v1.pt",
                        help="Path to classifier model weights")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite database")
    parser.add_argument("--ood-cal", type=str, default="models/ood_calibration.json",
                        help="Path to OOD calibration JSON")
    parser.add_argument("--json-output", type=str, default=None,
                        help="Write results to JSON file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipe = AstroSETIPipeline(
        model_path=args.model,
        db_path=args.db,
        ood_calibration_path=args.ood_cal,
    )

    results = pipe.process_batch(args.files)

    for r in results:
        fp = r["file_info"]["filepath"]
        s = r["summary"]
        t = r["timing"]
        print(f"\n{'=' * 60}")
        print(f"File: {Path(fp).name}")
        print(f"  Status:     {s['status']}")
        print(f"  Raw hits:   {s.get('total_hits_raw', 0)}")
        print(f"  Filtered:   {s.get('total_hits_filtered', 0)}")
        print(f"  RFI:        {s.get('rfi_count', 0)}")
        print(f"  Candidates: {s.get('candidate_count', 0)}")
        print(f"  Anomalies:  {s.get('anomaly_count', 0)}")
        print(f"  Time:       {t.get('total', 0):.3f}s")

        if r["candidates"]:
            print(f"\n  Top candidates:")
            for c in sorted(r["candidates"], key=lambda x: x.get("snr", 0), reverse=True)[:5]:
                print(
                    f"    freq={c['frequency_hz']:.6f} Hz  "
                    f"drift={c['drift_rate']:.4f} Hz/s  "
                    f"SNR={c['snr']:.1f}  "
                    f"class={c.get('classification', 'N/A')}  "
                    f"conf={c.get('confidence', 0):.3f}"
                )

    if args.json_output:
        _serializable = _make_json_serializable(results)
        with open(args.json_output, "w") as f:
            json.dump(_serializable, f, indent=2)
        print(f"\nResults written to {args.json_output}")


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == "__main__":
    main()
