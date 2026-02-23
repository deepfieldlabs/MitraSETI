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
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fix HDF5 plugin path before any HDF5 library is imported.
# Without this, h5py fails on systems where /usr/local/hdf5/lib/plugin
# does not exist.
if "HDF5_PLUGIN_PATH" not in os.environ:
    os.environ["HDF5_PLUGIN_PATH"] = ""

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

    # Maximum data points to load at once.  Kept low so the brute-force
    # de-Doppler search finishes in reasonable time (~2-5 min per file).
    _MAX_DATA_POINTS = 16 * 1024 * 1024  # ~16M → ~64 MB float32

    def _read_blimpy(self, filepath: str) -> Dict[str, Any]:
        """Read any BL file (.fil or .h5) using blimpy Waterfall.

        Handles all BL-specific formats (gpuspec, rawspec, etc.) that the
        Rust reader does not support.  For very large files, limits the data
        to a manageable subset.
        """
        from blimpy import Waterfall

        wf = Waterfall(filepath, load_data=False)
        hdr = wf.header

        n_chans = int(hdr.get("nchans", 0))
        n_ints = int(wf.n_ints_in_file) if hasattr(wf, "n_ints_in_file") else 16
        total_points = n_chans * n_ints

        if total_points > self._MAX_DATA_POINTS:
            # For huge files, read only a subset of channels around 1420 MHz
            # (hydrogen line / "water hole") or the center band.
            fch1 = float(hdr.get("fch1", 0))
            foff = float(hdr.get("foff", 0))
            if foff != 0:
                max_chans = self._MAX_DATA_POINTS // max(n_ints, 1)
                center_freq = fch1 + foff * n_chans / 2
                half_bw = abs(foff) * max_chans / 2
                f_start = center_freq - half_bw
                f_stop = center_freq + half_bw
                logger.info(
                    f"  Large file ({n_chans} chans × {n_ints} ints). "
                    f"Reading {max_chans} chans around {center_freq:.1f} MHz"
                )
                wf = Waterfall(filepath, f_start=f_start, f_stop=f_stop)
            else:
                wf = Waterfall(filepath, max_load=1.0)
        else:
            wf = Waterfall(filepath)

        data = wf.data
        if data.ndim == 3:
            data = data[:, 0, :]

        header_dict = {
            "fch1": float(hdr.get("fch1", 0.0)),
            "foff": float(hdr.get("foff", 0.0)),
            "tsamp": float(hdr.get("tsamp", 0.0)),
            "nchans": int(data.shape[-1]),
            "n_times": int(data.shape[0]),
            "source_name": hdr.get("source_name", "unknown"),
            "nbits": int(hdr.get("nbits", 32)),
            "tstart": float(hdr.get("tstart", 59000.0)),
            "nifs": int(hdr.get("nifs", 1)),
        }

        if isinstance(header_dict["source_name"], bytes):
            header_dict["source_name"] = header_dict["source_name"].decode(
                "utf-8", errors="replace"
            )

        return {
            "header": header_dict,
            "data": data.astype(np.float32),
            "n_times": int(data.shape[0]),
            "n_chans": int(data.shape[-1]),
        }

    def _read_file(self, filepath: str) -> Dict[str, Any]:
        """Read a filterbank or HDF5 file, choosing the appropriate reader.

        Tries the Rust reader first for .fil files (fastest), then falls
        back to blimpy which handles all BL formats.
        """
        ext = Path(filepath).suffix.lower()

        if ext == ".h5":
            # Always use blimpy for .h5 — handles BL format + HDF5 plugins
            return self._read_blimpy(filepath)
        elif ext == ".fil":
            # Try Rust first; fall back to blimpy if it fails
            if self._rust_available:
                try:
                    return self._read_fil_rust(filepath)
                except Exception as e:
                    logger.info(
                        f"Rust reader failed for {Path(filepath).name} "
                        f"({e}), falling back to blimpy"
                    )
            return self._read_blimpy(filepath)
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

        # Tighter limit for files with very few time integrations (e.g. gpuspec
        # files with 3 ints × 67M channels) — the brute-force de-Doppler scales
        # poorly with channel count when time steps are scarce.
        n_times = data.shape[0] if data.ndim == 2 else 1
        max_pts = 4_000_000 if n_times <= 8 else 16_000_000
        total_points = data.shape[0] * data.shape[1] if data.ndim == 2 else data.size
        if total_points > max_pts and data.ndim == 2:
            # Choose downsample factor to bring data under the limit
            factor = 1
            while data.shape[1] // factor * data.shape[0] > max_pts:
                factor *= 2
            logger.warning(
                f"Data too large for Rust de-Doppler ({total_points:,} points). "
                f"Downsampling channels by {factor}x."
            )
            trim = data.shape[1] - (data.shape[1] % factor)
            data = data[:, :trim].reshape(data.shape[0], -1, factor).mean(axis=2)
            h["nchans"] = data.shape[1]
            h["foff"] = h["foff"] * factor
            file_info["n_chans"] = data.shape[1]
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

    # Frequency clustering tolerance: hits within this many channels of
    # each other and with similar drift rates are considered duplicates.
    _CLUSTER_FREQ_TOL_CHANNELS = 64
    _CLUSTER_DRIFT_TOL = 0.5  # Hz/s

    def _cluster_hits(
        self, candidates: List[Dict[str, Any]], header: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Deduplicate de-Doppler hits by frequency/drift proximity.

        Many raw hits are the same physical signal detected at slightly
        different frequency/drift combinations.  Group nearby hits and
        keep only the strongest (highest SNR) per cluster.
        """
        if len(candidates) <= 1:
            return candidates

        fch1 = header.get("fch1", 1420.0)
        foff = abs(header.get("foff", 0.00028))

        for c in candidates:
            c["_chan_idx"] = (
                int((fch1 - c["frequency_hz"] / 1e6) / foff)
                if foff > 0 else 0
            )

        candidates.sort(key=lambda c: c["_chan_idx"])

        clusters: List[List[Dict[str, Any]]] = []
        current_cluster: List[Dict[str, Any]] = [candidates[0]]

        for c in candidates[1:]:
            prev = current_cluster[-1]
            freq_close = (
                abs(c["_chan_idx"] - prev["_chan_idx"])
                <= self._CLUSTER_FREQ_TOL_CHANNELS
            )
            drift_close = (
                abs(c["drift_rate"] - prev["drift_rate"])
                <= self._CLUSTER_DRIFT_TOL
            )
            if freq_close and drift_close:
                current_cluster.append(c)
            else:
                clusters.append(current_cluster)
                current_cluster = [c]
        clusters.append(current_cluster)

        result = []
        for cluster in clusters:
            best = max(cluster, key=lambda c: c.get("snr", 0))
            best.pop("_chan_idx", None)
            result.append(best)

        for c in candidates:
            c.pop("_chan_idx", None)

        return result

    # Directory where spectrograms are cached for later retraining.
    # Must match the path used by streaming_observation.py (from paths.py).
    from paths import DATA_DIR as _DATA_DIR
    _SPECTROGRAM_CACHE_DIR = _DATA_DIR / "spectrogram_cache"

    def _classify_candidates(
        self, candidates: List[Dict[str, Any]], data: np.ndarray, header: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Two-stage classification: rule-based on ALL, ML only on survivors.

        Stage 1 (instant): Apply rule-based classification to every signal.
                 This checks SNR, drift rate, and RFI patterns.
        Stage 2 (targeted): Run full ML inference (feature extraction,
                 CNN+Transformer classification, OOD detection) only on
                 signals that pass the candidate criteria from Stage 1.
        """
        if not candidates:
            return candidates

        n_chans = header.get("nchans", data.shape[1])
        fch1 = header.get("fch1", 1420.0)
        foff = abs(header.get("foff", 0.00028))

        # -- Stage 1: Rule-based classification on ALL signals -----
        ml_queue: List[int] = []

        for idx, cand in enumerate(candidates):
            drift = abs(cand.get("drift_rate", 0))
            snr_val = cand.get("snr", 0)
            has_meaningful_drift = 0.05 <= drift <= 10.0
            high_snr = snr_val >= 25.0
            exceptional_snr = snr_val >= 50.0
            rfi_like = drift < 0.001 and snr_val > 50

            cand["classification"] = (
                "rfi_stationary" if rfi_like
                else "narrowband_drifting" if drift > 0.01
                else "narrowband_stationary"
            )
            cand["confidence"] = min(snr_val / 50.0, 1.0)
            cand["rfi_probability"] = (
                0.9 if rfi_like else (0.3 if drift < 0.01 else 0.1)
            )
            cand["all_scores"] = {}
            cand["is_candidate"] = (
                has_meaningful_drift and high_snr and not rfi_like
            )
            cand["ood_score"] = 0.0
            cand["is_anomaly"] = exceptional_snr and has_meaningful_drift
            cand["ood_method_scores"] = {}

            if cand["is_candidate"] or cand["is_anomaly"]:
                ml_queue.append(idx)

        n_rule_candidates = len(ml_queue)
        logger.info(
            f"  Stage 1 (rule-based): {len(candidates)} signals → "
            f"{n_rule_candidates} pass candidate criteria"
        )

        # -- Stage 2: Batch ML inference on rule-based survivors ----
        _MAX_ML_CANDIDATES = 500   # only classify the strongest by SNR
        _ML_BATCH_SIZE = 128       # sub-batch to avoid OOM on large files

        if self._model_loaded and ml_queue:
            # Sort ML queue by SNR descending and cap at _MAX_ML_CANDIDATES
            ml_queue.sort(
                key=lambda idx: candidates[idx].get("snr", 0), reverse=True
            )
            if len(ml_queue) > _MAX_ML_CANDIDATES:
                logger.info(
                    f"  Stage 2 (ML): capping {len(ml_queue)} → "
                    f"{_MAX_ML_CANDIDATES} strongest candidates for inference"
                )
                ml_queue = ml_queue[:_MAX_ML_CANDIDATES]

            logger.info(
                f"  Stage 2 (ML): batch inference on "
                f"{len(ml_queue)} candidates "
                f"(sub-batches of {_ML_BATCH_SIZE})…"
            )
            cache_dir = self._SPECTROGRAM_CACHE_DIR
            cache_dir.mkdir(parents=True, exist_ok=True)

            # 2a. Extract all spectrograms upfront
            spectrograms = []
            for idx in ml_queue:
                cand = candidates[idx]
                freq_hz = cand["frequency_hz"]
                if foff > 0:
                    freq_idx = int(
                        (fch1 - freq_hz / 1e6) / foff
                    ) if fch1 > freq_hz / 1e6 else n_chans // 2
                else:
                    freq_idx = n_chans // 2
                freq_idx = max(0, min(freq_idx, n_chans - 1))
                spectrograms.append(self._extract_spectrogram(data, freq_idx))

            # 2b. Feature extraction (per-signal, lightweight)
            for i, idx in enumerate(ml_queue):
                features = self._feature_extractor.extract(
                    spectrograms[i], header
                )
                candidates[idx]["features"] = asdict(features)

            # 2c. Batch classification in sub-batches to avoid OOM
            cls_results = []
            for batch_start in range(0, len(spectrograms), _ML_BATCH_SIZE):
                batch_end = min(batch_start + _ML_BATCH_SIZE, len(spectrograms))
                sub_batch = spectrograms[batch_start:batch_end]
                try:
                    sub_results = self._classifier.classify_batch(sub_batch)
                    cls_results.extend(sub_results)
                except Exception as e:
                    logger.warning(
                        f"  Batch inference failed ({batch_start}-{batch_end}): "
                        f"{e}, falling back to individual classify"
                    )
                    for spec in sub_batch:
                        try:
                            cls_results.append(self._classifier.classify(spec))
                        except Exception:
                            cls_results.append(None)

            # 2d. Apply results + OOD (reuses logits, no duplicate fwd pass)
            for i, idx in enumerate(ml_queue):
                cand = candidates[idx]
                cls_result = cls_results[i] if i < len(cls_results) else None

                if cls_result is None:
                    continue

                cand["classification"] = cls_result.signal_type.name.lower()
                cand["confidence"] = cls_result.confidence
                cand["rfi_probability"] = cls_result.rfi_probability
                cand["all_scores"] = cls_result.all_scores
                cand["is_candidate"] = self._classifier.is_candidate(
                    cls_result
                )

                ood_result = self._ood_detector.detect_from_scores(
                    spectrograms[i], cls_result.all_scores
                )
                cand["ood_score"] = ood_result.ood_score
                cand["is_anomaly"] = ood_result.is_anomaly
                cand["ood_method_scores"] = ood_result.method_scores

                # Cache spectrogram + label for future retraining
                try:
                    label = cls_result.signal_type.value
                    cache_file = (
                        cache_dir / f"spec_{hash(cand['frequency_hz']):016x}.npz"
                    )
                    if not cache_file.exists():
                        np.savez_compressed(
                            cache_file,
                            spectrogram=spectrograms[i],
                            label=label,
                            snr=cand["snr"],
                            drift=cand["drift_rate"],
                        )
                except Exception:
                    pass

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

        fname = Path(filepath).name

        # Stage 1: Read file
        logger.info(f"  [1/4] Reading {fname}…")
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
        logger.info(
            f"  [1/4] Read complete: {file_info['n_times']}t × "
            f"{file_info['n_chans']:,}ch ({timings['read_file']:.1f}s)"
        )

        header = file_info["header"]
        data = file_info["data"]

        # Stage 2: De-Doppler search
        logger.info(f"  [2/4] Running de-Doppler search…")
        t0 = time.perf_counter()
        raw_candidates = self._run_dedoppler(file_info)
        timings["dedoppler_search"] = time.perf_counter() - t0
        logger.info(
            f"  [2/4] De-Doppler: {len(raw_candidates)} hits "
            f"({timings['dedoppler_search']:.1f}s)"
        )

        # Stage 3a: RFI filter
        t0 = time.perf_counter()
        filtered_candidates = self._run_rfi_filter(raw_candidates)
        timings["rfi_filter"] = time.perf_counter() - t0
        n_after_rfi = len(filtered_candidates)
        if len(raw_candidates) != n_after_rfi:
            logger.info(
                f"  [3/4] RFI filter: {len(raw_candidates)} → "
                f"{n_after_rfi} ({timings['rfi_filter']:.1f}s)"
            )

        # Stage 3b: Cluster nearby hits to remove duplicates
        t0 = time.perf_counter()
        filtered_candidates = self._cluster_hits(filtered_candidates, header)
        timings["clustering"] = time.perf_counter() - t0
        n_after_cluster = len(filtered_candidates)
        if n_after_rfi != n_after_cluster:
            logger.info(
                f"  [3/4] Clustering: {n_after_rfi} → "
                f"{n_after_cluster} unique signals "
                f"({(1 - n_after_cluster / max(n_after_rfi, 1)) * 100:.0f}% "
                f"reduction, {timings['clustering']:.2f}s)"
            )

        # Stage 4: Two-stage classification (rule-based on ALL, ML on survivors)
        logger.info(
            f"  [4/4] Classifying {len(filtered_candidates)} signals…"
        )
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

        # ── Efficiency metrics ───────────────────────────────────────
        data_points = file_info["n_times"] * file_info["n_chans"]
        dd_time = timings.get("dedoppler_search", 0.001)
        ml_time = timings.get("ml_inference", 0.001)
        n_filtered = len(filtered_candidates)

        snr_values = [c.get("snr", 0) for c in classified]
        conf_values = [c.get("confidence", 0) for c in classified]
        drift_values = [abs(c.get("drift_rate", 0)) for c in classified]

        cls_dist = {}
        for c in classified:
            cls_name = c.get("classification", "unknown")
            cls_dist[cls_name] = cls_dist.get(cls_name, 0) + 1

        metrics = {
            "dedoppler": {
                "data_points": data_points,
                "throughput_mpts_per_s": round(data_points / 1e6 / dd_time, 2),
                "raw_crossings": len(raw_candidates),
                "after_rfi_filter": n_after_rfi,
                "after_clustering": n_filtered,
                "clustering_reduction": round(
                    1 - n_filtered / max(n_after_rfi, 1), 3
                ) if n_after_rfi > 0 else 0,
                "rfi_rejected": len(raw_candidates) - n_after_rfi,
                "algorithm": "brute_force",
            },
            "ml_classifier": {
                "model_trained": self._model_loaded,
                "using_rule_based": not self._model_loaded,
                "signals_classified": n_filtered,
                "throughput_sig_per_s": round(
                    n_filtered / ml_time, 1
                ) if ml_time > 0 else 0,
                "classification_dist": cls_dist,
                "candidate_rate": round(
                    n_candidates / max(n_filtered, 1), 4
                ),
                "confidence_mean": round(
                    np.mean(conf_values), 3
                ) if conf_values else 0,
                "confidence_min": round(min(conf_values), 3) if conf_values else 0,
            },
            "ood_detector": {
                "calibrated": self._ood_calibrated,
                "anomalies_flagged": n_anomalies,
                "anomaly_rate": round(
                    n_anomalies / max(n_filtered, 1), 4
                ),
            },
            "snr_stats": {
                "max": round(max(snr_values), 2) if snr_values else 0,
                "mean": round(np.mean(snr_values), 2) if snr_values else 0,
                "median": round(float(np.median(snr_values)), 2) if snr_values else 0,
                "above_25": sum(1 for s in snr_values if s >= 25),
                "above_50": sum(1 for s in snr_values if s >= 50),
            },
            "drift_stats": {
                "max": round(max(drift_values), 4) if drift_values else 0,
                "in_candidate_range": sum(
                    1 for d in drift_values if 0.05 <= d <= 10.0
                ),
                "total_with_drift": sum(1 for d in drift_values if d > 0.01),
            },
        }

        logger.info(
            f"  Processed {fname}: "
            f"{len(raw_candidates)} raw → {n_filtered} filtered → "
            f"{n_candidates} candidates ({timings['total']:.2f}s)"
        )

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
            "metrics": metrics,
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
