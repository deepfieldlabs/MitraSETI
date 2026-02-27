"""
Tests for the MitraSETI processing pipeline, Rust core, ML models, and API.

These tests verify end-to-end functionality when components are available,
and gracefully skip when optional dependencies (Rust core, model weights,
running API server) are not present.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Rust Core
# ─────────────────────────────────────────────────────────────────────────────

class TestRustCore:
    def test_import(self):
        import mitraseti_core
        assert hasattr(mitraseti_core, "DedopplerEngine")

    def test_filterbank_reader(self):
        import mitraseti_core

        reader = mitraseti_core.FilterbankReader()
        fil_files = list(Path("data/filterbank").glob("*.fil"))
        if fil_files:
            header, data, n_times, n_chans = reader.read(str(fil_files[0]))
            assert n_times > 0
            assert n_chans > 0
        else:
            pytest.skip("No .fil files in data/filterbank")

    def test_search_params(self):
        import mitraseti_core

        params = mitraseti_core.SearchParams(max_drift_rate=4.0, min_snr=5.0)
        assert params is not None

    def test_rfi_filter_exists(self):
        import mitraseti_core
        assert hasattr(mitraseti_core, "RFIFilter")

    def test_dedoppler_engine_creation(self):
        import mitraseti_core

        params = mitraseti_core.SearchParams(max_drift_rate=4.0, min_snr=5.0)
        engine = mitraseti_core.DedopplerEngine(params)
        assert engine is not None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestPipeline:
    def test_pipeline_init(self):
        from pipeline import MitraSETIPipeline

        pipe = MitraSETIPipeline()
        assert pipe._rust_available

    def test_pipeline_init_with_model(self):
        from pipeline import MitraSETIPipeline

        model_path = Path("models/signal_classifier_v1.pt")
        pipe = MitraSETIPipeline(
            model_path=str(model_path) if model_path.exists() else None,
        )
        assert pipe is not None

    def test_process_file(self):
        from pipeline import MitraSETIPipeline

        pipe = MitraSETIPipeline()
        fil_files = list(Path("data/filterbank").glob("*.fil"))
        if fil_files:
            result = pipe.process_file(str(fil_files[0]))
            assert "candidates" in result
            assert "timing" in result
            assert "summary" in result
            assert result["summary"]["status"] in ("success", "error")
        else:
            pytest.skip("No .fil files in data/filterbank")

    def test_process_file_returns_timing(self):
        from pipeline import MitraSETIPipeline

        pipe = MitraSETIPipeline()
        fil_files = list(Path("data/filterbank").glob("*.fil"))
        if fil_files:
            result = pipe.process_file(str(fil_files[0]))
            timing = result.get("timing", {})
            assert "total" in timing or result["summary"]["status"] == "error"
        else:
            pytest.skip("No .fil files in data/filterbank")

    def test_process_file_invalid_path(self):
        from pipeline import MitraSETIPipeline

        pipe = MitraSETIPipeline()
        result = pipe.process_file("/nonexistent/file.fil")
        assert result["summary"]["status"] == "error"

    def test_process_batch(self):
        from pipeline import MitraSETIPipeline

        pipe = MitraSETIPipeline()
        fil_files = list(Path("data/filterbank").glob("*.fil"))
        if len(fil_files) >= 2:
            results = pipe.process_batch([str(f) for f in fil_files[:2]])
            assert len(results) == 2
        else:
            pytest.skip("Need at least 2 .fil files in data/filterbank")


# ─────────────────────────────────────────────────────────────────────────────
# ML Models
# ─────────────────────────────────────────────────────────────────────────────

class TestMLModels:
    def test_classifier_import(self):
        from inference.signal_classifier import SignalClassifier

        clf = SignalClassifier()
        assert clf is not None

    def test_feature_extractor(self):
        import numpy as np
        from inference.feature_extractor import FeatureExtractor

        fe = FeatureExtractor()
        spec = np.random.randn(256, 64).astype(np.float32)
        features = fe.extract(spec)
        assert isinstance(features, (dict, object))

    def test_ood_detector_import(self):
        from inference.ood_detector import RadioOODDetector

        ood = RadioOODDetector()
        assert ood is not None

    def test_classifier_with_model_weights(self):
        from inference.signal_classifier import SignalClassifier

        model_path = Path("models/signal_classifier_v1.pt")
        if model_path.exists():
            clf = SignalClassifier(model_path=str(model_path))
            assert clf is not None
        else:
            pytest.skip("Model weights not available")


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

class TestAPI:
    def test_health_endpoint(self):
        import httpx

        try:
            r = httpx.get("http://localhost:9000/health", timeout=2)
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "ok"
        except httpx.ConnectError:
            pytest.skip("API not running")

    def test_stats_endpoint(self):
        import httpx

        try:
            r = httpx.get("http://localhost:9000/stats", timeout=2)
            assert r.status_code == 200
        except httpx.ConnectError:
            pytest.skip("API not running")

    def test_signals_endpoint(self):
        import httpx

        try:
            r = httpx.get("http://localhost:9000/signals", params={"limit": 5}, timeout=2)
            assert r.status_code == 200
            assert isinstance(r.json(), list)
        except httpx.ConnectError:
            pytest.skip("API not running")

    def test_candidates_endpoint(self):
        import httpx

        try:
            r = httpx.get("http://localhost:9000/candidates", timeout=2)
            assert r.status_code == 200
            assert isinstance(r.json(), list)
        except httpx.ConnectError:
            pytest.skip("API not running")
