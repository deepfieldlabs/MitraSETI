"""Core tests for MitraSETI.

Tests for project structure, imports, the Rust core bindings,
the ML inference layer, and the processing pipeline.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProjectStructure:
    """Verify the project structure is intact."""

    def test_requirements_file_exists(self):
        req_file = Path(__file__).parent.parent / "requirements.txt"
        assert req_file.exists(), "requirements.txt not found"
        content = req_file.read_text()
        assert len(content.strip()) > 0, "requirements.txt is empty"

    def test_requirements_contains_core_deps(self):
        req_file = Path(__file__).parent.parent / "requirements.txt"
        content = req_file.read_text()
        required_packages = ["torch", "numpy", "scipy", "fastapi", "pydantic"]
        for pkg in required_packages:
            assert pkg in content, f"Missing required dependency: {pkg}"

    def test_paths_module(self):
        from paths import (
            PROJECT_ROOT,
        )

        assert PROJECT_ROOT.exists()

    def test_pipeline_module_importable(self):
        from pipeline import MitraSETIPipeline

        assert MitraSETIPipeline is not None


class TestImports:
    """Test that core Python dependencies can be imported."""

    def test_import_numpy(self):
        import numpy as np

        assert np.__version__ is not None

    def test_import_scipy(self):
        import scipy

        assert scipy.__version__ is not None

    def test_import_pydantic(self):
        import pydantic

        assert pydantic.__version__ is not None

    def test_import_torch(self):
        import torch

        assert torch.__version__ is not None


class TestRustCoreIntegration:
    """Tests for the compiled Rust extension module."""

    def test_rust_core_importable(self):
        import mitraseti_core

        assert mitraseti_core is not None

    def test_dedoppler_engine(self):
        import mitraseti_core

        params = mitraseti_core.SearchParams(max_drift_rate=4.0, min_snr=5.0)
        engine = mitraseti_core.DedopplerEngine(params)
        assert engine is not None

    def test_rfi_filter(self):
        import mitraseti_core

        rfi_filter = mitraseti_core.RFIFilter()
        assert rfi_filter is not None

    def test_filterbank_reader(self):
        import mitraseti_core

        reader = mitraseti_core.FilterbankReader()
        assert reader is not None

    def test_filterbank_read_real_file(self):
        import mitraseti_core

        fil_files = list(Path("data/filterbank").glob("*.fil"))
        if not fil_files:
            pytest.skip("No .fil files available")
        reader = mitraseti_core.FilterbankReader()
        header, data, n_times, n_chans = reader.read(str(fil_files[0]))
        assert n_times > 0
        assert n_chans > 0


class TestMLClassifier:
    """Tests for the signal classifier and feature extractor."""

    def test_signal_classifier_init(self):
        from inference.signal_classifier import SignalClassifier

        clf = SignalClassifier()
        assert clf is not None

    def test_feature_extractor_init(self):
        from inference.feature_extractor import FeatureExtractor

        fe = FeatureExtractor()
        assert fe is not None

    def test_feature_extraction_on_synthetic_data(self):
        import numpy as np

        from inference.feature_extractor import FeatureExtractor

        fe = FeatureExtractor()
        spec = np.random.randn(256, 64).astype(np.float32)
        features = fe.extract(spec)
        assert features is not None

    def test_ood_detector_init(self):
        from inference.ood_detector import RadioOODDetector

        ood = RadioOODDetector()
        assert ood is not None


class TestPipelineEndToEnd:
    """End-to-end pipeline tests with real files when available."""

    def test_pipeline_init_default(self):
        from pipeline import MitraSETIPipeline

        pipe = MitraSETIPipeline()
        assert pipe._rust_available

    def test_pipeline_process_returns_structure(self):
        from pipeline import MitraSETIPipeline

        pipe = MitraSETIPipeline()
        fil_files = list(Path("data/filterbank").glob("*.fil"))
        if not fil_files:
            pytest.skip("No .fil files in data/filterbank")
        result = pipe.process_file(str(fil_files[0]))
        assert "file_info" in result
        assert "candidates" in result
        assert "timing" in result
        assert "summary" in result

    def test_pipeline_error_on_missing_file(self):
        from pipeline import MitraSETIPipeline

        pipe = MitraSETIPipeline()
        result = pipe.process_file("/tmp/nonexistent_file_12345.fil")
        assert result["summary"]["status"] == "error"


class TestCatalogModule:
    """Tests for catalog cross-reference."""

    def test_catalog_query_import(self):
        from catalog.radio_catalogs import RadioCatalogQuery

        q = RadioCatalogQuery()
        assert q is not None


class TestAstroLensIntegration:
    """Tests for AstroLens cross-reference."""

    def test_astrolens_crossref_import(self):
        try:
            from catalog.sky_position import astrolens_crossref

            result = astrolens_crossref(ra=180.0, dec=45.0)
            assert isinstance(result, list)
        except ImportError:
            pytest.skip("AstroLens cross-reference module not available")
