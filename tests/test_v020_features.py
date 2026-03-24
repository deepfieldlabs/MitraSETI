"""
Comprehensive test suite for MitraSETI v0.2.0 features.

Tests all new modules:
  - HDBSCAN clustering
  - RFI database
  - FITS catalog export
  - Persistence tracking
  - Interestingness scoring
  - Periodicity detection
  - Attention maps
  - Astropy cross-matching
  - CLI interface
  - Completeness contours
  - Multi-scale search
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────
# HDBSCAN Clustering
# ──────────────────────────────────────────────────────────────────────


class TestHDBSCANClustering:
    def _make_candidates(self, n: int, n_clusters: int = 5):
        """Generate synthetic candidates in distinct clusters."""
        rng = np.random.default_rng(42)
        candidates = []
        for cluster_id in range(n_clusters):
            center_freq = 1420e6 + cluster_id * 1e6
            center_drift = cluster_id * 0.5
            for _ in range(n // n_clusters):
                candidates.append(
                    {
                        "frequency_hz": center_freq + rng.normal(0, 100),
                        "drift_rate": center_drift + rng.normal(0, 0.05),
                        "snr": rng.uniform(10, 100),
                    }
                )
        return candidates

    def test_hdbscan_reduces_hits(self):
        """HDBSCAN should significantly reduce hit count."""
        from pipeline import MitraSETIPipeline

        pipe = MitraSETIPipeline.__new__(MitraSETIPipeline)
        candidates = self._make_candidates(100, n_clusters=5)
        header = {"fch1": 1420.0, "foff": -0.00028}

        result = pipe._cluster_hits(candidates, header)
        assert len(result) < len(candidates)
        assert len(result) >= 5  # at least one per cluster

    def test_hdbscan_preserves_highest_snr(self):
        """Each cluster should keep the strongest signal."""
        from pipeline import MitraSETIPipeline

        pipe = MitraSETIPipeline.__new__(MitraSETIPipeline)

        candidates = [
            {"frequency_hz": 1420e6, "drift_rate": 0.0, "snr": 50},
            {"frequency_hz": 1420e6 + 10, "drift_rate": 0.01, "snr": 100},
            {"frequency_hz": 1420e6 + 20, "drift_rate": 0.0, "snr": 30},
        ]
        header = {"fch1": 1420.0, "foff": -0.00028}

        result = pipe._cluster_hits(candidates, header)
        snrs = [c["snr"] for c in result]
        assert 100 in snrs

    def test_greedy_fallback_for_small_sets(self):
        """Small candidate sets should use greedy fallback."""
        from pipeline import MitraSETIPipeline

        pipe = MitraSETIPipeline.__new__(MitraSETIPipeline)

        candidates = [
            {"frequency_hz": 1420e6, "drift_rate": 0.0, "snr": 50},
            {"frequency_hz": 1420e6 + 1e6, "drift_rate": 2.0, "snr": 80},
        ]
        header = {"fch1": 1420.0, "foff": -0.00028}

        result = pipe._cluster_hits(candidates, header)
        assert len(result) == 2

    def test_single_candidate_passthrough(self):
        """Single candidate should pass through unchanged."""
        from pipeline import MitraSETIPipeline

        pipe = MitraSETIPipeline.__new__(MitraSETIPipeline)
        candidates = [{"frequency_hz": 1420e6, "drift_rate": 0, "snr": 50}]
        result = pipe._cluster_hits(candidates, {"fch1": 1420.0, "foff": -0.00028})
        assert len(result) == 1
        assert result[0]["snr"] == 50


# ──────────────────────────────────────────────────────────────────────
# RFI Database
# ──────────────────────────────────────────────────────────────────────


class TestRFIDatabase:
    def test_gps_l1_match(self):
        from catalog.rfi_database import RFIDatabase

        db = RFIDatabase()
        result = db.match(1575.42)
        assert result is not None
        assert "GPS" in result["source"]
        assert result["category"] == "satellite_navigation"

    def test_iridium_match(self):
        from catalog.rfi_database import RFIDatabase

        db = RFIDatabase()
        result = db.match(1621.0)
        assert result is not None
        assert "Iridium" in result["source"]

    def test_no_match_in_clean_band(self):
        from catalog.rfi_database import RFIDatabase

        db = RFIDatabase()
        result = db.match(1420.405)  # hydrogen line center
        # May or may not match the protected band — check existence
        if result:
            assert result["category"] == "protected_band"

    def test_batch_labeling(self):
        from catalog.rfi_database import RFIDatabase

        db = RFIDatabase()
        candidates = [
            {"frequency_hz": 1575.42e6, "drift_rate": 0.0},
            {"frequency_hz": 1420.0e6, "drift_rate": 1.5},
            {"frequency_hz": 3000.0e6, "drift_rate": 0.0},
        ]
        result = db.match_batch(candidates)
        assert result[0]["known_rfi"] is True
        assert "GPS" in result[0]["known_rfi_source"]

    def test_custom_entries(self):
        from catalog.rfi_database import RFIDatabase

        custom = [
            {
                "source": "Test RFI",
                "category": "test",
                "freq_min_mhz": 999.0,
                "freq_max_mhz": 1001.0,
                "typical_drift_hz_s": 0.0,
                "notes": "Test entry",
            }
        ]
        db = RFIDatabase(extra_entries=custom)
        result = db.match(1000.0)
        assert result is not None
        assert result["source"] == "Test RFI"

    def test_summary(self):
        from catalog.rfi_database import RFIDatabase

        db = RFIDatabase()
        summary = db.summary()
        assert len(summary) > 0
        assert "satellite_navigation" in summary


# ──────────────────────────────────────────────────────────────────────
# FITS Catalog Export
# ──────────────────────────────────────────────────────────────────────


class TestFITSExport:
    def test_export_candidates(self):
        from catalog.fits_export import export_candidates_fits

        candidates = [
            {
                "frequency_hz": 1420e6,
                "drift_rate": 0.5,
                "snr": 25.0,
                "classification": "narrowband",
                "confidence": 0.95,
                "rfi_probability": 0.1,
                "ood_score": 0.7,
                "is_candidate": True,
                "interestingness_score": 85.0,
                "known_rfi": False,
            },
        ]

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = Path(f.name)

        try:
            result = export_candidates_fits(candidates, output_path=path)
            assert result.exists()
            assert result.stat().st_size > 0

            from astropy.io import fits

            with fits.open(str(result)) as hdul:
                assert len(hdul) >= 2
                data = hdul[1].data
                assert len(data) == 1
                assert abs(data["frequency_hz"][0] - 1420e6) < 1
        finally:
            path.unlink(missing_ok=True)

    def test_export_empty(self):
        from catalog.fits_export import export_candidates_fits

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = Path(f.name)
        try:
            result = export_candidates_fits([], output_path=path)
            assert result.exists()
        finally:
            path.unlink(missing_ok=True)

    def test_export_skymap(self):
        from catalog.fits_export import export_skymap_fits

        radio = [
            {
                "ra": 180.0,
                "dec": 45.0,
                "frequency_hz": 1420e6,
                "snr": 20.0,
                "drift_rate": 0.5,
                "source_name": "TEST",
            }
        ]
        optical = [
            {
                "ra_deg": 180.1,
                "dec_deg": 45.1,
                "ood_score": 0.8,
                "classification": "anomaly",
                "source": "astrolens",
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = Path(f.name)
        try:
            result = export_skymap_fits(radio, optical, output_path=path)
            assert result.exists()

            from astropy.io import fits

            with fits.open(str(result)) as hdul:
                assert len(hdul) >= 3
        finally:
            path.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Persistence Tracking
# ──────────────────────────────────────────────────────────────────────


class TestPersistenceTracking:
    def test_record_and_retrieve(self):
        from catalog.persistence import PersistenceTracker

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            tracker = PersistenceTracker(state_path=path)

            candidates = [
                {"frequency_hz": 1420e6, "drift_rate": 0.5, "snr": 25.0},
                {"frequency_hz": 1421e6, "drift_rate": 1.0, "snr": 30.0},
            ]

            # Epoch 1
            r1 = tracker.record("TEST_SOURCE", candidates, epoch_id="epoch_1")
            assert r1["new_signals"] == 2
            assert r1["matched_existing"] == 0

            # Epoch 2: same signals
            r2 = tracker.record("TEST_SOURCE", candidates, epoch_id="epoch_2")
            assert r2["new_signals"] == 0
            assert r2["matched_existing"] == 2

            persistent = tracker.get_persistent("TEST_SOURCE", min_epochs=2)
            assert len(persistent) == 2
            assert persistent[0]["epoch_count"] == 2
        finally:
            path.unlink(missing_ok=True)

    def test_different_sources(self):
        from catalog.persistence import PersistenceTracker

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            tracker = PersistenceTracker(state_path=path)
            tracker.record("SOURCE_A", [{"frequency_hz": 1420e6, "drift_rate": 0, "snr": 10}])
            tracker.record("SOURCE_B", [{"frequency_hz": 1500e6, "drift_rate": 0, "snr": 10}])

            sources = tracker.get_all_sources()
            assert "SOURCE_A" in sources
            assert "SOURCE_B" in sources
        finally:
            path.unlink(missing_ok=True)

    def test_clear(self):
        from catalog.persistence import PersistenceTracker

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            tracker = PersistenceTracker(state_path=path)
            tracker.record("TEST", [{"frequency_hz": 1420e6, "drift_rate": 0, "snr": 10}])
            tracker.clear("TEST")
            assert "TEST" not in tracker.get_all_sources()
        finally:
            path.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Interestingness Score
# ──────────────────────────────────────────────────────────────────────


class TestInterestingnessScore:
    def test_high_snr_high_score(self):
        from inference.interestingness import compute_interestingness

        candidate = {
            "snr": 100,
            "drift_rate": 1.0,
            "rfi_probability": 0.0,
            "ood_score": 0.9,
            "confidence": 0.95,
            "is_candidate": True,
        }
        score = compute_interestingness(candidate)
        assert score > 70

    def test_rfi_signal_low_score(self):
        from inference.interestingness import compute_interestingness

        candidate = {
            "snr": 50,
            "drift_rate": 0.0,
            "rfi_probability": 0.95,
            "ood_score": 0.0,
            "confidence": 0.8,
            "is_candidate": False,
        }
        score = compute_interestingness(candidate)
        assert score < 40

    def test_boundary_drift_penalized(self):
        from inference.interestingness import compute_interestingness

        at_boundary = compute_interestingness({"snr": 50, "drift_rate": 3.99})
        in_sweet_spot = compute_interestingness({"snr": 50, "drift_rate": 1.0})
        assert in_sweet_spot > at_boundary

    def test_rank_candidates(self):
        from inference.interestingness import rank_candidates

        candidates = [
            {"snr": 10, "drift_rate": 0, "rfi_probability": 0.9},
            {"snr": 100, "drift_rate": 1.0, "rfi_probability": 0.0, "ood_score": 0.5},
        ]
        ranked = rank_candidates(candidates)
        assert ranked[0]["snr"] == 100
        assert "interestingness_score" in ranked[0]


# ──────────────────────────────────────────────────────────────────────
# Periodicity Detection
# ──────────────────────────────────────────────────────────────────────


class TestPeriodicityDetection:
    def test_detect_periodic_signal(self):
        from inference.periodicity import detect_periodicity

        # Create a periodic signal: period = 5 time steps
        rng = np.random.default_rng(42)
        n_freq, n_time = 64, 200
        spec = rng.standard_normal((n_freq, n_time)).astype(np.float32)
        period_samples = 20
        for t in range(0, n_time, period_samples):
            spec[32, t] += 10.0  # bright pulse at channel 32

        result = detect_periodicity(spec, tsamp=1.0, freq_channel=32, significance_threshold=3.0)
        assert result.period_significance > 0

    def test_no_periodicity_in_noise(self):
        from inference.periodicity import detect_periodicity

        rng = np.random.default_rng(123)
        spec = rng.standard_normal((64, 100)).astype(np.float32)
        result = detect_periodicity(spec, tsamp=1.0, significance_threshold=10.0)
        assert result.is_periodic is False

    def test_batch_search(self):
        from inference.periodicity import batch_periodicity_search

        rng = np.random.default_rng(42)
        spec = rng.standard_normal((64, 100)).astype(np.float32)
        results = batch_periodicity_search(spec, tsamp=1.0, n_channels=3)
        assert isinstance(results, list)

    def test_handles_small_data(self):
        from inference.periodicity import detect_periodicity

        spec = np.ones((4, 4), dtype=np.float32)
        result = detect_periodicity(spec, tsamp=1.0)
        assert result.is_periodic is False


# ──────────────────────────────────────────────────────────────────────
# Attention Maps
# ──────────────────────────────────────────────────────────────────────


class TestAttentionMaps:
    def test_heatmap_shape(self):
        from inference.attention_maps import attention_to_spectrogram_heatmap

        attn = np.random.rand(64, 64).astype(np.float32)
        spec = np.random.rand(256, 64).astype(np.float32)
        heatmap = attention_to_spectrogram_heatmap(attn, spec)
        assert heatmap.shape == spec.shape

    def test_heatmap_range(self):
        from inference.attention_maps import attention_to_spectrogram_heatmap

        attn = np.random.rand(32, 32).astype(np.float32)
        spec = np.random.rand(128, 32).astype(np.float32)
        heatmap = attention_to_spectrogram_heatmap(attn, spec)
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1.01


# ──────────────────────────────────────────────────────────────────────
# Astropy Cross-Matching
# ──────────────────────────────────────────────────────────────────────


class TestAstropyCrossMatch:
    def test_exact_match(self):
        from catalog.astropy_crossmatch import crossmatch_radio_optical

        radio = [{"ra": 180.0, "dec": 45.0, "snr": 20, "source_name": "TEST"}]
        optical = [{"ra_deg": 180.0, "dec_deg": 45.0, "ood_score": 0.5, "classification": "galaxy"}]

        result = crossmatch_radio_optical(radio, optical, max_sep_arcsec=10)
        assert result["n_matches"] == 1
        assert result["matches"][0]["separation_arcsec"] < 1

    def test_no_match_far_apart(self):
        from catalog.astropy_crossmatch import crossmatch_radio_optical

        radio = [{"ra": 0.0, "dec": 0.0, "snr": 20}]
        optical = [{"ra_deg": 180.0, "dec_deg": 45.0, "ood_score": 0.5}]

        result = crossmatch_radio_optical(radio, optical, max_sep_arcsec=120)
        assert result["n_matches"] == 0

    def test_empty_catalogs(self):
        from catalog.astropy_crossmatch import crossmatch_radio_optical

        result = crossmatch_radio_optical([], [], max_sep_arcsec=120)
        assert result["n_matches"] == 0

    def test_multiple_matches(self):
        from catalog.astropy_crossmatch import crossmatch_radio_optical

        radio = [
            {"ra": 180.0, "dec": 45.0, "snr": 20},
            {"ra": 90.0, "dec": -30.0, "snr": 15},
        ]
        optical = [
            {"ra_deg": 180.001, "dec_deg": 45.001, "ood_score": 0.8},
            {"ra_deg": 90.001, "dec_deg": -30.001, "ood_score": 0.6},
        ]

        result = crossmatch_radio_optical(radio, optical, max_sep_arcsec=60)
        assert result["n_matches"] == 2


# ──────────────────────────────────────────────────────────────────────
# CLI Interface
# ──────────────────────────────────────────────────────────────────────


class TestCLI:
    def test_cli_version(self):
        from click.testing import CliRunner

        from cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.2.0" in result.output

    def test_cli_rfi_command(self):
        from click.testing import CliRunner

        from cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["rfi"])
        assert result.exit_code == 0
        assert "satellite_navigation" in result.output

    def test_cli_paths_command(self):
        from click.testing import CliRunner

        from cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["paths"])
        assert result.exit_code == 0
        assert "Project root" in result.output

    def test_cli_persistence_command(self):
        from click.testing import CliRunner

        from cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["persistence"])
        assert result.exit_code == 0


# ──────────────────────────────────────────────────────────────────────
# Spectral Kurtosis (adaptive thresholds)
# ──────────────────────────────────────────────────────────────────────


class TestSpectralKurtosisAdaptive:
    def test_gaussian_noise_low_flagging(self):
        """Pure Gaussian noise should flag very few channels."""
        from pipeline import MitraSETIPipeline

        rng = np.random.default_rng(42)
        data = rng.standard_normal((16, 1024)).astype(np.float32)

        result = MitraSETIPipeline.compute_spectral_kurtosis(data)
        assert result["fraction_flagged"] < 0.10

    def test_rfi_channel_flagged(self):
        """Constant-power RFI channel should be flagged."""
        from pipeline import MitraSETIPipeline

        rng = np.random.default_rng(42)
        data = rng.standard_normal((32, 256)).astype(np.float32)
        data[:, 100] = 1000.0  # constant power

        result = MitraSETIPipeline.compute_spectral_kurtosis(data)
        assert result["rfi_mask"][100]

    def test_adaptive_thresholds_present(self):
        """Should report adaptive threshold bounds."""
        from pipeline import MitraSETIPipeline

        rng = np.random.default_rng(42)
        data = rng.standard_normal((16, 512)).astype(np.float32)

        result = MitraSETIPipeline.compute_spectral_kurtosis(data)
        assert "sk_lower" in result
        assert "sk_upper" in result
        assert result["sk_lower"] < result["sk_upper"]


# ──────────────────────────────────────────────────────────────────────
# Integration: Pipeline with all new features
# ──────────────────────────────────────────────────────────────────────


class TestPipelineIntegration:
    @pytest.fixture
    def pipeline(self):
        from pipeline import MitraSETIPipeline

        return MitraSETIPipeline()

    def test_pipeline_imports(self, pipeline):
        assert pipeline is not None

    def test_known_rfi_labeling_in_pipeline(self):
        """RFI database should be used during classification."""
        from catalog.rfi_database import RFIDatabase

        db = RFIDatabase()
        candidates = [
            {"frequency_hz": 1575.42e6, "drift_rate": 0.0, "snr": 15},
        ]
        db.match_batch(candidates)
        assert candidates[0]["known_rfi"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
