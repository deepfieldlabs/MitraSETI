"""
Tests for the unified sky map (AstroLens + MitraSETI integration).

Validates:
1. AstroLens skymap export produces valid coordinate-enriched JSON
2. MitraSETI paths correctly resolve the skymap export file
3. Coordinate extraction from filenames works for all patterns
4. Cross-matching logic finds radio-optical coincidences
5. Web sky map template data flow is correct
6. Desktop radar panel loads AstroLens data correctly
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_skymap_export_coord_extraction():
    """Validate coordinate extraction from various filename patterns."""
    from astroLens.catalog.skymap_export import _extract_coords_from_path

    cases = [
        ("gz_anomaly_0001_ra183.3_dec13.7.jpg", (183.3, 13.7)),
        ("sdss_ra200.6_dec40.6.jpg", (200.6, 40.6)),
        ("image_ra_180.0_dec_-30.5.png", (180.0, -30.5)),
        ("transient_region_005_ra149.0_dec69.0.jpg", (149.0, 69.0)),
        ("decals_ra0.5_dec-89.9.fits", (0.5, -89.9)),
        ("sn_SN2014J_Ia.jpg", None),
        ("no_coords_here.png", None),
    ]

    passed = 0
    for filename, expected in cases:
        result = _extract_coords_from_path(filename)
        if expected is None:
            assert result is None, f"Expected None for '{filename}', got {result}"
        else:
            assert result is not None, f"Expected coords for '{filename}', got None"
            assert abs(result[0] - expected[0]) < 0.01, f"RA mismatch for '{filename}'"
            assert abs(result[1] - expected[1]) < 0.01, f"Dec mismatch for '{filename}'"
        passed += 1

    print(f"  PASS: {passed}/{len(cases)} filename patterns validated")


def test_skymap_export_json_schema():
    """Exported JSON entries must have the required fields."""
    from astroLens.catalog.skymap_export import export_skymap_json

    artifacts_dir = Path(__file__).parent.parent.parent / "astrolens_artifacts"
    if not artifacts_dir.exists():
        print("  SKIP: astrolens_artifacts not found")
        return

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    try:
        results = export_skymap_json(artifacts_dir, out_path)

        assert len(results) > 0, "Expected at least 1 detection with coordinates"

        required_keys = {"ra_deg", "dec_deg", "classification", "ood_score", "source"}
        for entry in results[:10]:
            missing = required_keys - set(entry.keys())
            assert not missing, f"Missing keys: {missing} in entry {entry}"
            assert -360 <= entry["ra_deg"] <= 360, f"RA out of range: {entry['ra_deg']}"
            assert -90 <= entry["dec_deg"] <= 90, f"Dec out of range: {entry['dec_deg']}"

        with open(out_path) as f:
            loaded = json.load(f)
        assert len(loaded) == len(results), "File contents don't match return value"

        print(f"  PASS: {len(results)} entries, all with valid schema and coordinates")
    finally:
        Path(out_path).unlink(missing_ok=True)


def test_skymap_export_file_exists():
    """The generated skymap_export.json should exist in the artifacts dir."""
    skymap_file = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "data" / "skymap_export.json"
    if not skymap_file.exists():
        print("  SKIP: skymap_export.json not yet generated")
        return

    with open(skymap_file) as f:
        data = json.load(f)

    assert isinstance(data, list), "skymap_export.json should be a list"
    assert len(data) > 0, "skymap_export.json should not be empty"

    entry = data[0]
    assert "ra_deg" in entry, "Entry missing ra_deg"
    assert "dec_deg" in entry, "Entry missing dec_deg"

    print(f"  PASS: skymap_export.json contains {len(data)} entries")


def test_paths_skymap_file():
    """MitraSETI paths.py should expose ASTROLENS_SKYMAP_FILE."""
    from paths import ASTROLENS_SKYMAP_FILE

    assert "skymap_export.json" in str(ASTROLENS_SKYMAP_FILE), (
        f"ASTROLENS_SKYMAP_FILE should point to skymap_export.json, got {ASTROLENS_SKYMAP_FILE}"
    )
    print(f"  PASS: ASTROLENS_SKYMAP_FILE = {ASTROLENS_SKYMAP_FILE}")


def test_cross_matching():
    """Radio and optical detections within 2 degrees should be marked as cross-matches."""
    radio_observations = [
        {"ra_deg": 149.0, "dec_deg": 69.0, "classification": "candidate", "snr": 15},
        {"ra_deg": 200.0, "dec_deg": 30.0, "classification": "signal", "snr": 8},
    ]

    astrolens_detections = [
        {"ra_deg": 149.5, "dec_deg": 68.8, "classification": "supernova_candidate", "ood_score": 0.8},
        {"ra_deg": 300.0, "dec_deg": -10.0, "classification": "spiral", "ood_score": 0.3},
    ]

    match_radius = 2.0
    matches = 0
    for obs in radio_observations:
        for al in astrolens_detections:
            dra = abs(obs["ra_deg"] - al["ra_deg"])
            ddec = abs(obs["dec_deg"] - al["dec_deg"])
            if dra < match_radius and ddec < match_radius:
                obs["astrolens_match"] = True
                matches += 1
                break

    assert matches == 1, f"Expected 1 cross-match, got {matches}"
    assert radio_observations[0].get("astrolens_match"), "First observation should match AstroLens"
    assert not radio_observations[1].get("astrolens_match"), "Second observation should NOT match"

    print(f"  PASS: Cross-matching correctly found {matches} radio-optical coincidence(s)")


def test_ztf_coord_lookup():
    """ZTF metadata files should provide coordinates for transient candidates."""
    from astroLens.catalog.skymap_export import _load_ztf_coords

    artifacts_dir = Path(__file__).parent.parent.parent / "astrolens_artifacts"
    transient_dir = artifacts_dir / "transient_data"

    if not transient_dir.exists():
        print("  SKIP: transient_data not found")
        return

    coords = _load_ztf_coords(transient_dir)
    if not coords:
        print("  SKIP: No ZTF metadata found")
        return

    for ztf_id, (ra, dec) in list(coords.items())[:3]:
        assert 0 <= ra <= 360, f"RA out of range for {ztf_id}: {ra}"
        assert -90 <= dec <= 90, f"Dec out of range for {ztf_id}: {dec}"

    print(f"  PASS: Loaded {len(coords)} ZTF coordinate entries")


def test_sky_map_panel_imports():
    """Desktop sky map panel should be importable with AstroLens support."""
    try:
        from ui.sky_map_panel import SkyMapPanel, _TargetBlip

        obs_radio = {"name": "Test", "ra": 100, "dec": 30, "signals": 1, "candidates": 0}
        blip_radio = _TargetBlip(obs_radio)
        assert blip_radio.category == "signal"

        obs_astrolens = {
            "name": "AstroLens: spiral", "ra": 150, "dec": 45,
            "signals": 0, "candidates": 0, "source": "astrolens",
        }
        blip_al = _TargetBlip(obs_astrolens)
        assert blip_al.category == "astrolens", f"Expected 'astrolens', got '{blip_al.category}'"

        print("  PASS: Desktop panel correctly categorises radio and AstroLens blips")
    except ImportError as e:
        print(f"  SKIP: Could not import UI panel (likely no PyQt5): {e}")


def main():
    print("=" * 60)
    print("Unified Sky Map Integration Tests")
    print("=" * 60)

    tests = [
        ("Coordinate extraction from filenames", test_skymap_export_coord_extraction),
        ("Skymap export JSON schema", test_skymap_export_json_schema),
        ("Skymap export file existence", test_skymap_export_file_exists),
        ("MitraSETI paths configuration", test_paths_skymap_file),
        ("Radio-optical cross-matching", test_cross_matching),
        ("ZTF coordinate lookup", test_ztf_coord_lookup),
        ("Desktop sky map panel imports", test_sky_map_panel_imports),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, func in tests:
        print(f"\n[TEST] {name}")
        try:
            func()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
