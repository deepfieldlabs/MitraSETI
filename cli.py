#!/usr/bin/env python3
"""
MitraSETI Command-Line Interface

Unified CLI for all MitraSETI operations. Designed for both
interactive use and cloud orchestration (Lambda, Batch, Step Functions).

Usage:
    mitraseti search --file observation.h5 --snr 10
    mitraseti stream --hours 1 --mode normal
    mitraseti benchmark --type speed
    mitraseti export --format fits --output results.fits
    mitraseti crossmatch --max-sep 120
    mitraseti report
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click

if "HDF5_PLUGIN_PATH" not in os.environ:
    os.environ["HDF5_PLUGIN_PATH"] = ""

sys.path.insert(0, str(Path(__file__).parent))


@click.group()
@click.version_option("0.2.0", prog_name="MitraSETI")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def cli(verbose: bool):
    """MitraSETI — Rust-accelerated SETI signal analysis pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@cli.command()
@click.option("--file", "-f", "filepath", required=True, help="Filterbank/HDF5 file")
@click.option("--snr", default=10.0, help="Minimum SNR threshold (default: 10)")
@click.option("--max-drift", default=4.0, help="Maximum drift rate Hz/s (default: 4)")
@click.option("--multi-scale", is_flag=True, help="Run multi-scale search (1x, 2x, 4x)")
@click.option("--output", "-o", default=None, help="Output JSON path")
@click.option("--fits", is_flag=True, help="Also export FITS catalog")
def search(filepath: str, snr: float, max_drift: float, multi_scale: bool,
           output: str, fits: bool):
    """Process a single filterbank/HDF5 file."""
    from pipeline import MitraSETIPipeline

    pipe = MitraSETIPipeline()
    result = pipe.process_file(filepath)

    summary = result.get("summary", {})
    click.echo(f"\nProcessed: {Path(filepath).name}")
    click.echo(f"  Raw hits:    {summary.get('total_hits_raw', 0)}")
    click.echo(f"  Filtered:    {summary.get('total_hits_filtered', 0)}")
    click.echo(f"  Candidates:  {summary.get('candidate_count', 0)}")
    click.echo(f"  Anomalies:   {summary.get('anomaly_count', 0)}")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        click.echo(f"  Results → {out_path}")

    if fits:
        from catalog.fits_export import export_candidates_fits
        fits_path = export_candidates_fits(
            result.get("candidates", []),
            result.get("file_info"),
        )
        click.echo(f"  FITS → {fits_path}")

    if multi_scale:
        click.echo("\nRunning multi-scale search...")
        from scripts.multiscale_search import run_multiscale
        ms_result = run_multiscale(filepath, max_drift=max_drift, min_snr=snr)
        for s in [1, 2, 4]:
            sd = ms_result["scales"].get(str(s), {})
            click.echo(f"  {s}×: {sd.get('n_hits', 0)} hits ({sd.get('time_s', 0):.3f}s)")
        click.echo(f"  Broadband-only: {ms_result.get('unique_coarse_scale_detections', 0)}")


@cli.command()
@click.option("--hours", default=1.0, help="Duration in hours (default: 1)")
@click.option("--mode", default="normal", type=click.Choice(["normal", "fast", "deep"]))
def stream(hours: float, mode: str):
    """Run streaming observation engine."""
    import subprocess
    cmd = [
        sys.executable, "scripts/streaming_observation.py",
        "--hours", str(hours), "--mode", mode,
    ]
    click.echo(f"Starting streaming observation ({hours}h, {mode} mode)...")
    subprocess.run(cmd, cwd=str(Path(__file__).parent))


@cli.command()
@click.option("--type", "bench_type", default="speed",
              type=click.Choice(["speed", "injection", "fpr", "completeness", "comparison"]))
@click.option("--repeats", default=3)
def benchmark(bench_type: str, repeats: int):
    """Run performance benchmarks."""
    import subprocess
    scripts = {
        "speed": ["scripts/speed_benchmark.py", "--repeats", str(repeats)],
        "injection": ["scripts/injection_recovery.py"],
        "fpr": ["scripts/fpr_roc_analysis.py"],
        "completeness": ["scripts/completeness_contours.py"],
        "comparison": ["scripts/turboseti_comparison.py"],
    }
    cmd = [sys.executable] + scripts[bench_type]
    click.echo(f"Running {bench_type} benchmark...")
    subprocess.run(cmd, cwd=str(Path(__file__).parent))


@cli.command("export")
@click.option("--format", "fmt", default="fits", type=click.Choice(["fits", "json", "csv"]))
@click.option("--output", "-o", default=None, help="Output path")
@click.option("--source", default="streaming", help="Data source: streaming, file")
def export_cmd(fmt: str, output: str, source: str):
    """Export results in various formats."""
    from paths import CANDIDATES_DIR, STREAMING_STATE

    candidates = []
    if source == "streaming" and STREAMING_STATE.exists():
        with open(STREAMING_STATE) as f:
            state = json.load(f)
            candidates = state.get("candidates", [])

    if not candidates:
        click.echo("No candidates to export.")
        return

    if fmt == "fits":
        from catalog.fits_export import export_candidates_fits
        out = Path(output) if output else None
        path = export_candidates_fits(candidates, output_path=out)
        click.echo(f"FITS catalog → {path}")
    elif fmt == "json":
        out = Path(output) if output else CANDIDATES_DIR / "export.json"
        with open(out, "w") as f:
            json.dump(candidates, f, indent=2, default=str)
        click.echo(f"JSON → {out}")
    elif fmt == "csv":
        import csv
        out = Path(output) if output else CANDIDATES_DIR / "export.csv"
        if candidates:
            keys = list(candidates[0].keys())
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(candidates)
        click.echo(f"CSV → {out}")


@cli.command()
@click.option("--max-sep", default=120.0, help="Max separation in arcseconds (default: 120)")
def crossmatch(max_sep: float):
    """Cross-match MitraSETI radio and AstroLens optical catalogs."""
    from catalog.astropy_crossmatch import (
        crossmatch_radio_optical,
        load_astrolens_candidates,
        load_radio_candidates,
    )

    radio = load_radio_candidates()
    optical = load_astrolens_candidates()

    click.echo(f"Radio candidates:  {len(radio)}")
    click.echo(f"Optical candidates: {len(optical)}")

    if not radio or not optical:
        click.echo("Need both radio and optical data for cross-matching.")
        return

    result = crossmatch_radio_optical(radio, optical, max_sep_arcsec=max_sep)
    click.echo(f"\nMatches found: {result['n_matches']}")
    if result["matches"]:
        click.echo(f"Mean separation: {result['mean_separation_arcsec']}\"")
        for m in result["matches"][:10]:
            click.echo(
                f"  RA={m['radio_ra']:.4f} Dec={m['radio_dec']:.4f} "
                f"sep={m['separation_arcsec']:.1f}\" "
                f"SNR={m['radio_snr']:.1f} OOD={m['optical_ood']:.3f}"
            )


@cli.command()
def report():
    """Generate consolidated publication report."""
    import subprocess
    cmd = [sys.executable, "scripts/generate_publication_report.py"]
    click.echo("Generating publication report...")
    subprocess.run(cmd, cwd=str(Path(__file__).parent))


@cli.command()
def rfi():
    """Show known RFI database summary."""
    from catalog.rfi_database import RFIDatabase
    db = RFIDatabase()
    summary = db.summary()
    click.echo(f"\nKnown RFI Database: {len(db.catalog)} entries\n")
    for cat, count in sorted(summary.items()):
        click.echo(f"  {cat:<25s} {count:3d} entries")


@cli.command()
def persistence():
    """Show signal persistence tracking summary."""
    from catalog.persistence import PersistenceTracker
    tracker = PersistenceTracker()
    sources = tracker.get_all_sources()

    if not sources:
        click.echo("No persistence data recorded yet.")
        return

    click.echo(f"\nPersistence Tracking: {len(sources)} sources\n")
    for name, info in sorted(sources.items()):
        click.echo(
            f"  {name:<30s} "
            f"epochs={info['total_epochs']:3d}  "
            f"signals={info['total_signals']:5d}  "
            f"persistent={info['persistent_signals']:3d}"
        )


@cli.command()
def paths():
    """Show configured artifact paths."""
    from paths import (
        ARTIFACTS_DIR,
        CANDIDATES_DIR,
        DATA_DIR,
        DB_PATH,
        FILTERBANK_DIR,
        MODELS_DIR,
        PROJECT_ROOT,
    )
    click.echo("\nMitraSETI Paths:")
    click.echo(f"  Project root:  {PROJECT_ROOT}")
    click.echo(f"  Artifacts:     {ARTIFACTS_DIR}")
    click.echo(f"  Data:          {DATA_DIR}")
    click.echo(f"  Database:      {DB_PATH}")
    click.echo(f"  Filterbank:    {FILTERBANK_DIR}")
    click.echo(f"  Models:        {MODELS_DIR}")
    click.echo(f"  Candidates:    {CANDIDATES_DIR}")


if __name__ == "__main__":
    cli()
