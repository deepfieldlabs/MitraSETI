#!/usr/bin/env python3
"""
Publication Report Generator — MitraSETI

Generates a comprehensive HTML report suitable for sharing, archiving,
or converting to PDF for journal submission.  Aggregates results from:

  - Streaming observation state
  - Injection/recovery tests
  - Speed benchmarks
  - Cadence filter results
  - Pipeline metrics
  - AstroLens cross-matches

Output:
  mitraseti_artifacts/reports/mitraseti_report_<timestamp>.html

Usage:
  python scripts/generate_publication_report.py
  python scripts/generate_publication_report.py --title "TRAPPIST-1 Analysis"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import ARTIFACTS_DIR, STREAMING_STATE

logger = logging.getLogger("mitraseti.report")

REPORTS_DIR = ARTIFACTS_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _find_latest(directory: Path, pattern: str) -> Optional[Path]:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def collect_data() -> Dict[str, Any]:
    """Collect all available results into a single data dict."""
    data: Dict[str, Any] = {}

    # Streaming state
    state = _load_json(STREAMING_STATE)
    if state:
        data["streaming"] = state

    # Injection/recovery
    inj_dir = ARTIFACTS_DIR / "injection_recovery"
    inj_file = _find_latest(inj_dir, "results_*.json")
    if inj_file:
        data["injection_recovery"] = _load_json(inj_file)

    # Speed benchmarks
    bench_dir = ARTIFACTS_DIR / "benchmarks"
    bench_file = _find_latest(bench_dir, "speed_results_*.json")
    if bench_file:
        data["speed_benchmark"] = _load_json(bench_file)

    # Cadence filter
    cadence_dir = ARTIFACTS_DIR / "cadence"
    cadence_files = sorted(cadence_dir.glob("cadence_*.json"), key=lambda p: p.stat().st_mtime)
    if cadence_files:
        data["cadence"] = [_load_json(f) for f in cadence_files if f.name != "cadence_summary.html"]
        data["cadence"] = [c for c in data["cadence"] if c is not None]

    # Completeness curve image
    comp_img = inj_dir / "completeness_curve.png" if inj_dir.exists() else None
    if comp_img and comp_img.exists():
        import base64
        data["completeness_img_b64"] = base64.b64encode(comp_img.read_bytes()).decode()

    # Speed comparison image
    speed_img = bench_dir / "speed_comparison.png" if bench_dir.exists() else None
    if speed_img and speed_img.exists():
        import base64
        data["speed_img_b64"] = base64.b64encode(speed_img.read_bytes()).decode()

    return data


def generate_report(data: Dict[str, Any], title: str) -> str:
    """Generate the HTML report string."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract key numbers
    streaming = data.get("streaming", {})
    total_obs = streaming.get("observations_completed", 0)
    total_signals = streaming.get("signals_detected", 0)
    total_candidates = streaming.get("verified_candidates", 0)
    elapsed = streaming.get("elapsed_hours", 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', sans-serif; background: #080c14; color: #e0e8f0; padding: 40px; max-width: 1100px; margin: 0 auto; line-height: 1.6; }}
  h1 {{ color: #4da6ff; font-weight: 300; font-size: 36px; letter-spacing: 3px; margin-bottom: 8px; }}
  h2 {{ color: #00d4ff; font-weight: 400; font-size: 20px; margin: 40px 0 16px 0; padding-bottom: 8px; border-bottom: 1px solid rgba(100,180,255,0.15); }}
  h3 {{ color: #e0e8f0; font-weight: 500; font-size: 15px; margin: 20px 0 10px 0; }}
  p {{ color: rgba(200,215,235,0.7); margin-bottom: 12px; }}
  .meta {{ color: rgba(140,165,200,0.4); font-size: 12px; letter-spacing: 1px; margin-bottom: 32px; }}
  .card {{ background: rgba(15,25,45,0.85); border: 1px solid rgba(100,180,255,0.15); border-radius: 16px; padding: 24px; margin: 16px 0; }}
  .stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }}
  .stat {{ flex: 1; min-width: 120px; padding: 16px; background: rgba(15,25,45,0.7); border: 1px solid rgba(100,180,255,0.12); border-radius: 12px; text-align: center; }}
  .stat .val {{ font-size: 28px; color: #00d4ff; font-weight: 300; }}
  .stat .lbl {{ font-size: 10px; color: rgba(140,165,200,0.5); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th {{ text-align: left; padding: 10px 14px; font-size: 10px; color: rgba(0,212,255,0.7); text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid rgba(100,180,255,0.12); }}
  td {{ padding: 8px 14px; font-size: 13px; border-bottom: 1px solid rgba(100,180,255,0.06); }}
  .pass {{ color: #00ff88; }} .fail {{ color: #ff3366; }} .warn {{ color: #ffaa00; }}
  .highlight {{ background: rgba(0,212,255,0.05); border-left: 3px solid #00d4ff; padding: 16px 20px; border-radius: 0 12px 12px 0; margin: 16px 0; }}
  img {{ max-width: 100%; border-radius: 12px; margin: 16px 0; }}
  .footer {{ text-align: center; margin-top: 60px; padding: 20px 0; color: rgba(140,165,200,0.3); font-size: 11px; letter-spacing: 1px; }}
  @media print {{
    body {{ background: white; color: #1a1a2e; padding: 20px; }}
    .card {{ border-color: #ddd; background: #f8f9fa; }}
    .stat {{ border-color: #ddd; background: #f0f0f5; }}
    .stat .val {{ color: #0066cc; }}
    h1 {{ color: #0066cc; }} h2 {{ color: #0088dd; }}
    p {{ color: #333; }}
  }}
</style>
</head>
<body>

<h1>{title}</h1>
<div class="meta">
  Generated: {now} &bull; MitraSETI v0.2.0 &bull; Taylor Tree + CNN-Transformer Architecture
</div>

<h2>1. Observation Summary</h2>
<div class="stats">
  <div class="stat"><div class="val">{total_obs}</div><div class="lbl">Observations</div></div>
  <div class="stat"><div class="val">{total_signals:,}</div><div class="lbl">Signals</div></div>
  <div class="stat"><div class="val">{total_candidates}</div><div class="lbl">Candidates</div></div>
  <div class="stat"><div class="val">{elapsed:.1f}h</div><div class="lbl">Runtime</div></div>
</div>
"""

    # Speed benchmark section
    speed = data.get("speed_benchmark")
    if speed:
        speedups = speed.get("speedups", [])
        html += """<h2>2. Taylor Tree Performance</h2>
<p>Wall-clock comparison of the Taylor tree O(N&middot;F&middot;log&thinsp;N) algorithm
against brute-force O(D&middot;N&middot;F) de-Doppler search across increasing data sizes.</p>
"""
        if "speed_img_b64" in data:
            html += f'<img src="data:image/png;base64,{data["speed_img_b64"]}" alt="Speed comparison">\n'

        if speedups:
            max_s = max(s["speedup_x"] for s in speedups)
            html += f"""<div class="highlight">
  Peak speedup: <strong class="pass">{max_s}&times;</strong> at {speedups[-1]['n_chans']:,} channels
  (N_time = {speed.get('n_times', 16)}, {speed.get('repeats', 3)} trials)
</div>
<table><thead><tr><th>Channels</th><th>Taylor (ms)</th><th>Brute-Force (ms)</th><th>Speedup</th></tr></thead><tbody>
"""
            for s in speedups:
                html += f"<tr><td>{s['n_chans']:,}</td><td>{s['taylor_ms']}</td><td>{s['brute_ms']}</td><td class='pass'>{s['speedup_x']}&times;</td></tr>\n"
            html += "</tbody></table>\n"

    # Injection/recovery section
    inj = data.get("injection_recovery")
    if inj:
        html += """<h2>3. Detection Efficiency (Injection/Recovery)</h2>
<p>Synthetic narrowband drifting signals injected into real Breakthrough Listen data
at controlled SNR and drift rates.  Recovery fraction measures pipeline completeness.</p>
"""
        if "completeness_img_b64" in data:
            html += f'<img src="data:image/png;base64,{data["completeness_img_b64"]}" alt="Completeness curves">\n'

        html += f"""<div class="card">
  <p>Injections per point: <strong>{inj.get('n_injections_per_point', 0)}</strong> &bull;
  Files used: <strong>{len(inj.get('files_used', []))}</strong></p>
</div>
"""

    # Cadence filter section
    cadence = data.get("cadence", [])
    if cadence:
        html += """<h2>4. ON/OFF Cadence Filter</h2>
<p>Standard Breakthrough Listen verification: signals must appear in ON-target scans
but not in OFF-target (reference) scans.  Direction-independent RFI is eliminated.</p>
"""
        total_survivors = sum(r.get("statistics", {}).get("cadence_survivors", 0) for r in cadence)
        total_on = sum(r.get("statistics", {}).get("total_on_signals", 0) for r in cadence)
        html += f"""<div class="stats">
  <div class="stat"><div class="val">{len(cadence)}</div><div class="lbl">Targets</div></div>
  <div class="stat"><div class="val">{total_on:,}</div><div class="lbl">ON Signals</div></div>
  <div class="stat"><div class="val {'pass' if total_survivors else 'fail'}">{total_survivors}</div><div class="lbl">Survivors</div></div>
</div>
"""
        for r in cadence:
            stats = r.get("statistics", {})
            html += f"""<h3>{r.get('target', 'Unknown')}</h3>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
  <tr><td>ON scans</td><td>{r.get('cadence', {}).get('on_scans', 0)}</td></tr>
  <tr><td>OFF scans</td><td>{r.get('cadence', {}).get('off_scans', 0)}</td></tr>
  <tr><td>Total ON signals</td><td>{stats.get('total_on_signals', 0):,}</td></tr>
  <tr><td>Multi-ON consensus</td><td>{stats.get('multi_on_consensus', 0)}</td></tr>
  <tr><td>OFF-matched (RFI)</td><td>{stats.get('off_matched_rfi', 0)}</td></tr>
  <tr><td>Cadence survivors</td><td class="{'pass' if stats.get('cadence_survivors') else 'fail'}">{stats.get('cadence_survivors', 0)}</td></tr>
</tbody></table>
"""
            cands = r.get("candidates", [])
            if cands:
                html += "<table><thead><tr><th>Freq (MHz)</th><th>Drift (Hz/s)</th><th>SNR</th><th>ON Detect</th></tr></thead><tbody>"
                for c in cands[:10]:
                    freq = c.get("frequency_hz", 0) / 1e6 if c.get("frequency_hz", 0) > 1e6 else c.get("frequency_mhz", 0)
                    html += f"""<tr><td>{freq:.6f}</td><td>{c.get('drift_rate', 0):.4f}</td>
<td class="pass">{c.get('snr', 0):.1f}</td>
<td>{c.get('on_detections', '?')}/{c.get('on_scans_total', '?')}</td></tr>"""
                html += "</tbody></table>"

    # Methodology
    html += """<h2>5. Methodology</h2>
<div class="card">
<h3>De-Doppler Search</h3>
<p>Taylor tree algorithm (Taylor 1974) implemented in Rust via PyO3.
Complexity: O(N&middot;F&middot;log&thinsp;N) vs brute-force O(D&middot;N&middot;F).
Recursive partial-sum construction with bit-decomposition for drift channels.</p>

<h3>Signal Classification</h3>
<p>Two-stage pipeline: (1) rule-based triage on all hits (SNR, drift rate, boundary checks),
(2) CNN+Transformer hybrid (1D spectral CNN + 2-layer Transformer encoder)
on surviving candidates. 9-class output with RFI probability estimation.</p>

<h3>Anomaly Detection</h3>
<p>Ensemble OOD detector: MSP + energy-based scoring + spectral distance
to calibrated reference templates. Voting threshold for final anomaly flag.</p>

<h3>Cadence Verification</h3>
<p>Standard ON/OFF direction filter: signals must appear in &ge;2 ON-target scans
and zero OFF-target scans to survive. Frequency tolerance: 50&thinsp;Hz,
drift rate tolerance: 0.5&thinsp;Hz/s.</p>
</div>
"""

    html += f"""
<div class="footer">
  MitraSETI &mdash; Intelligent SETI Signal Analysis<br>
  Deep Field Labs &bull; {now[:10]}
</div>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="MitraSETI Publication Report Generator")
    parser.add_argument("--title", type=str, default="MitraSETI Analysis Report",
                        help="Report title")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Collecting results...")
    data = collect_data()

    sections = []
    if "streaming" in data:
        sections.append("streaming state")
    if "injection_recovery" in data:
        sections.append("injection/recovery")
    if "speed_benchmark" in data:
        sections.append("speed benchmark")
    if "cadence" in data:
        sections.append(f"cadence ({len(data['cadence'])} targets)")
    logger.info(f"  Found: {', '.join(sections) if sections else 'no data yet'}")

    html = generate_report(data, args.title)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = REPORTS_DIR / f"mitraseti_report_{ts}.html"
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Report saved to {output_path}")
    print(f"\nReport: {output_path}")
    print(f"Open in browser: file://{output_path}")


if __name__ == "__main__":
    main()
