#!/usr/bin/env python3
"""
astroSETI Streaming Report Generator

Generates self-contained HTML reports with embedded charts for daily and
final observation summaries.  Uses matplotlib for chart generation,
embedded as base64 in standalone HTML.

Adapted from AstroLens streaming_report.py for the radio SETI domain.

Reports include:
- Signal count over days
- Candidate count over days
- RFI rejection rate trend
- Processing speed trend
- Classification distribution pie
- Top candidates table
- AstroLens cross-reference highlights

Charts use a dark crystalline theme matching the astroSETI aesthetic.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Chart generation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Crystalline dark palette
_BG_DARK = "#0a0e17"
_BG_CARD = "#111827"
_BORDER = "#1e293b"
_TEXT_PRIMARY = "#e2e8f0"
_TEXT_SECONDARY = "#94a3b8"
_ACCENT_BLUE = "#38bdf8"
_ACCENT_GREEN = "#34d399"
_ACCENT_RED = "#f87171"
_ACCENT_AMBER = "#fbbf24"
_ACCENT_PURPLE = "#a78bfa"
_ACCENT_CYAN = "#22d3ee"
_GRID_COLOR = "#1e293b"


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=120, bbox_inches="tight",
        facecolor=_BG_DARK, edgecolor="none",
    )
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


def _style_axis(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply crystalline dark styling to a matplotlib axis."""
    ax.set_facecolor(_BG_CARD)
    if title:
        ax.set_title(title, color=_TEXT_PRIMARY, fontsize=13, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=_TEXT_SECONDARY, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=_TEXT_SECONDARY, fontsize=11)
    ax.tick_params(colors=_TEXT_SECONDARY)
    ax.spines["bottom"].set_color(_BORDER)
    ax.spines["left"].set_color(_BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color=_GRID_COLOR)


def _chart_img(b64: str, alt: str) -> str:
    """Wrap a base64 chart image in HTML."""
    if b64:
        return (
            f'<div class="chart-container">'
            f'<img src="data:image/png;base64,{b64}" alt="{alt}">'
            f'</div>'
        )
    return '<div class="card">Chart unavailable (install matplotlib)</div>'


# ── Signal count chart ────────────────────────────────────────────────────

def _create_signal_count_chart(snapshots: List[dict]) -> str:
    """Signal count over days (bar + cumulative line)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = [s.get("day", i + 1) for i, s in enumerate(snapshots)]
        signals = [s.get("signals_found", 0) for s in snapshots]

        cumulative = []
        total = 0
        for s in signals:
            total += s
            cumulative.append(total)

        fig, ax = plt.subplots(figsize=(8, 4), facecolor=_BG_DARK)
        _style_axis(ax, "Signals Detected Over Days", "Day", "Signals")

        ax.bar(days, signals, color=_ACCENT_BLUE, alpha=0.7, label="Daily")
        ax.plot(days, cumulative, "o-", color=_ACCENT_AMBER, linewidth=2,
                markersize=5, label="Cumulative")
        ax.legend(
            facecolor=_BG_CARD, edgecolor=_BORDER,
            labelcolor=_TEXT_PRIMARY, fontsize=10,
        )

        fig.tight_layout()
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64
    except ImportError:
        return ""


# ── Candidate count chart ────────────────────────────────────────────────

def _create_candidate_chart(snapshots: List[dict]) -> str:
    """Candidate count over days."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = [s.get("day", i + 1) for i, s in enumerate(snapshots)]
        candidates = [s.get("candidates_found", 0) for s in snapshots]

        cumulative = []
        total = 0
        for c in candidates:
            total += c
            cumulative.append(total)

        fig, ax = plt.subplots(figsize=(8, 4), facecolor=_BG_DARK)
        _style_axis(ax, "ET Candidates Over Days", "Day", "Candidates")

        ax.bar(days, candidates, color=_ACCENT_GREEN, alpha=0.7, label="Daily")
        ax.plot(days, cumulative, "o-", color=_ACCENT_CYAN, linewidth=2,
                markersize=5, label="Cumulative")
        ax.legend(
            facecolor=_BG_CARD, edgecolor=_BORDER,
            labelcolor=_TEXT_PRIMARY, fontsize=10,
        )

        fig.tight_layout()
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64
    except ImportError:
        return ""


# ── RFI rejection rate ───────────────────────────────────────────────────

def _create_rfi_chart(snapshots: List[dict]) -> str:
    """RFI rejection rate trend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = [s.get("day", i + 1) for i, s in enumerate(snapshots)]
        rfi_rates = []
        for s in snapshots:
            signals = s.get("signals_found", 0)
            rfi = s.get("rfi_rejected", 0)
            rate = (rfi / signals * 100) if signals > 0 else 0
            rfi_rates.append(rate)

        fig, ax = plt.subplots(figsize=(8, 4), facecolor=_BG_DARK)
        _style_axis(ax, "RFI Rejection Rate", "Day", "RFI %")

        ax.plot(days, rfi_rates, "o-", color=_ACCENT_RED, linewidth=2, markersize=6)
        ax.fill_between(days, rfi_rates, alpha=0.1, color=_ACCENT_RED)

        fig.tight_layout()
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64
    except ImportError:
        return ""


# ── Processing speed ─────────────────────────────────────────────────────

def _create_speed_chart(snapshots: List[dict]) -> str:
    """Processing speed trend (files/hour)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = [s.get("day", i + 1) for i, s in enumerate(snapshots)]
        fph = [s.get("files_per_hour", 0) for s in snapshots]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor=_BG_DARK)
        _style_axis(ax, "Processing Speed", "Day", "Files / Hour")

        ax.plot(days, fph, "o-", color=_ACCENT_CYAN, linewidth=2, markersize=6)
        ax.fill_between(days, fph, alpha=0.1, color=_ACCENT_CYAN)

        fig.tight_layout()
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64
    except ImportError:
        return ""


# ── Classification pie chart ─────────────────────────────────────────────

def _create_classification_pie(snapshots: List[dict]) -> str:
    """Classification distribution pie chart (aggregated across all days)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Aggregate classification counts across all snapshots
        agg: Dict[str, int] = {}
        for s in snapshots:
            for sig_type, count in s.get("classification_counts", {}).items():
                agg[sig_type] = agg.get(sig_type, 0) + count

        if not agg:
            return ""

        labels = list(agg.keys())
        values = list(agg.values())

        colors = [
            _ACCENT_BLUE, _ACCENT_GREEN, _ACCENT_RED, _ACCENT_AMBER,
            _ACCENT_PURPLE, _ACCENT_CYAN, "#f472b6", "#818cf8",
            "#a3e635",
        ]

        fig, ax = plt.subplots(figsize=(6, 6), facecolor=_BG_DARK)
        ax.set_facecolor(_BG_DARK)

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors[: len(labels)],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": _TEXT_PRIMARY, "fontsize": 10},
        )
        for t in autotexts:
            t.set_fontsize(9)
            t.set_color(_TEXT_SECONDARY)

        ax.set_title(
            "Signal Classification Distribution",
            color=_TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=16,
        )

        fig.tight_layout()
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64
    except ImportError:
        return ""


# ── Sensitivity (SNR threshold) evolution ────────────────────────────────

def _create_sensitivity_chart(snapshots: List[dict]) -> str:
    """SNR threshold evolution over days."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = [s.get("day", i + 1) for i, s in enumerate(snapshots)]
        starts = [s.get("sensitivity_start", 10.0) for s in snapshots]
        ends = [s.get("sensitivity_end", 10.0) for s in snapshots]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor=_BG_DARK)
        _style_axis(ax, "Sensitivity Evolution (min SNR)", "Day", "SNR Threshold")

        ax.plot(days, ends, "o-", color=_ACCENT_PURPLE, linewidth=2, markersize=6,
                label="End-of-day threshold")
        ax.fill_between(days, starts, ends, alpha=0.15, color=_ACCENT_PURPLE)

        ax.legend(
            facecolor=_BG_CARD, edgecolor=_BORDER,
            labelcolor=_TEXT_PRIMARY, fontsize=10,
        )

        fig.tight_layout()
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64
    except ImportError:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# CSS – crystalline dark theme
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Helvetica, Arial, sans-serif;
    background: #0a0e17;
    color: #e2e8f0;
    margin: 0;
    padding: 24px;
    line-height: 1.6;
}
.container { max-width: 1100px; margin: 0 auto; }
h1 {
    color: #38bdf8;
    font-size: 28px;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}
h2 {
    color: #e2e8f0;
    font-size: 20px;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 8px;
    margin-top: 36px;
}
h3 { color: #94a3b8; font-size: 16px; }
.subtitle { color: #94a3b8; font-size: 14px; margin-bottom: 24px; }
.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
}
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.stat-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.stat-value {
    font-size: 28px;
    font-weight: 700;
    color: #38bdf8;
}
.stat-label {
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stat-card.highlight .stat-value { color: #34d399; }
.stat-card.warning .stat-value { color: #fbbf24; }
.stat-card.danger .stat-value { color: #f87171; }
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
}
th, td {
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid #1e293b;
}
th {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
td { color: #e2e8f0; font-size: 13px; }
tr:hover td { background: rgba(56, 189, 248, 0.04); }
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}
.tag-candidate { background: rgba(52, 211, 153, 0.2); color: #34d399; }
.tag-rfi { background: rgba(248, 113, 113, 0.2); color: #f87171; }
.tag-signal { background: rgba(56, 189, 248, 0.2); color: #38bdf8; }
.tag-ood { background: rgba(167, 139, 250, 0.2); color: #a78bfa; }
.chart-container { margin: 16px 0; text-align: center; }
.chart-container img { max-width: 100%; border-radius: 8px; }
.correction-item {
    padding: 8px 12px;
    margin: 4px 0;
    background: rgba(251, 191, 36, 0.08);
    border-left: 3px solid #fbbf24;
    border-radius: 0 6px 6px 0;
    font-size: 13px;
}
.summary-box {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.06), rgba(52, 211, 153, 0.06));
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 24px;
    margin: 24px 0;
}
.summary-box h3 { color: #38bdf8; margin-top: 0; }
.footer {
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #1e293b;
    text-align: center;
    color: #475569;
    font-size: 12px;
}
.footer a { color: #38bdf8; text-decoration: none; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_daily_report(
    day_number: int,
    snapshot: dict,
    artifacts_dir,
    streaming_state: Optional[dict] = None,
) -> str:
    """
    Generate a daily HTML report with charts.

    Args:
        day_number: Current day number.
        snapshot: Latest DailySnapshot as dict.
        artifacts_dir: Directory to save the report.
        streaming_state: Full StreamingState dict (for trend charts).

    Returns:
        Path to the generated HTML file.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    state = streaming_state or {}
    snapshots = state.get("daily_snapshots", [snapshot] if snapshot else [])
    target_days = state.get("target_days", 7)
    mode = state.get("current_mode", "normal")
    runtime = state.get("total_runtime_hours", 0)

    # Generate charts
    signal_chart = _create_signal_count_chart(snapshots) if snapshots else ""
    candidate_chart = _create_candidate_chart(snapshots) if snapshots else ""
    rfi_chart = _create_rfi_chart(snapshots) if snapshots else ""
    speed_chart = _create_speed_chart(snapshots) if snapshots else ""
    pie_chart = _create_classification_pie(snapshots) if snapshots else ""
    sensitivity_chart = _create_sensitivity_chart(snapshots) if snapshots else ""

    # Cumulative stats
    total_files = state.get("total_files_processed", snapshot.get("files_processed", 0))
    total_signals = state.get("total_signals", snapshot.get("signals_found", 0))
    total_candidates = state.get("total_candidates", snapshot.get("candidates_found", 0))
    total_rfi = state.get("total_rfi_rejected", snapshot.get("rfi_rejected", 0))
    total_ood = state.get("total_ood_anomalies", snapshot.get("ood_anomalies", 0))

    # Today's stats
    day_signals = snapshot.get("signals_found", 0)
    day_candidates = snapshot.get("candidates_found", 0)
    day_rfi = snapshot.get("rfi_rejected", 0)
    day_rate = snapshot.get("candidate_rate", 0)

    # Top candidates table
    candidates_rows = ""
    for i, c in enumerate(snapshot.get("top_candidates", [])[:10], 1):
        candidates_rows += f"""
        <tr>
            <td>{i}</td>
            <td><strong>{c.get('ood_score', 0):.4f}</strong></td>
            <td>{c.get('signal_type', '?')}</td>
            <td>{c.get('snr', 0):.1f}</td>
            <td>{c.get('drift_rate', 0):.4f}</td>
            <td>{c.get('confidence', 0):.1%}</td>
            <td>{Path(c.get('file', '')).name if c.get('file') else '?'}</td>
        </tr>"""

    # Corrections
    corrections_html = ""
    for snap in snapshots:
        for c in snap.get("corrections_applied", []):
            corrections_html += (
                f'<div class="correction-item">'
                f'<strong>Day {snap.get("day", "?")}:</strong> {c}</div>'
            )
    if not corrections_html:
        corrections_html = '<div class="card">No corrections needed yet.</div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>astroSETI Observation Report - Day {day_number}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">

<h1>astroSETI Streaming Observation</h1>
<div class="subtitle">
    Day {day_number} of {target_days} |
    {datetime.now().strftime('%Y-%m-%d %H:%M')} |
    Mode: {mode.upper()} |
    Runtime: {runtime:.1f}h
</div>

<h2>Overview</h2>
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{total_files:,}</div>
        <div class="stat-label">Files Processed</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{total_signals:,}</div>
        <div class="stat-label">Signals Detected</div>
    </div>
    <div class="stat-card highlight">
        <div class="stat-value">{total_candidates}</div>
        <div class="stat-label">ET Candidates</div>
    </div>
    <div class="stat-card danger">
        <div class="stat-value">{total_rfi:,}</div>
        <div class="stat-label">RFI Rejected</div>
    </div>
    <div class="stat-card warning">
        <div class="stat-value">{total_ood}</div>
        <div class="stat-label">OOD Anomalies</div>
    </div>
</div>

<h2>Today (Day {day_number})</h2>
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{day_signals:,}</div>
        <div class="stat-label">Signals Today</div>
    </div>
    <div class="stat-card {'highlight' if day_candidates > 0 else ''}">
        <div class="stat-value">{day_candidates}</div>
        <div class="stat-label">Candidates Today</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{day_rfi}</div>
        <div class="stat-label">RFI Today</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{day_rate:.2f}%</div>
        <div class="stat-label">Candidate Rate</div>
    </div>
</div>

<h2>Trends</h2>
{_chart_img(signal_chart, "Signal Count")}
{_chart_img(candidate_chart, "Candidate Count")}
{_chart_img(rfi_chart, "RFI Rejection Rate")}
{_chart_img(speed_chart, "Processing Speed")}

<h2>Classification Distribution</h2>
{_chart_img(pie_chart, "Classification Pie")}

<h2>Sensitivity Evolution</h2>
{_chart_img(sensitivity_chart, "Sensitivity")}

<h2>Top Candidates</h2>
<div class="card">
<table>
<thead>
    <tr><th>#</th><th>OOD Score</th><th>Type</th><th>SNR</th><th>Drift (Hz/s)</th><th>Confidence</th><th>File</th></tr>
</thead>
<tbody>
    {candidates_rows if candidates_rows else '<tr><td colspan="7" style="text-align:center;color:#475569">No candidates yet</td></tr>'}
</tbody>
</table>
</div>

<h2>Self-Corrections</h2>
{corrections_html}

<div class="footer">
    Generated by <a href="https://github.com/samantaba/astroSETI">astroSETI</a><br>
    If this tool helps your research, please star the repo.
</div>

</div>
</body>
</html>"""

    date_str = datetime.now().strftime("%Y-%m-%d")
    report_path = artifacts_dir / f"day_{day_number}_{date_str}.html"
    report_path.write_text(html, encoding="utf-8")
    return str(report_path)


def generate_final_summary(
    state: dict,
    artifacts_dir,
) -> str:
    """
    Generate the final publishing-ready HTML summary report.

    Args:
        state: Full StreamingState dict.
        artifacts_dir: Directory to save the report.

    Returns:
        Path to the generated HTML file.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    snapshots = state.get("daily_snapshots", [])
    best_candidates = state.get("best_candidates", [])
    total_files = state.get("total_files_processed", 0)
    total_signals = state.get("total_signals", 0)
    total_candidates = state.get("total_candidates", 0)
    total_rfi = state.get("total_rfi_rejected", 0)
    total_ood = state.get("total_ood_anomalies", 0)
    runtime = state.get("total_runtime_hours", 0)
    total_days = state.get("current_day", 0)
    mode = state.get("current_mode", "normal")
    mode_history = state.get("mode_history", [])
    total_corrections = state.get("total_corrections", 0)

    # Charts
    signal_chart = _create_signal_count_chart(snapshots) if snapshots else ""
    candidate_chart = _create_candidate_chart(snapshots) if snapshots else ""
    rfi_chart = _create_rfi_chart(snapshots) if snapshots else ""
    speed_chart = _create_speed_chart(snapshots) if snapshots else ""
    pie_chart = _create_classification_pie(snapshots) if snapshots else ""
    sensitivity_chart = _create_sensitivity_chart(snapshots) if snapshots else ""

    candidate_rate = (
        (total_candidates / total_signals * 100) if total_signals > 0 else 0
    )
    rfi_rate = (total_rfi / total_signals * 100) if total_signals > 0 else 0

    # Candidates table
    candidates_rows = ""
    for i, c in enumerate(best_candidates[:20], 1):
        candidates_rows += f"""
        <tr>
            <td>{i}</td>
            <td><strong>{c.get('ood_score', 0):.4f}</strong></td>
            <td>{c.get('signal_type', '?')}</td>
            <td>{c.get('snr', 0):.1f}</td>
            <td>{c.get('drift_rate', 0):.4f}</td>
            <td>{c.get('confidence', 0):.1%}</td>
            <td>{Path(c.get('file', '')).name if c.get('file') else '?'}</td>
        </tr>"""

    # Per-day breakdown
    day_rows = ""
    for s in snapshots:
        day_rows += f"""
        <tr>
            <td>{s.get('day', '?')}</td>
            <td>{s.get('date', '?')}</td>
            <td>{s.get('files_processed', 0):,}</td>
            <td>{s.get('signals_found', 0):,}</td>
            <td>{s.get('candidates_found', 0)}</td>
            <td>{s.get('rfi_rejected', 0)}</td>
            <td>{s.get('candidate_rate', 0):.2f}%</td>
            <td>{s.get('files_per_hour', 0):.0f}</td>
            <td>{len(s.get('corrections_applied', []))}</td>
        </tr>"""

    # Mode changes
    mode_rows = ""
    for m in mode_history:
        mode_rows += f"""
        <tr>
            <td>Day {m.get('day', '?')}</td>
            <td>{m.get('from', '?')} &rarr; {m.get('to', '?')}</td>
            <td>{m.get('reason', '')}</td>
        </tr>"""

    # Corrections log
    all_corrections = []
    for snap in snapshots:
        for c in snap.get("corrections_applied", []):
            all_corrections.append(
                f'<div class="correction-item">'
                f'<strong>Day {snap.get("day", "?")}:</strong> {c}</div>'
            )
    corrections_html = "".join(all_corrections)
    if not corrections_html:
        corrections_html = '<div class="card">No corrections were needed.</div>'

    # AstroLens cross-reference section
    xref_total = state.get("astrolens_crossref_total", 0)
    xref_html = ""
    if xref_total > 0:
        xref_html = f"""
<h2>AstroLens Cross-Reference</h2>
<div class="card">
    <p>
        <strong>{xref_total}</strong> candidate(s) had positional matches with
        optical anomalies detected by
        <a href="https://github.com/samantaba/astroLens" style="color:{_ACCENT_BLUE}">AstroLens</a>.
        Multi-wavelength confirmation strengthens the scientific case for
        these detections.
    </p>
</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>astroSETI Observation - Final Summary</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">

<h1>astroSETI Streaming Observation</h1>
<div class="subtitle">
    Final Summary | {total_days} days |
    {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<div class="summary-box">
    <h3>Executive Summary</h3>
    <p>
        Over <strong>{total_days} days</strong> ({runtime:.1f} hours), astroSETI processed
        <strong>{total_files:,}</strong> filterbank files yielding
        <strong>{total_signals:,}</strong> radio signals.
        The system identified <strong>{total_candidates} ET candidates</strong>
        (rate: {candidate_rate:.2f}%), rejected <strong>{total_rfi:,} RFI signals</strong>
        ({rfi_rate:.1f}%), and flagged <strong>{total_ood} OOD anomalies</strong>
        for further investigation.
    </p>
    <p>
        The self-correcting intelligence applied <strong>{total_corrections} adjustments</strong>
        to optimize detection throughout the run.
        Final mode: <strong>{mode.upper()}</strong>.
    </p>
</div>

<h2>Key Metrics</h2>
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{total_days}</div>
        <div class="stat-label">Days</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{total_files:,}</div>
        <div class="stat-label">Files Processed</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{total_signals:,}</div>
        <div class="stat-label">Signals</div>
    </div>
    <div class="stat-card highlight">
        <div class="stat-value">{total_candidates}</div>
        <div class="stat-label">ET Candidates</div>
    </div>
    <div class="stat-card danger">
        <div class="stat-value">{total_rfi:,}</div>
        <div class="stat-label">RFI Rejected</div>
    </div>
    <div class="stat-card warning">
        <div class="stat-value">{total_ood}</div>
        <div class="stat-label">OOD Anomalies</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{candidate_rate:.2f}%</div>
        <div class="stat-label">Candidate Rate</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{total_corrections}</div>
        <div class="stat-label">Self-Corrections</div>
    </div>
</div>

<h2>Trends Over Time</h2>
{_chart_img(signal_chart, "Signal Count")}
{_chart_img(candidate_chart, "Candidate Count")}
{_chart_img(rfi_chart, "RFI Rejection Rate")}
{_chart_img(speed_chart, "Processing Speed")}

<h2>Classification Distribution</h2>
{_chart_img(pie_chart, "Classification Pie")}

<h2>Sensitivity Evolution</h2>
{_chart_img(sensitivity_chart, "Sensitivity")}

<h2>Per-Day Breakdown</h2>
<div class="card">
<table>
<thead>
    <tr><th>Day</th><th>Date</th><th>Files</th><th>Signals</th><th>Candidates</th><th>RFI</th><th>Rate</th><th>Files/Hr</th><th>Corrections</th></tr>
</thead>
<tbody>
    {day_rows if day_rows else '<tr><td colspan="9" style="text-align:center;color:#475569">No data yet</td></tr>'}
</tbody>
</table>
</div>

<h2>Top {min(20, len(best_candidates))} Candidates</h2>
<div class="card">
<table>
<thead>
    <tr><th>#</th><th>OOD Score</th><th>Type</th><th>SNR</th><th>Drift (Hz/s)</th><th>Confidence</th><th>File</th></tr>
</thead>
<tbody>
    {candidates_rows if candidates_rows else '<tr><td colspan="7" style="text-align:center;color:#475569">No candidates yet</td></tr>'}
</tbody>
</table>
</div>

{xref_html}

<h2>Self-Correction Log</h2>
{corrections_html}

{"<h2>Mode Changes</h2><div class='card'><table><thead><tr><th>Day</th><th>Change</th><th>Reason</th></tr></thead><tbody>" + mode_rows + "</tbody></table></div>" if mode_rows else ""}

<div class="footer">
    Generated by <a href="https://github.com/samantaba/astroSETI">astroSETI</a><br>
    If this tool helps your research, please
    <a href="https://github.com/samantaba/astroSETI">star the repo</a>.
</div>

</div>
</body>
</html>"""

    summary_path = artifacts_dir / f"final_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    summary_path.write_text(html, encoding="utf-8")

    # Also save JSON for programmatic access
    json_path = artifacts_dir / "final_summary.json"
    json_data = {
        "generated_at": datetime.now().isoformat(),
        "total_days": total_days,
        "total_runtime_hours": runtime,
        "total_files_processed": total_files,
        "total_signals": total_signals,
        "total_candidates": total_candidates,
        "total_rfi_rejected": total_rfi,
        "total_ood_anomalies": total_ood,
        "candidate_rate": candidate_rate,
        "rfi_rate": rfi_rate,
        "mode": mode,
        "corrections": total_corrections,
        "best_candidates": best_candidates[:20],
        "daily_snapshots": snapshots,
    }
    json_path.write_text(
        json.dumps(json_data, indent=2, default=str), encoding="utf-8"
    )

    return str(summary_path)
