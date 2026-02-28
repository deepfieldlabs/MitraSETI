"""
MitraSETI Dashboard — The First Thing Users See

Hero section, stat cards, quick actions, recent activity,
system health indicators, and mini waterfall preview.
"""

from __future__ import annotations

import io
import subprocess
import sys
import webbrowser
from pathlib import Path

import matplotlib
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure

from .theme import COLORS, make_stat_card

# ── Health indicator dot ──────────────────────────────────────────────────────


class _HealthDot(QFrame):
    """Tiny colored dot for system health."""

    def __init__(self, size: int = 8, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._color = QColor(COLORS["text_tertiary"])

    def set_status(self, ok: bool):
        self._color = QColor(COLORS["success"] if ok else COLORS["danger"])
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self._color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(1, 1, self.width() - 2, self.height() - 2)


# ── Dashboard Panel ──────────────────────────────────────────────────────────


class DashboardPanel(QWidget):
    """Main dashboard — the landing page of MitraSETI."""

    navigate_to = pyqtSignal(int)  # Emit panel index to switch to

    _STATE_FILE = (
        Path(__file__).parent.parent.parent
        / "mitraseti_artifacts"
        / "data"
        / "streaming_state.json"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

        # Render mini waterfall preview after a short delay
        QTimer.singleShot(500, self._render_mini_waterfall)

        # Auto-refresh stats from streaming state
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_stats)
        self._refresh_timer.start(5000)
        QTimer.singleShot(800, self._refresh_stats)

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(28)

        # ══════════════════════════════════════════════════════════════════
        # HERO SECTION
        # ══════════════════════════════════════════════════════════════════
        hero = QFrame()
        hero.setMinimumHeight(160)
        hero.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 50, 100, 0.4),
                    stop:0.5 rgba(0, 80, 140, 0.25),
                    stop:1 rgba(124, 58, 237, 0.15));
                border: 1px solid rgba(77, 166, 255, 0.12);
                border-radius: 18px;
            }
        """)
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(40, 30, 40, 30)
        hero_layout.setSpacing(8)

        # Title with glow feel
        hero_title = QLabel("MitraSETI")
        hero_title.setStyleSheet("""
            font-size: 42px;
            font-weight: 800;
            color: #4da6ff;
            letter-spacing: 2px;
        """)
        hero_layout.addWidget(hero_title)

        hero_tagline = QLabel("Intelligent SETI Signal Analysis")
        hero_tagline.setStyleSheet("""
            font-size: 16px;
            font-weight: 300;
            color: rgba(200, 215, 235, 0.6);
            letter-spacing: 1px;
        """)
        hero_layout.addWidget(hero_tagline)

        hero_sub = QLabel(
            "Detect narrowband drifting signals  ·  ML-powered RFI rejection  ·  "
            "Cross-reference with AstroLens transients"
        )
        hero_sub.setStyleSheet("""
            font-size: 12px;
            color: rgba(140, 165, 200, 0.45);
            margin-top: 8px;
        """)
        hero_sub.setWordWrap(True)
        hero_layout.addWidget(hero_sub)

        layout.addWidget(hero)

        # ══════════════════════════════════════════════════════════════════
        # STAT CARDS — 2x3 grid
        # ══════════════════════════════════════════════════════════════════
        stats_label = QLabel("OVERVIEW")
        stats_label.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: rgba(0,212,255,0.5); letter-spacing: 2px;"
        )
        layout.addWidget(stats_label)

        stats_grid = QGridLayout()
        stats_grid.setSpacing(12)

        self._stat_cards = {}
        stat_defs = [
            ("observations", "Total Observations", "0", "#4da6ff"),
            ("signals", "Signals Detected", "0", "#7c3aed"),
            ("candidates", "Candidates Found", "0", "#34d399"),
            ("rfi_rate", "RFI Rejection Rate", "0%", "#f87171"),
            ("speed", "Processing Speed", "—", "#fbbf24"),
            ("yolo", "YOLO Transients", "0", "#4da6ff"),
        ]

        for i, (key, label, default, color) in enumerate(stat_defs):
            card = make_stat_card(label, default, color)
            row, col = divmod(i, 3)
            stats_grid.addWidget(card, row, col)
            self._stat_cards[key] = card

        layout.addLayout(stats_grid)

        # ══════════════════════════════════════════════════════════════════
        # QUICK ACTIONS
        # ══════════════════════════════════════════════════════════════════
        actions_label = QLabel("QUICK ACTIONS")
        actions_label.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: rgba(0,212,255,0.5); letter-spacing: 2px;"
        )
        layout.addWidget(actions_label)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(12)

        load_btn = self._make_action_btn(
            "📡  Waterfall Viewer",
            "#4da6ff",
            "Open the spectrogram viewer to inspect filterbank (.fil) or HDF5 (.h5) files",
        )
        load_btn.clicked.connect(lambda: self.navigate_to.emit(1))
        actions_row.addWidget(load_btn)

        stream_btn = self._make_action_btn(
            "📶  Start Streaming", "#34d399", "Begin continuous observation processing"
        )
        stream_btn.clicked.connect(lambda: self.navigate_to.emit(5))
        actions_row.addWidget(stream_btn)

        candidates_btn = self._make_action_btn(
            "🔬  View Candidates", "#7c3aed", "Browse candidate signals in gallery"
        )
        candidates_btn.clicked.connect(lambda: self.navigate_to.emit(2))
        actions_row.addWidget(candidates_btn)

        report_btn = self._make_action_btn(
            "📊  Generate Report", "#fbbf24", "Export analysis summary"
        )
        report_btn.clicked.connect(self._generate_report)
        actions_row.addWidget(report_btn)

        layout.addLayout(actions_row)

        # Second row: benchmark + download
        actions_row2 = QHBoxLayout()
        actions_row2.setSpacing(12)

        bench_btn = self._make_action_btn(
            "⚡  Run Benchmark", "#f87171", "Compare MitraSETI vs turboSETI performance"
        )
        bench_btn.clicked.connect(self._run_benchmark)
        actions_row2.addWidget(bench_btn)

        download_btn = self._make_action_btn(
            "📥  Download BL Data", "#7c3aed", "Download real Breakthrough Listen observation files"
        )
        download_btn.clicked.connect(self._download_bl_data)
        actions_row2.addWidget(download_btn)

        actions_row2.addStretch()
        layout.addLayout(actions_row2)

        # ══════════════════════════════════════════════════════════════════
        # BOTTOM ROW: Recent Activity + Health + Mini Waterfall
        # ══════════════════════════════════════════════════════════════════
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)

        # Recent activity feed
        activity_card = QFrame()
        activity_card.setStyleSheet("""
            QFrame {
                background: rgba(15, 25, 45, 0.55);
                border: 1px solid rgba(100, 180, 255, 0.1);
                border-radius: 14px;
            }
        """)
        activity_layout = QVBoxLayout(activity_card)
        activity_layout.setContentsMargins(20, 16, 20, 16)
        activity_layout.setSpacing(8)

        act_title = QLabel("Recent Activity")
        act_title.setStyleSheet("font-size: 13px; font-weight: 600; color: rgba(0,212,255,0.8);")
        activity_layout.addWidget(act_title)

        # Demo activity items
        demo_activities = [
            ("✦", "Candidate signal detected", "SNR 24.7 @ 1420.405 MHz", "#34d399", "2 min ago"),
            ("✕", "RFI rejected", "GPS L1 sidelobe @ 1575.42 MHz", "#f87171", "5 min ago"),
            ("✦", "Candidate signal detected", "SNR 18.2 @ 1667.3 MHz", "#34d399", "12 min ago"),
            ("⚡", "Pipeline completed", "observation_2024_0312.fil", "#4da6ff", "15 min ago"),
            ("✕", "RFI rejected", "Broadband transient, terrestrial", "#f87171", "18 min ago"),
            ("✦", "Candidate signal detected", "SNR 31.4 @ 1420.8 MHz", "#34d399", "22 min ago"),
            ("⚡", "Pipeline completed", "observation_2024_0311.fil", "#4da6ff", "28 min ago"),
            ("🔗", "AstroLens cross-ref", "YOLO transient match #12", "#7c3aed", "35 min ago"),
            ("✕", "RFI rejected", "WiFi 2.4G spillover", "#f87171", "42 min ago"),
            ("✦", "Candidate signal detected", "SNR 15.8 @ 408.0 MHz", "#34d399", "50 min ago"),
        ]

        for icon, title_text, detail, color, time_ago in demo_activities:
            row = QHBoxLayout()
            row.setSpacing(10)

            icon_lbl = QLabel(icon)
            icon_lbl.setFixedWidth(20)
            icon_lbl.setStyleSheet(f"font-size: 12px; color: {color};")
            row.addWidget(icon_lbl)

            text_col = QVBoxLayout()
            text_col.setSpacing(1)
            t = QLabel(title_text)
            t.setStyleSheet("font-size: 12px; color: #e0e8f0; font-weight: 500;")
            text_col.addWidget(t)
            d = QLabel(detail)
            d.setStyleSheet("font-size: 10px; color: rgba(200,215,235,0.4);")
            text_col.addWidget(d)
            row.addLayout(text_col)

            row.addStretch()

            time_lbl = QLabel(time_ago)
            time_lbl.setStyleSheet("font-size: 10px; color: rgba(140,165,200,0.35);")
            row.addWidget(time_lbl)

            activity_layout.addLayout(row)

        activity_layout.addStretch()
        bottom_row.addWidget(activity_card, 2)

        # Right column: health + mini waterfall
        right_col = QVBoxLayout()
        right_col.setSpacing(16)

        # System health card
        health_card = QFrame()
        health_card.setStyleSheet("""
            QFrame {
                background: rgba(15, 25, 45, 0.55);
                border: 1px solid rgba(100, 180, 255, 0.1);
                border-radius: 14px;
            }
        """)
        health_layout = QVBoxLayout(health_card)
        health_layout.setContentsMargins(20, 16, 20, 16)
        health_layout.setSpacing(10)

        health_title = QLabel("System Health")
        health_title.setStyleSheet("font-size: 13px; font-weight: 600; color: rgba(0,212,255,0.8);")
        health_layout.addWidget(health_title)

        self._health_items = {}
        health_defs = [
            ("api", "API Status", True),
            ("gpu", "GPU Status", True),
            ("models", "Models Loaded", True),
            ("astrolens", "AstroLens Connection", False),
        ]

        for key, label, ok in health_defs:
            row = QHBoxLayout()
            row.setSpacing(8)
            dot = _HealthDot(8)
            dot.set_status(ok)
            row.addWidget(dot)
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 12px; color: rgba(200,215,235,0.7);")
            row.addWidget(lbl)
            row.addStretch()
            status_lbl = QLabel("Online" if ok else "Offline")
            status_lbl.setStyleSheet(
                f"font-size: 11px; font-weight: 500; "
                f"color: {COLORS['success'] if ok else COLORS['danger']};"
            )
            row.addWidget(status_lbl)
            health_layout.addLayout(row)
            self._health_items[key] = (dot, status_lbl)

        right_col.addWidget(health_card)

        # Mini waterfall preview
        waterfall_card = QFrame()
        waterfall_card.setStyleSheet("""
            QFrame {
                background: rgba(15, 25, 45, 0.55);
                border: 1px solid rgba(100, 180, 255, 0.1);
                border-radius: 14px;
            }
        """)
        wf_layout = QVBoxLayout(waterfall_card)
        wf_layout.setContentsMargins(16, 16, 16, 16)
        wf_layout.setSpacing(8)

        wf_title = QLabel("Latest Spectrogram")
        wf_title.setStyleSheet("font-size: 13px; font-weight: 600; color: rgba(0,212,255,0.8);")
        wf_layout.addWidget(wf_title)

        self._mini_waterfall = QLabel()
        self._mini_waterfall.setFixedSize(320, 180)
        self._mini_waterfall.setAlignment(Qt.AlignCenter)
        self._mini_waterfall.setStyleSheet("background: rgba(8,12,20,0.8); border-radius: 10px;")
        self._mini_waterfall.setText("Generating preview…")
        self._mini_waterfall.setStyleSheet(
            "background: rgba(8,12,20,0.8); border-radius: 10px; "
            "color: rgba(140,165,200,0.3); font-size: 12px;"
        )
        wf_layout.addWidget(self._mini_waterfall)

        view_btn = QPushButton("Open in Waterfall Viewer →")
        view_btn.setCursor(Qt.PointingHandCursor)
        view_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: rgba(77, 166, 255, 0.7);
                font-size: 12px;
                font-weight: 500;
                text-align: left;
                padding: 4px 0;
            }
            QPushButton:hover {
                color: #4da6ff;
            }
        """)
        view_btn.clicked.connect(lambda: self.navigate_to.emit(1))
        wf_layout.addWidget(view_btn)

        right_col.addWidget(waterfall_card)
        right_col.addStretch()

        bottom_row.addLayout(right_col, 1)
        layout.addLayout(bottom_row, 1)

        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Helper: action button ─────────────────────────────────────────────

    @staticmethod
    def _make_action_btn(text: str, color: str, tooltip: str) -> QPushButton:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(tooltip)
        btn.setMinimumHeight(52)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba({r},{g},{b}, 0.08),
                    stop:1 rgba({r},{g},{b}, 0.04));
                border: 1px solid rgba({r},{g},{b}, 0.2);
                border-radius: 12px;
                padding: 14px 20px;
                color: {color};
                font-size: 14px;
                font-weight: 600;
                text-align: left;
            }}
            QPushButton:hover {{
                background: rgba({r},{g},{b}, 0.15);
                border-color: rgba({r},{g},{b}, 0.4);
            }}
        """)
        return btn

    # ── Generate Report ────────────────────────────────────────────────

    def _generate_report(self):
        """Generate an HTML summary report from current streaming results."""
        import json
        from datetime import datetime

        artifacts = Path(__file__).parent.parent.parent / "mitraseti_artifacts"
        reports_dir = artifacts / "streaming_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        state_file = artifacts / "data" / "streaming_state.json"
        cands_file = artifacts / "candidates" / "verified_candidates.json"

        state = {}
        candidates = []
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
            except Exception:
                pass
        if cands_file.exists():
            try:
                with open(cands_file) as f:
                    candidates = json.load(f)
            except Exception:
                pass

        if not state and not candidates:
            QMessageBox.information(
                self,
                "No Data",
                "No streaming data or candidates found.\n"
                "Run the streaming pipeline first to generate results.",
            )
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        report_path = reports_dir / f"summary_report_{ts}.html"

        total_signals = state.get("total_signals", 0)
        total_candidates = state.get("total_candidates", 0)
        total_rfi = state.get("total_rfi_rejected", 0)
        files_processed = state.get("files_processed", 0)
        runtime_h = state.get("total_runtime_hours", 0)
        rfi_pct = (total_rfi / total_signals * 100) if total_signals > 0 else 0

        candidates.sort(key=lambda c: c.get("snr", 0), reverse=True)

        cand_rows = ""
        for i, c in enumerate(candidates[:20], 1):
            cand_rows += f"""<tr>
                <td>{i}</td>
                <td>{c.get("target_name", c.get("category", "Unknown"))}</td>
                <td>{c.get("frequency_hz", 0) / 1e6:.4f}</td>
                <td>{c.get("snr", 0):.1f}</td>
                <td>{c.get("drift_rate", 0):.4f}</td>
                <td>{c.get("classification", "N/A")}</td>
                <td>{c.get("file", "N/A")}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>MitraSETI Summary Report — {ts}</title>
<style>
body {{ font-family: 'Inter', -apple-system, sans-serif; background: #0a0e18; color: #e0e8f0; margin: 0; padding: 40px; }}
h1 {{ color: #00d4ff; font-weight: 300; letter-spacing: 2px; }}
h2 {{ color: #7c3aed; margin-top: 32px; }}
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
.stat {{ background: rgba(15,25,45,0.8); border: 1px solid rgba(100,180,255,0.15); border-radius: 12px; padding: 20px; text-align: center; }}
.stat .val {{ font-size: 32px; font-weight: 300; color: #00d4ff; }}
.stat .lbl {{ font-size: 11px; color: rgba(200,215,235,0.5); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
th {{ text-align: left; padding: 10px 14px; font-size: 11px; color: rgba(0,212,255,0.7); text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid rgba(100,180,255,0.15); }}
td {{ padding: 10px 14px; font-size: 13px; border-bottom: 1px solid rgba(100,180,255,0.06); }}
tr:hover td {{ background: rgba(0,212,255,0.03); }}
.footer {{ text-align: center; margin-top: 40px; font-size: 11px; color: rgba(140,165,200,0.4); }}
</style></head><body>
<h1>MitraSETI — Summary Report</h1>
<p style="color: rgba(200,215,235,0.6);">Generated {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>

<div class="stats">
  <div class="stat"><div class="val">{files_processed}</div><div class="lbl">Files Processed</div></div>
  <div class="stat"><div class="val">{total_signals:,}</div><div class="lbl">Signals Analyzed</div></div>
  <div class="stat"><div class="val">{len(candidates)}</div><div class="lbl">Verified Candidates</div></div>
  <div class="stat"><div class="val">{rfi_pct:.1f}%</div><div class="lbl">RFI Rejection Rate</div></div>
</div>

<h2>Pipeline Performance</h2>
<div class="stats">
  <div class="stat"><div class="val">{runtime_h:.1f}h</div><div class="lbl">Total Runtime</div></div>
  <div class="stat"><div class="val">{total_rfi:,}</div><div class="lbl">RFI Rejected</div></div>
  <div class="stat"><div class="val">{total_candidates}</div><div class="lbl">ML Candidates</div></div>
  <div class="stat"><div class="val">{state.get("cadence_passed", 0)}</div><div class="lbl">Cadence Passed</div></div>
</div>

<h2>Top Verified Candidates</h2>
<table>
<thead><tr><th>#</th><th>Target</th><th>Freq (MHz)</th><th>SNR</th><th>Drift (Hz/s)</th><th>Classification</th><th>Source File</th></tr></thead>
<tbody>{cand_rows if cand_rows else '<tr><td colspan="7" style="text-align:center; color: rgba(200,215,235,0.4);">No candidates found</td></tr>'}</tbody>
</table>

<div class="footer">MitraSETI v0.1.0 — Deep Field Labs — Intelligent SETI Signal Analysis</div>
</body></html>"""

        try:
            report_path.write_text(html, encoding="utf-8")
            webbrowser.open(f"file://{report_path}")
            QMessageBox.information(
                self,
                "Report Generated",
                f"Summary report saved and opened in browser:\n{report_path}",
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to generate report:\n{e}")

    # ── Benchmark & Download ─────────────────────────────────────────────

    def _run_benchmark(self):
        """Launch the benchmark suite in a subprocess."""
        script = Path(__file__).parent.parent / "scripts" / "benchmark.py"
        artifacts = Path(__file__).parent.parent.parent / "mitraseti_artifacts" / "benchmarks"
        artifacts.mkdir(parents=True, exist_ok=True)
        try:
            self._bench_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(script),
                    "--sizes",
                    "tiny",
                    "small",
                    "medium",
                    "--output-dir",
                    str(artifacts),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            QMessageBox.information(
                self,
                "Benchmark Started",
                "Benchmark is running in the background.\n"
                "An HTML report will be generated in:\n"
                f"{artifacts}\n\n"
                "This may take 1-3 minutes depending on your system.",
            )
            QTimer.singleShot(3000, self._check_benchmark_done)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to launch benchmark:\n{e}")

    def _check_benchmark_done(self):
        if hasattr(self, "_bench_proc") and self._bench_proc:
            if self._bench_proc.poll() is not None:
                artifacts = (
                    Path(__file__).parent.parent.parent / "mitraseti_artifacts" / "benchmarks"
                )
                reports = sorted(artifacts.glob("benchmark_report_*.html"), reverse=True)
                if reports:
                    webbrowser.open(f"file://{reports[0]}")
                self._bench_proc = None
            else:
                QTimer.singleShot(3000, self._check_benchmark_done)

    def _download_bl_data(self):
        """Download Breakthrough Listen sample data."""
        script = Path(__file__).parent.parent / "scripts" / "download_bl_data.py"
        try:
            self._dl_proc = subprocess.Popen(
                [sys.executable, str(script), "--count", "5"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            QMessageBox.information(
                self,
                "Download Started",
                "Downloading Breakthrough Listen sample data (5 files).\n"
                "Files include: Voyager-1, TRAPPIST-1, Proxima Centauri, etc.\n\n"
                "This may take a few minutes depending on your connection.\n"
                "Files will appear in the Waterfall Viewer when complete.",
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start download:\n{e}")

    # ── Mini waterfall rendering ──────────────────────────────────────────

    def _render_mini_waterfall(self):
        """Generate a small synthetic waterfall preview."""
        rng = np.random.default_rng(42)
        n_t, n_f = 128, 256
        data = rng.normal(0, 1, (n_t, n_f)).astype(np.float32)

        # Inject a couple of drifting signals
        for snr, f0, dr in [(15, 100, 0.8), (20, 180, -1.2), (12, 60, 0.3)]:
            for t in range(n_t):
                fc = int(f0 + dr * t / n_t * 30)
                if 0 <= fc < n_f:
                    data[t, max(0, fc - 1) : min(n_f, fc + 2)] += snr

        fig = Figure(figsize=(320 / 72, 180 / 72), dpi=72)
        ax = fig.add_subplot(111)
        from matplotlib.colors import LinearSegmentedColormap

        seti_cmap = LinearSegmentedColormap.from_list(
            "seti",
            [
                (0.0, "#020810"),
                (0.2, "#081838"),
                (0.4, "#0a3060"),
                (0.6, "#0060a0"),
                (0.8, "#00b0e0"),
                (1.0, "#00ffff"),
            ],
        )
        ax.imshow(data, aspect="auto", origin="lower", cmap=seti_cmap, interpolation="nearest")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            facecolor="#080c14",
            bbox_inches="tight",
            pad_inches=0,
        )
        buf.seek(0)

        img = QImage.fromData(buf.read())
        if not img.isNull():
            pm = QPixmap.fromImage(img).scaled(
                320, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._mini_waterfall.setPixmap(pm)
            self._mini_waterfall.setStyleSheet(
                "background: rgba(8,12,20,0.8); border-radius: 10px;"
            )

    # ── Auto-refresh from streaming state ───────────────────────────────

    def _refresh_stats(self):
        """Read streaming state and update dashboard stat cards."""
        import json

        if not self._STATE_FILE.exists():
            return
        try:
            with open(self._STATE_FILE) as f:
                state = json.load(f)
        except Exception:
            return

        files = state.get("total_files_processed", 0)
        signals = state.get("total_signals", 0)
        candidates = state.get("total_candidates", 0)
        rfi = state.get("total_rfi_rejected", 0)
        runtime_hrs = state.get("total_runtime_hours", 0)

        # RFI rejection rate as percentage
        total_classified = rfi + signals
        rfi_pct = f"{rfi / total_classified * 100:.1f}%" if total_classified > 0 else "0%"

        # Processing speed as avg seconds per file
        if files > 0 and runtime_hrs > 0:
            secs = (runtime_hrs * 3600) / files
            speed = f"{secs:.1f}s/file" if secs < 60 else f"{secs / 60:.1f}m/file"
        else:
            speed = "—"

        self.update_stat("observations", str(files))
        self.update_stat("signals", f"{signals:,}" if signals else "0")
        self.update_stat("candidates", str(candidates))
        self.update_stat("rfi_rate", rfi_pct)
        self.update_stat("speed", speed)

    # ── Public API ────────────────────────────────────────────────────────

    def update_stat(self, key: str, value: str):
        """Update a specific stat card value."""
        card = self._stat_cards.get(key)
        if card:
            val_lbl = card.findChild(QLabel, "value")
            if val_lbl:
                val_lbl.setText(value)

    def set_health(self, key: str, ok: bool):
        """Update a health indicator."""
        if key in self._health_items:
            dot, lbl = self._health_items[key]
            dot.set_status(ok)
            lbl.setText("Online" if ok else "Offline")
            lbl.setStyleSheet(
                f"font-size: 11px; font-weight: 500; "
                f"color: {COLORS['success'] if ok else COLORS['danger']};"
            )
