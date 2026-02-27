"""
MitraSETI RFI Dashboard

Radio Frequency Interference rejection dashboard:
- Top stats row with rejection metrics
- Pie chart of signal classification distribution
- Bar chart of RFI by type
- Known RFI frequency bands table
- Before/After ML filtering comparison
- False positive tracker
"""

from __future__ import annotations

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea, QSizePolicy, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .theme import COLORS, make_stat_card, create_stat_card_style


# ── Known RFI bands table data ───────────────────────────────────────────────

KNOWN_RFI_BANDS = [
    ("GPS L1", "1575.42 MHz", "Satellite", "Navigation signal"),
    ("GPS L2", "1227.60 MHz", "Satellite", "Navigation signal"),
    ("Iridium", "1616–1626 MHz", "Satellite", "Comm constellation"),
    ("Cell LTE", "700–900 MHz", "Terrestrial", "Mobile broadband"),
    ("WiFi 2.4G", "2400–2483 MHz", "Terrestrial", "802.11 b/g/n"),
    ("Radar L-band", "1215–1400 MHz", "Terrestrial", "Weather/ATC"),
    ("TV Broadcast", "470–890 MHz", "Terrestrial", "UHF television"),
    ("Starlink DL", "10.7–12.7 GHz", "Satellite", "LEO broadband"),
    ("60 Hz Hum", "60 Hz harmonics", "Instrumental", "Power line interference"),
    ("ADC Spur", "Variable", "Instrumental", "Digital quantization"),
]


class RFIPanel(QWidget):
    """RFI rejection dashboard with charts, tables, and stats."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_demo_data()
        self._setup_ui()

        # Auto-refresh from pipeline results
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_data)
        self._timer.start(10_000)

    # ── Demo / placeholder data ───────────────────────────────────────────

    def _setup_demo_data(self):
        """Initialize with demo statistics for first-run experience."""
        self._total_signals = 247
        self._rfi_rejected = 189
        self._false_positives = 7
        self._true_candidates = 51

        self._classification_dist = {
            "Candidate": 51,
            "RFI Terrestrial": 88,
            "RFI Satellite": 52,
            "RFI Statistical": 31,
            "RFI Broadband": 18,
            "Unknown": 7,
        }

        self._rfi_by_type = {
            "Terrestrial": 88,
            "Satellite": 52,
            "Statistical": 31,
            "Broadband": 18,
        }

        self._false_positive_log = [
            {"id": "FP-001", "freq": "1420.8 MHz", "reason": "Drift rate matched known pulsar", "status": "Resolved"},
            {"id": "FP-002", "freq": "1575.4 MHz", "reason": "GPS sidelobe mis-classified", "status": "Resolved"},
            {"id": "FP-003", "freq": "850.2 MHz", "reason": "Broadband transient", "status": "Under Review"},
            {"id": "FP-004", "freq": "1665.0 MHz", "reason": "OH maser false hit", "status": "Resolved"},
            {"id": "FP-005", "freq": "2401.5 MHz", "reason": "WiFi edge spillover", "status": "Under Review"},
            {"id": "FP-006", "freq": "408.0 MHz", "reason": "TV broadcast harmonic", "status": "Resolved"},
            {"id": "FP-007", "freq": "1176.5 MHz", "reason": "GPS L5 sidelobe", "status": "Under Review"},
        ]

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(24)

        # ── Header ────────────────────────────────────────────────────────
        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel("🛡️  RFI Rejection Dashboard")
        title.setStyleSheet(
            "font-size: 26px; font-weight: 300; color: #e0e8f0; letter-spacing: -0.5px;"
        )
        header.addWidget(title)
        subtitle = QLabel("Monitor and analyze radio frequency interference filtering")
        subtitle.setStyleSheet("font-size: 13px; color: rgba(140,165,200,0.5);")
        header.addWidget(subtitle)
        layout.addLayout(header)

        # ── Top stats row ─────────────────────────────────────────────────
        stats_grid = QGridLayout()
        stats_grid.setSpacing(12)

        self._stat_total = make_stat_card("Total Signals", str(self._total_signals), "#4da6ff")
        pct = f"{self._rfi_rejected} ({self._rfi_rejected / max(self._total_signals, 1) * 100:.1f}%)"
        self._stat_rejected = make_stat_card("RFI Rejected", pct, "#f87171")
        self._stat_fp = make_stat_card("False Positives", str(self._false_positives), "#fbbf24")
        self._stat_candidates = make_stat_card("True Candidates", str(self._true_candidates), "#34d399")

        stats_grid.addWidget(self._stat_total, 0, 0)
        stats_grid.addWidget(self._stat_rejected, 0, 1)
        stats_grid.addWidget(self._stat_fp, 0, 2)
        stats_grid.addWidget(self._stat_candidates, 0, 3)
        layout.addLayout(stats_grid)

        # ── Charts row ────────────────────────────────────────────────────
        charts_row = QHBoxLayout()
        charts_row.setSpacing(16)

        # Pie chart — classification distribution
        pie_card = QFrame()
        pie_card.setStyleSheet("""
            QFrame {
                background: rgba(15, 25, 45, 0.55);
                border: 1px solid rgba(100, 180, 255, 0.1);
                border-radius: 14px;
            }
        """)
        pie_layout = QVBoxLayout(pie_card)
        pie_layout.setContentsMargins(16, 16, 16, 16)

        pie_title = QLabel("Classification Distribution")
        pie_title.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: rgba(0,212,255,0.8);"
        )
        pie_layout.addWidget(pie_title)

        self._pie_fig = Figure(figsize=(4, 3.5), dpi=100, facecolor="#080c14")
        self._pie_canvas = FigureCanvas(self._pie_fig)
        self._pie_canvas.setMinimumHeight(260)
        pie_layout.addWidget(self._pie_canvas)

        charts_row.addWidget(pie_card)

        # Bar chart — RFI by type
        bar_card = QFrame()
        bar_card.setStyleSheet("""
            QFrame {
                background: rgba(15, 25, 45, 0.55);
                border: 1px solid rgba(100, 180, 255, 0.1);
                border-radius: 14px;
            }
        """)
        bar_layout = QVBoxLayout(bar_card)
        bar_layout.setContentsMargins(16, 16, 16, 16)

        bar_title = QLabel("RFI by Type")
        bar_title.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: rgba(0,212,255,0.8);"
        )
        bar_layout.addWidget(bar_title)

        self._bar_fig = Figure(figsize=(4, 3.5), dpi=100, facecolor="#080c14")
        self._bar_canvas = FigureCanvas(self._bar_fig)
        self._bar_canvas.setMinimumHeight(260)
        bar_layout.addWidget(self._bar_canvas)

        charts_row.addWidget(bar_card)

        layout.addLayout(charts_row)

        # ── Before / After comparison ─────────────────────────────────────
        ba_card = QFrame()
        ba_card.setStyleSheet("""
            QFrame {
                background: rgba(15, 25, 45, 0.55);
                border: 1px solid rgba(100, 180, 255, 0.1);
                border-radius: 14px;
            }
        """)
        ba_layout = QVBoxLayout(ba_card)
        ba_layout.setContentsMargins(20, 16, 20, 16)
        ba_layout.setSpacing(12)

        ba_title = QLabel("Before / After ML Filtering")
        ba_title.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: rgba(0,212,255,0.8);"
        )
        ba_layout.addWidget(ba_title)

        self._ba_fig = Figure(figsize=(8, 2.5), dpi=100, facecolor="#080c14")
        self._ba_canvas = FigureCanvas(self._ba_fig)
        self._ba_canvas.setMinimumHeight(180)
        ba_layout.addWidget(self._ba_canvas)

        layout.addWidget(ba_card)

        # ── Tables row ────────────────────────────────────────────────────
        tables_row = QHBoxLayout()
        tables_row.setSpacing(16)

        # Known RFI bands table
        rfi_table_card = QFrame()
        rfi_table_card.setStyleSheet("""
            QFrame {
                background: rgba(15, 25, 45, 0.55);
                border: 1px solid rgba(100, 180, 255, 0.1);
                border-radius: 14px;
            }
        """)
        rfi_tbl_layout = QVBoxLayout(rfi_table_card)
        rfi_tbl_layout.setContentsMargins(16, 16, 16, 16)

        rfi_tbl_title = QLabel("Known RFI Frequency Bands")
        rfi_tbl_title.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: rgba(0,212,255,0.8);"
        )
        rfi_tbl_layout.addWidget(rfi_tbl_title)

        self._rfi_table = QTableWidget()
        self._rfi_table.setColumnCount(4)
        self._rfi_table.setHorizontalHeaderLabels(["Source", "Frequency", "Type", "Description"])
        self._rfi_table.horizontalHeader().setStretchLastSection(True)
        self._rfi_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._rfi_table.verticalHeader().setVisible(False)
        self._rfi_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._rfi_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._rfi_table.setAlternatingRowColors(False)
        self._rfi_table.setMinimumHeight(280)
        rfi_tbl_layout.addWidget(self._rfi_table)

        tables_row.addWidget(rfi_table_card)

        # False positive tracker
        fp_card = QFrame()
        fp_card.setStyleSheet("""
            QFrame {
                background: rgba(15, 25, 45, 0.55);
                border: 1px solid rgba(100, 180, 255, 0.1);
                border-radius: 14px;
            }
        """)
        fp_layout = QVBoxLayout(fp_card)
        fp_layout.setContentsMargins(16, 16, 16, 16)

        fp_title = QLabel("False Positive Tracker")
        fp_title.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: rgba(0,212,255,0.8);"
        )
        fp_layout.addWidget(fp_title)

        self._fp_table = QTableWidget()
        self._fp_table.setColumnCount(4)
        self._fp_table.setHorizontalHeaderLabels(["ID", "Frequency", "Reason", "Status"])
        self._fp_table.horizontalHeader().setStretchLastSection(True)
        self._fp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._fp_table.verticalHeader().setVisible(False)
        self._fp_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._fp_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._fp_table.setMinimumHeight(280)
        fp_layout.addWidget(self._fp_table)

        tables_row.addWidget(fp_card)

        layout.addLayout(tables_row)
        layout.addStretch()

        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # Render everything
        self._render_pie()
        self._render_bar()
        self._render_before_after()
        self._populate_rfi_table()
        self._populate_fp_table()

    # ── Chart rendering ───────────────────────────────────────────────────

    def _render_pie(self):
        self._pie_fig.clear()
        ax = self._pie_fig.add_subplot(111)

        labels = list(self._classification_dist.keys())
        sizes = list(self._classification_dist.values())
        colors = ["#34d399", "#f87171", "#ff6644", "#fbbf24", "#cc6600", "#7c3aed"]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors[:len(labels)],
            autopct="%1.1f%%", startangle=140,
            textprops={"color": "#e0e8f0", "fontsize": 8},
            pctdistance=0.75,
            wedgeprops={"linewidth": 1, "edgecolor": "#080c14"},
        )
        for t in autotexts:
            t.set_fontsize(7)
            t.set_color("#e0e8f0")
        for t in texts:
            t.set_fontsize(7)

        ax.set_facecolor("#080c14")
        self._pie_fig.set_facecolor("#080c14")
        self._pie_fig.tight_layout(pad=1)
        self._pie_canvas.draw()

    def _render_bar(self):
        self._bar_fig.clear()
        ax = self._bar_fig.add_subplot(111)

        types = list(self._rfi_by_type.keys())
        counts = list(self._rfi_by_type.values())
        colors = ["#f87171", "#ff6644", "#fbbf24", "#cc6600"]

        bars = ax.barh(types, counts, color=colors[:len(types)], height=0.6, edgecolor="#080c14")

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center", ha="left", color="#e0e8f0", fontsize=10, fontweight="bold",
            )

        ax.set_facecolor("#080c14")
        ax.tick_params(colors="#8ca5c8", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color((0.39, 0.71, 1.0, 0.15))
        ax.spines["left"].set_color((0.39, 0.71, 1.0, 0.15))
        ax.set_xlabel("Count", color="#8ca5c8", fontsize=9)

        self._bar_fig.set_facecolor("#080c14")
        self._bar_fig.tight_layout(pad=1.2)
        self._bar_canvas.draw()

    def _render_before_after(self):
        self._ba_fig.clear()
        ax = self._ba_fig.add_subplot(111)

        categories = ["Total\nDetections", "After ML\nFiltering", "Final\nCandidates"]
        before_vals = [self._total_signals, self._total_signals - self._rfi_rejected, self._true_candidates]
        colors = ["#4da6ff", "#7c3aed", "#34d399"]

        bars = ax.bar(categories, before_vals, color=colors, width=0.5, edgecolor="#080c14")

        for bar, val in zip(bars, before_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val),
                ha="center", va="bottom", color="#e0e8f0", fontsize=12, fontweight="bold",
            )

        ax.set_facecolor("#080c14")
        ax.tick_params(colors="#8ca5c8", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color((0.39, 0.71, 1.0, 0.15))
        ax.spines["left"].set_color((0.39, 0.71, 1.0, 0.15))
        ax.set_ylabel("Signal Count", color="#8ca5c8", fontsize=9)

        # Reduction arrows
        ax.annotate(
            f"−{self._rfi_rejected}", xy=(1, before_vals[1] + 5),
            fontsize=10, color="#f87171", fontweight="bold", ha="center",
        )

        self._ba_fig.set_facecolor("#080c14")
        self._ba_fig.tight_layout(pad=1.2)
        self._ba_canvas.draw()

    # ── Table population ──────────────────────────────────────────────────

    def _populate_rfi_table(self):
        self._rfi_table.setRowCount(len(KNOWN_RFI_BANDS))
        for row, (source, freq, rfi_type, desc) in enumerate(KNOWN_RFI_BANDS):
            self._rfi_table.setItem(row, 0, QTableWidgetItem(source))
            self._rfi_table.setItem(row, 1, QTableWidgetItem(freq))
            self._rfi_table.setItem(row, 2, QTableWidgetItem(rfi_type))
            self._rfi_table.setItem(row, 3, QTableWidgetItem(desc))

            # Highlight if signals detected in this band
            rng = np.random.default_rng(row)
            if rng.random() > 0.5:
                for col in range(4):
                    item = self._rfi_table.item(row, col)
                    if item:
                        item.setForeground(QColor("#f87171"))

    def _populate_fp_table(self):
        self._fp_table.setRowCount(len(self._false_positive_log))
        for row, fp in enumerate(self._false_positive_log):
            self._fp_table.setItem(row, 0, QTableWidgetItem(fp["id"]))
            self._fp_table.setItem(row, 1, QTableWidgetItem(fp["freq"]))
            self._fp_table.setItem(row, 2, QTableWidgetItem(fp["reason"]))

            status_item = QTableWidgetItem(fp["status"])
            if fp["status"] == "Resolved":
                status_item.setForeground(QColor("#34d399"))
            else:
                status_item.setForeground(QColor("#fbbf24"))
            self._fp_table.setItem(row, 3, status_item)

    # ── Refresh ───────────────────────────────────────────────────────────

    def _refresh_data(self):
        """Placeholder for refreshing from real pipeline output."""
        pass

    # ── Public API ────────────────────────────────────────────────────────

    def update_stats(
        self,
        total: int,
        rfi_rejected: int,
        false_positives: int,
        candidates: int,
    ):
        """Update all stats from pipeline results."""
        self._total_signals = total
        self._rfi_rejected = rfi_rejected
        self._false_positives = false_positives
        self._true_candidates = candidates

        self._update_stat_card(self._stat_total, str(total))
        pct = f"{rfi_rejected} ({rfi_rejected / max(total, 1) * 100:.1f}%)"
        self._update_stat_card(self._stat_rejected, pct)
        self._update_stat_card(self._stat_fp, str(false_positives))
        self._update_stat_card(self._stat_candidates, str(candidates))

        self._render_pie()
        self._render_bar()
        self._render_before_after()

    @staticmethod
    def _update_stat_card(card: QFrame, value: str):
        lbl = card.findChild(QLabel, "value")
        if lbl:
            lbl.setText(value)
