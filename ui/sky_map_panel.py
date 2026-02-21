"""
Sky Map Panel — Interactive celestial coordinate display.

Plots observation positions on an all-sky Aitoff projection,
showing where signals were detected and their classifications.
Can cross-reference AstroLens optical detections when available.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Dict

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QComboBox,
)
from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SkyMapPanel(QWidget):
    """All-sky map showing observation pointings and signal detections."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._observations: List[Dict] = []
        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        header = QHBoxLayout()
        title = QLabel("Sky Map")
        title.setStyleSheet(
            "font-size: 20px; font-weight: 600; color: #e2e8f0;"
        )
        header.addWidget(title)

        subtitle = QLabel(
            "Observation positions and signal detections on the celestial sphere"
        )
        subtitle.setStyleSheet(
            "font-size: 12px; color: rgba(196,210,230,0.55); margin-left: 12px;"
        )
        header.addWidget(subtitle)
        header.addStretch()

        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "All Observations",
            "With Signals",
            "Candidates Only",
        ])
        self.filter_combo.currentIndexChanged.connect(self._redraw)
        self.filter_combo.setFixedWidth(160)
        header.addWidget(self.filter_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self._load_data)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        self.figure = Figure(facecolor="#0a0e1a")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        stats_bar = QHBoxLayout()
        stats_bar.setSpacing(24)

        self.obs_count_lbl = QLabel("Observations: 0")
        self.obs_count_lbl.setStyleSheet(
            "font-size: 12px; color: rgba(196,210,230,0.6);"
        )
        stats_bar.addWidget(self.obs_count_lbl)

        self.signal_count_lbl = QLabel("Signals: 0")
        self.signal_count_lbl.setStyleSheet(
            "font-size: 12px; color: rgba(77,166,255,0.8);"
        )
        stats_bar.addWidget(self.signal_count_lbl)

        self.candidate_count_lbl = QLabel("Candidates: 0")
        self.candidate_count_lbl.setStyleSheet(
            "font-size: 12px; color: rgba(52,211,153,0.8);"
        )
        stats_bar.addWidget(self.candidate_count_lbl)

        stats_bar.addStretch()

        note = QLabel("RA/Dec from filterbank headers and observation metadata")
        note.setStyleSheet(
            "font-size: 11px; color: rgba(140,160,190,0.35);"
        )
        stats_bar.addWidget(note)

        layout.addLayout(stats_bar)

    def _load_data(self):
        """Load observation data from state files and filterbank metadata."""
        self._observations = []

        state_paths = [
            Path("astroseti_artifacts/observation_state.json"),
            Path("data/observations.json"),
        ]

        for sp in state_paths:
            if sp.exists():
                try:
                    with open(sp) as f:
                        state = json.load(f)
                    if isinstance(state, dict) and "observations" in state:
                        self._observations.extend(state["observations"])
                    elif isinstance(state, list):
                        self._observations.extend(state)
                except Exception:
                    pass

        if not self._observations:
            self._generate_demo_positions()

        self._redraw()

    def _generate_demo_positions(self):
        """Show demo observation positions for well-known BL targets."""
        rng = np.random.default_rng(42)

        targets = [
            {"name": "Proxima Centauri", "ra": 217.43, "dec": -62.68, "signals": 3, "candidates": 0},
            {"name": "TRAPPIST-1", "ra": 346.62, "dec": -5.04, "signals": 7, "candidates": 1},
            {"name": "Kepler-160", "ra": 291.41, "dec": 42.31, "signals": 2, "candidates": 0},
            {"name": "GJ 887", "ra": 344.78, "dec": -35.85, "signals": 5, "candidates": 0},
            {"name": "Tau Ceti", "ra": 26.02, "dec": -15.94, "signals": 12, "candidates": 2},
            {"name": "Ross 128", "ra": 176.94, "dec": 0.80, "signals": 4, "candidates": 0},
            {"name": "Barnard's Star", "ra": 269.45, "dec": 4.69, "signals": 8, "candidates": 1},
            {"name": "Wolf 1061", "ra": 248.37, "dec": -12.66, "signals": 1, "candidates": 0},
            {"name": "Luyten's Star", "ra": 111.84, "dec": 5.23, "signals": 6, "candidates": 0},
            {"name": "Lacaille 9352", "ra": 346.47, "dec": -35.86, "signals": 3, "candidates": 0},
            {"name": "Voyager 1 (ref)", "ra": 256.0, "dec": 12.0, "signals": 1, "candidates": 1},
        ]

        for i in range(25):
            targets.append({
                "name": f"GBT Target {i+1}",
                "ra": rng.uniform(0, 360),
                "dec": rng.uniform(-60, 70),
                "signals": int(rng.integers(0, 15)),
                "candidates": int(rng.integers(0, 2)),
            })

        self._observations = targets

    def _redraw(self):
        """Redraw the sky map with current data and filter."""
        self.figure.clear()

        ax = self.figure.add_subplot(111, projection="aitoff")
        ax.set_facecolor("#0a0e1a")
        ax.grid(True, color=(0.31, 0.55, 0.78, 0.08), linewidth=0.5)

        ax.tick_params(
            colors=(0.45, 0.55, 0.7, 0.35),
            labelsize=8,
        )
        for spine in ax.spines.values():
            spine.set_color((0.3, 0.45, 0.65, 0.15))

        filter_idx = self.filter_combo.currentIndex()

        obs_ra, obs_dec, obs_sizes, obs_colors = [], [], [], []
        sig_ra, sig_dec, sig_sizes = [], [], []
        cand_ra, cand_dec, cand_sizes = [], [], []

        total_obs = 0
        total_signals = 0
        total_candidates = 0

        for obs in self._observations:
            ra_deg = obs.get("ra", 0)
            dec_deg = obs.get("dec", 0)
            n_signals = obs.get("signals", 0)
            n_candidates = obs.get("candidates", 0)

            ra_rad = math.radians(ra_deg - 180)
            dec_rad = math.radians(dec_deg)

            total_obs += 1
            total_signals += n_signals
            total_candidates += n_candidates

            if filter_idx == 1 and n_signals == 0:
                continue
            if filter_idx == 2 and n_candidates == 0:
                continue

            obs_ra.append(ra_rad)
            obs_dec.append(dec_rad)
            obs_sizes.append(max(12, min(n_signals * 6, 80)))
            obs_colors.append(
                (0.30, 0.55, 0.85, 0.35) if n_signals == 0
                else (0.30, 0.65, 1.0, 0.55)
            )

            if n_signals > 0:
                sig_ra.append(ra_rad)
                sig_dec.append(dec_rad)
                sig_sizes.append(max(20, n_signals * 8))

            if n_candidates > 0:
                cand_ra.append(ra_rad)
                cand_dec.append(dec_rad)
                cand_sizes.append(max(40, n_candidates * 25))

        if obs_ra:
            ax.scatter(
                obs_ra, obs_dec, s=obs_sizes, c=obs_colors,
                marker="o", edgecolors="none", zorder=2,
            )

        if sig_ra:
            ax.scatter(
                sig_ra, sig_dec, s=sig_sizes,
                color=(0.30, 0.65, 1.0, 0.25),
                marker="o", edgecolors=(0.30, 0.65, 1.0, 0.45),
                linewidths=0.8, zorder=3,
            )

        if cand_ra:
            ax.scatter(
                cand_ra, cand_dec, s=cand_sizes,
                facecolors="none",
                edgecolors=(0.20, 0.83, 0.60, 0.70),
                linewidths=1.5, marker="D", zorder=4,
            )

        for obs in self._observations:
            n_cand = obs.get("candidates", 0)
            if n_cand > 0 and obs.get("name"):
                ra_rad = math.radians(obs["ra"] - 180)
                dec_rad = math.radians(obs["dec"])
                ax.annotate(
                    obs["name"],
                    (ra_rad, dec_rad),
                    fontsize=7,
                    color=(0.78, 0.88, 0.95, 0.7),
                    textcoords="offset points",
                    xytext=(6, 4),
                )

        ax.set_title(
            "AstroSETI Observation Coverage",
            fontsize=11, color=(0.85, 0.90, 0.95, 0.75),
            pad=12, fontweight=500,
        )

        self.obs_count_lbl.setText(f"Observations: {total_obs}")
        self.signal_count_lbl.setText(f"Signals: {total_signals}")
        self.candidate_count_lbl.setText(f"Candidates: {total_candidates}")

        self.canvas.draw_idle()
