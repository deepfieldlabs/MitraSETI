"""
Radar Visualizer — Animated radar-style display of celestial observations.

Replaces the static Aitoff sky map with a real-time radar sweep
that shows observation positions, signal detections, and candidate
targets with phosphor-persistence fading.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QComboBox, QToolTip,
)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QRadialGradient, QConicalGradient,
    QLinearGradient, QFont, QFontMetrics, QPainterPath, QBrush,
)


_BG_COLOR = QColor(8, 12, 24)
_GRID_COLOR = QColor(30, 70, 50, 45)
_GRID_BRIGHT = QColor(30, 90, 55, 70)
_SWEEP_COLOR = QColor(0, 255, 80)
_TEXT_DIM = QColor(80, 140, 100, 120)
_TEXT_BRIGHT = QColor(140, 220, 160, 200)

_CATEGORY_COLORS = {
    "candidate": QColor(52, 211, 153),
    "signal": QColor(77, 166, 255),
    "rfi": QColor(255, 100, 80, 100),
    "observation": QColor(60, 130, 90, 140),
    "anomaly": QColor(255, 200, 40),
}

_RING_COUNT = 6
_SWEEP_SPEED = 0.8  # radians per second
_TRAIL_ARC = math.radians(35)
_BLIP_PERSISTENCE = 4.0  # seconds a blip stays bright after sweep passes
_FPS = 30


class _TargetBlip:
    """A single target on the radar."""
    __slots__ = (
        "name", "ra_deg", "dec_deg", "angle", "radius",
        "n_signals", "n_candidates", "category", "last_swept",
        "frequency_mhz", "snr", "classification",
    )

    def __init__(self, obs: Dict):
        self.name: str = obs.get("name", "")
        self.ra_deg: float = float(obs.get("ra", 0))
        self.dec_deg: float = float(obs.get("dec", 0))
        self.n_signals: int = int(obs.get("signals", 0))
        self.n_candidates: int = int(obs.get("candidates", 0))
        self.frequency_mhz: float = float(obs.get("frequency_mhz", 0))
        self.snr: float = float(obs.get("snr", 0))
        self.classification: str = obs.get("classification", "")

        self.angle = math.radians(self.ra_deg)
        dec_norm = (90.0 - self.dec_deg) / 180.0
        self.radius = max(0.05, min(dec_norm, 0.95))

        if self.n_candidates > 0:
            self.category = "candidate"
        elif self.n_signals > 0:
            self.category = "signal"
        else:
            self.category = "observation"

        self.last_swept: float = 0.0


class RadarWidget(QWidget):
    """Custom QPainter radar display with animated sweep and phosphor blips."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._blips: List[_TargetBlip] = []
        self._sweep_angle = 0.0
        self._start_time = time.monotonic()
        self._hovered_blip: Optional[_TargetBlip] = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000 // _FPS)

    def set_blips(self, blips: List[_TargetBlip]):
        self._blips = blips

    def _tick(self):
        now = time.monotonic()
        dt = now - self._start_time
        self._sweep_angle = (_SWEEP_SPEED * dt) % (2 * math.pi)

        for blip in self._blips:
            angle_diff = (self._sweep_angle - blip.angle) % (2 * math.pi)
            if angle_diff < math.radians(3):
                blip.last_swept = now

        self.update()

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        side = min(w, h) - 20
        cx, cy = w / 2, h / 2
        r_max = side / 2

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), _BG_COLOR)

        self._draw_grid(painter, cx, cy, r_max)
        self._draw_sweep(painter, cx, cy, r_max)
        self._draw_blips(painter, cx, cy, r_max)
        self._draw_center_dot(painter, cx, cy)
        self._draw_labels(painter, cx, cy, r_max)

        painter.end()

    def _draw_grid(self, p: QPainter, cx: float, cy: float, r_max: float):
        pen = QPen(_GRID_COLOR, 1.0)
        p.setPen(pen)
        for i in range(1, _RING_COUNT + 1):
            r = r_max * i / _RING_COUNT
            p.drawEllipse(QPointF(cx, cy), r, r)

        p.setPen(QPen(_GRID_BRIGHT, 0.5))
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            x1 = cx + r_max * 0.04 * math.cos(rad)
            y1 = cy - r_max * 0.04 * math.sin(rad)
            x2 = cx + r_max * math.cos(rad)
            y2 = cy - r_max * math.sin(rad)
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        pen_cross = QPen(_GRID_COLOR, 1.2)
        p.setPen(pen_cross)
        p.drawLine(QPointF(cx - r_max, cy), QPointF(cx + r_max, cy))
        p.drawLine(QPointF(cx, cy - r_max), QPointF(cx, cy + r_max))

    def _draw_sweep(self, p: QPainter, cx: float, cy: float, r_max: float):
        trail_steps = 30
        for i in range(trail_steps):
            frac = i / trail_steps
            a = self._sweep_angle - frac * _TRAIL_ARC
            alpha = int(120 * (1.0 - frac) ** 2)
            color = QColor(0, 255, 80, alpha)
            pen = QPen(color, max(1.0, 2.5 * (1.0 - frac)))
            p.setPen(pen)
            x = cx + r_max * math.cos(a)
            y = cy - r_max * math.sin(a)
            p.drawLine(QPointF(cx, cy), QPointF(x, y))

        glow_grad = QRadialGradient(cx, cy, r_max * 0.15)
        glow_grad.setColorAt(0.0, QColor(0, 255, 80, 25))
        glow_grad.setColorAt(1.0, QColor(0, 255, 80, 0))
        p.setBrush(QBrush(glow_grad))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), r_max * 0.15, r_max * 0.15)

    def _draw_blips(self, p: QPainter, cx: float, cy: float, r_max: float):
        now = time.monotonic()

        for blip in self._blips:
            bx = cx + r_max * blip.radius * math.cos(blip.angle)
            by = cy - r_max * blip.radius * math.sin(blip.angle)

            age = now - blip.last_swept if blip.last_swept > 0 else _BLIP_PERSISTENCE + 1
            brightness = max(0.0, 1.0 - age / _BLIP_PERSISTENCE)

            base_color = _CATEGORY_COLORS.get(blip.category, _CATEGORY_COLORS["observation"])

            if brightness > 0.01:
                alpha = int(40 + 215 * brightness)
                color = QColor(base_color.red(), base_color.green(), base_color.blue(), alpha)

                dot_r = 3.0 + 4.0 * brightness
                if blip.category == "candidate":
                    dot_r += 2.0

                if brightness > 0.5:
                    glow_r = dot_r * 3.0 * brightness
                    glow = QRadialGradient(bx, by, glow_r)
                    glow.setColorAt(0.0, QColor(base_color.red(), base_color.green(), base_color.blue(), int(60 * brightness)))
                    glow.setColorAt(1.0, QColor(base_color.red(), base_color.green(), base_color.blue(), 0))
                    p.setBrush(QBrush(glow))
                    p.setPen(Qt.NoPen)
                    p.drawEllipse(QPointF(bx, by), glow_r, glow_r)

                p.setBrush(QBrush(color))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(bx, by), dot_r, dot_r)

                if blip.category == "candidate" and brightness > 0.3:
                    ring_alpha = int(180 * brightness)
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor(52, 211, 153, ring_alpha), 1.5))
                    ring_r = dot_r + 4.0 + 3.0 * math.sin(now * 3.0)
                    p.drawEllipse(QPointF(bx, by), ring_r, ring_r)

                if brightness > 0.7 and blip.name:
                    p.setPen(QPen(QColor(200, 240, 210, int(200 * brightness)), 1))
                    font = QFont("Menlo", 8)
                    p.setFont(font)
                    p.drawText(QPointF(bx + dot_r + 4, by - 2), blip.name)

            else:
                dim_alpha = 25 if blip.category == "observation" else 45
                p.setBrush(QBrush(QColor(base_color.red(), base_color.green(), base_color.blue(), dim_alpha)))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(bx, by), 2.0, 2.0)

    def _draw_center_dot(self, p: QPainter, cx: float, cy: float):
        glow = QRadialGradient(cx, cy, 10)
        glow.setColorAt(0.0, QColor(0, 255, 80, 80))
        glow.setColorAt(0.5, QColor(0, 200, 60, 30))
        glow.setColorAt(1.0, QColor(0, 200, 60, 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), 10, 10)

        p.setBrush(QBrush(QColor(0, 255, 80, 200)))
        p.drawEllipse(QPointF(cx, cy), 2.5, 2.5)

    def _draw_labels(self, p: QPainter, cx: float, cy: float, r_max: float):
        font = QFont("Menlo", 8)
        p.setFont(font)
        p.setPen(QPen(_TEXT_DIM, 1))

        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            lx = cx + (r_max + 14) * math.cos(rad)
            ly = cy - (r_max + 14) * math.sin(rad)
            label = f"{deg}°"
            fm = QFontMetrics(font)
            tw = fm.horizontalAdvance(label)
            p.drawText(QPointF(lx - tw / 2, ly + fm.ascent() / 2), label)

        p.setPen(QPen(QColor(60, 130, 90, 80), 1))
        for i in range(1, _RING_COUNT + 1):
            r = r_max * i / _RING_COUNT
            dec_val = 90.0 - i * 180.0 / _RING_COUNT
            label = f"{dec_val:+.0f}°"
            p.drawText(QPointF(cx + 4, cy - r + 12), label)

    def mouseMoveEvent(self, event):
        w, h = self.width(), self.height()
        side = min(w, h) - 20
        cx, cy = w / 2, h / 2
        r_max = side / 2
        mx, my = event.x(), event.y()

        closest = None
        closest_dist = 20.0

        for blip in self._blips:
            bx = cx + r_max * blip.radius * math.cos(blip.angle)
            by = cy - r_max * blip.radius * math.sin(blip.angle)
            dist = math.hypot(mx - bx, my - by)
            if dist < closest_dist:
                closest = blip
                closest_dist = dist

        if closest:
            lines = [f"Target: {closest.name or 'Unknown'}"]
            lines.append(f"RA: {closest.ra_deg:.2f}°  Dec: {closest.dec_deg:.2f}°")
            if closest.n_signals:
                lines.append(f"Signals: {closest.n_signals}")
            if closest.n_candidates:
                lines.append(f"Candidates: {closest.n_candidates}")
            if closest.snr:
                lines.append(f"SNR: {closest.snr:.1f}")
            if closest.classification:
                lines.append(f"Class: {closest.classification}")
            QToolTip.showText(event.globalPos(), "\n".join(lines), self)
        else:
            QToolTip.hideText()


class SkyMapPanel(QWidget):
    """Radar-style space visualization of observation targets and detections."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._observations: List[Dict] = []
        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Space Radar")
        title.setStyleSheet(
            "font-size: 20px; font-weight: 600; color: #34d399;"
        )
        header.addWidget(title)

        subtitle = QLabel(
            "Live sweep of observed targets — RA/Dec polar projection"
        )
        subtitle.setStyleSheet(
            "font-size: 12px; color: rgba(100,180,130,0.5); margin-left: 12px;"
        )
        header.addWidget(subtitle)
        header.addStretch()

        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "All Targets",
            "With Signals",
            "Candidates Only",
        ])
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        self.filter_combo.setFixedWidth(160)
        header.addWidget(self.filter_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self._load_data)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        self.radar = RadarWidget()
        layout.addWidget(self.radar, stretch=1)

        stats_bar = QHBoxLayout()
        stats_bar.setSpacing(24)

        self.obs_count_lbl = QLabel("Targets: 0")
        self.obs_count_lbl.setStyleSheet(
            "font-size: 12px; color: rgba(100,180,130,0.6);"
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

        legend_items = [
            ("●", "#3c825a", "Observation"),
            ("●", "#4da6ff", "Signal"),
            ("◆", "#34d399", "Candidate"),
        ]
        for symbol, color, label in legend_items:
            lbl = QLabel(f'<span style="color:{color}">{symbol}</span> {label}')
            lbl.setStyleSheet("font-size: 11px; color: rgba(160,200,180,0.5);")
            stats_bar.addWidget(lbl)

        layout.addLayout(stats_bar)

    def _load_data(self):
        """Load observation data from streaming state and verified candidates."""
        self._observations = []

        from paths import DATA_DIR, CANDIDATES_DIR

        state_file = DATA_DIR / "streaming_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                cats = state.get("categories", {})
                for cat_name, cat_data in cats.items():
                    if isinstance(cat_data, dict) and cat_data.get("files_processed", 0) > 0:
                        self._observations.append({
                            "name": cat_name,
                            "ra": hash(cat_name) % 360,
                            "dec": (hash(cat_name + "d") % 120) - 60,
                            "signals": cat_data.get("signals_detected", 0),
                            "candidates": cat_data.get("verified_candidates", 0),
                        })
            except Exception:
                pass

        cand_file = CANDIDATES_DIR / "verified_candidates.json"
        if cand_file.exists():
            try:
                with open(cand_file) as f:
                    candidates = json.load(f)
                for i, c in enumerate(candidates):
                    target = c.get("target_name", c.get("category", f"Target-{i}"))
                    freq_mhz = c.get("frequency_hz", 0) / 1e6 if c.get("frequency_hz") else 0
                    seed_str = f"{target}_{freq_mhz:.4f}_{i}"
                    ra = float(c.get("ra", (hash(seed_str) % 3600) / 10.0))
                    dec = float(c.get("dec", (hash(seed_str + "d") % 1200 - 600) / 10.0))
                    self._observations.append({
                        "name": target,
                        "ra": ra,
                        "dec": dec,
                        "signals": 1,
                        "candidates": 1,
                        "frequency_mhz": freq_mhz,
                        "snr": c.get("snr", 0),
                        "classification": c.get("classification", ""),
                    })
            except Exception:
                pass

        if not self._observations:
            self._generate_demo_positions()

        self._apply_filter()

    def _generate_demo_positions(self):
        """Demo observation positions for well-known BL targets."""
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

        for i in range(30):
            targets.append({
                "name": f"GBT-{i+1:03d}",
                "ra": float(rng.uniform(0, 360)),
                "dec": float(rng.uniform(-60, 70)),
                "signals": int(rng.integers(0, 15)),
                "candidates": int(rng.integers(0, 2)),
            })

        self._observations = targets

    def _apply_filter(self):
        """Apply the dropdown filter and update the radar blips."""
        filter_idx = self.filter_combo.currentIndex()
        filtered = []
        total_signals = 0
        total_candidates = 0

        for obs in self._observations:
            n_sig = obs.get("signals", 0)
            n_cand = obs.get("candidates", 0)
            total_signals += n_sig
            total_candidates += n_cand

            if filter_idx == 1 and n_sig == 0:
                continue
            if filter_idx == 2 and n_cand == 0:
                continue
            filtered.append(obs)

        blips = [_TargetBlip(obs) for obs in filtered]
        self.radar.set_blips(blips)

        self.obs_count_lbl.setText(f"Targets: {len(self._observations)}")
        self.signal_count_lbl.setText(f"Signals: {total_signals}")
        self.candidate_count_lbl.setText(f"Candidates: {total_candidates}")
