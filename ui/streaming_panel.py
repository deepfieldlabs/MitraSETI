"""
astroSETI Streaming Panel — Continuous Observation Monitor

Adapted from AstroLens streaming_panel.py with crystalline glass theme.
Controls continuous SETI signal processing with:
- Start/Stop controls
- Duration and mode selectors
- Real-time stat cards
- Live log viewer
- Progress tracking
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QGridLayout, QProgressBar, QTextEdit,
    QSpinBox, QComboBox, QGroupBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush

from .theme import COLORS, make_stat_card, create_glow_button


# ── Paths ─────────────────────────────────────────────────────────────────────

_ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "astroseti_artifacts"
_DATA_DIR = _ARTIFACTS_DIR / "data"
_STREAMING_STATE = _DATA_DIR / "streaming_state.json"
_STREAMING_LOG = _DATA_DIR / "streaming_observation.log"
_REPORTS_DIR = _ARTIFACTS_DIR / "streaming_reports" / "daily"


# ── Pulsing status dot ───────────────────────────────────────────────────────

class _LiveDot(QFrame):
    """Animated status dot with crystalline glow."""

    def __init__(self, size: int = 12, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._color = QColor(COLORS["text_tertiary"])

    def set_active(self, active: bool):
        self._color = QColor(COLORS["success"] if active else COLORS["text_tertiary"])
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor(self._color)
        p.setBrush(QBrush(c))
        p.setPen(Qt.NoPen)
        p.drawEllipse(1, 1, self.width() - 2, self.height() - 2)


# ── Streaming Panel ──────────────────────────────────────────────────────────

class StreamingPanel(QWidget):
    """Continuous SETI observation monitor — crystalline glass theme."""

    anomaly_found = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: Optional[subprocess.Popen] = None
        self._log_file_pos = 0

        self._setup_ui()

        # Refresh timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(5000)

        QTimer.singleShot(500, self._refresh)

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 24, 28, 24)

        # ── Header ────────────────────────────────────────────────────────
        header = QHBoxLayout()

        title_col = QVBoxLayout()
        title_row = QHBoxLayout()
        self._status_dot = _LiveDot(size=10)
        title_row.addWidget(self._status_dot)
        title_label = QLabel("📶  Streaming Monitor")
        title_label.setStyleSheet(
            "font-size: 20px; font-weight: 600; color: #e0e8f0;"
        )
        title_row.addWidget(title_label)
        title_row.addStretch()
        title_col.addLayout(title_row)

        self._status_label = QLabel("Idle — waiting to start")
        self._status_label.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text_tertiary']}; margin-left: 22px;"
        )
        title_col.addWidget(self._status_label)
        header.addLayout(title_col)
        header.addStretch()

        layout.addLayout(header)

        # ── Controls bar ──────────────────────────────────────────────────
        controls = QFrame()
        controls.setStyleSheet("""
            QFrame {
                background: rgba(15, 25, 45, 0.65);
                border: 1px solid rgba(100, 180, 255, 0.1);
                border-radius: 12px;
            }
        """)
        ctrl_layout = QHBoxLayout(controls)
        ctrl_layout.setContentsMargins(20, 14, 20, 14)
        ctrl_layout.setSpacing(14)

        # Duration
        dur_lbl = QLabel("Duration:")
        dur_lbl.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text_secondary']}; border: none;"
        )
        ctrl_layout.addWidget(dur_lbl)

        self._days_spin = QSpinBox()
        self._days_spin.setRange(1, 30)
        self._days_spin.setValue(7)
        self._days_spin.setSuffix(" days")
        self._days_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(8, 12, 20, 0.8);
                border: 1px solid rgba(100, 180, 255, 0.12);
                border-radius: 6px;
                padding: 6px 10px;
                color: #e0e8f0;
                font-size: 12px;
            }
            QSpinBox:focus {
                border-color: rgba(0, 212, 255, 0.4);
            }
        """)
        ctrl_layout.addWidget(self._days_spin)

        # Mode
        mode_lbl = QLabel("Mode:")
        mode_lbl.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text_secondary']}; border: none;"
        )
        ctrl_layout.addWidget(mode_lbl)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Normal", "Aggressive", "Turbo"])
        self._mode_combo.setFixedWidth(120)
        ctrl_layout.addWidget(self._mode_combo)

        ctrl_layout.addStretch()

        # Start / Stop button
        self._start_btn = QPushButton("▶  Start Streaming")
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 200, 120, 0.85), stop:1 rgba(0, 255, 136, 0.9));
                border: 1px solid rgba(0, 255, 136, 0.3);
                border-radius: 8px;
                padding: 9px 24px;
                color: #080c14;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: rgba(0, 255, 136, 1);
                border-color: rgba(0, 255, 136, 0.5);
            }
            QPushButton:disabled {
                background: rgba(15, 25, 45, 0.5);
                color: rgba(140,165,200,0.3);
                border-color: transparent;
            }
        """)
        self._start_btn.clicked.connect(self._toggle_streaming)
        ctrl_layout.addWidget(self._start_btn)

        layout.addWidget(controls)

        # ── Progress section ──────────────────────────────────────────────
        progress_frame = QFrame()
        progress_frame.setStyleSheet("""
            QFrame {
                background: rgba(15, 25, 45, 0.4);
                border: 1px solid rgba(100, 180, 255, 0.08);
                border-radius: 10px;
            }
        """)
        prog_layout = QVBoxLayout(progress_frame)
        prog_layout.setContentsMargins(20, 14, 20, 14)
        prog_layout.setSpacing(8)

        prog_header = QHBoxLayout()
        self._elapsed_label = QLabel("Elapsed: 0:00:00")
        self._elapsed_label.setStyleSheet(
            "font-size: 12px; color: rgba(200,215,235,0.5); border: none;"
        )
        prog_header.addWidget(self._elapsed_label)
        prog_header.addStretch()
        self._files_label = QLabel("Files: 0 processed")
        self._files_label.setStyleSheet(
            "font-size: 12px; color: rgba(200,215,235,0.5); border: none;"
        )
        prog_header.addWidget(self._files_label)
        prog_header.addStretch()
        self._current_file_label = QLabel("Current: —")
        self._current_file_label.setStyleSheet(
            "font-size: 12px; color: rgba(0,212,255,0.6); border: none;"
        )
        prog_header.addWidget(self._current_file_label)
        prog_layout.addLayout(prog_header)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("Day %v of %m")
        self._progress_bar.setFixedHeight(18)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(8, 12, 20, 0.7);
                border: 1px solid rgba(100, 180, 255, 0.08);
                border-radius: 5px;
                text-align: center;
                color: rgba(200,215,235,0.6);
                font-size: 11px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 180, 220, 0.8),
                    stop:0.6 rgba(0, 212, 255, 0.9),
                    stop:1 rgba(124, 58, 237, 0.8));
                border-radius: 4px;
            }
        """)
        prog_layout.addWidget(self._progress_bar)

        layout.addWidget(progress_frame)

        # ── Stats grid ────────────────────────────────────────────────────
        stats_grid = QGridLayout()
        stats_grid.setSpacing(10)

        self._stat_cards = {}
        stat_defs = [
            ("signals", "Signals Processed", "0", "#4da6ff"),
            ("candidates", "Candidates Found", "0", "#34d399"),
            ("rfi", "RFI Rejected", "0", "#f87171"),
            ("rate", "Processing Rate", "0/min", "#7c3aed"),
        ]

        for i, (key, label, default, color) in enumerate(stat_defs):
            card = make_stat_card(label, default, color)
            stats_grid.addWidget(card, 0, i)
            self._stat_cards[key] = card

        layout.addLayout(stats_grid)

        # ── Live log viewer ───────────────────────────────────────────────
        _group_ss = f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 600;
                color: #e0e8f0;
                border: 1px solid rgba(100, 180, 255, 0.08);
                border-radius: 12px;
                padding-top: 26px;
                margin-top: 8px;
                background: rgba(15, 25, 45, 0.35);
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                color: rgba(0, 212, 255, 0.7);
                font-size: 11px;
                letter-spacing: 0.5px;
            }}
        """
        _mono_ss = """
            QTextEdit {
                background: rgba(8, 12, 20, 0.7);
                border: 1px solid rgba(100, 180, 255, 0.06);
                border-radius: 8px;
                color: #e0e8f0;
                font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """

        log_group = QGroupBox("Live Output")
        log_group.setStyleSheet(_group_ss)
        log_layout = QVBoxLayout(log_group)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet(_mono_ss)
        self._log_text.setMinimumHeight(160)
        self._log_text.setPlaceholderText(
            "  Streaming log will appear here when processing starts…"
        )
        log_layout.addWidget(self._log_text)

        # Open reports button
        reports_row = QHBoxLayout()
        reports_row.addStretch()
        open_reports_btn = QPushButton("📂  Open Reports Folder")
        open_reports_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 212, 255, 0.08);
                border: 1px solid rgba(0, 212, 255, 0.2);
                border-radius: 8px;
                padding: 8px 16px;
                color: #4da6ff;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(0, 212, 255, 0.16);
                border-color: rgba(0, 212, 255, 0.4);
            }
        """)
        open_reports_btn.clicked.connect(self._open_reports_folder)
        reports_row.addWidget(open_reports_btn)
        log_layout.addLayout(reports_row)

        layout.addWidget(log_group, 1)

    # ── Start / Stop ──────────────────────────────────────────────────────

    def _toggle_streaming(self):
        if self._process and self._process.poll() is None:
            self._stop_streaming()
        else:
            self._start_streaming()

    def _start_streaming(self):
        days = self._days_spin.value()
        hours = days * 24
        mode = self._mode_combo.currentText().lower()

        self._log_file_pos = 0
        self._log_text.setPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] Launching streaming observation…\n"
        )

        try:
            script = str(Path(__file__).parent.parent / "scripts" / "streaming_observation.py")
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            log_handle = open(_STREAMING_LOG, "a")
            self._log_handle = log_handle
            self._process = subprocess.Popen(
                [sys.executable, "-u", script, "--hours", str(hours), "--mode", mode],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:
            self._log_text.append(f"[ERROR] Failed to launch: {exc}")
            return

        self._start_btn.setText("⏹  Stop Streaming")
        self._start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 40, 80, 0.85), stop:1 rgba(255, 51, 102, 0.9));
                border: 1px solid rgba(255, 51, 102, 0.3);
                border-radius: 8px;
                padding: 9px 24px;
                color: #ffffff;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: rgba(255, 51, 102, 1);
            }
        """)
        self._status_dot.set_active(True)
        self._status_label.setText(f"Running — {mode} mode, {days} days (PID {self._process.pid})")
        self._status_label.setStyleSheet(
            f"font-size: 12px; color: {COLORS['success']}; margin-left: 22px;"
        )
        self._days_spin.setEnabled(False)
        self._mode_combo.setEnabled(False)

    def _stop_streaming(self):
        if self._process and self._process.poll() is None:
            import signal as _sig
            try:
                self._process.send_signal(_sig.SIGTERM)
            except OSError:
                self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()

        self._process = None
        if hasattr(self, "_log_handle") and self._log_handle:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
        self._start_btn.setText("▶  Start Streaming")
        self._start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 200, 120, 0.85), stop:1 rgba(0, 255, 136, 0.9));
                border: 1px solid rgba(0, 255, 136, 0.3);
                border-radius: 8px;
                padding: 9px 24px;
                color: #080c14;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: rgba(0, 255, 136, 1);
                border-color: rgba(0, 255, 136, 0.5);
            }
        """)
        self._status_dot.set_active(False)
        self._status_label.setText("Stopped")
        self._status_label.setStyleSheet(
            f"font-size: 12px; color: {COLORS['warning']}; margin-left: 22px;"
        )
        self._days_spin.setEnabled(True)
        self._mode_combo.setEnabled(True)

    # ── Refresh ───────────────────────────────────────────────────────────

    def _refresh(self):
        """Refresh stats from state files and live log."""
        # Check if our process died
        if self._process and self._process.poll() is not None:
            self._stop_streaming()

        # Read streaming state if available
        state = self._read_json(_STREAMING_STATE)

        current_day = state.get("current_day", 0)
        target_days = state.get("target_days", 0) or self._days_spin.value()
        self._progress_bar.setMaximum(int(max(target_days, 1)))
        self._progress_bar.setValue(min(current_day, int(max(target_days, 1))))
        self._progress_bar.setFormat(f"Day {current_day} of {int(target_days)}")

        # Map state field names (script writes total_candidates, total_rfi_rejected, etc.)
        signals = state.get("total_signals", 0)
        candidates = state.get("total_candidates", state.get("candidates_found", 0))
        rfi = state.get("total_rfi_rejected", state.get("rfi_rejected", 0))
        rate = state.get("processing_rate", "0/min")

        self._update_stat("signals", f"{signals:,}" if signals else "0")
        self._update_stat("candidates", str(candidates), "#34d399" if candidates > 0 else "#4da6ff")
        self._update_stat("rfi", str(rfi), "#f87171" if rfi > 0 else "#4da6ff")
        self._update_stat("rate", str(rate))

        # Elapsed -- compute from started_at if not explicitly stored
        elapsed_str = state.get("elapsed", "")
        if not elapsed_str and state.get("started_at"):
            try:
                from datetime import datetime as _dt
                started = _dt.fromisoformat(state["started_at"])
                delta = _dt.now() - started
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                mins, secs = divmod(remainder, 60)
                elapsed_str = f"{hours}:{mins:02d}:{secs:02d}"
            except Exception:
                elapsed_str = "0:00:00"
        self._elapsed_label.setText(f"Elapsed: {elapsed_str or '0:00:00'}")

        files_done = state.get("total_files_processed", state.get("files_processed", 0))
        self._files_label.setText(f"Files: {files_done} processed")
        current_file = state.get("current_file", "—")
        self._current_file_label.setText(f"Current: {current_file}")

        # Read live log
        self._refresh_log()

    def _refresh_log(self):
        if not _STREAMING_LOG.exists():
            return

        try:
            file_size = _STREAMING_LOG.stat().st_size
            if file_size == self._log_file_pos:
                return

            read_from = max(0, file_size - 8192)
            with open(_STREAMING_LOG, "r", errors="replace") as f:
                f.seek(read_from)
                if read_from > 0:
                    f.readline()
                content = f.read()

            self._log_file_pos = file_size

            lines = content.strip().split("\n")
            display = "\n".join(lines[-60:])
            self._log_text.setPlainText(display)

            scrollbar = self._log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception:
            pass

    def _update_stat(self, key: str, value: str, color: str = ""):
        card = self._stat_cards.get(key)
        if card:
            val_lbl = card.findChild(QLabel, "value")
            if val_lbl:
                val_lbl.setText(str(value))
                if color:
                    val_lbl.setStyleSheet(
                        f"font-size: 26px; font-weight: 700; color: {color}; border: none;"
                    )

    def _open_reports_folder(self):
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        reports = sorted(_REPORTS_DIR.glob("*.html"), reverse=True)
        if reports:
            webbrowser.open(f"file://{reports[0]}")
        else:
            import platform
            if platform.system() == "Darwin":
                subprocess.Popen(["open", str(_REPORTS_DIR)])
            elif platform.system() == "Linux":
                subprocess.Popen(["xdg-open", str(_REPORTS_DIR)])
            else:
                subprocess.Popen(["explorer", str(_REPORTS_DIR)])

    @staticmethod
    def _read_json(path: Path) -> dict:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
