"""
MitraSETI Settings Panel

Configuration for pipeline, streaming, paths, and display preferences.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QSpinBox, QDoubleSpinBox, QComboBox,
    QLineEdit, QCheckBox, QGroupBox, QFileDialog, QMessageBox,
)
from PyQt5.QtCore import Qt

from .theme import COLORS

# Paths
_ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "mitraseti_artifacts"
_SETTINGS_FILE = _ARTIFACTS_DIR / "data" / "settings.json"


def _load_settings() -> dict:
    if _SETTINGS_FILE.exists():
        try:
            with open(_SETTINGS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_settings(settings: dict):
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


class SettingsPanel(QWidget):
    """Application settings and configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = _load_settings()
        self._setup_ui()
        self._load_values()

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(20)

        # Header
        title = QLabel("⚙️  Settings")
        title.setStyleSheet("font-size: 22px; font-weight: 600; color: #e0e8f0;")
        layout.addWidget(title)
        subtitle = QLabel("Configure pipeline parameters, paths, and display preferences")
        subtitle.setStyleSheet(f"font-size: 13px; color: {COLORS['text_secondary']};")
        layout.addWidget(subtitle)

        group_ss = f"""
            QGroupBox {{
                font-size: 14px; font-weight: 600; color: #4da6ff;
                border: 1px solid rgba(100, 180, 255, 0.1);
                border-radius: 12px;
                padding-top: 28px; margin-top: 8px;
                background: rgba(15, 25, 45, 0.45);
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 16px;
                padding: 0 8px; color: #4da6ff;
            }}
        """

        # ── Pipeline Settings ──
        pipeline_group = QGroupBox("Pipeline")
        pipeline_group.setStyleSheet(group_ss)
        pg = QVBoxLayout(pipeline_group)
        pg.setSpacing(12)

        # Min SNR
        row = QHBoxLayout()
        row.addWidget(self._lbl("Minimum SNR Threshold"))
        self._snr_spin = QDoubleSpinBox()
        self._snr_spin.setRange(1.0, 50.0)
        self._snr_spin.setSingleStep(0.5)
        self._snr_spin.setValue(10.0)
        self._snr_spin.setStyleSheet(self._input_ss())
        row.addWidget(self._snr_spin)
        pg.addLayout(row)

        # Max Drift Rate
        row = QHBoxLayout()
        row.addWidget(self._lbl("Max Drift Rate (Hz/s)"))
        self._drift_spin = QDoubleSpinBox()
        self._drift_spin.setRange(0.1, 20.0)
        self._drift_spin.setSingleStep(0.5)
        self._drift_spin.setValue(4.0)
        self._drift_spin.setStyleSheet(self._input_ss())
        row.addWidget(self._drift_spin)
        pg.addLayout(row)

        # ML Device
        row = QHBoxLayout()
        row.addWidget(self._lbl("ML Device"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "mps", "cuda"])
        self._device_combo.setFixedWidth(150)
        row.addWidget(self._device_combo)
        pg.addLayout(row)

        layout.addWidget(pipeline_group)

        # ── Streaming Settings ──
        stream_group = QGroupBox("Streaming")
        stream_group.setStyleSheet(group_ss)
        sg = QVBoxLayout(stream_group)
        sg.setSpacing(12)

        row = QHBoxLayout()
        row.addWidget(self._lbl("Default Duration (days)"))
        self._dur_spin = QSpinBox()
        self._dur_spin.setRange(1, 30)
        self._dur_spin.setValue(7)
        self._dur_spin.setStyleSheet(self._input_ss())
        row.addWidget(self._dur_spin)
        sg.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(self._lbl("Default Mode"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["normal", "aggressive", "turbo"])
        self._mode_combo.setFixedWidth(150)
        row.addWidget(self._mode_combo)
        sg.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(self._lbl("Auto-download BL data"))
        self._auto_dl_check = QCheckBox()
        self._auto_dl_check.setChecked(True)
        row.addWidget(self._auto_dl_check)
        row.addStretch()
        sg.addLayout(row)

        layout.addWidget(stream_group)

        # ── Paths ──
        paths_group = QGroupBox("Paths")
        paths_group.setStyleSheet(group_ss)
        ppg = QVBoxLayout(paths_group)
        ppg.setSpacing(12)

        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from paths import FILTERBANK_DIR, ARTIFACTS_DIR, MODELS_DIR
            fb_path = str(FILTERBANK_DIR)
            art_path = str(ARTIFACTS_DIR)
            mdl_path = str(MODELS_DIR)
        except Exception:
            fb_path = str(_ARTIFACTS_DIR / "data" / "filterbank")
            art_path = str(_ARTIFACTS_DIR)
            mdl_path = str(_ARTIFACTS_DIR / "models")

        for label_text, path_val in [
            ("Filterbank Directory", fb_path),
            ("Artifacts Directory", art_path),
            ("Models Directory", mdl_path),
        ]:
            row = QHBoxLayout()
            row.addWidget(self._lbl(label_text))
            path_edit = QLineEdit(path_val)
            path_edit.setReadOnly(True)
            path_edit.setStyleSheet(
                "background: rgba(8,12,20,0.6); border: 1px solid rgba(100,180,255,0.08); "
                "border-radius: 6px; padding: 6px 10px; color: rgba(200,215,235,0.5); "
                "font-size: 11px;"
            )
            row.addWidget(path_edit, 1)
            ppg.addLayout(row)

        layout.addWidget(paths_group)

        # ── Display Settings ──
        display_group = QGroupBox("Display")
        display_group.setStyleSheet(group_ss)
        dg = QVBoxLayout(display_group)
        dg.setSpacing(12)

        row = QHBoxLayout()
        row.addWidget(self._lbl("Default Colormap"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["seti", "viridis", "inferno", "plasma", "magma", "turbo"])
        self._cmap_combo.setFixedWidth(150)
        row.addWidget(self._cmap_combo)
        dg.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(self._lbl("Show Drift Lines by Default"))
        self._drift_lines_check = QCheckBox()
        self._drift_lines_check.setChecked(True)
        row.addWidget(self._drift_lines_check)
        row.addStretch()
        dg.addLayout(row)

        layout.addWidget(display_group)

        # ── AstroLens Integration ──
        al_group = QGroupBox("AstroLens Integration")
        al_group.setStyleSheet(group_ss)
        alg = QVBoxLayout(al_group)
        alg.setSpacing(12)

        row = QHBoxLayout()
        row.addWidget(self._lbl("AstroLens API URL"))
        self._al_url = QLineEdit("http://localhost:8000")
        self._al_url.setStyleSheet(self._input_ss())
        row.addWidget(self._al_url, 1)
        alg.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(self._lbl("Enable Cross-Reference"))
        self._al_xref_check = QCheckBox()
        self._al_xref_check.setChecked(True)
        row.addWidget(self._al_xref_check)
        row.addStretch()
        alg.addLayout(row)

        layout.addWidget(al_group)

        # ── Breakthrough Listen Data ──
        bl_group = QGroupBox("Breakthrough Listen Data")
        bl_group.setStyleSheet(group_ss)
        blg = QVBoxLayout(bl_group)
        blg.setSpacing(12)

        bl_info = QLabel(
            "Download observation files from the BL Open Data Archive.\n"
            "Files are saved to the data/breakthrough_listen_data_files/ directory.\n"
            "Requires aria2c (brew install aria2)."
        )
        bl_info.setStyleSheet(f"font-size: 12px; color: {COLORS['text_secondary']};")
        bl_info.setWordWrap(True)
        blg.addWidget(bl_info)

        try:
            from paths import FILTERBANK_DIR
            bl_dir = FILTERBANK_DIR
        except Exception:
            bl_dir = _ARTIFACTS_DIR / "data" / "breakthrough_listen_data_files"

        file_count = 0
        total_size_gb = 0.0
        if bl_dir.exists():
            for f in bl_dir.iterdir():
                if f.suffix in (".fil", ".h5", ".hdf5") and f.stat().st_size > 100_000:
                    file_count += 1
                    total_size_gb += f.stat().st_size / (1024 ** 3)

        self._bl_status = QLabel(
            f"Currently: {file_count} files ({total_size_gb:.1f} GB)"
        )
        self._bl_status.setStyleSheet("font-size: 12px; color: #34d399;")
        blg.addWidget(self._bl_status)

        dl_btn = QPushButton("📥  Download BL Data Files")
        dl_btn.setCursor(Qt.PointingHandCursor)
        dl_btn.setMinimumHeight(40)
        dl_btn.setStyleSheet("""
            QPushButton {
                background: rgba(52, 211, 153, 0.1);
                border: 1px solid rgba(52, 211, 153, 0.25);
                border-radius: 10px; padding: 8px 24px;
                color: #34d399; font-size: 13px; font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(52, 211, 153, 0.2);
                border-color: rgba(52, 211, 153, 0.4);
            }
        """)
        dl_btn.clicked.connect(self._download_bl_data)
        blg.addWidget(dl_btn)

        layout.addWidget(bl_group)

        # ── Save button ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("💾  Save Settings")
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.setMinimumHeight(44)
        save_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(77, 166, 255, 0.8), stop:1 rgba(99, 102, 241, 0.8));
                border: 1px solid rgba(77, 166, 255, 0.3);
                border-radius: 10px; padding: 10px 32px;
                color: #ffffff; font-size: 14px; font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(77, 166, 255, 0.95);
                border-color: rgba(77, 166, 255, 0.6);
            }
        """)
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)

        reset_btn = QPushButton("Reset Defaults")
        reset_btn.setCursor(Qt.PointingHandCursor)
        reset_btn.setMinimumHeight(44)
        reset_btn.setStyleSheet("""
            QPushButton {
                background: rgba(248, 113, 113, 0.08);
                border: 1px solid rgba(248, 113, 113, 0.2);
                border-radius: 10px; padding: 10px 24px;
                color: #f87171; font-size: 13px; font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(248, 113, 113, 0.18);
                border-color: rgba(248, 113, 113, 0.4);
            }
        """)
        reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()

        layout.addLayout(btn_row)
        layout.addStretch()

        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _lbl(self, text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet(f"font-size: 13px; color: {COLORS['text_secondary']}; min-width: 200px;")
        l.setFixedWidth(220)
        return l

    @staticmethod
    def _input_ss() -> str:
        return (
            "background: rgba(8,12,20,0.7); border: 1px solid rgba(100,180,255,0.12); "
            "border-radius: 6px; padding: 6px 10px; color: #e0e8f0; font-size: 12px;"
        )

    def _load_values(self):
        s = self._settings
        self._snr_spin.setValue(s.get("min_snr", 10.0))
        self._drift_spin.setValue(s.get("max_drift_rate", 4.0))
        self._device_combo.setCurrentText(s.get("device", "auto"))
        self._dur_spin.setValue(s.get("streaming_days", 7))
        self._mode_combo.setCurrentText(s.get("streaming_mode", "normal"))
        self._auto_dl_check.setChecked(s.get("auto_download_bl", True))
        self._cmap_combo.setCurrentText(s.get("colormap", "seti"))
        self._drift_lines_check.setChecked(s.get("show_drift_lines", True))
        self._al_url.setText(s.get("astrolens_api_url", "http://localhost:8000"))
        self._al_xref_check.setChecked(s.get("astrolens_crossref", True))

    def _save(self):
        self._settings = {
            "min_snr": self._snr_spin.value(),
            "max_drift_rate": self._drift_spin.value(),
            "device": self._device_combo.currentText(),
            "streaming_days": self._dur_spin.value(),
            "streaming_mode": self._mode_combo.currentText(),
            "auto_download_bl": self._auto_dl_check.isChecked(),
            "colormap": self._cmap_combo.currentText(),
            "show_drift_lines": self._drift_lines_check.isChecked(),
            "astrolens_api_url": self._al_url.text(),
            "astrolens_crossref": self._al_xref_check.isChecked(),
        }
        try:
            _save_settings(self._settings)
            QMessageBox.information(self, "Saved", "Settings saved successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save settings:\n{e}")

    def _reset_defaults(self):
        self._settings = {}
        self._load_values()
        QMessageBox.information(self, "Reset", "Settings restored to defaults.")

    def _download_bl_data(self):
        """Download BL data files with confirmation dialog."""
        import subprocess
        import shutil

        script_path = Path(__file__).parent.parent / "data" / "download_all_BL_data.sh"
        if not script_path.exists():
            QMessageBox.warning(
                self, "Missing Script",
                f"Download script not found:\n{script_path}\n\n"
                "Please ensure data/download_all_BL_data.sh exists."
            )
            return

        if not shutil.which("aria2c"):
            QMessageBox.warning(
                self, "aria2c Not Found",
                "aria2c is required for downloads.\n\n"
                "Install with:\n  brew install aria2  (macOS)\n"
                "  apt install aria2   (Linux)"
            )
            return

        reply = QMessageBox.question(
            self, "Confirm Download",
            "This will download Breakthrough Listen observation files.\n\n"
            "Total size: ~40+ GB\n"
            "Files already on disk will be skipped.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            subprocess.Popen(
                ["bash", str(script_path)],
                cwd=str(script_path.parent),
            )
            QMessageBox.information(
                self, "Download Started",
                "Download is running in the background.\n"
                "Check the terminal for progress.\n\n"
                "Files will be saved to:\n"
                "mitraseti_artifacts/data/breakthrough_listen_data_files/"
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start download:\n{e}")
