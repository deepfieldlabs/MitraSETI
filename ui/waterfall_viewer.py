"""
astroSETI Waterfall Viewer — THE HERO PANEL

Spectrogram / waterfall display with:
- Embedded matplotlib canvas for the main spectrogram
- Color-map selector, zoom/pan, drift-line overlays
- Signal detail side-panel
- ON/OFF source split view
- Demo synthetic spectrogram on startup
"""

from __future__ import annotations

import io
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QCheckBox, QSplitter, QScrollArea,
    QFileDialog, QSizePolicy, QGridLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

from .theme import COLORS, create_glass_card, create_glow_button


# ── Custom "seti" colormap — dark blue → cyan glow ──────────────────────────

_SETI_CMAP_COLORS = [
    (0.0, "#020810"),
    (0.2, "#081838"),
    (0.4, "#0a3060"),
    (0.6, "#0060a0"),
    (0.8, "#00b0e0"),
    (1.0, "#00ffff"),
]
_seti_cmap = LinearSegmentedColormap.from_list(
    "seti", [(v, c) for v, c in _SETI_CMAP_COLORS]
)


# ── Synthetic demo data generator ────────────────────────────────────────────

def generate_demo_waterfall(
    n_time: int = 512,
    n_freq: int = 1024,
    n_signals: int = 4,
    noise_level: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, list[dict]]:
    """
    Generate a realistic-looking synthetic waterfall spectrogram.

    Returns (data_array, list_of_injected_signal_params).
    """
    rng = np.random.default_rng(seed)

    # Base noise — Gaussian
    data = rng.normal(0, noise_level, (n_time, n_freq)).astype(np.float32)

    # Add broadband RFI-like horizontal lines
    for _ in range(3):
        t = rng.integers(0, n_time)
        data[t, :] += rng.uniform(3, 8)

    # Add narrowband persistent RFI (vertical lines)
    for _ in range(5):
        f = rng.integers(50, n_freq - 50)
        data[:, f] += rng.uniform(1.5, 4)

    signals = []
    for i in range(n_signals):
        # Drifting narrowband signal
        snr = rng.uniform(8, 30)
        drift_rate = rng.uniform(-2.0, 2.0)  # Hz/s equivalent in pixels
        start_freq = rng.integers(100, n_freq - 100)
        width = rng.integers(1, 4)
        rfi_score = rng.uniform(0, 1)
        classification = rng.choice(
            ["candidate", "candidate", "rfi_terrestrial", "rfi_satellite", "unknown"]
        )
        confidence = rng.uniform(0.5, 0.99)

        for t in range(n_time):
            f_center = int(start_freq + drift_rate * t / n_time * 60)
            if 0 <= f_center < n_freq:
                f_lo = max(0, f_center - width)
                f_hi = min(n_freq, f_center + width + 1)
                data[t, f_lo:f_hi] += snr * noise_level

        signals.append({
            "id": i,
            "snr": round(float(snr), 2),
            "drift_rate": round(float(drift_rate), 4),
            "start_freq": float(start_freq),
            "width": int(width),
            "rfi_score": round(float(rfi_score), 3),
            "classification": classification,
            "confidence": round(float(confidence), 3),
        })

    return data, signals


# ── Signal detail side panel ─────────────────────────────────────────────────

class SignalDetailPanel(QFrame):
    """Right-side panel showing selected signal parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setStyleSheet(f"""
            QFrame {{
                background: rgba(12, 20, 38, 0.9);
                border-left: 1px solid rgba(100, 180, 255, 0.08);
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Signal Details")
        title.setStyleSheet(
            "font-size: 15px; font-weight: 600; color: #00d4ff;"
            "letter-spacing: 0.5px;"
        )
        layout.addWidget(title)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: rgba(0, 212, 255, 0.12);")
        layout.addWidget(sep)

        # Parameter rows
        self._params: dict[str, QLabel] = {}
        param_defs = [
            ("SNR", "snr"),
            ("Drift Rate (Hz/s)", "drift_rate"),
            ("Frequency (MHz)", "start_freq"),
            ("Classification", "classification"),
            ("RFI Score", "rfi_score"),
            ("Confidence", "confidence"),
        ]

        for display_name, key in param_defs:
            row = QHBoxLayout()
            lbl = QLabel(display_name)
            lbl.setStyleSheet(
                "font-size: 11px; color: rgba(200,215,235,0.5); font-weight: 500;"
            )
            row.addWidget(lbl)
            row.addStretch()
            val = QLabel("—")
            val.setStyleSheet("font-size: 13px; color: #e0e8f0; font-weight: 600;")
            val.setAlignment(Qt.AlignRight)
            row.addWidget(val)
            layout.addLayout(row)
            self._params[key] = val

        # Classification badge
        self._badge = QLabel("")
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setStyleSheet(
            "padding: 8px; border-radius: 8px; font-weight: 600; font-size: 12px;"
        )
        self._badge.hide()
        layout.addWidget(self._badge)

        layout.addStretch()

        self._empty_label = QLabel("Click a signal on the\nspectrogram to inspect.")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(
            "color: rgba(140,165,200,0.35); font-size: 12px; padding: 20px 0;"
        )
        layout.addWidget(self._empty_label)

    def show_signal(self, sig: dict):
        self._empty_label.hide()
        self._badge.show()

        self._params["snr"].setText(f"{sig.get('snr', 0):.2f}")
        self._params["drift_rate"].setText(f"{sig.get('drift_rate', 0):.4f}")
        self._params["start_freq"].setText(f"{sig.get('start_freq', 0):.1f}")
        self._params["rfi_score"].setText(f"{sig.get('rfi_score', 0):.3f}")
        self._params["confidence"].setText(f"{sig.get('confidence', 0):.1%}")

        cls = sig.get("classification", "unknown")
        self._params["classification"].setText(cls.replace("_", " ").title())

        if "candidate" in cls:
            self._badge.setText("✦ CANDIDATE")
            self._badge.setStyleSheet(
                "background: rgba(0,255,136,0.12); color: #00ff88; "
                "padding: 8px; border-radius: 8px; border: 1px solid rgba(0,255,136,0.25); "
                "font-weight: 600; font-size: 12px;"
            )
        elif "rfi" in cls:
            self._badge.setText("✕ RFI")
            self._badge.setStyleSheet(
                "background: rgba(255,51,102,0.12); color: #ff3366; "
                "padding: 8px; border-radius: 8px; border: 1px solid rgba(255,51,102,0.25); "
                "font-weight: 600; font-size: 12px;"
            )
        else:
            self._badge.setText("? UNKNOWN")
            self._badge.setStyleSheet(
                "background: rgba(255,170,0,0.12); color: #ffaa00; "
                "padding: 8px; border-radius: 8px; border: 1px solid rgba(255,170,0,0.25); "
                "font-weight: 600; font-size: 12px;"
            )

    def clear(self):
        for val in self._params.values():
            val.setText("—")
        self._badge.hide()
        self._empty_label.show()


# ── Main Waterfall Viewer ────────────────────────────────────────────────────

class WaterfallViewer(QWidget):
    """Hero panel — spectrogram / waterfall display."""

    signal_detected = pyqtSignal(dict)  # Emitted when pipeline detects a signal

    COLORMAPS = ["seti", "viridis", "inferno", "plasma", "magma", "turbo"]

    def __init__(self, parent=None):
        super().__init__(parent)

        self._data: np.ndarray | None = None
        self._signals: list[dict] = []
        self._show_drift_lines = True
        self._current_cmap = "seti"
        self._split_view = False

        self._setup_ui()

        # Auto-generate demo on startup
        QTimer.singleShot(300, self._load_demo)

    # ── UI setup ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ──────────────────────────────────────────────────────
        toolbar = QFrame()
        toolbar.setFixedHeight(56)
        toolbar.setStyleSheet("""
            QFrame {
                background: rgba(10, 16, 28, 0.95);
                border-bottom: 1px solid rgba(100, 180, 255, 0.06);
            }
        """)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(20, 0, 20, 0)
        tb_layout.setSpacing(12)

        title = QLabel("📡  Waterfall Viewer")
        title.setStyleSheet(
            "font-size: 16px; font-weight: 600; color: #e0e8f0; border: none;"
        )
        tb_layout.addWidget(title)
        tb_layout.addSpacing(24)

        # File open
        self._open_btn = QPushButton("Open File")
        self._open_btn.setStyleSheet(self._toolbar_btn_style())
        self._open_btn.clicked.connect(self._open_file)
        tb_layout.addWidget(self._open_btn)

        # Process
        self._process_btn = QPushButton("⚡ Process")
        self._process_btn.setStyleSheet(self._toolbar_btn_style("#00d4ff"))
        self._process_btn.clicked.connect(self._run_pipeline)
        tb_layout.addWidget(self._process_btn)

        tb_layout.addSpacing(16)

        # Colormap
        cmap_lbl = QLabel("Colormap:")
        cmap_lbl.setStyleSheet(
            "font-size: 12px; color: rgba(200,215,235,0.5); border:none;"
        )
        tb_layout.addWidget(cmap_lbl)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(self.COLORMAPS)
        self._cmap_combo.setCurrentIndex(0)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self._cmap_combo.setFixedWidth(110)
        tb_layout.addWidget(self._cmap_combo)

        # Drift-line overlay toggle
        self._drift_check = QCheckBox("Drift Lines")
        self._drift_check.setChecked(True)
        self._drift_check.setStyleSheet(
            "QCheckBox { color: rgba(200,215,235,0.6); font-size: 12px; }"
        )
        self._drift_check.toggled.connect(self._toggle_drift_lines)
        tb_layout.addWidget(self._drift_check)

        # ON/OFF split toggle
        self._split_check = QCheckBox("ON/OFF Split")
        self._split_check.setChecked(False)
        self._split_check.setStyleSheet(
            "QCheckBox { color: rgba(200,215,235,0.6); font-size: 12px; }"
        )
        self._split_check.toggled.connect(self._toggle_split)
        tb_layout.addWidget(self._split_check)

        tb_layout.addStretch()

        # File name display
        self._file_label = QLabel("demo_waterfall.fil")
        self._file_label.setStyleSheet(
            "font-size: 11px; color: rgba(0,212,255,0.6); border:none;"
        )
        tb_layout.addWidget(self._file_label)

        root.addWidget(toolbar)

        # ── Main content: spectrogram + detail panel ─────────────────────
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Spectrogram area (left)
        spec_container = QWidget()
        spec_layout = QVBoxLayout(spec_container)
        spec_layout.setContentsMargins(12, 12, 4, 12)
        spec_layout.setSpacing(4)

        # Matplotlib figure
        self._fig = Figure(facecolor="#080c14")
        self._ax = self._fig.add_subplot(111)
        self._ax_off = None  # for split view
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.mpl_connect("button_press_event", self._on_canvas_click)

        # Matplotlib nav toolbar (zoom, pan, etc.)
        self._mpl_toolbar = NavigationToolbar(self._canvas, self)
        self._mpl_toolbar.setStyleSheet("""
            QToolBar {
                background: rgba(10, 16, 28, 0.8);
                border: 1px solid rgba(100, 180, 255, 0.08);
                border-radius: 8px;
                spacing: 4px;
                padding: 2px;
            }
            QToolButton {
                background: transparent;
                border: none;
                padding: 4px;
                border-radius: 4px;
            }
            QToolButton:hover {
                background: rgba(0, 212, 255, 0.15);
            }
            QToolButton:checked {
                background: rgba(0, 212, 255, 0.2);
                border: 1px solid rgba(0, 212, 255, 0.3);
            }
        """)

        spec_layout.addWidget(self._mpl_toolbar)
        spec_layout.addWidget(self._canvas, 1)

        body.addWidget(spec_container, 1)

        # Detail panel (right)
        self._detail_panel = SignalDetailPanel()
        body.addWidget(self._detail_panel)

        root.addLayout(body, 1)

    # ── Styles ────────────────────────────────────────────────────────────

    @staticmethod
    def _toolbar_btn_style(color: str = "#e0e8f0") -> str:
        if color == "#e0e8f0":
            return """
                QPushButton {
                    background: rgba(15, 25, 45, 0.7);
                    border: 1px solid rgba(100, 180, 255, 0.12);
                    border-radius: 8px;
                    padding: 7px 16px;
                    color: rgba(200,215,235,0.8);
                    font-size: 12px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: rgba(20, 35, 60, 0.85);
                    border-color: rgba(100, 180, 255, 0.3);
                    color: #e0e8f0;
                }
            """
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"""
            QPushButton {{
                background: rgba({r},{g},{b}, 0.12);
                border: 1px solid rgba({r},{g},{b}, 0.25);
                border-radius: 8px;
                padding: 7px 16px;
                color: {color};
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: rgba({r},{g},{b}, 0.22);
                border-color: rgba({r},{g},{b}, 0.45);
            }}
        """

    # ── Data loading ──────────────────────────────────────────────────────

    def _load_demo(self):
        """Generate and display synthetic demo spectrogram."""
        self._data, self._signals = generate_demo_waterfall()
        self._file_label.setText("demo_waterfall.fil (synthetic)")
        self._render()

        # Emit signals for gallery
        for sig in self._signals:
            self.signal_detected.emit(sig)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Filterbank / HDF5",
            "",
            "SETI Data (*.fil *.h5 *.hdf5);;All Files (*)",
        )
        if not path:
            return
        self._file_label.setText(path.split("/")[-1])
        # For now, load demo data (replace with real blimpy/h5py loader)
        self._data, self._signals = generate_demo_waterfall(seed=hash(path) % 2**31)
        self._render()

    def _run_pipeline(self):
        """Simulate running the full detection pipeline."""
        if self._data is None:
            return
        # In production, this would call the backend pipeline
        # For now just re-render with fresh signals
        _, new_sigs = generate_demo_waterfall(
            n_signals=np.random.randint(2, 7),
            seed=np.random.randint(0, 10000),
        )
        self._signals = new_sigs
        self._render()
        for sig in self._signals:
            self.signal_detected.emit(sig)

    # ── Rendering ─────────────────────────────────────────────────────────

    def _get_cmap(self):
        if self._current_cmap == "seti":
            return _seti_cmap
        return self._current_cmap

    def _render(self):
        """Render the spectrogram onto the matplotlib canvas."""
        if self._data is None:
            return

        self._fig.clear()

        if self._split_view:
            self._ax = self._fig.add_subplot(121)
            self._ax_off = self._fig.add_subplot(122)
        else:
            self._ax = self._fig.add_subplot(111)
            self._ax_off = None

        cmap = self._get_cmap()

        # Main (ON-source) spectrogram
        self._ax.imshow(
            self._data,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
        )
        self._ax.set_xlabel("Frequency Channel", color="#e0e8f0", fontsize=10)
        self._ax.set_ylabel("Time Step", color="#e0e8f0", fontsize=10)
        self._ax.set_title("ON Source", color="#00d4ff", fontsize=12, fontweight="bold")
        self._ax.tick_params(colors="#8ca5c8", labelsize=8)
        self._ax.set_facecolor("#080c14")

        # Drift-line overlays
        if self._show_drift_lines and self._signals:
            n_time = self._data.shape[0]
            for sig in self._signals:
                f0 = sig["start_freq"]
                dr = sig["drift_rate"]
                t = np.arange(n_time)
                f = f0 + dr * t / n_time * 60
                color = (
                    "#00ff88"
                    if "candidate" in sig.get("classification", "")
                    else "#ff3366"
                    if "rfi" in sig.get("classification", "")
                    else "#ffaa00"
                )
                self._ax.plot(
                    f, t,
                    color=color, linewidth=1.2, alpha=0.7,
                    linestyle="--",
                )

        # OFF-source (just noise) for split view
        if self._ax_off is not None:
            rng = np.random.default_rng(99)
            off_data = rng.normal(0, 1, self._data.shape).astype(np.float32)
            # Add some broadband
            for _ in range(3):
                off_data[rng.integers(0, off_data.shape[0]), :] += rng.uniform(2, 5)

            self._ax_off.imshow(
                off_data,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                interpolation="nearest",
            )
            self._ax_off.set_xlabel("Frequency Channel", color="#e0e8f0", fontsize=10)
            self._ax_off.set_ylabel("Time Step", color="#e0e8f0", fontsize=10)
            self._ax_off.set_title(
                "OFF Source (Reference)",
                color="#ff3366", fontsize=12, fontweight="bold",
            )
            self._ax_off.tick_params(colors="#8ca5c8", labelsize=8)
            self._ax_off.set_facecolor("#080c14")

        self._fig.set_facecolor("#080c14")
        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()

    # ── Interaction ───────────────────────────────────────────────────────

    def _on_cmap_changed(self, name: str):
        self._current_cmap = name
        self._render()

    def _toggle_drift_lines(self, checked: bool):
        self._show_drift_lines = checked
        self._render()

    def _toggle_split(self, checked: bool):
        self._split_view = checked
        self._render()

    def _on_canvas_click(self, event):
        """When user clicks on the spectrogram, find nearest signal."""
        if event.xdata is None or event.ydata is None:
            return
        if not self._signals:
            return

        click_f = event.xdata
        click_t = event.ydata

        best = None
        best_dist = float("inf")
        n_time = self._data.shape[0] if self._data is not None else 512

        for sig in self._signals:
            f0 = sig["start_freq"]
            dr = sig["drift_rate"]
            f_at_t = f0 + dr * click_t / n_time * 60
            dist = abs(f_at_t - click_f)
            if dist < best_dist:
                best_dist = dist
                best = sig

        if best and best_dist < 30:
            self._detail_panel.show_signal(best)

    # ── Public API ────────────────────────────────────────────────────────

    def render_thumbnail(self, width: int = 64, height: int = 64) -> QPixmap | None:
        """Render a small thumbnail of current spectrogram as QPixmap."""
        if self._data is None:
            return None

        fig = Figure(figsize=(width / 72, height / 72), dpi=72)
        ax = fig.add_subplot(111)
        ax.imshow(
            self._data, aspect="auto", origin="lower",
            cmap=self._get_cmap(), interpolation="nearest",
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="#080c14", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = QImage.fromData(buf.read())
        return QPixmap.fromImage(img)
