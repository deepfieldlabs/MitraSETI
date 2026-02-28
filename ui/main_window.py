"""
MitraSETI Main Window

Crystalline glass interface with sidebar navigation,
stacked content panels, and glowing status indicators.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QPainter
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .theme import COLORS, get_stylesheet

# ── Pulsing status dot ────────────────────────────────────────────────────────


class _StatusDot(QFrame):
    """Small animated status indicator dot."""

    def __init__(self, size: int = 8, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._color = QColor(COLORS["text_tertiary"])

    def set_color(self, hex_color: str):
        self._color = QColor(hex_color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self._color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(1, 1, self.width() - 2, self.height() - 2)


# ── Main Window ──────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    """MitraSETI main application window — crystalline glass design."""

    NAV_ITEMS = [
        ("\u2302", "Dashboard", 0),
        ("\u2637", "Waterfall Viewer", 1),
        ("\u2609", "Signal Gallery", 2),
        ("\u2616", "RFI Dashboard", 3),
        ("\u25ce", "Space Radar", 4),
        ("\u25b6", "Streaming", 5),
        ("\u2699", "Settings", 6),
        ("\u24d8", "About", 7),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MitraSETI — Intelligent SETI Signal Analysis")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Apply crystalline glass stylesheet
        self.setStyleSheet(get_stylesheet())

        self._active_nav_index = 0
        self._setup_ui()
        self._setup_statusbar()

        # Simulate connection check
        QTimer.singleShot(800, self._check_status)

    # ── UI assembly ───────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        sidebar = self._create_sidebar()
        root.addWidget(sidebar)

        # Content stack
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("background: #080c14;")
        root.addWidget(self.content_stack, 1)

        # Import and add panels (lazy — panels self-register)
        self._add_panels()

    # ── Sidebar ───────────────────────────────────────────────────────────

    def _create_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setObjectName("sidebarRoot")
        sidebar.setStyleSheet("""
            #sidebarRoot {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(10, 16, 28, 0.98),
                    stop:1 rgba(8, 12, 20, 0.98));
                border-right: 1px solid rgba(100, 180, 255, 0.06);
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(6)

        # ── Title with glow ──────────────────────────────────────────────
        title = QLabel("MitraSETI")
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: 700;
            color: #4da6ff;
            letter-spacing: 1px;
            padding: 8px 4px 2px 4px;
        """)
        layout.addWidget(title)

        tagline = QLabel("Signal Intelligence")
        tagline.setStyleSheet("""
            font-size: 10px;
            font-weight: 500;
            color: rgba(0, 212, 255, 0.45);
            letter-spacing: 2px;
            text-transform: uppercase;
            padding: 0 4px 8px 4px;
        """)
        layout.addWidget(tagline)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: rgba(100, 180, 255, 0.08);")
        layout.addWidget(sep)
        layout.addSpacing(12)

        # ── Nav label ────────────────────────────────────────────────────
        nav_label = QLabel("NAVIGATION")
        nav_label.setStyleSheet("""
            font-size: 9px;
            font-weight: 600;
            color: rgba(140, 165, 200, 0.35);
            letter-spacing: 2px;
            padding: 0 4px;
            margin-bottom: 6px;
        """)
        layout.addWidget(nav_label)

        # ── Nav buttons ──────────────────────────────────────────────────
        self._nav_buttons: list[QPushButton] = []

        for icon, label, idx in self.NAV_ITEMS:
            btn = QPushButton(f"  {icon}  {label}")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("nav_index", idx)
            btn.clicked.connect(lambda checked, i=idx: self._on_nav_clicked(i))
            self._nav_buttons.append(btn)
            layout.addWidget(btn)

        self._apply_nav_styles()

        layout.addStretch()

        # ── Processing indicator ─────────────────────────────────────────
        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background: rgba(100, 180, 255, 0.06);")
        layout.addWidget(sep2)
        layout.addSpacing(8)

        self._processing_label = QLabel("No active processing")
        self._processing_label.setStyleSheet("""
            font-size: 11px;
            color: rgba(140, 165, 200, 0.4);
            padding: 4px;
        """)
        layout.addWidget(self._processing_label)

        return sidebar

    def _apply_nav_styles(self):
        """Style all nav buttons; highlight the active one with cyan glow."""
        for i, btn in enumerate(self._nav_buttons):
            if i == self._active_nav_index:
                btn.setStyleSheet("""
                    QPushButton {
                        background: rgba(0, 212, 255, 0.1);
                        border: 1px solid rgba(0, 212, 255, 0.2);
                        border-left: 3px solid #4da6ff;
                        border-radius: 8px;
                        padding: 11px 14px;
                        text-align: left;
                        color: #4da6ff;
                        font-size: 13px;
                        font-weight: 600;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background: transparent;
                        border: 1px solid transparent;
                        border-radius: 8px;
                        padding: 11px 14px;
                        text-align: left;
                        color: rgba(200, 215, 235, 0.5);
                        font-size: 13px;
                        font-weight: 400;
                    }
                    QPushButton:hover {
                        background: rgba(20, 35, 60, 0.4);
                        border-color: rgba(100, 180, 255, 0.08);
                        color: rgba(200, 215, 235, 0.8);
                    }
                """)

    def _on_nav_clicked(self, index: int):
        self._active_nav_index = index
        self._apply_nav_styles()
        if index < self.content_stack.count():
            self.content_stack.setCurrentIndex(index)

    # ── Panels ────────────────────────────────────────────────────────────

    def _add_panels(self):
        """Import and add all content panels to the stack."""
        from .about_panel import AboutPanel
        from .dashboard import DashboardPanel
        from .rfi_panel import RFIPanel
        from .settings_panel import SettingsPanel
        from .signal_gallery import SignalGallery
        from .sky_map_panel import SkyMapPanel
        from .streaming_panel import StreamingPanel
        from .waterfall_viewer import WaterfallViewer

        self.dashboard = DashboardPanel()
        self.waterfall_viewer = WaterfallViewer()
        self.signal_gallery = SignalGallery()
        self.rfi_panel = RFIPanel()
        self.sky_map = SkyMapPanel()
        self.streaming_panel = StreamingPanel()
        self.settings_panel = SettingsPanel()
        self.about_panel = AboutPanel()

        self.content_stack.addWidget(self.dashboard)  # 0
        self.content_stack.addWidget(self.waterfall_viewer)  # 1
        self.content_stack.addWidget(self.signal_gallery)  # 2
        self.content_stack.addWidget(self.rfi_panel)  # 3
        self.content_stack.addWidget(self.sky_map)  # 4
        self.content_stack.addWidget(self.streaming_panel)  # 5
        self.content_stack.addWidget(self.settings_panel)  # 6
        self.content_stack.addWidget(self.about_panel)  # 7

        # Wire cross-panel signals
        self.dashboard.navigate_to.connect(self._on_nav_clicked)
        self.waterfall_viewer.signal_detected.connect(self.signal_gallery.add_signal)
        self.signal_gallery.open_in_viewer.connect(self._open_signal_in_viewer)

    def _open_signal_in_viewer(self, signal_data: dict):
        """Navigate to waterfall viewer with the selected signal's file."""
        file_name = signal_data.get("file", "")
        if file_name and hasattr(self.waterfall_viewer, "load_file_by_name"):
            self.waterfall_viewer.load_file_by_name(file_name)
        self._on_nav_clicked(1)

    # ── Status bar ────────────────────────────────────────────────────────

    def _setup_statusbar(self):
        bar = QStatusBar()
        self.setStatusBar(bar)

        # Connection
        conn_row = QHBoxLayout()
        conn_row.setSpacing(6)
        self._conn_dot = _StatusDot(8)
        conn_row.addWidget(self._conn_dot)
        self._conn_label = QLabel("Connecting…")
        self._conn_label.setStyleSheet("color: rgba(140,165,200,0.5); font-size: 12px;")
        conn_row.addWidget(self._conn_label)
        conn_widget = QWidget()
        conn_widget.setLayout(conn_row)
        conn_widget.setStyleSheet("background: transparent;")
        bar.addWidget(conn_widget)

        # Spacer
        bar.addWidget(QLabel(""), 1)

        # GPU info
        self._gpu_label = QLabel("GPU: scanning…")
        self._gpu_label.setStyleSheet("color: rgba(140,165,200,0.4); font-size: 12px;")
        bar.addPermanentWidget(self._gpu_label)

        # Processing indicator
        self._proc_label = QLabel("Idle")
        self._proc_label.setStyleSheet("color: rgba(140,165,200,0.4); font-size: 12px;")
        bar.addPermanentWidget(self._proc_label)

    def _check_status(self):
        """Simulate status check (replace with real API check)."""
        self._conn_dot.set_color(COLORS["success"])
        self._conn_label.setText("Connected")
        self._conn_label.setStyleSheet("color: #34d399; font-size: 12px;")
        self._gpu_label.setText("GPU: available")
        self._proc_label.setText("Idle")

    # ── Public API ────────────────────────────────────────────────────────

    def navigate_to(self, panel_index: int):
        """Programmatically switch to a panel."""
        self._on_nav_clicked(panel_index)

    def set_processing(self, active: bool, text: str = ""):
        """Update processing indicator in sidebar and status bar."""
        if active:
            self._processing_label.setText(f"⚡ {text or 'Processing…'}")
            self._processing_label.setStyleSheet("font-size: 11px; color: #4da6ff; padding: 4px;")
            self._proc_label.setText(text or "Processing…")
            self._proc_label.setStyleSheet("color: #4da6ff; font-size: 12px;")
        else:
            self._processing_label.setText("No active processing")
            self._processing_label.setStyleSheet(
                "font-size: 11px; color: rgba(140,165,200,0.4); padding: 4px;"
            )
            self._proc_label.setText("Idle")
            self._proc_label.setStyleSheet("color: rgba(140,165,200,0.4); font-size: 12px;")
