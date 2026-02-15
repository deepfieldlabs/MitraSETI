"""
astroSETI Signal Gallery

Grid view of detected signal thumbnails with filtering, sorting,
and batch actions. Each card shows a mini spectrogram + signal info.
"""

from __future__ import annotations

import io
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QScrollArea, QGridLayout, QSizePolicy,
    QLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure

from .theme import COLORS, create_stat_card_style


# ── Flow Layout (responsive grid) ────────────────────────────────────────────

class FlowLayout(QLayout):
    """Layout that arranges widgets left-to-right, wrapping to next row."""

    def __init__(self, parent=None, margin: int = 0, spacing: int = 12):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect, test_only=False):
        m = self.contentsMargins()
        effective = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = effective.x()
        y = effective.y()
        row_height = 0

        for item in self._items:
            sz = item.sizeHint()
            next_x = x + sz.width() + self._spacing
            if next_x - self._spacing > effective.right() and row_height > 0:
                x = effective.x()
                y += row_height + self._spacing
                next_x = x + sz.width() + self._spacing
                row_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), sz))
            x = next_x
            row_height = max(row_height, sz.height())

        return y + row_height - rect.y() + m.bottom()


# ── Signal thumbnail card ────────────────────────────────────────────────────

class SignalCard(QFrame):
    """A single card showing a mini spectrogram + signal summary."""

    clicked = pyqtSignal(dict)

    def __init__(self, signal_data: dict, parent=None):
        super().__init__(parent)
        self.signal_data = signal_data
        self.setFixedSize(210, 240)
        self.setCursor(Qt.PointingHandCursor)
        self._selected = False

        cls = signal_data.get("classification", "unknown")
        if "candidate" in cls:
            border = "rgba(0, 255, 136, 0.2)"
            hover = "rgba(0, 255, 136, 0.35)"
        elif "rfi" in cls:
            border = "rgba(255, 51, 102, 0.2)"
            hover = "rgba(255, 51, 102, 0.35)"
        else:
            border = "rgba(100, 180, 255, 0.12)"
            hover = "rgba(0, 212, 255, 0.3)"

        self.setStyleSheet(f"""
            SignalCard {{
                background: rgba(15, 25, 45, 0.55);
                border: 1px solid {border};
                border-radius: 12px;
            }}
            SignalCard:hover {{
                border-color: {hover};
                background: rgba(20, 35, 60, 0.7);
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Thumbnail
        self._thumb = QLabel()
        self._thumb.setFixedSize(190, 110)
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setStyleSheet(
            "background: rgba(8, 12, 20, 0.8); border-radius: 8px;"
        )
        layout.addWidget(self._thumb)
        self._generate_thumbnail()

        # Classification badge
        badge = QLabel(cls.replace("_", " ").title())
        badge.setAlignment(Qt.AlignCenter)
        if "candidate" in cls:
            badge.setStyleSheet(
                "background: rgba(0,255,136,0.12); color: #00ff88; "
                "font-size: 10px; font-weight: 600; padding: 3px 8px; "
                "border-radius: 8px; border: 1px solid rgba(0,255,136,0.2);"
            )
        elif "rfi" in cls:
            badge.setStyleSheet(
                "background: rgba(255,51,102,0.12); color: #ff3366; "
                "font-size: 10px; font-weight: 600; padding: 3px 8px; "
                "border-radius: 8px; border: 1px solid rgba(255,51,102,0.2);"
            )
        else:
            badge.setStyleSheet(
                "background: rgba(255,170,0,0.12); color: #ffaa00; "
                "font-size: 10px; font-weight: 600; padding: 3px 8px; "
                "border-radius: 8px; border: 1px solid rgba(255,170,0,0.2);"
            )
        layout.addWidget(badge)

        # Stats row
        stats = QHBoxLayout()
        stats.setSpacing(8)

        snr_lbl = QLabel(f"SNR {signal_data.get('snr', 0):.1f}")
        snr_lbl.setStyleSheet("font-size: 11px; color: #00d4ff; font-weight: 600;")
        stats.addWidget(snr_lbl)

        stats.addStretch()

        drift_lbl = QLabel(f"Drift {signal_data.get('drift_rate', 0):.2f}")
        drift_lbl.setStyleSheet("font-size: 11px; color: rgba(200,215,235,0.6);")
        stats.addWidget(drift_lbl)

        layout.addLayout(stats)

        # RFI score bar
        rfi = signal_data.get("rfi_score", 0)
        rfi_row = QHBoxLayout()
        rfi_label = QLabel("RFI")
        rfi_label.setStyleSheet("font-size: 9px; color: rgba(200,215,235,0.4);")
        rfi_row.addWidget(rfi_label)

        bar_bg = QFrame()
        bar_bg.setFixedHeight(4)
        bar_bg.setStyleSheet(
            "background: rgba(8,12,20,0.6); border-radius: 2px;"
        )
        bar_bg.setMinimumWidth(100)

        bar_fill = QFrame(bar_bg)
        fill_width = max(2, int(rfi * 100))
        bar_fill.setFixedSize(fill_width, 4)
        bar_fill.move(0, 0)
        fill_color = "#ff3366" if rfi > 0.7 else "#ffaa00" if rfi > 0.3 else "#00ff88"
        bar_fill.setStyleSheet(
            f"background: {fill_color}; border-radius: 2px;"
        )

        rfi_row.addWidget(bar_bg, 1)
        layout.addLayout(rfi_row)

    def _generate_thumbnail(self):
        """Render a tiny spectrogram thumbnail from signal params."""
        rng = np.random.default_rng(
            int(abs(self.signal_data.get("start_freq", 42)))
        )

        n_t, n_f = 32, 64
        data = rng.normal(0, 1, (n_t, n_f)).astype(np.float32)

        # Inject the signal
        snr = self.signal_data.get("snr", 10)
        drift = self.signal_data.get("drift_rate", 0)
        f0 = n_f // 2
        for t in range(n_t):
            fc = int(f0 + drift * t / n_t * 10)
            if 0 <= fc < n_f:
                data[t, max(0, fc - 1):min(n_f, fc + 2)] += snr * 0.5

        fig = Figure(figsize=(190 / 72, 110 / 72), dpi=72)
        ax = fig.add_subplot(111)
        ax.imshow(data, aspect="auto", origin="lower", cmap="inferno", interpolation="nearest")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="#080c14", bbox_inches="tight", pad_inches=0)
        buf.seek(0)

        img = QImage.fromData(buf.read())
        if not img.isNull():
            pm = QPixmap.fromImage(img).scaled(
                190, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._thumb.setPixmap(pm)
        else:
            self._thumb.setText("—")
            self._thumb.setStyleSheet(
                "color: rgba(140,165,200,0.3); font-size: 12px; "
                "background: rgba(8,12,20,0.8); border-radius: 8px;"
            )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.signal_data)


# ── Signal Gallery Panel ─────────────────────────────────────────────────────

class SignalGallery(QWidget):
    """Grid gallery of detected signal cards with filtering and sorting."""

    open_in_viewer = pyqtSignal(dict)

    SORT_OPTIONS = ["SNR", "Drift Rate", "RFI Score", "Time"]
    FILTER_OPTIONS = ["All", "ML-Filtered", "Candidates Only", "RFI", "Verified"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signals: list[dict] = []
        self._cards: list[SignalCard] = []
        self._current_filter = "All"
        self._current_sort = "SNR"
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(20)

        # ── Header ────────────────────────────────────────────────────────
        header = QHBoxLayout()

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title = QLabel("Signal Gallery")
        title.setStyleSheet(
            "font-size: 26px; font-weight: 300; color: #e0e8f0; letter-spacing: -0.5px;"
        )
        title_col.addWidget(title)
        subtitle = QLabel("Browse detected signals across all observations")
        subtitle.setStyleSheet("font-size: 13px; color: rgba(140,165,200,0.5);")
        title_col.addWidget(subtitle)
        header.addLayout(title_col)
        header.addStretch()

        # Sort
        sort_lbl = QLabel("Sort:")
        sort_lbl.setStyleSheet("font-size: 12px; color: rgba(200,215,235,0.5);")
        header.addWidget(sort_lbl)

        self._sort_combo = QComboBox()
        self._sort_combo.addItems(self.SORT_OPTIONS)
        self._sort_combo.setFixedWidth(130)
        self._sort_combo.currentTextChanged.connect(self._on_sort_changed)
        header.addWidget(self._sort_combo)

        header.addSpacing(12)

        # Count label
        self._count_label = QLabel("0 signals")
        self._count_label.setStyleSheet(
            "font-size: 12px; color: rgba(0,212,255,0.6); font-weight: 500;"
        )
        header.addWidget(self._count_label)

        root.addLayout(header)

        # ── Filter bar ────────────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)

        self._filter_btns: list[QPushButton] = []
        for f in self.FILTER_OPTIONS:
            btn = QPushButton(f)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("filter_name", f)
            btn.clicked.connect(lambda checked, name=f: self._on_filter_clicked(name))
            self._filter_btns.append(btn)
            filter_row.addWidget(btn)

        filter_row.addStretch()

        # Batch actions
        self._verify_btn = QPushButton("✓ Verify All Candidates")
        self._verify_btn.setStyleSheet(self._batch_btn_style("#00ff88"))
        self._verify_btn.clicked.connect(self._verify_all_candidates)
        filter_row.addWidget(self._verify_btn)

        self._reject_btn = QPushButton("✕ Reject All RFI")
        self._reject_btn.setStyleSheet(self._batch_btn_style("#ff3366"))
        self._reject_btn.clicked.connect(self._reject_all_rfi)
        filter_row.addWidget(self._reject_btn)

        self._export_btn = QPushButton("Export Selected")
        self._export_btn.setStyleSheet(self._batch_btn_style("#00d4ff"))
        filter_row.addWidget(self._export_btn)

        root.addLayout(filter_row)
        self._apply_filter_styles()

        # ── Scrollable grid ───────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background: transparent;")
        self._flow_layout = FlowLayout(self._grid_widget, margin=0, spacing=14)
        self._scroll.setWidget(self._grid_widget)
        root.addWidget(self._scroll, 1)

        # ── Empty state ───────────────────────────────────────────────────
        self._empty = QLabel(
            "📡  No signals detected yet.\n\n"
            "Load a filterbank file in the Waterfall Viewer to begin."
        )
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setStyleSheet("""
            color: rgba(140, 165, 200, 0.4);
            font-size: 15px;
            font-weight: 300;
            background: rgba(15, 25, 45, 0.3);
            border: 1px dashed rgba(100, 180, 255, 0.1);
            border-radius: 16px;
            padding: 60px;
        """)
        root.addWidget(self._empty)

    # ── Styles ────────────────────────────────────────────────────────────

    @staticmethod
    def _batch_btn_style(color: str) -> str:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"""
            QPushButton {{
                background: rgba({r},{g},{b}, 0.08);
                border: 1px solid rgba({r},{g},{b}, 0.2);
                border-radius: 8px;
                padding: 7px 14px;
                color: {color};
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: rgba({r},{g},{b}, 0.18);
                border-color: rgba({r},{g},{b}, 0.4);
            }}
        """

    def _apply_filter_styles(self):
        for btn in self._filter_btns:
            name = btn.property("filter_name")
            if name == self._current_filter:
                btn.setStyleSheet("""
                    QPushButton {
                        background: rgba(0, 212, 255, 0.15);
                        border: 1px solid rgba(0, 212, 255, 0.3);
                        border-radius: 8px;
                        padding: 7px 16px;
                        color: #00d4ff;
                        font-size: 12px;
                        font-weight: 600;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background: rgba(15, 25, 45, 0.5);
                        border: 1px solid rgba(100, 180, 255, 0.08);
                        border-radius: 8px;
                        padding: 7px 16px;
                        color: rgba(200, 215, 235, 0.5);
                        font-size: 12px;
                    }
                    QPushButton:hover {
                        background: rgba(20, 35, 60, 0.6);
                        color: rgba(200, 215, 235, 0.8);
                    }
                """)

    # ── Data management ───────────────────────────────────────────────────

    def add_signal(self, signal_data: dict):
        """Add a signal from detection pipeline."""
        # Avoid duplicates by id
        existing_ids = {s.get("id") for s in self._signals}
        if signal_data.get("id") in existing_ids:
            return
        self._signals.append(signal_data)
        self._rebuild_grid()

    def set_signals(self, signals: list[dict]):
        """Replace all signals."""
        self._signals = list(signals)
        self._rebuild_grid()

    def _get_filtered_sorted(self) -> list[dict]:
        """Apply current filter and sort to signals list."""
        filtered = list(self._signals)

        if self._current_filter == "Candidates Only":
            filtered = [s for s in filtered if "candidate" in s.get("classification", "")]
        elif self._current_filter == "RFI":
            filtered = [s for s in filtered if "rfi" in s.get("classification", "")]
        elif self._current_filter == "ML-Filtered":
            filtered = [s for s in filtered if s.get("rfi_score", 1) < 0.5]
        elif self._current_filter == "Verified":
            filtered = [s for s in filtered if s.get("verified", False)]

        sort_key = {
            "SNR": lambda s: s.get("snr", 0),
            "Drift Rate": lambda s: abs(s.get("drift_rate", 0)),
            "RFI Score": lambda s: s.get("rfi_score", 0),
            "Time": lambda s: s.get("id", 0),
        }.get(self._current_sort, lambda s: s.get("snr", 0))

        filtered.sort(key=sort_key, reverse=True)
        return filtered

    def _rebuild_grid(self):
        """Clear and rebuild the card grid."""
        # Remove old cards
        while self._flow_layout.count():
            item = self._flow_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._cards.clear()

        visible = self._get_filtered_sorted()
        self._count_label.setText(f"{len(visible)} signal{'s' if len(visible) != 1 else ''}")

        if not visible:
            self._empty.show()
            self._grid_widget.hide()
            return

        self._empty.hide()
        self._grid_widget.show()

        for sig in visible:
            card = SignalCard(sig)
            card.clicked.connect(self._on_card_clicked)
            self._cards.append(card)
            self._flow_layout.addWidget(card)

    # ── Interactions ──────────────────────────────────────────────────────

    def _on_filter_clicked(self, name: str):
        self._current_filter = name
        self._apply_filter_styles()
        self._rebuild_grid()

    def _on_sort_changed(self, sort_name: str):
        self._current_sort = sort_name
        self._rebuild_grid()

    def _on_card_clicked(self, signal_data: dict):
        self.open_in_viewer.emit(signal_data)

    def _verify_all_candidates(self):
        for sig in self._signals:
            if "candidate" in sig.get("classification", ""):
                sig["verified"] = True
        self._rebuild_grid()

    def _reject_all_rfi(self):
        self._signals = [
            s for s in self._signals if "rfi" not in s.get("classification", "")
        ]
        self._rebuild_grid()
