"""
MitraSETI Theme — Crystalline Glass Aesthetic

Frosted glass panels, translucent effects, blue-cyan glow accents.
Designed to evoke deep-space observation and alien signal detection.
"""

from PyQt5.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    # Backgrounds — deep space, muted and professional
    "background": "#0a0e1a",
    "surface": "rgba(14, 22, 38, 0.88)",
    "surface_solid": "#0e1626",
    "elevated": "rgba(18, 30, 50, 0.78)",
    "hover": "rgba(22, 38, 62, 0.55)",

    # Glass effects — subtle, not glowing
    "glass_border": "rgba(80, 140, 200, 0.12)",
    "glass_border_hover": "rgba(80, 140, 200, 0.25)",
    "glass_highlight": "rgba(100, 160, 220, 0.06)",

    # Primary accent — steel blue (professional, observatory feel)
    "primary": "#4da6ff",
    "primary_dim": "rgba(77, 166, 255, 0.55)",
    "primary_bg": "rgba(77, 166, 255, 0.07)",
    "primary_border": "rgba(77, 166, 255, 0.20)",

    # Secondary — muted indigo
    "secondary": "#6366f1",
    "secondary_dim": "rgba(99, 102, 241, 0.55)",
    "secondary_bg": "rgba(99, 102, 241, 0.08)",
    "secondary_border": "rgba(99, 102, 241, 0.22)",

    # Semantic — desaturated for professionalism
    "success": "#34d399",
    "success_bg": "rgba(52, 211, 153, 0.07)",
    "success_border": "rgba(52, 211, 153, 0.20)",
    "warning": "#fbbf24",
    "warning_bg": "rgba(251, 191, 36, 0.07)",
    "warning_border": "rgba(251, 191, 36, 0.20)",
    "danger": "#f87171",
    "danger_bg": "rgba(248, 113, 113, 0.07)",
    "danger_border": "rgba(248, 113, 113, 0.20)",

    # Text hierarchy
    "text_primary": "#e2e8f0",
    "text_secondary": "rgba(196, 210, 230, 0.68)",
    "text_tertiary": "rgba(140, 160, 190, 0.42)",
    "text_accent": "#4da6ff",

    # Glow — very subtle
    "glow_cyan": "rgba(77, 166, 255, 0.18)",
    "glow_purple": "rgba(99, 102, 241, 0.18)",
    "glow_green": "rgba(52, 211, 153, 0.18)",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN QSS STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════

def get_stylesheet() -> str:
    """Return the complete crystalline glass QSS stylesheet."""
    return """
/* ════════════════════════════════════════════════════════════════
   GLOBAL — Crystalline Glass Foundation
   ════════════════════════════════════════════════════════════════ */

* {
    font-family: 'Helvetica Neue', sans-serif;
}

QWidget {
    background-color: #0a0e1a;
    color: #e2e8f0;
    font-size: 13px;
    font-weight: 400;
}

QMainWindow {
    background-color: #0a0e1a;
}

QLabel {
    background: transparent;
    color: #e2e8f0;
    font-weight: 400;
}

/* ════════════════════════════════════════════════════════════════
   SCROLLBARS — Thin, translucent, futuristic
   ════════════════════════════════════════════════════════════════ */

QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 4px 1px;
}
QScrollBar::handle:vertical {
    background: rgba(77, 166, 255, 0.12);
    border-radius: 3px;
    min-height: 40px;
}
QScrollBar::handle:vertical:hover {
    background: rgba(77, 166, 255, 0.28);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
    height: 0;
}

QScrollBar:horizontal {
    background: transparent;
    height: 6px;
    margin: 1px 4px;
}
QScrollBar::handle:horizontal {
    background: rgba(77, 166, 255, 0.12);
    border-radius: 3px;
    min-width: 40px;
}
QScrollBar::handle:horizontal:hover {
    background: rgba(77, 166, 255, 0.28);
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
    width: 0;
}

/* ════════════════════════════════════════════════════════════════
   BUTTONS — Frosted glass with glow hover
   ════════════════════════════════════════════════════════════════ */

QPushButton {
    background: rgba(14, 22, 38, 0.72);
    border: 1px solid rgba(80, 140, 200, 0.10);
    border-radius: 8px;
    padding: 10px 20px;
    color: rgba(196, 210, 230, 0.82);
    font-weight: 500;
    font-size: 13px;
}
QPushButton:hover {
    background: rgba(18, 30, 50, 0.88);
    border-color: rgba(77, 166, 255, 0.32);
    color: #e2e8f0;
}
QPushButton:pressed {
    background: rgba(77, 166, 255, 0.10);
    border-color: rgba(77, 166, 255, 0.42);
}
QPushButton:disabled {
    background: rgba(10, 16, 28, 0.4);
    color: rgba(140, 165, 200, 0.3);
    border-color: transparent;
}

/* Primary action button — cyan glow */
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(0, 180, 220, 0.85), stop:1 rgba(77, 166, 255, 0.9));
    border: 1px solid rgba(77, 166, 255, 0.4);
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(77, 166, 255, 0.95), stop:1 rgba(80, 230, 255, 1));
    border-color: rgba(77, 166, 255, 0.6);
}

/* Danger button */
QPushButton#danger {
    background: rgba(248, 113, 113, 0.12);
    border: 1px solid rgba(248, 113, 113, 0.25);
    color: #f87171;
}
QPushButton#danger:hover {
    background: rgba(248, 113, 113, 0.22);
    border-color: rgba(248, 113, 113, 0.45);
}

/* Success button */
QPushButton#success {
    background: rgba(52, 211, 153, 0.1);
    border: 1px solid rgba(52, 211, 153, 0.25);
    color: #34d399;
}
QPushButton#success:hover {
    background: rgba(52, 211, 153, 0.2);
    border-color: rgba(52, 211, 153, 0.4);
}

/* ════════════════════════════════════════════════════════════════
   INPUTS — Glass fields with cyan focus glow
   ════════════════════════════════════════════════════════════════ */

QLineEdit, QTextEdit, QSpinBox {
    background: rgba(8, 12, 20, 0.85);
    border: 1px solid rgba(100, 180, 255, 0.12);
    border-radius: 10px;
    padding: 10px 14px;
    color: #e0e8f0;
    font-size: 13px;
    selection-background-color: rgba(77, 166, 255, 0.3);
}
QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {
    border-color: rgba(77, 166, 255, 0.5);
    background: rgba(10, 18, 30, 0.95);
}
QLineEdit::placeholder {
    color: rgba(140, 165, 200, 0.4);
}

QComboBox {
    background: rgba(8, 12, 20, 0.85);
    border: 1px solid rgba(100, 180, 255, 0.12);
    border-radius: 10px;
    padding: 10px 14px;
    color: rgba(200, 215, 235, 0.85);
    min-width: 120px;
}
QComboBox:hover {
    border-color: rgba(77, 166, 255, 0.35);
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid rgba(77, 166, 255, 0.5);
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background: #0b1222;
    border: 1px solid rgba(100, 180, 255, 0.2);
    border-radius: 10px;
    padding: 4px;
    color: #e0e8f0;
    selection-background-color: rgba(77, 166, 255, 0.2);
}

/* ════════════════════════════════════════════════════════════════
   GROUP BOX — Glass panel containers
   ════════════════════════════════════════════════════════════════ */

QGroupBox {
    background: rgba(15, 25, 45, 0.5);
    border: 1px solid rgba(100, 180, 255, 0.1);
    border-radius: 14px;
    margin-top: 20px;
    padding: 22px 16px 16px 16px;
    font-weight: 500;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 18px;
    top: 6px;
    padding: 0 10px;
    color: rgba(77, 166, 255, 0.8);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ════════════════════════════════════════════════════════════════
   PROGRESS BAR — Cyan gradient energy bar
   ════════════════════════════════════════════════════════════════ */

QProgressBar {
    background: rgba(8, 12, 20, 0.7);
    border: 1px solid rgba(100, 180, 255, 0.1);
    border-radius: 6px;
    height: 10px;
    text-align: center;
    font-size: 11px;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(0, 180, 220, 0.9),
        stop:0.5 rgba(77, 166, 255, 1),
        stop:1 rgba(124, 58, 237, 0.85));
    border-radius: 5px;
}

/* ════════════════════════════════════════════════════════════════
   CHECKBOX & RADIO — Cyan accent indicators
   ════════════════════════════════════════════════════════════════ */

QCheckBox {
    spacing: 10px;
    color: rgba(200, 215, 235, 0.85);
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid rgba(100, 180, 255, 0.2);
    background: rgba(8, 12, 20, 0.8);
}
QCheckBox::indicator:checked {
    background: rgba(77, 166, 255, 0.6);
    border-color: rgba(77, 166, 255, 0.4);
}
QCheckBox::indicator:hover {
    border-color: rgba(77, 166, 255, 0.5);
}

QRadioButton {
    spacing: 10px;
    color: rgba(200, 215, 235, 0.85);
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 1px solid rgba(100, 180, 255, 0.2);
    background: rgba(8, 12, 20, 0.8);
}
QRadioButton::indicator:checked {
    background: #4da6ff;
    border: 4px solid rgba(8, 12, 20, 0.9);
}

/* ════════════════════════════════════════════════════════════════
   TAB WIDGET — Glass tabs with glow-on-select
   ════════════════════════════════════════════════════════════════ */

QTabWidget::pane {
    background: rgba(15, 25, 45, 0.4);
    border: 1px solid rgba(100, 180, 255, 0.1);
    border-radius: 14px;
    padding: 20px;
    margin-top: -1px;
}
QTabBar::tab {
    background: transparent;
    border: none;
    padding: 14px 28px;
    margin-right: 4px;
    color: rgba(140, 165, 200, 0.5);
    font-weight: 500;
    font-size: 13px;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected {
    color: #4da6ff;
    border-bottom: 2px solid #4da6ff;
    background: rgba(77, 166, 255, 0.05);
}
QTabBar::tab:hover:!selected {
    color: rgba(200, 215, 235, 0.7);
    background: rgba(77, 166, 255, 0.03);
}

/* ════════════════════════════════════════════════════════════════
   LIST WIDGET — Glass list items
   ════════════════════════════════════════════════════════════════ */

QListWidget {
    background: transparent;
    border: none;
    outline: none;
}
QListWidget::item {
    padding: 14px 18px;
    border-radius: 10px;
    margin: 2px 4px;
    color: rgba(200, 215, 235, 0.55);
}
QListWidget::item:selected {
    background: rgba(77, 166, 255, 0.1);
    color: #4da6ff;
    border-left: 2px solid #4da6ff;
}
QListWidget::item:hover:!selected {
    background: rgba(20, 35, 60, 0.4);
    color: rgba(200, 215, 235, 0.8);
}

/* ════════════════════════════════════════════════════════════════
   TABLE WIDGET — Frosted glass table
   ════════════════════════════════════════════════════════════════ */

QTableWidget {
    background: rgba(8, 12, 20, 0.6);
    border: 1px solid rgba(100, 180, 255, 0.1);
    border-radius: 10px;
    gridline-color: rgba(100, 180, 255, 0.06);
    color: #e0e8f0;
    font-size: 12px;
}
QTableWidget::item {
    padding: 8px 12px;
    border-bottom: 1px solid rgba(100, 180, 255, 0.05);
}
QTableWidget::item:selected {
    background: rgba(77, 166, 255, 0.12);
    color: #4da6ff;
}
QHeaderView::section {
    background: rgba(15, 25, 45, 0.8);
    border: none;
    border-bottom: 1px solid rgba(100, 180, 255, 0.1);
    padding: 10px 12px;
    color: rgba(77, 166, 255, 0.7);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ════════════════════════════════════════════════════════════════
   STATUS BAR — Subtle glass bottom bar
   ════════════════════════════════════════════════════════════════ */

QStatusBar {
    background: rgba(8, 12, 20, 0.95);
    border-top: 1px solid rgba(100, 180, 255, 0.08);
    padding: 6px 16px;
    color: rgba(140, 165, 200, 0.5);
    font-size: 12px;
}

/* ════════════════════════════════════════════════════════════════
   TOOLTIPS — Glass tooltip
   ════════════════════════════════════════════════════════════════ */

QToolTip {
    background: rgba(15, 25, 45, 0.95);
    border: 1px solid rgba(77, 166, 255, 0.25);
    border-radius: 8px;
    padding: 8px 12px;
    color: #e0e8f0;
    font-size: 12px;
}

/* ════════════════════════════════════════════════════════════════
   SPLITTER — Subtle glass divider
   ════════════════════════════════════════════════════════════════ */

QSplitter::handle {
    background: rgba(100, 180, 255, 0.06);
    width: 2px;
    height: 2px;
}
QSplitter::handle:hover {
    background: rgba(77, 166, 255, 0.3);
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS — Reusable glass component styles
# ═══════════════════════════════════════════════════════════════════════════════

def create_glass_card(
    parent=None,
    border_color="rgba(100, 180, 255, 0.12)",
    hover_border="rgba(77, 166, 255, 0.3)",
    bg="rgba(15, 25, 45, 0.65)",
) -> QFrame:
    """Create a frosted glass card QFrame."""
    card = QFrame(parent)
    card.setStyleSheet(f"""
        QFrame {{
            background: {bg};
            border: 1px solid {border_color};
            border-radius: 14px;
        }}
        QFrame:hover {{
            border-color: {hover_border};
        }}
    """)
    return card


def create_glow_button(
    text: str,
    color: str = "#4da6ff",
    parent=None,
) -> QPushButton:
    """Create a button with a cyan/color glow on hover."""
    btn = QPushButton(text, parent)
    # Parse hex to rgba for glow
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: rgba({r}, {g}, {b}, 0.1);
            border: 1px solid rgba({r}, {g}, {b}, 0.25);
            border-radius: 10px;
            padding: 10px 20px;
            color: {color};
            font-weight: 500;
            font-size: 13px;
        }}
        QPushButton:hover {{
            background: rgba({r}, {g}, {b}, 0.2);
            border-color: rgba({r}, {g}, {b}, 0.5);
            color: {color};
        }}
        QPushButton:pressed {{
            background: rgba({r}, {g}, {b}, 0.28);
        }}
    """)
    return btn


def create_stat_card_style(
    accent_color: str = "#4da6ff",
) -> str:
    """Return a style dict for stat cards with a specific accent color."""
    r = int(accent_color[1:3], 16)
    g = int(accent_color[3:5], 16)
    b = int(accent_color[5:7], 16)
    return f"""
        QFrame {{
            background: rgba(15, 25, 45, 0.55);
            border: 1px solid rgba({r}, {g}, {b}, 0.12);
            border-radius: 12px;
        }}
        QFrame:hover {{
            border-color: rgba({r}, {g}, {b}, 0.3);
            background: rgba(15, 25, 45, 0.7);
        }}
    """


def make_stat_card(
    label: str,
    value: str,
    accent: str = "#4da6ff",
    parent=None,
) -> QFrame:
    """Build a complete stat card widget: value on top, label below."""
    card = QFrame(parent)
    card.setStyleSheet(create_stat_card_style(accent))

    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(4)

    value_lbl = QLabel(value)
    value_lbl.setObjectName("value")
    value_lbl.setAlignment(Qt.AlignCenter)
    value_lbl.setStyleSheet(
        f"font-size: 26px; font-weight: 700; color: {accent}; border: none;"
    )
    layout.addWidget(value_lbl)

    name_lbl = QLabel(label)
    name_lbl.setObjectName("label")
    name_lbl.setAlignment(Qt.AlignCenter)
    name_lbl.setStyleSheet(
        "font-size: 10px; color: rgba(200,215,235,0.5); "
        "text-transform: uppercase; letter-spacing: 1px; border: none;"
    )
    layout.addWidget(name_lbl)

    return card
