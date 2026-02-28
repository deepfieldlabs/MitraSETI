"""
MitraSETI About Panel

Displays project info, creator details, technology stack,
and cloud pilot program information.
"""

from __future__ import annotations

import webbrowser

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QGridLayout, QSizePolicy,
)
from PyQt5.QtCore import Qt

from .theme import COLORS, create_glass_card


class AboutPanel(QWidget):
    """About MitraSETI — project info and creator details."""

    _LINKEDIN = "https://www.linkedin.com/in/samantabatabaeian/"
    _GITHUB = "https://github.com/deepfieldlabs"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(36, 36, 36, 36)
        layout.setSpacing(16)

        # Header
        header = QLabel("About MitraSETI")
        header.setStyleSheet(f"""
            font-size: 28px; font-weight: 300;
            color: {COLORS['primary']}; margin-bottom: 4px;
        """)
        layout.addWidget(header)

        sub = QLabel("Intelligent SETI Signal Analysis Platform")
        sub.setStyleSheet(f"font-size: 13px; color: rgba(140,165,200,0.45);")
        layout.addWidget(sub)
        layout.addSpacing(12)

        # Row 1: Project + Creator
        row1 = QHBoxLayout()
        row1.setSpacing(16)

        # Project card
        proj = create_glass_card()
        pl = QVBoxLayout(proj)
        pl.setContentsMargins(22, 22, 22, 22)
        pl.setSpacing(8)

        pl.addWidget(self._section_title("Project"))
        pl.addWidget(self._body_text(
            "MitraSETI is a Rust + Python hybrid platform for intelligent "
            "SETI signal analysis. It combines high-performance Rust DSP "
            "pipelines with Python ML models to detect, classify, and verify "
            "candidate extraterrestrial signals in real time.\n\n"
            "Designed for streaming radio telescope data, it features "
            "GPU-accelerated inference, adaptive RFI rejection, and "
            "AstroLens cross-matching."
        ))

        badges = QHBoxLayout()
        badges.setSpacing(8)
        badges.addWidget(self._badge("v0.1.0", COLORS["success"]))
        badges.addWidget(self._badge("MIT License", COLORS["primary"]))
        badges.addStretch()
        pl.addLayout(badges)
        row1.addWidget(proj)

        # Creator card
        creator = create_glass_card()
        cl = QVBoxLayout(creator)
        cl.setContentsMargins(22, 22, 22, 22)
        cl.setSpacing(6)

        cl.addWidget(self._section_title("Creator"))

        name = QLabel("Saman Tabatabaeian")
        name.setStyleSheet("font-size: 18px; font-weight: 500; color: #e0e8f0;")
        cl.addWidget(name)

        role = QLabel("Founder & Technical Lead, Deep Field Labs")
        role.setStyleSheet("font-size: 12px; color: rgba(140,165,200,0.55);")
        cl.addWidget(role)

        role2 = QLabel("Cloud & DevOps Engineer  |  AI/ML Developer")
        role2.setStyleSheet("font-size: 11px; color: rgba(140,165,200,0.45);")
        cl.addWidget(role2)
        cl.addSpacing(10)

        org_lbl = QLabel("Organization")
        org_lbl.setStyleSheet("font-size: 12px; color: rgba(200,215,235,0.55);")
        cl.addWidget(org_lbl)
        org_val = QLabel("Deep Field Labs")
        org_val.setStyleSheet(f"font-size: 15px; font-weight: 500; color: {COLORS['secondary']};")
        cl.addWidget(org_val)
        cl.addSpacing(8)

        linkedin_btn = QPushButton("  LinkedIn Profile")
        linkedin_btn.setCursor(Qt.PointingHandCursor)
        linkedin_btn.setStyleSheet(self._link_style())
        linkedin_btn.clicked.connect(lambda: webbrowser.open(self._LINKEDIN))
        cl.addWidget(linkedin_btn)

        github_btn = QPushButton("  github.com/deepfieldlabs")
        github_btn.setCursor(Qt.PointingHandCursor)
        github_btn.setStyleSheet(self._link_style())
        github_btn.clicked.connect(lambda: webbrowser.open(self._GITHUB))
        cl.addWidget(github_btn)

        cl.addStretch()
        row1.addWidget(creator)
        layout.addLayout(row1)

        # Row 2: Technology + Companion
        row2 = QHBoxLayout()
        row2.setSpacing(16)

        # Technology card
        tech = create_glass_card()
        tl = QVBoxLayout(tech)
        tl.setContentsMargins(22, 22, 22, 22)
        tl.setSpacing(10)
        tl.addWidget(self._section_title("Technology"))

        tech_grid = QGridLayout()
        tech_grid.setSpacing(10)
        for i, (name_t, color, desc) in enumerate([
            ("Rust", COLORS["warning"], "DSP & Signal Pipeline"),
            ("Python", COLORS["success"], "ML Inference & Web UI"),
            ("FastAPI", COLORS["primary"], "API & Web Server"),
            ("PyTorch", COLORS["secondary"], "GPU-Accelerated Models"),
        ]):
            card = self._tech_card(name_t, color, desc)
            tech_grid.addWidget(card, i // 2, i % 2)
        tl.addLayout(tech_grid)
        tl.addStretch()
        row2.addWidget(tech)

        # Companion card
        companion = create_glass_card()
        cpl = QVBoxLayout(companion)
        cpl.setContentsMargins(22, 22, 22, 22)
        cpl.setSpacing(8)
        cpl.addWidget(self._section_title("Companion Project"))

        al_name = QLabel("AstroLens")
        al_name.setStyleSheet("font-size: 15px; font-weight: 500; color: #e0e8f0;")
        cpl.addWidget(al_name)
        cpl.addWidget(self._body_text(
            "Gravitational lensing detection and analysis platform. "
            "AstroLens cross-matches with MitraSETI candidate signals "
            "to correlate SETI observations with known astrophysical "
            "phenomena and deep-sky survey data."
        ))

        gh_btn = QPushButton("  View on GitHub")
        gh_btn.setCursor(Qt.PointingHandCursor)
        gh_btn.setStyleSheet(self._link_style())
        gh_btn.clicked.connect(lambda: webbrowser.open(self._GITHUB))
        cpl.addWidget(gh_btn)
        cpl.addStretch()
        row2.addWidget(companion)
        layout.addLayout(row2)

        # Cloud Pilot
        pilot = create_glass_card()
        pilot.setStyleSheet(pilot.styleSheet() + """
            border-color: rgba(99, 102, 241, 0.25);
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 rgba(99,102,241,0.06), stop:1 rgba(0,212,255,0.04));
        """)
        pil = QVBoxLayout(pilot)
        pil.setContentsMargins(22, 22, 22, 22)
        pil.setSpacing(8)

        pilot_title = QLabel("Cloud Version — Pilot Program")
        pilot_title.setStyleSheet(f"font-size: 16px; font-weight: 500; color: {COLORS['secondary']};")
        pil.addWidget(pilot_title)

        pil.addWidget(self._body_text(
            "We are developing a cloud-hosted version of MitraSETI for "
            "research teams and institutions that need scalable, GPU-accelerated "
            "SETI analysis without local infrastructure setup. The cloud pilot "
            "includes managed pipelines, team dashboards, and API access."
        ))

        for feat in [
            "Managed GPU inference clusters",
            "Scalable multi-node streaming pipelines",
            "Team collaboration dashboards",
            "REST API for custom integrations",
            "Priority support & onboarding",
        ]:
            fl = QLabel(f"  {feat}")
            fl.setStyleSheet("font-size: 12px; color: rgba(140,165,200,0.5);")
            pil.addWidget(fl)

        pil.addSpacing(8)
        contact_btn = QPushButton("  Contact Us on LinkedIn")
        contact_btn.setCursor(Qt.PointingHandCursor)
        contact_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(77,166,255,0.15);
                border: 1px solid rgba(77,166,255,0.3);
                border-radius: 8px; padding: 10px 20px;
                color: {COLORS['primary']}; font-size: 13px; font-weight: 500;
                text-align: left;
            }}
            QPushButton:hover {{
                background: rgba(77,166,255,0.25);
            }}
        """)
        contact_btn.clicked.connect(lambda: webbrowser.open(self._LINKEDIN))
        pil.addWidget(contact_btn)
        layout.addWidget(pilot)

        layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _section_title(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            font-size: 15px; font-weight: 500;
            color: {COLORS['primary']}; margin-bottom: 8px;
        """)
        return lbl

    @staticmethod
    def _body_text(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("font-size: 13px; color: rgba(200,215,235,0.6); line-height: 1.7;")
        return lbl

    @staticmethod
    def _badge(text: str, color: str) -> QLabel:
        lbl = QLabel(f" {text} ")
        lbl.setStyleSheet(f"""
            background: rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))}, 0.12);
            color: {color}; border-radius: 6px; padding: 3px 10px;
            font-size: 11px; font-weight: 500;
        """)
        return lbl

    @staticmethod
    def _link_style() -> str:
        return f"""
            QPushButton {{
                background: transparent; border: none;
                color: {COLORS['primary']}; font-size: 13px;
                text-align: left; padding: 4px 0;
            }}
            QPushButton:hover {{ color: #7cb8ff; }}
        """

    @staticmethod
    def _tech_card(name: str, color: str, desc: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: rgba(14,22,38,0.5);
                border: 1px solid rgba(80,140,200,0.1);
                border-radius: 10px; padding: 14px;
            }}
        """)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(2)
        n = QLabel(name)
        n.setStyleSheet(f"font-size: 12px; font-weight: 500; color: {color};")
        lay.addWidget(n)
        d = QLabel(desc)
        d.setStyleSheet("font-size: 11px; color: rgba(140,165,200,0.45);")
        lay.addWidget(d)
        return card
