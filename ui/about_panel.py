"""
MitraSETI About Panel

Clean, attractive page displaying project info, creator details,
technology stack, and cloud pilot program information.
"""

from __future__ import annotations

import webbrowser

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


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
        scroll.setStyleSheet("QScrollArea { background: #080c14; border: none; }")

        container = QWidget()
        container.setStyleSheet("background: #080c14;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(60, 48, 60, 48)
        layout.setSpacing(0)

        # ── Title ──────────────────────────────────────────────
        title = QLabel("MitraSETI")
        title.setStyleSheet("""
            font-size: 36px; font-weight: 200;
            color: #4da6ff; letter-spacing: 2px;
        """)
        layout.addWidget(title)

        subtitle = QLabel("Intelligent SETI Signal Analysis Platform")
        subtitle.setStyleSheet("""
            font-size: 14px; color: rgba(140,165,200,0.4);
            letter-spacing: 1px; margin-top: 4px;
        """)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        # Version + License inline
        ver_row = QHBoxLayout()
        ver_row.setSpacing(12)
        ver_row.setContentsMargins(0, 0, 0, 0)
        for txt, col in [("v0.1.0", "#34d399"), ("MIT License", "#4da6ff")]:
            b = QLabel(txt)
            b.setStyleSheet(f"""
                background: rgba({self._hex_to_rgb(col)}, 0.10);
                color: {col}; border-radius: 10px;
                padding: 4px 14px; font-size: 11px; font-weight: 500;
            """)
            ver_row.addWidget(b)
        ver_row.addStretch()
        layout.addLayout(ver_row)

        # ── Divider ────────────────────────────────────────────
        layout.addSpacing(28)
        layout.addWidget(self._divider())
        layout.addSpacing(28)

        # ── Description ────────────────────────────────────────
        desc = QLabel(
            "MitraSETI is a Rust + Python hybrid platform for intelligent SETI signal analysis. "
            "It combines high-performance Rust DSP pipelines with Python ML models to detect, "
            "classify, and verify candidate extraterrestrial signals in real time. Designed for "
            "streaming radio telescope data, it features GPU-accelerated inference, adaptive RFI "
            "rejection, and AstroLens cross-matching."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("""
            font-size: 14px; color: rgba(200,215,235,0.55);
            line-height: 1.8;
        """)
        layout.addWidget(desc)

        # ── Tech tags ──────────────────────────────────────────
        layout.addSpacing(24)
        tech_row = QHBoxLayout()
        tech_row.setSpacing(10)
        tech_row.setContentsMargins(0, 0, 0, 0)
        for name, col in [
            ("Rust", "#fbbf24"),
            ("Python", "#34d399"),
            ("FastAPI", "#4da6ff"),
            ("PyTorch", "#6366f1"),
            ("rayon", "#f87171"),
            ("PyO3", "#a78bfa"),
        ]:
            t = QLabel(name)
            t.setStyleSheet(f"""
                background: rgba({self._hex_to_rgb(col)}, 0.08);
                border: 1px solid rgba({self._hex_to_rgb(col)}, 0.15);
                color: {col}; border-radius: 14px;
                padding: 5px 14px; font-size: 11px; font-weight: 500;
            """)
            tech_row.addWidget(t)
        tech_row.addStretch()
        layout.addLayout(tech_row)

        # ── Divider ────────────────────────────────────────────
        layout.addSpacing(28)
        layout.addWidget(self._divider())
        layout.addSpacing(28)

        # ── Creator ────────────────────────────────────────────
        section_lbl = QLabel("CREATOR")
        section_lbl.setStyleSheet("""
            font-size: 10px; font-weight: 600; color: rgba(77,166,255,0.5);
            letter-spacing: 3px;
        """)
        layout.addWidget(section_lbl)
        layout.addSpacing(12)

        name_lbl = QLabel("Saman Tabatabaeian")
        name_lbl.setStyleSheet("font-size: 22px; font-weight: 400; color: #e0e8f0;")
        layout.addWidget(name_lbl)

        role_lbl = QLabel("Founder & Technical Lead, Deep Field Labs")
        role_lbl.setStyleSheet("font-size: 13px; color: rgba(140,165,200,0.5); margin-top: 2px;")
        layout.addWidget(role_lbl)

        role2_lbl = QLabel("Cloud & DevOps Engineer  \u00b7  AI/ML Developer")
        role2_lbl.setStyleSheet("font-size: 12px; color: rgba(140,165,200,0.35); margin-top: 2px;")
        layout.addWidget(role2_lbl)

        layout.addSpacing(16)

        links_row = QHBoxLayout()
        links_row.setSpacing(12)
        links_row.setContentsMargins(0, 0, 0, 0)

        linkedin_btn = self._action_button("LinkedIn Profile", self._LINKEDIN)
        github_btn = self._action_button("GitHub", self._GITHUB)
        links_row.addWidget(linkedin_btn)
        links_row.addWidget(github_btn)
        links_row.addStretch()
        layout.addLayout(links_row)

        # ── Divider ────────────────────────────────────────────
        layout.addSpacing(28)
        layout.addWidget(self._divider())
        layout.addSpacing(28)

        # ── Companion ──────────────────────────────────────────
        comp_lbl = QLabel("COMPANION PROJECT")
        comp_lbl.setStyleSheet("""
            font-size: 10px; font-weight: 600; color: rgba(77,166,255,0.5);
            letter-spacing: 3px;
        """)
        layout.addWidget(comp_lbl)
        layout.addSpacing(12)

        al_title = QLabel("AstroLens")
        al_title.setStyleSheet("font-size: 18px; font-weight: 400; color: #e0e8f0;")
        layout.addWidget(al_title)

        al_desc = QLabel(
            "Gravitational lensing detection and analysis platform. AstroLens cross-matches "
            "with MitraSETI candidate signals to correlate SETI observations with known "
            "astrophysical phenomena and deep-sky survey data."
        )
        al_desc.setWordWrap(True)
        al_desc.setStyleSheet("""
            font-size: 13px; color: rgba(200,215,235,0.45);
            line-height: 1.7; margin-top: 6px;
        """)
        layout.addWidget(al_desc)

        # ── Divider ────────────────────────────────────────────
        layout.addSpacing(28)
        layout.addWidget(self._divider())
        layout.addSpacing(28)

        # ── Cloud Pilot ────────────────────────────────────────
        cloud_lbl = QLabel("CLOUD PILOT PROGRAM")
        cloud_lbl.setStyleSheet("""
            font-size: 10px; font-weight: 600; color: rgba(99,102,241,0.6);
            letter-spacing: 3px;
        """)
        layout.addWidget(cloud_lbl)
        layout.addSpacing(12)

        cloud_title = QLabel("Run MitraSETI at Scale \u2014 In the Cloud")
        cloud_title.setStyleSheet("font-size: 18px; font-weight: 400; color: #e0e8f0;")
        layout.addWidget(cloud_title)

        cloud_desc = QLabel(
            "We are developing a cloud-hosted version of MitraSETI for research teams "
            "and institutions that need scalable, GPU-accelerated SETI analysis without "
            "local infrastructure setup."
        )
        cloud_desc.setWordWrap(True)
        cloud_desc.setStyleSheet("""
            font-size: 13px; color: rgba(200,215,235,0.45);
            line-height: 1.7; margin-top: 6px;
        """)
        layout.addWidget(cloud_desc)

        layout.addSpacing(14)
        for feat in [
            "Managed GPU inference clusters",
            "Scalable multi-node streaming pipelines",
            "Team collaboration dashboards",
            "REST API for custom integrations",
            "Priority support & onboarding",
        ]:
            fl = QLabel(f"\u2713  {feat}")
            fl.setStyleSheet("font-size: 12px; color: rgba(140,165,200,0.4); padding: 2px 0;")
            layout.addWidget(fl)

        layout.addSpacing(18)
        contact_btn = QPushButton("  Contact Us on LinkedIn")
        contact_btn.setCursor(Qt.PointingHandCursor)
        contact_btn.setStyleSheet("""
            QPushButton {
                background: rgba(99,102,241,0.12);
                border: 1px solid rgba(99,102,241,0.25);
                border-radius: 10px; padding: 12px 24px;
                color: #a78bfa; font-size: 13px; font-weight: 500;
                text-align: left;
            }
            QPushButton:hover {
                background: rgba(99,102,241,0.20);
                border-color: rgba(99,102,241,0.4);
            }
        """)
        contact_btn.clicked.connect(lambda: webbrowser.open(self._LINKEDIN))
        layout.addWidget(contact_btn)

        layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _divider() -> QFrame:
        d = QFrame()
        d.setFixedHeight(1)
        d.setStyleSheet("background: rgba(77,166,255,0.06);")
        return d

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        return ",".join(str(int(h[i : i + 2], 16)) for i in (0, 2, 4))

    def _action_button(self, text: str, url: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton {
                background: rgba(77,166,255,0.08);
                border: 1px solid rgba(77,166,255,0.15);
                border-radius: 8px; padding: 8px 18px;
                color: #4da6ff; font-size: 12px; font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(77,166,255,0.15);
                border-color: rgba(77,166,255,0.3);
            }
        """)
        btn.clicked.connect(lambda: webbrowser.open(url))
        return btn
