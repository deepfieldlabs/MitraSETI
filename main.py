"""astroSETI — Desktop Application Entry Point.

Launches the PyQt5-based desktop application for interactive
SETI signal analysis and visualization.

Author: Saman Tabatabaeian
"""

import sys

from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import get_stylesheet


def main():
    """Launch the astroSETI desktop application."""
    app = QApplication(sys.argv)
    app.setApplicationName("astroSETI")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("astroSETI")
    app.setStyleSheet(get_stylesheet())

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
