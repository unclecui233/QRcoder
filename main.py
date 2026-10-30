"""Entry point for the QR analysis and generation application."""

from __future__ import annotations

import sys

from qrcoder.ui.main_window import run

if __name__ == "__main__":
    sys.exit(run(sys.argv))
