"""Bounded real-cocoa screenshot harness for the GUI redesign pass.

Drives the real MainWindow through every section, grabs each page to
/tmp/csi_gui_shots/, then exits. Hard-bounded with faulthandler so a hang
dumps stacks and dies instead of wedging.

Usage:  venv/bin/python tests/_shot_harness.py [outdir] [width height]
"""
from __future__ import annotations

import faulthandler
import os
import sys
from pathlib import Path

faulthandler.dump_traceback_later(90, exit=True)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from csi_gui.ui import theme  # noqa: E402
from csi_gui.ui.main_window import MainWindow  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/csi_gui_shots")
W = int(sys.argv[2]) if len(sys.argv) > 3 else 1280
H = int(sys.argv[3]) if len(sys.argv) > 3 else 820
OUT.mkdir(parents=True, exist_ok=True)

app = QApplication(sys.argv[:1])
theme.apply(app)
win = MainWindow()
win.resize(W, H)
win.show()

SECTIONS = ["calibrate", "record", "sessions", "live_validate",
            "record_empty"]
state = {"i": 0}


def grab_current() -> None:
    name = SECTIONS[state["i"]]
    pix = win.grab()
    path = OUT / f"{state['i']}_{name}_{W}x{H}.png"
    pix.save(str(path))
    print(f"saved {path}", flush=True)
    state["i"] += 1
    if state["i"] >= len(SECTIONS):
        QTimer.singleShot(200, app.quit)
        return
    nxt = SECTIONS[state["i"]]
    if nxt == "record_empty":
        win._sidebar.setCurrentRow(1)  # Record
        win.record_page._mode_empty.setChecked(True)
        win.record_page._set_mode("empty")
    else:
        win._sidebar.setCurrentRow(state["i"])
    # Give the page time to settle (preflight rows, lazy loads).
    QTimer.singleShot(2500, grab_current)


QTimer.singleShot(2500, grab_current)
rc = app.exec()
os._exit(rc)
