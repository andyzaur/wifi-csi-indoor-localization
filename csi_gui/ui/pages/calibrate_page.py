"""Calibrate page: status + launch buttons for the three calibration tools.

The three calibration JSONs are READ-ONLY here (status display via
``csi_gui.calibration_status``); we never write or import the tools. Each tool is
LAUNCHED as a detached subprocess that opens its own OpenCV / terminal UI:

    subprocess.Popen([sys.executable, <tool>, ...], cwd=ROOT)

Tools + exact args (from each tool's argparse):
  * floor  -> aruco_setup.py   --camera <shared url>   (PRIMARY: camera moved)
  * lens   -> lens_calibrate.py                        (one-time, no required args)
  * marker -> marker_layout.py                         (interactive stdin; no args)

Floor calibration is PRIMARY (it changes whenever the camera moves) and is shown
prominently at the top; lens + marker are the secondary one-time prerequisites.

The shared camera URL lives on the AppContext so it stays in sync with the Record
page. "Refresh status" re-reads the JSONs.
"""

from __future__ import annotations

import subprocess
import sys

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from csi_gui import calibration_status as cs

# Tool script filenames (launched relative to ROOT, which is the subprocess cwd).
_FLOOR_TOOL = "aruco_setup.py"
_LENS_TOOL = "lens_calibrate.py"
_MARKER_TOOL = "marker_layout.py"


def _fmt_fields(status: cs.CalibrationStatus) -> str:
    """Multi-line key: value summary of a status's fields (+ mtime)."""
    if not status.present:
        return "Not calibrated — run the tool below to create it."
    if status.error is not None:
        return f"File present but unreadable ({status.error})."
    lines = []
    for key, val in status.fields.items():
        lines.append(f"{key}: {val}")
    lines.append(f"last modified: {status.mtime_str}")
    return "\n".join(lines)


class _CalibCard(QFrame):
    """One calibration block: title, status badge, fields, launch button."""

    def __init__(self, title: str, subtitle: str, button_text: str,
                 primary: bool, on_launch, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("calibCardPrimary" if primary else "calibCard")
        self._primary = primary

        self._badge = QLabel("…")
        self._badge.setObjectName("calibBadge")
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_lbl = QLabel(title)
        title_lbl.setObjectName(
            "calibTitlePrimary" if primary else "calibTitle")
        subtitle_lbl = QLabel(subtitle)
        subtitle_lbl.setObjectName("calibSubtitle")
        subtitle_lbl.setWordWrap(True)

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.setSpacing(2)
        header_text.addWidget(title_lbl)
        header_text.addWidget(subtitle_lbl)
        header.addLayout(header_text, 1)
        header.addWidget(self._badge, 0, Qt.AlignmentFlag.AlignTop)

        self._fields = QLabel("")
        self._fields.setObjectName("calibFields")
        self._fields.setWordWrap(True)
        self._fields.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)

        self._button = QPushButton(button_text)
        self._button.setObjectName(
            "calibButtonPrimary" if primary else "calibButton")
        self._button.clicked.connect(on_launch)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._button)
        btn_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addWidget(self._fields)
        layout.addLayout(btn_row)

    def update_status(self, status: cs.CalibrationStatus) -> None:
        self._badge.setText(status.summary)
        if status.present and status.error is None:
            badge_state = "ok"
        elif status.present:
            badge_state = "warn"
        else:
            badge_state = "missing"
        self._badge.setProperty("state", badge_state)
        # Re-polish so the dynamic property restyles the badge.
        self._badge.style().unpolish(self._badge)
        self._badge.style().polish(self._badge)
        self._fields.setText(_fmt_fields(status))


class CalibratePage(QWidget):
    """Status dashboard + launchers for floor / lens / marker calibration."""

    def __init__(self, context, parent=None) -> None:
        super().__init__(parent)
        self._context = context

        # No in-page heading: the shell's topbar already titles this page; the
        # per-card subtitles carry the "floor is primary / others one-time" info.
        # Shared camera URL (same default as Record; mutates the AppContext).
        self._camera_edit = QLineEdit(self._context.camera_url)
        self._camera_edit.setPlaceholderText(
            "Camera URL or index (e.g. http://127.0.0.1:8080/video)")
        self._camera_edit.editingFinished.connect(self._sync_context_url)
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        cam_row.addWidget(self._camera_edit, 1)

        refresh_btn = QPushButton("Refresh status")
        refresh_btn.setObjectName("refreshButton")
        refresh_btn.clicked.connect(self.refresh)
        cam_row.addWidget(refresh_btn)

        # --- cards ------------------------------------------------------------
        self._floor_card = _CalibCard(
            "Floor calibration",
            "Camera extrinsics (homography + grid). Re-run when the camera moves. "
            "Launches aruco_setup.py with the camera above.",
            "Run floor calibration", primary=True,
            on_launch=self._launch_floor)
        self._lens_card = _CalibCard(
            "Lens calibration",
            "One-time intrinsic lens profile (chessboard). Launches lens_calibrate.py.",
            "Run lens calibration", primary=False,
            on_launch=self._launch_lens)
        self._marker_card = _CalibCard(
            "Marker layout",
            "One-time floor-marker map (interactive, in a terminal). "
            "Launches marker_layout.py.",
            "Run marker layout", primary=False,
            on_launch=self._launch_marker)

        secondary = QHBoxLayout()
        secondary.setSpacing(12)
        secondary.addWidget(self._lens_card, 1)
        secondary.addWidget(self._marker_card, 1)

        self._error = QLabel("")
        self._error.setObjectName("calibError")
        self._error.setWordWrap(True)
        self._error.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addLayout(cam_row)
        layout.addWidget(self._floor_card)
        layout.addLayout(secondary)
        layout.addWidget(self._error)
        layout.addStretch(1)

        self.refresh()

    # -- shared-context sync ---------------------------------------------------
    def sync_from_context(self) -> None:
        if self._camera_edit.text() != self._context.camera_url:
            self._camera_edit.setText(self._context.camera_url)

    def _sync_context_url(self) -> None:
        self._context.camera_url = self._camera_edit.text().strip()

    # -- status ----------------------------------------------------------------
    @Slot()
    def refresh(self) -> None:
        statuses = cs.all_status(self._context.root)
        self._floor_card.update_status(statuses["floor"])
        self._lens_card.update_status(statuses["lens"])
        self._marker_card.update_status(statuses["marker"])

    # -- launchers -------------------------------------------------------------
    def _launch(self, argv: list[str]) -> None:
        """Popen a calibration tool with the repo ROOT as cwd."""
        self._error.setVisible(False)
        try:
            subprocess.Popen([sys.executable, *argv], cwd=self._context.root)
        except OSError as exc:
            self._error.setText(f"Failed to launch {argv[0]}: {exc}")
            self._error.setVisible(True)

    @Slot()
    def _launch_floor(self) -> None:
        self._sync_context_url()
        camera = self._context.camera_url
        argv = [_FLOOR_TOOL]
        if camera:
            argv += ["--camera", camera]
        self._launch(argv)

    @Slot()
    def _launch_lens(self) -> None:
        self._sync_context_url()
        camera = self._context.camera_url
        argv = [_LENS_TOOL]
        if camera:
            argv += ["--camera", camera]
        self._launch(argv)

    @Slot()
    def _launch_marker(self) -> None:
        # marker_layout.py is interactive (reads anchors/distances from stdin);
        # it takes no CLI args. Launch it so it inherits this terminal's stdio.
        self._launch([_MARKER_TOOL])
