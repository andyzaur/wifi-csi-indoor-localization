"""MetadataForm: the post-Stop "record what you remember" form.

Shown after a session is STOPPED. One field per human metadata key (the same
keys :data:`session_metadata.HUMAN_FIELDS` carries), plus a Save button that
calls :func:`session_metadata.write_metadata` with the entered values, the repo
ROOT, and the ArUco body-center offsets ``(20.0, -15.0)`` (matching the tracker
defaults the camera ground truth was computed with), writing ``metadata.json``
into the session directory.

The field->dict mapping logic lives in :func:`collect_fields` (Qt-free) so it is
unit-testable; the QWidget is a thin wrapper that builds one input per key and
wires the Save button.
"""

from __future__ import annotations

from typing import Optional

from session_metadata import HUMAN_FIELDS, write_metadata
from csi_gui.app_context import ROOT
from csi_gui.ui import theme
from csi_gui.ui.theme import C

# The ArUco body-center offsets the camera ground truth uses (aruco_track
# defaults). Recorded into metadata so a session is self-describing.
OFFSETS = (20.0, -15.0)

# Keys that benefit from a multi-line input.
_MULTILINE_KEYS = {"furniture_notes", "notes", "board_placement"}


def collect_fields(values: dict) -> dict:
    """Map raw text ``values`` to the human dict ``write_metadata`` expects.

    Every HUMAN_FIELDS key is present in the result; a blank/missing entry
    becomes ``None`` (so write_metadata keeps the slot visible as a TODO rather
    than writing an empty string). Unknown keys in ``values`` are ignored.
    """
    out = {}
    for key, _prompt in HUMAN_FIELDS:
        raw = values.get(key)
        text = (raw or "").strip() if isinstance(raw, str) else raw
        out[key] = text if text else None
    return out


def save_metadata(session_dir: str, values: dict,
                  offsets: tuple = OFFSETS, root: str = ROOT,
                  write_fn=None) -> dict:
    """Collect ``values`` -> human dict and write the session metadata.json.

    Thin, Qt-free seam over :func:`session_metadata.write_metadata` so the form
    (and tests) call one function. ``write_fn`` is injectable for tests; when
    None it resolves the module-level ``write_metadata`` at call time (so a
    monkeypatch of this module's name takes effect).
    """
    if write_fn is None:
        write_fn = write_metadata
    human = collect_fields(values)
    return write_fn(session_dir, root, human, offsets=offsets)


# ---------------------------------------------------------------------------
# Qt widget (lazy import — the mapping logic above needs no Qt).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from PySide6.QtCore import Signal, Slot
    from PySide6.QtWidgets import (
        QFormLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    _HAVE_QT = True
except Exception:  # pragma: no cover
    _HAVE_QT = False


METADATA_QSS = f"""
#metaPanel {{ background: {C.SURFACE_1}; border: 1px solid {C.BORDER}; border-radius: {theme.RADIUS_CARD}px; }}
#metaHeading {{ font-size: 15px; font-weight: bold; color: {C.TEXT}; }}
#metaSub {{ color: {C.TEXT_DIM}; font-size: 11px; }}
#metaSaved {{ color: {C.ACCENT}; font-size: 12px; }}
#metaError {{ color: {C.BAD}; font-size: 12px; }}
#metaSaveBtn {{
    background: {C.ACCENT}; border: 1px solid {C.ACCENT_STRONG}; color: {C.TEXT_ON_ACCENT}; font-weight: 600;
    border-radius: {theme.RADIUS_CTRL}px; padding: 7px 16px;
}}
#metaSaveBtn:hover {{ background: {C.ACCENT_STRONG}; }}
"""


if _HAVE_QT:

    class MetadataForm(QWidget):
        """One input per human metadata key + a Save -> write_metadata button."""

        # Emitted with the written metadata dict after a successful Save.
        saved = Signal(object)

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.setObjectName("metaPanel")
            self.setStyleSheet(METADATA_QSS)

            self._session_dir: Optional[str] = None
            self._inputs: dict[str, object] = {}
            self._build_ui()

        def _build_ui(self) -> None:
            heading = QLabel("Session metadata")
            heading.setObjectName("metaHeading")
            sub = QLabel("Fill while you remember — room, people, walk style, "
                         "board placement. Saved to metadata.json.")
            sub.setObjectName("metaSub")
            sub.setWordWrap(True)

            form = QFormLayout()
            form.setSpacing(6)
            # The rail is narrow: stack each label ABOVE its field so the inputs
            # get the full width instead of being squished beside long prompts.
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
            for key, prompt in HUMAN_FIELDS:
                if key in _MULTILINE_KEYS:
                    field = QPlainTextEdit()
                    field.setFixedHeight(48)
                else:
                    field = QLineEdit()
                    field.setPlaceholderText(prompt)
                self._inputs[key] = field
                form.addRow(QLabel(prompt), field)

            self._save_btn = QPushButton("Save metadata")
            self._save_btn.setObjectName("metaSaveBtn")
            self._save_btn.clicked.connect(self._on_save)

            self._status = QLabel("")
            self._status.setObjectName("metaSaved")
            self._status.setWordWrap(True)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setSpacing(8)
            layout.addWidget(heading)
            layout.addWidget(sub)
            layout.addLayout(form)
            layout.addWidget(self._save_btn)
            layout.addWidget(self._status)

        # -- session binding ---------------------------------------------------
        def set_session(self, session_dir: str, purpose: Optional[str] = None) -> None:
            """Point the form at ``session_dir`` (where metadata.json is written).

            ``purpose`` (the Record page's purpose field) prefills the matching
            input so the user isn't retyping it.
            """
            self._session_dir = session_dir
            self._status.setText("")
            self._status.setObjectName("metaSaved")
            if purpose and "purpose" in self._inputs:
                self._set_value("purpose", purpose)

        def _set_value(self, key: str, value: str) -> None:
            field = self._inputs.get(key)
            if field is None:
                return
            if isinstance(field, QPlainTextEdit):
                field.setPlainText(value)
            else:
                field.setText(value)

        def values(self) -> dict:
            """Read the current text of every field into a ``{key: text}`` dict."""
            out = {}
            for key, field in self._inputs.items():
                if isinstance(field, QPlainTextEdit):
                    out[key] = field.toPlainText()
                else:
                    out[key] = field.text()
            return out

        def clear(self) -> None:
            """Wipe every input + the status line (used by Record's New session)."""
            self._session_dir = None
            for field in self._inputs.values():
                field.clear()  # QLineEdit + QPlainTextEdit both have .clear()
            self._status.setText("")
            self._status.setObjectName("metaSaved")

        # -- save --------------------------------------------------------------
        @Slot()
        def _on_save(self) -> None:
            if self._session_dir is None:
                self._show_error("no session selected")
                return
            try:
                meta = save_metadata(self._session_dir, self.values())
            except Exception as exc:  # noqa: BLE001 — surface write errors in UI
                self._show_error(f"save failed: {exc}")
                return
            self._status.setObjectName("metaSaved")
            self._status.setText("Saved metadata.json ✓")
            self._status.style().unpolish(self._status)
            self._status.style().polish(self._status)
            self.saved.emit(meta)

        def _show_error(self, msg: str) -> None:
            self._status.setObjectName("metaError")
            self._status.setText(msg)
            self._status.style().unpolish(self._status)
            self._status.style().polish(self._status)
