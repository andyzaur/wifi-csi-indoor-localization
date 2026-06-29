"""ValidatePanel: render a validate_session.build_report() Report inline.

Shown after Stop (and re-runnable via a Validate button). It renders each check
row with PASS/✓ WARN/⚠ FAIL/✗ coloring + the overall verdict, reusing the
csi_gui status palette (green / amber / red, matching the pre-flight badges).

The build_report call runs on a worker thread (driven by the SessionController),
so this panel only *renders* a Report handed to it on the GUI thread; it never
blocks the event loop.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from validate_session import OK, WARN, FAIL, GLYPH

from csi_gui.ui import theme
from csi_gui.ui.theme import C

# Map a validate level to the shared badge QSS state (same palette as pre-flight).
_LEVEL_STATE = {OK: "ok", WARN: "warn", FAIL: "bad"}

VALIDATE_QSS = f"""
#validatePanel {{ background: {C.SURFACE_1}; border: 1px solid {C.BORDER}; border-radius: {theme.RADIUS_CARD}px; }}
#valHeading {{ font-size: 15px; font-weight: bold; color: {C.TEXT}; }}
#valSub {{ color: {C.TEXT_DIM}; font-size: 11px; }}

#valGlyph {{ font-size: 14px; min-width: 16px; }}
#valGlyph[lvl="ok"] {{ color: {C.OK}; }}
#valGlyph[lvl="warn"] {{ color: {C.WARN}; }}
#valGlyph[lvl="bad"] {{ color: {C.BAD}; }}

#valLabel {{ color: {C.TEXT}; font-size: 12px; }}
#valDetail {{ color: {C.TEXT_DIM}; font-family: {theme.MONO}; font-size: 11px; }}

#valVerdict {{
    border-radius: 10px; padding: 12px; font-size: 15px; font-weight: bold;
    background: {C.SURFACE_2}; color: {C.TEXT_DIM}; border: 1px solid {C.BORDER};
}}
#valVerdict[lvl="ok"] {{ background: {C.OK_SOFT}; color: {C.OK}; border: 1px solid {C.ACCENT_LINE}; }}
#valVerdict[lvl="warn"] {{ background: {C.WARN_SOFT}; color: {C.WARN_TEXT}; border: 1px solid {C.WARN_LINE}; }}
#valVerdict[lvl="bad"] {{ background: {C.BAD_SOFT}; color: {C.BAD_TEXT}; border: 1px solid {C.BAD_LINE}; }}

#valRunBtn {{
    background: {C.ACCENT}; border: 1px solid {C.ACCENT_STRONG}; border-radius: {theme.RADIUS_CTRL}px;
    padding: 7px 14px; color: {C.TEXT_ON_ACCENT}; font-weight: 600;
}}
#valRunBtn:hover {{ background: {C.ACCENT_STRONG}; }}
"""

_VERDICT_TEXT = {
    OK: "SESSION OK ✓",
    WARN: "SESSION WARN ⚠",
    FAIL: "SESSION FAIL ✗",
}


class ValidatePanel(QWidget):
    """Render a build_report() Report with per-row coloring + overall verdict."""

    # Emitted when the Validate button is pressed (the Record page drives the
    # actual build_report run on a worker thread via the SessionController).
    validateRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("validatePanel")
        self.setStyleSheet(VALIDATE_QSS)
        self._row_widgets: list = []
        self._build_ui()

    def _build_ui(self) -> None:
        heading = QLabel("Validation")
        heading.setObjectName("valHeading")
        sub = QLabel("validate_session.build_report — checks CSI, camera, "
                     "clapper + alignment before you trust the session.")
        sub.setObjectName("valSub")
        sub.setWordWrap(True)

        self._run_btn = QPushButton("Validate session")
        self._run_btn.setObjectName("valRunBtn")
        self._run_btn.clicked.connect(self.validateRequested)

        self._verdict = QLabel("Not validated yet")
        self._verdict.setObjectName("valVerdict")
        self._verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._verdict.setProperty("lvl", "none")

        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(4)
        self._grid.setColumnStretch(2, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        head = QVBoxLayout()
        head.setSpacing(2)
        head.addWidget(heading)
        head.addWidget(sub)
        layout.addLayout(head)
        layout.addWidget(self._run_btn)
        layout.addWidget(self._verdict)
        layout.addLayout(self._grid)

    def set_running(self) -> None:
        """Show a 'validating…' placeholder while the worker runs."""
        self._verdict.setText("Validating…")
        self._verdict.setProperty("lvl", "none")
        self._verdict.style().unpolish(self._verdict)
        self._verdict.style().polish(self._verdict)

    def set_idle(self) -> None:
        """Reset to the not-run-yet prompt (the report is opt-in / on demand)."""
        self._clear_rows()
        self._verdict.setText("Press Analyze to run the quality report")
        self._verdict.setProperty("lvl", "none")
        self._verdict.style().unpolish(self._verdict)
        self._verdict.style().polish(self._verdict)

    def _clear_rows(self) -> None:
        for widgets in self._row_widgets:
            for w in widgets:
                self._grid.removeWidget(w)
                w.deleteLater()
        self._row_widgets = []

    @Slot(object)
    def render_report_rows(self, result: dict) -> None:
        """Render the PICKLABLE rows dict from :func:`session_worker.compute_report`.

        ``result`` is ``{"rows": [{"status","label","message"}, ...],
        "verdict": "OK"|"WARN"|"FAIL"}`` — the cross-process payload, with no live
        ``Report`` object. Reuses the same per-row coloring + verdict styling as
        :meth:`render_report`.
        """
        self._clear_rows()
        for r, row in enumerate(result.get("rows", [])):
            level = row.get("status", OK)
            label = row.get("label", "")
            detail = row.get("message", "")
            self._add_row(r, level, label, detail)
        verdict = result.get("verdict", OK)
        self._verdict.setText(_VERDICT_TEXT.get(verdict, "SESSION ?"))
        self._verdict.setProperty("lvl", _LEVEL_STATE.get(verdict, "none"))
        self._verdict.style().unpolish(self._verdict)
        self._verdict.style().polish(self._verdict)

    def _add_row(self, r: int, level: str, label: str, detail: str) -> None:
        """Add one colored check row (glyph + label + detail) at grid row ``r``."""
        state = _LEVEL_STATE.get(level, "ok")
        glyph = QLabel(GLYPH.get(level, "?"))
        glyph.setObjectName("valGlyph")
        glyph.setProperty("lvl", state)
        glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel(label)
        lbl.setObjectName("valLabel")
        det = QLabel(detail or "")
        det.setObjectName("valDetail")
        det.setWordWrap(True)
        self._grid.addWidget(glyph, r, 0)
        self._grid.addWidget(lbl, r, 1)
        self._grid.addWidget(det, r, 2)
        self._row_widgets.append((glyph, lbl, det))

    @Slot(object)
    def render_report(self, report) -> None:
        """Render a validate_session.Report: one colored row per check + verdict."""
        self._clear_rows()
        for r, (level, label, detail) in enumerate(report.rows):
            self._add_row(r, level, label, detail)

        verdict = report.worst()
        self._verdict.setText(_VERDICT_TEXT.get(verdict, "SESSION ?"))
        self._verdict.setProperty("lvl", _LEVEL_STATE.get(verdict, "none"))
        self._verdict.style().unpolish(self._verdict)
        self._verdict.style().polish(self._verdict)
