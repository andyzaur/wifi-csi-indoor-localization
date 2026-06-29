"""PreflightPanel: the Stage-1 guided pre-flight UI for the Record section.

One row per check: a coloured status badge (green / yellow / red / checking) +
label + live detail, and a "Fix" button wherever a remediation exists (Connect
Wi-Fi, Set static IP [admin], Start iproxy). Plus:

  * a "Recheck all" button (force an immediate sweep),
  * a prominent **READY TO RECORD** banner that turns green only when every
    CRITICAL check passes,
  * a persistent **⚠ STATIC IP ACTIVE — Revert now** bar shown whenever
    :func:`netconfig.is_static_active`, so the user can always get internet back.

All the work happens off the GUI thread (see ``scheduler``); this widget only
renders the latest ``checkUpdated`` signal and runs the one-click fixes. The
camera check uses the shared AppContext camera URL.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from csi_gui.preflight import actions, netconfig
from csi_gui.preflight.engine import (
    CAMERA,
    IPROXY,
    STATIC_IP,
    WIFI,
    PreflightEngine,
)
from csi_gui.preflight.probes import GREEN, RED, YELLOW
from csi_gui.preflight.scheduler import PreflightScheduler
from csi_gui.ui import theme
from csi_gui.ui.theme import C

# Map a probe status (or the synthetic 'CHECKING') to the badge's QSS state.
_BADGE_STATE = {
    GREEN: "ok",
    YELLOW: "warn",
    RED: "bad",
    "CHECKING": "checking",
}
_BADGE_GLYPH = {
    GREEN: "●",
    YELLOW: "●",
    RED: "●",
    "CHECKING": "…",
}

PREFLIGHT_QSS = f"""
#preflightPanel {{ background: {C.SURFACE_1}; border: 1px solid {C.BORDER}; border-radius: {theme.RADIUS_CARD}px; }}
#preflightHeading {{ font-size: 16px; font-weight: bold; color: {C.TEXT}; }}
#preflightSub {{ color: {C.TEXT_DIM}; font-size: 12px; }}

#pfRow {{ border-top: 1px solid {C.HAIR}; }}
#pfRowLabel {{ color: {C.TEXT}; font-size: 13px; }}
#pfRowDetail {{ color: {C.TEXT_DIM}; font-family: {theme.MONO}; font-size: 11px; }}

#pfDot {{
    font-size: 16px; min-width: 18px;
    color: {C.TEXT_FAINT};
}}
#pfDot[pf="ok"] {{ color: {C.OK}; }}
#pfDot[pf="warn"] {{ color: {C.WARN}; }}
#pfDot[pf="bad"] {{ color: {C.BAD}; }}
#pfDot[pf="checking"] {{ color: {C.INFO}; }}

#pfCritTag {{ color: {C.TEXT_FAINT}; font-size: 10px; }}

#pfFixBtn {{
    background: {C.SURFACE_3}; border: 1px solid {C.BORDER_STRONG}; border-radius: 7px;
    padding: 4px 10px; color: {C.TEXT}; font-size: 12px;
}}
#pfFixBtn:hover {{ background: {C.BORDER_STRONG}; }}

#pfReady {{
    border-radius: 10px; padding: 12px; font-size: 16px; font-weight: bold;
    background: {C.BAD_SOFT}; color: {C.BAD_TEXT}; border: 1px solid {C.BAD_LINE};
}}
#pfReady[pf="ready"] {{
    background: {C.OK_SOFT}; color: {C.OK}; border: 1px solid {C.ACCENT_LINE};
}}
#pfReady[pf="paused"] {{
    background: {C.INFO_SOFT}; color: {C.INFO}; border: 1px solid {C.INFO_LINE};
}}

#pfStaticBar {{
    background: {C.WARN_SOFT}; border: 1px solid {C.WARN_LINE}; border-radius: 10px; padding: 8px;
}}
#pfStaticBarText {{ color: {C.WARN_TEXT}; font-size: 12px; font-weight: bold; }}
#pfStaticBtn {{
    background: {C.WARN_SOFT}; border: 1px solid {C.WARN}; border-radius: 7px;
    padding: 4px 12px; color: {C.WARN_TEXT}; font-weight: bold;
}}
#pfStaticBtn:hover {{ background: {C.WARN_LINE}; }}
"""


class _CheckRow(QWidget):
    """One self-contained pre-flight row widget: dot | (label over detail) | fix.

    Each row is its OWN widget (the design's ``.pf-row``: ``18px 1fr auto`` + a
    hairline top border + vertical padding) stacked in a column. This is
    deliberate over a shared QGridLayout: word-wrapped QLabels in a grid
    miscompute their height-for-width, so rows OVERLAPPED in the narrow rail. A
    per-row widget sizes itself to its wrapped content, so rows never collide.
    """

    def __init__(self, check_id: str, label: str, critical: bool,
                 fix_btn=None) -> None:
        super().__init__()
        self.setObjectName("pfRow")
        # A bare QWidget ignores QSS background/border unless it's a styled
        # background widget — without this the #pfRow hairline separator never
        # painted and the rows visually ran together.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.check_id = check_id
        self.critical = critical
        self.status = "CHECKING"

        self.dot = QLabel(_BADGE_GLYPH["CHECKING"])
        self.dot.setObjectName("pfDot")
        self.dot.setProperty("pf", "checking")
        self.dot.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.dot.setFixedWidth(16)

        self.label = QLabel(label + ("  (required)" if critical else ""))
        self.label.setObjectName("pfRowLabel")
        self.label.setWordWrap(True)

        self.detail = QLabel("checking…")
        self.detail.setObjectName("pfRowDetail")
        self.detail.setWordWrap(True)

        main = QVBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(4)
        main.addWidget(self.label)
        main.addWidget(self.detail)

        # Fix button (created on demand by the panel) sits at the row's right,
        # top-aligned next to the label.
        self.fix_btn: QPushButton | None = fix_btn

        row = QHBoxLayout(self)
        row.setContentsMargins(2, 13, 2, 13)
        row.setSpacing(10)
        row.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignTop)
        row.addLayout(main, 1)
        if fix_btn is not None:
            row.addWidget(fix_btn, 0, Qt.AlignmentFlag.AlignTop)

    def _fit_heights(self) -> None:
        """Force the wrapped labels to their TRUE wrapped height.

        A word-wrapped QLabel inside a ``QScrollArea(widgetResizable=True)``
        reports a 1-line ``sizeHint`` (Qt can't resolve heightForWidth through the
        scroll area), so the layout under-allocated and clipped 2-line labels —
        the rows visually ran together. Once the row has its real width we pin
        each label's ``minimumHeight`` to its ``heightForWidth``. It only ever
        grows to a stable value, so this can't oscillate into a resize loop.
        """
        for lbl in (self.label, self.detail):
            w = lbl.width()
            # Guard against pre-layout tiny widths: heightForWidth(10px) would
            # wrap to a dozen lines and pin a bogus huge minimum. The real label
            # width in the rail is always > ~150px.
            if w > 120:
                h = lbl.heightForWidth(w)
                if h > 0 and lbl.minimumHeight() != h:
                    lbl.setMinimumHeight(h)

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._fit_heights()

    def showEvent(self, event):  # noqa: N802 - Qt override
        super().showEvent(event)
        self._fit_heights()


class PreflightPanel(QWidget):
    """Guided pre-flight widget driven by a PreflightEngine + PreflightScheduler."""

    def __init__(self, context, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("preflightPanel")
        self._context = context
        self.setStyleSheet(PREFLIGHT_QSS)

        self._engine = PreflightEngine(
            camera_url_getter=lambda: self._context.camera_url,
            root=getattr(self._context, "root", None))
        self._scheduler = PreflightScheduler(self._engine)
        self._scheduler.checkUpdated.connect(self._on_check_updated)

        # Latest status per check id (for the READY gate).
        self._statuses: dict[str, str] = {}
        self._rows: dict[str, _CheckRow] = {}

        # Paused while a recording is in progress (FIX 1): the scheduler's
        # ping/curl/networksetup subprocesses every few seconds otherwise stall
        # the Python process and collapse the tracker's frame rate. While paused
        # the scheduler is stopped and the board listener releases :5500 + CPU.
        self._paused = False

        self._build_ui()

        # Poll the sentinel so the revert bar appears/disappears even when set
        # from elsewhere (the out-of-process watchdog, another process).
        self._sentinel_timer = QTimer(self)
        self._sentinel_timer.setInterval(1500)
        self._sentinel_timer.timeout.connect(self._refresh_static_bar)
        self._refresh_static_bar()

    # -- UI build --------------------------------------------------------------
    def _build_ui(self) -> None:
        heading = QLabel("Pre-flight checklist")
        heading.setObjectName("preflightHeading")
        sub = QLabel("Section A of the session checklist. Record is safe once all "
                     "required checks are green.")
        sub.setObjectName("preflightSub")
        sub.setWordWrap(True)

        self._recheck_btn = QPushButton("Recheck all")
        self._recheck_btn.setObjectName("pfFixBtn")
        self._recheck_btn.clicked.connect(self._scheduler.recheck_now)

        header = QHBoxLayout()
        htext = QVBoxLayout()
        htext.setSpacing(2)
        htext.addWidget(heading)
        htext.addWidget(sub)
        header.addLayout(htext, 1)
        header.addWidget(self._recheck_btn, 0, Qt.AlignmentFlag.AlignTop)

        # READY banner.
        self._ready = QLabel("NOT READY — complete the required checks")
        self._ready.setObjectName("pfReady")
        self._ready.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ready.setWordWrap(True)  # don't force the narrow rail wider
        self._ready.setProperty("pf", "notready")

        # Static-IP revert bar (hidden unless the sentinel is active).
        self._static_bar = self._build_static_bar()

        # Rows — one self-contained _CheckRow widget per check, stacked in a
        # column with hairline separators (the design's .pf-row). Per-row widgets
        # avoid the wrapped-QLabel-in-a-grid height miscalculation that made the
        # rows overlap in the narrow rail.
        rows_box = QWidget()
        rows_layout = QVBoxLayout(rows_box)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(0)
        for check in self._engine.checks:
            fix = self._make_fix_button(check.id)
            row = _CheckRow(check.id, check.label, check.critical, fix_btn=fix)
            self._rows[check.id] = row
            rows_layout.addWidget(row)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addLayout(header)
        layout.addWidget(self._static_bar)
        layout.addWidget(self._ready)
        layout.addWidget(rows_box)

        # Vertical Minimum (not Maximum): the panel must NEVER shrink below the
        # height its rows need — otherwise, when the rail is shorter than the
        # checklist, the layout squeezes the rows and clips the wrapped text.
        # Overflow instead scrolls the rail (its V-scrollbar is always on).
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

    def _build_static_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("pfStaticBar")
        text = QLabel("⚠ STATIC IP ACTIVE — your normal Wi-Fi will have no internet "
                      "until you revert.")
        text.setObjectName("pfStaticBarText")
        text.setWordWrap(True)
        self._revert_btn = QPushButton("Revert now")
        self._revert_btn.setObjectName("pfStaticBtn")
        self._revert_btn.clicked.connect(self._on_revert_static)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.addWidget(text, 1)
        lay.addWidget(self._revert_btn, 0)
        bar.setVisible(False)
        return bar

    def _make_fix_button(self, check_id: str) -> QPushButton | None:
        spec = {
            WIFI: ("Connect Wi-Fi", self._fix_connect_wifi),
            STATIC_IP: ("Set static IP", self._fix_set_static_ip),
            IPROXY: ("Start iproxy", self._fix_start_iproxy),
            CAMERA: ("Start iproxy", self._fix_start_iproxy),
        }.get(check_id)
        if spec is None:
            return None
        text, handler = spec
        btn = QPushButton(text)
        btn.setObjectName("pfFixBtn")
        btn.clicked.connect(handler)
        return btn

    # -- lifecycle (called by the host page on show/hide/close) ----------------
    def start(self) -> None:
        """Start the board listener + the check scheduler (call when shown).

        A no-op for the scheduler/listener while paused (a recording is running):
        re-showing the page must not revive the subprocess sweeps that pause()
        deliberately stopped. The sentinel timer (revert bar) still runs.
        """
        self._sentinel_timer.start()
        self._refresh_static_bar()
        if self._paused:
            self._render_paused()
            return
        self._engine.start_board_listener()
        self._scheduler.start()

    def stop(self) -> None:
        """Stop checks + release UDP :5500 (call on hide / app close).

        Releasing the port is REQUIRED before a Stage-2 recording collector binds
        it; the board listener is a pure pre-flight probe. Releasing the UDP
        socket is the part that MUST happen, so it is never guarded away — only
        the Qt-timer stops (which can hit a deleted C++ object at teardown) are
        wrapped defensively.
        """
        self._scheduler.stop()
        self._engine.stop_board_listener()
        # A full stop supersedes "paused": the next start() should run normally.
        self._paused = False
        try:
            self._sentinel_timer.stop()
        except RuntimeError:
            pass

    # -- pause/resume during recording (FIX 1) ---------------------------------
    def pause(self) -> None:
        """Suspend pre-flight while a recording runs (idempotent + safe).

        Pre-flight is a BEFORE-recording activity: its check scheduler spawns
        ping/curl/networksetup subprocesses every few seconds, which stalls the
        Python process in bursts and collapses the tracker's frame rate. Pausing
        stops the scheduler AND releases the board listener (frees UDP :5500 +
        CPU — also required so a Stage-2 session collector can bind :5500), then
        shows a clear "paused" state on the banner + rows.
        """
        if self._paused:
            return
        self._paused = True
        # Stop the check scheduler and free :5500 (both calls are themselves
        # idempotent + teardown-safe, so this is safe even if already stopped).
        self._scheduler.stop()
        self._engine.stop_board_listener()
        self._context.log("pre-flight: paused (recording)")
        self._render_paused()

    def resume(self) -> None:
        """Restart pre-flight after a recording stops (idempotent + safe).

        Re-binds the board listener + restarts the check scheduler and clears
        the paused banner. A no-op if the panel was never paused.
        """
        if not self._paused:
            return
        self._paused = False
        self._context.log("pre-flight: resumed")
        # Re-arm the listener + scheduler (start() guards against double-start).
        self._engine.start_board_listener()
        self._scheduler.start()
        # Roll rows back to a neutral "checking…" look until fresh results land;
        # the immediate sweep in scheduler.start() repaints them shortly.
        for row in self._rows.values():
            row.status = "CHECKING"
            row.dot.setText(_BADGE_GLYPH["CHECKING"])
            row.dot.setProperty("pf", "checking")
            row.dot.style().unpolish(row.dot)
            row.dot.style().polish(row.dot)
            row.detail.setText("rechecking…")
        self._update_ready()

    def _render_paused(self) -> None:
        """Show the 'paused while recording' state on the banner + every row."""
        self._ready.setText("Pre-flight paused while recording")
        self._ready.setProperty("pf", "paused")
        self._ready.style().unpolish(self._ready)
        self._ready.style().polish(self._ready)
        for row in self._rows.values():
            row.dot.setText("‖")
            row.dot.setProperty("pf", "checking")
            row.dot.style().unpolish(row.dot)
            row.dot.style().polish(row.dot)
            row.detail.setText("paused — pre-flight runs before recording")
            if row.fix_btn is not None:
                row.fix_btn.setVisible(False)

    @property
    def is_paused(self) -> bool:
        return self._paused

    # -- session mode (walk vs empty-room) --------------------------------------
    def set_camera_required(self, required: bool) -> None:
        """Toggle the camera check's READY gating (empty-room = CSI-only).

        The camera row stays visible (informational) but is re-labelled so the
        operator can see it no longer blocks recording; the READY banner is
        re-evaluated immediately.
        """
        self._engine.camera_required = required
        row = self._rows.get(CAMERA)
        if row is not None:
            base = "Camera stream HTTP 200"
            row.label.setText(base + ("  (required)" if required
                                      else "  (not needed — empty room)"))
        self._update_ready()

    # -- scheduler results -----------------------------------------------------
    @Slot(str, str, str, str)
    def _on_check_updated(self, check_id: str, status: str, detail: str,
                          hint: str) -> None:
        row = self._rows.get(check_id)
        if row is None:
            return
        # A late result can land just after pause() stopped the scheduler; don't
        # let it overwrite the "paused" row state.
        if self._paused:
            return

        # Log only real transitions (Fix 3) so the file isn't spammed every sweep.
        if self._statuses.get(check_id) != status:
            self._context.log(f"check {check_id}: {status} {detail}")
        row.status = status
        self._statuses[check_id] = status

        state = _BADGE_STATE.get(status, "checking")
        row.dot.setText(_BADGE_GLYPH.get(status, "…"))
        row.dot.setProperty("pf", state)
        row.dot.style().unpolish(row.dot)
        row.dot.style().polish(row.dot)

        tip = f"  →  {hint}" if hint and status != GREEN else ""
        row.detail.setText(f"{detail}{tip}")
        if row.fix_btn is not None:
            # Only offer the fix while the check is not green.
            row.fix_btn.setVisible(status != GREEN)

        self._update_ready()

    def _update_ready(self) -> None:
        if self._paused:
            # The paused banner takes precedence over READY/NOT-READY.
            return
        ready = self._engine.all_critical_green(self._statuses)
        if ready:
            self._ready.setText("READY TO RECORD ✓")
            self._ready.setProperty("pf", "ready")
        else:
            self._ready.setText("NOT READY — complete the required checks")
            self._ready.setProperty("pf", "notready")
        self._ready.style().unpolish(self._ready)
        self._ready.style().polish(self._ready)

    @property
    def ready(self) -> bool:
        return self._engine.all_critical_green(self._statuses)

    # -- static-IP bar ---------------------------------------------------------
    def _refresh_static_bar(self) -> None:
        self._static_bar.setVisible(netconfig.is_static_active())

    @Slot()
    def _on_revert_static(self) -> None:
        # Revert the static IP to DHCP AND rejoin the home Wi-Fi (restores internet)
        # if one is configured in ~/.csi_gui_local.json.
        netconfig.revert_and_reconnect()
        self._refresh_static_bar()
        QTimer.singleShot(300, self._scheduler.recheck_now)

    # -- one-click fixes -------------------------------------------------------
    @Slot()
    def _fix_connect_wifi(self) -> None:
        actions.connect_wifi()
        QTimer.singleShot(1500, self._scheduler.recheck_now)

    @Slot()
    def _fix_set_static_ip(self) -> None:
        # One elevated osascript: sets the IP + arms the crash-safe watchdog.
        netconfig.set_static_ip()
        self._refresh_static_bar()
        QTimer.singleShot(1000, self._scheduler.recheck_now)

    @Slot()
    def _fix_start_iproxy(self) -> None:
        actions.start_iproxy()
        QTimer.singleShot(1200, self._scheduler.recheck_now)
