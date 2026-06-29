"""Record page: the full Stage-2 data-collection SESSION flow.

SESSION_CHECKLIST.md section B, driven from one page:

    Pre-flight (Section A panel, green)        — required before recording
        -> Start: pause pre-flight (RELEASES :5500) -> SessionController.start()
           which binds :5500 (CsiCollector) + opens the camera (ArucoTracker),
           both on daemon threads. The live monitor (per-board Hz/age, clapper
           banner, camera detection %, CSI rows, elapsed) confirms flow.
        -> Stop: SessionController.stop() (joins both backends, releases :5500)
           -> resume pre-flight -> reveal the metadata form + inline validate.

The proven Phase-3 live-preview path is reused verbatim: the SessionController's
ArucoTracker (owns_window=False, emit_preview=True, display_scale=0.25) feeds
CameraBridge.on_frame -> LiveFrameProvider -> queued frameReady/positionUpdated
signals -> the QML live view. The tracker now ALSO writes camera.csv (its
``log=`` points into the session dir).

The controller's backend callbacks are plain Python (worker-thread) callables;
this page wraps them as a small relay QObject whose Qt signals are *queued*, so
all UI updates (monitor, validate) run on the GUI thread.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, Qt, QUrl, Signal, Slot
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from csi_gui.adapters.frame_provider import LiveFrameProvider
from csi_gui.adapters.signal_bridge import CameraBridge
from csi_gui.session import SessionController, next_session_name
from csi_gui.ui import theme
from csi_gui.ui.theme import C
from csi_gui.ui.metadata_form import MetadataForm
from csi_gui.ui.monitor_panel import MonitorPanel
from csi_gui.ui.preflight_panel import PreflightPanel
from csi_gui.ui.validate_panel import ValidatePanel

_QML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live_view.qml")
_PROVIDER_ID = "live"

# Preview-fps choices for the selector (default 30).
_FPS_CHOICES = (10, 15, 20, 30)
_DEFAULT_FPS = 30

_SESSION_BAR_QSS = f"""
#sessionBar {{ background: {C.SURFACE_1}; border: 1px solid {C.BORDER}; border-radius: {theme.RADIUS_CARD}px; }}
#sessionBigBtn {{
    background: {C.ACCENT}; border: 1px solid {C.ACCENT_STRONG}; color: {C.TEXT_ON_ACCENT}; font-weight: 600;
    border-radius: {theme.RADIUS_CTRL}px; padding: 11px 26px; font-size: 14px;
}}
#sessionBigBtn:hover {{ background: {C.ACCENT_STRONG}; }}
#sessionBigBtn:disabled {{
    background: {C.SURFACE_2}; border-color: {C.BORDER}; color: {C.TEXT_FAINT};
}}
#sessionStopBtn {{
    background: {C.BAD_SOFT}; border: 1px solid {C.BAD}; color: {C.BAD_TEXT}; font-weight: 600;
    border-radius: {theme.RADIUS_CTRL}px; padding: 11px 26px; font-size: 14px;
}}
#sessionStopBtn:hover {{ background: {C.BAD_LINE}; }}
#sessionStopBtn:disabled {{
    background: {C.SURFACE_2}; border-color: {C.BORDER}; color: {C.TEXT_FAINT};
}}
#sessionState {{
    font-family: {theme.MONO}; font-size: 12px; font-weight: bold;
    padding: 4px 10px; border-radius: 8px; background: {C.SURFACE_2}; color: {C.TEXT_DIM};
}}
#sessionState[st="recording"] {{ background: {C.OK_SOFT}; color: {C.OK}; }}
#sessionState[st="stopped"] {{ background: {C.WARN_SOFT}; color: {C.WARN_TEXT}; }}
#sessionState[st="validated"] {{ background: {C.ACCENT_SOFT}; color: {C.ACCENT_TEXT}; }}
#barFieldLabel {{ color: {C.TEXT_DIM}; font-size: 12px; }}
#modeBtn {{
    background: {C.SURFACE_2}; border: 1px solid {C.BORDER}; color: {C.TEXT_DIM};
    border-radius: {theme.RADIUS_CTRL}px; padding: 6px 14px; font-size: 12.5px;
}}
#modeBtn:checked {{
    background: {C.ACCENT_SOFT}; border: 1px solid {C.ACCENT_LINE};
    color: {C.TEXT}; font-weight: 600;
}}
#modeBtn:hover:!checked {{ background: {C.SURFACE_3}; }}
#phaseCard {{
    background: {C.SURFACE_1}; border: 1px solid {C.BORDER};
    border-radius: {theme.RADIUS_CARD}px;
}}
#phaseStep {{
    color: {C.TEXT_FAINT}; font-family: {theme.MONO}; font-size: 10.5px;
    font-weight: 600; letter-spacing: 1px;
}}
#recordStatus {{ color: {C.TEXT_DIM}; font-family: {theme.MONO}; font-size: 11.5px; }}
"""


class _RailScrollArea(QScrollArea):
    """Vertical scroll area that HONORS its inner widget's heightForWidth.

    A plain ``QScrollArea(widgetResizable=True)`` ignores heightForWidth, so a
    column of word-wrapped rows (the pre-flight checklist) gets vertically
    squeezed and the text clips / rows run together. We pin the inner widget's
    minimum height to its layout's heightForWidth at the current viewport width,
    so the content always gets its true height and simply scrolls when it
    overflows the narrow rail. The viewport width is stable (the rail is a fixed
    width and the V-scrollbar is always on), so this can't oscillate.
    """

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        inner = self.widget()
        if inner is None:
            return
        lay = inner.layout()
        w = self.viewport().width()
        if lay is not None and lay.hasHeightForWidth() and w > 0:
            hfw = lay.heightForWidth(w)
            if hfw > 0 and inner.minimumHeight() != hfw:
                inner.setMinimumHeight(hfw)


class _BackendRelay(QObject):
    """Marshal LOW-RATE SessionController worker-thread callbacks onto the GUI
    thread.

    The controller fires plain-Python callbacks on the collector/tracker worker
    threads. The LOW-RATE ones (board_stats ~1/s, clap rare, state, validated)
    are re-emitted here as *queued* signals so their slots run on the GUI thread,
    never touching Qt off-thread.

    The HIGH-RATE callbacks are deliberately NOT relayed here: on_csi (~100/s)
    and on_position (~22/s) would flood the event loop and starve the live
    preview repaint (the "no live video while recording" bug). Those are wired
    straight to the thread-safe :class:`MonitorState` counters, which the
    monitor's low-Hz repaint timer samples.
    """

    boardStats = Signal(object)
    clap = Signal(object)
    stateChanged = Signal(str)
    validated = Signal(object)


class RecordPage(QWidget):
    """Stage-2 session page: pre-flight -> Start -> live monitor -> Stop ->
    metadata + validate, owning the SessionController lifecycle."""

    def __init__(self, context, parent=None,
                 controller_factory=None) -> None:
        super().__init__(parent)
        self._context = context
        self._controller_factory = controller_factory or SessionController
        self.setStyleSheet(_SESSION_BAR_QSS)

        # --- Stage-1 guided pre-flight panel (above the live view) ------------
        self._preflight = PreflightPanel(context)
        self._preflight_toggle = QPushButton("▾  Pre-flight (Section A)")
        self._preflight_toggle.setObjectName("preflightToggle")
        self._preflight_toggle.setCheckable(True)
        self._preflight_toggle.setChecked(True)
        self._preflight_toggle.clicked.connect(self._toggle_preflight)
        # NOTE: the preflight is NOT wrapped in its own QScrollArea. The whole
        # right rail has ONE scroll area (right_scroll). Two nested
        # setWidgetResizable(True) QScrollAreas caused an infinite resize-feedback
        # loop (QScrollArea::eventFilter -> resize -> eventFilter -> ...) that
        # pinned ~2 CPU cores and beachballed the app.

        # --- worker->GUI relay for the live PREVIEW (built once) --------------
        self._provider = LiveFrameProvider()
        self._bridge = CameraBridge(self._provider, target_fps=float(_DEFAULT_FPS))

        # --- worker->GUI relay for the SESSION backends -----------------------
        self._relay = _BackendRelay()

        # --- session controller (created on Start) ----------------------------
        self._controller: SessionController | None = None
        # Which panel the right rail shows: "setup" (pre-flight) / "recording"
        # (live monitor) / "review" (metadata + validate). The rail shows exactly
        # ONE per phase — otherwise pre-flight stays full-height and pushes the
        # monitor / review panels off the bottom of the rail.
        self._phase = "setup"

        # --- session bar controls ---------------------------------------------
        # Session MODE: a normal walk (camera ground truth) or an empty-room
        # CSI-only baseline capture (no camera at all).
        self._mode = "walk"
        self._mode_walk = QPushButton("Walk (camera)")
        self._mode_walk.setObjectName("modeBtn")
        self._mode_walk.setCheckable(True)
        self._mode_walk.setChecked(True)
        self._mode_empty = QPushButton("Empty room (CSI only)")
        self._mode_empty.setObjectName("modeBtn")
        self._mode_empty.setCheckable(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self._mode_walk)
        self._mode_group.addButton(self._mode_empty)
        self._mode_walk.clicked.connect(lambda: self._set_mode("walk"))
        self._mode_empty.clicked.connect(lambda: self._set_mode("empty"))

        self._purpose_edit = QLineEdit()
        self._purpose_edit.setPlaceholderText("purpose (e.g. walk_grid_slow)")
        self._purpose_edit.setMaximumWidth(220)
        self._name_edit = QLineEdit(self._suggest_name())
        self._name_edit.setPlaceholderText("YYYYMMDD_HHMM_purpose")

        # Action buttons live in the RIGHT rail's phase card (one per phase) —
        # not crammed into this bar (they used to squeeze the camera URL field
        # down to a few pixels).
        self._start_btn = QPushButton("Start session")
        self._start_btn.setObjectName("sessionBigBtn")
        self._stop_btn = QPushButton("Stop session")
        self._stop_btn.setObjectName("sessionStopBtn")
        self._stop_btn.setEnabled(False)
        # Reset everything for the next capture without restarting the app.
        self._new_btn = QPushButton("New session")
        self._new_btn.setObjectName("newSessionBtn")
        self._new_btn.setToolTip("Clear the name / purpose / metadata + scorecard "
                                 "and prep a fresh capture.")
        self._new_btn.clicked.connect(self.new_session)
        self._state_label = QLabel("ready")
        self._state_label.setObjectName("sessionState")
        self._state_label.setProperty("st", "ready")

        # Camera + preview-fps row.
        self._source_edit = QLineEdit(self._context.camera_url)
        self._source_edit.setMinimumWidth(240)
        self._source_edit.setPlaceholderText("Camera URL or index "
                                             "(e.g. http://127.0.0.1:8080/video)")
        self._fps_combo = QComboBox()
        for f in _FPS_CHOICES:
            self._fps_combo.addItem(f"{f} fps", f)
        self._fps_combo.setCurrentIndex(_FPS_CHOICES.index(_DEFAULT_FPS))
        self._fps_combo.currentIndexChanged.connect(self._on_fps_changed)

        self._status = QLabel("idle")
        self._status.setObjectName("recordStatus")

        def _field_label(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setObjectName("barFieldLabel")
            return lbl

        session_bar = QWidget()
        session_bar.setObjectName("sessionBar")
        bar_layout = QVBoxLayout(session_bar)
        bar_layout.setContentsMargins(12, 10, 12, 10)
        bar_layout.setSpacing(8)
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(self._mode_walk)
        row1.addWidget(self._mode_empty)
        row1.addStretch(1)
        row1.addWidget(self._state_label)
        bar_layout.addLayout(row1)
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(_field_label("Purpose"))
        row2.addWidget(self._purpose_edit)
        row2.addWidget(_field_label("Name"))
        row2.addWidget(self._name_edit, 1)
        bar_layout.addLayout(row2)
        row3 = QHBoxLayout()
        row3.setSpacing(8)
        row3.addWidget(_field_label("Camera"))
        row3.addWidget(self._source_edit, 1)
        row3.addWidget(_field_label("Preview"))
        row3.addWidget(self._fps_combo)
        bar_layout.addLayout(row3)

        # --- live view (QQuickWidget hosting live_view.qml) -------------------
        # This is the DOMINANT element: it must be readable from across the room,
        # so it expands to fill the whole LEFT column. detection stays full-res
        # (display_scale=0.25 in the tracker); only the on-screen widget grows.
        self._quick = QQuickWidget()
        self._quick.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
        self._quick.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self._quick.setMinimumSize(480, 360)
        self._quick.engine().addImageProvider(_PROVIDER_ID, self._provider)
        self._quick.setSource(QUrl.fromLocalFile(_QML_PATH))

        # --- live monitor (shown during recording) ----------------------------
        self._monitor = MonitorPanel()
        self._monitor.setVisible(False)

        # --- post-stop panels (metadata + validate) ---------------------------
        self._metadata = MetadataForm()
        self._metadata.setVisible(False)
        self._validate = ValidatePanel()
        self._validate.setVisible(False)
        self._post_stop = QWidget()
        # Stacked vertically: the post-stop panels live in the narrow RIGHT rail
        # now, so side-by-side would be too cramped.
        post_layout = QVBoxLayout(self._post_stop)
        post_layout.setContentsMargins(0, 0, 0, 0)
        post_layout.setSpacing(8)
        post_layout.addWidget(self._metadata)
        post_layout.addWidget(self._validate)
        self._post_stop.setVisible(False)

        # --- layout: LARGE video top-left, side panels on a narrow RIGHT rail --
        # LEFT column (stretch): compact session bar on top, then the big video
        # filling the rest — so the live feed dominates and is readable across
        # the room. RIGHT column (fixed ~340px): pre-flight + live monitor and,
        # after Stop, the metadata form + validate panel, each on a scroll rail.
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(session_bar)
        left_layout.addWidget(self._quick, 1)   # video stretches to dominate

        # Phase header card: which step the operator is on + the ONE action that
        # makes sense right now (Start / Stop / New session). Replaces the old
        # everything-at-once button row that crushed the camera URL field.
        self._phase_step = QLabel("STEP 1 · SETUP")
        self._phase_step.setObjectName("phaseStep")
        phase_card = QWidget()
        phase_card.setObjectName("phaseCard")
        phase_lay = QVBoxLayout(phase_card)
        phase_lay.setContentsMargins(14, 12, 14, 12)
        phase_lay.setSpacing(8)
        phase_lay.addWidget(self._phase_step)
        # The three actions stack; _apply_phase raises the current one.
        self._action_stack = QStackedWidget()
        self._action_stack.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Fixed)
        for btn in (self._start_btn, self._stop_btn, self._new_btn):
            self._action_stack.addWidget(btn)
        phase_lay.addWidget(self._action_stack)
        phase_lay.addWidget(self._status)

        right_col = QWidget()
        right_col.setObjectName("recordRightRail")
        right_col.setFixedWidth(360)
        right_inner = QWidget()
        right_layout = QVBoxLayout(right_inner)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(phase_card)
        right_layout.addWidget(self._preflight)
        right_layout.addWidget(self._monitor)
        right_layout.addWidget(self._post_stop)
        right_layout.addStretch(1)
        right_scroll = _RailScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Reserve the vertical scrollbar + forbid the horizontal one so the
        # viewport width is STABLE — a word-wrap height-for-width change can't
        # toggle a scrollbar and oscillate the resize (the single-scroll variant
        # of the loop above).
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        right_scroll.setWidget(right_inner)
        right_outer = QVBoxLayout(right_col)
        right_outer.setContentsMargins(0, 0, 0, 0)
        right_outer.addWidget(right_scroll)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(left_col, 1)   # left column takes the stretch
        layout.addWidget(right_col)     # right rail stays narrow/fixed

        # --- preview bridge queued signals (cross-thread -> GUI thread) -------
        self._bridge.frameReady.connect(
            self._on_frame_ready, Qt.ConnectionType.QueuedConnection)
        self._bridge.positionUpdated.connect(
            self._on_position_updated, Qt.ConnectionType.QueuedConnection)
        self._bridge.fpsMeasured.connect(
            self._on_fps_measured, Qt.ConnectionType.QueuedConnection)

        # --- backend relay queued signals -> GUI panels ----------------------
        # Only the LOW-RATE callbacks hop the thread via queued signals. The
        # high-rate on_csi/on_position are bound straight to the thread-safe
        # MonitorState in start_session() (no per-event GUI work).
        self._relay.boardStats.connect(
            self._monitor.ingest_board_stats, Qt.ConnectionType.QueuedConnection)
        self._relay.clap.connect(
            self._monitor.ingest_clap, Qt.ConnectionType.QueuedConnection)
        self._relay.stateChanged.connect(
            self._on_state_changed, Qt.ConnectionType.QueuedConnection)
        self._relay.validated.connect(
            self._on_validated, Qt.ConnectionType.QueuedConnection)

        self._start_btn.clicked.connect(self.start_session)
        self._stop_btn.clicked.connect(self.stop_session)
        self._validate.validateRequested.connect(self._run_validate)
        self._source_edit.editingFinished.connect(self._sync_context_url)
        self._purpose_edit.editingFinished.connect(self._refresh_suggested_name)

    # -- name suggestion -------------------------------------------------------
    def _suggest_name(self) -> str:
        purpose = self._purpose_edit.text().strip() if hasattr(self, "_purpose_edit") else ""
        if not purpose and getattr(self, "_mode", "walk") == "empty":
            purpose = "empty_room"
        try:
            return next_session_name(purpose,
                                     sessions_dir=os.path.join(self._context.root, "sessions"))
        except Exception:
            return next_session_name(purpose)

    # -- session mode (walk vs empty-room CSI-only) -----------------------------
    def _set_mode(self, mode: str) -> None:
        """Switch between a camera-tracked walk and an empty-room capture.

        Empty room = CSI + clapper only: the camera tracker never starts, the
        camera pre-flight check stops gating READY, and validation runs the
        CSI-only report. Disabled while a recording is in progress.
        """
        if mode == self._mode:
            return
        if self._controller is not None and self._controller.is_recording:
            # Revert the visual toggle — mode is locked mid-recording.
            (self._mode_walk if self._mode == "walk" else self._mode_empty).setChecked(True)
            return
        self._mode = mode
        empty = mode == "empty"
        self._preflight.set_camera_required(not empty)
        self._source_edit.setEnabled(not empty)
        self._fps_combo.setEnabled(not empty)
        root = self._root
        if root is not None:
            root.setProperty("cameraEnabled", not empty)
        # Default the purpose for the common case; never clobber a custom one.
        if empty and not self._purpose_edit.text().strip():
            self._purpose_edit.setText("empty_room")
        elif not empty and self._purpose_edit.text().strip() == "empty_room":
            self._purpose_edit.clear()
        self._refresh_suggested_name()
        self._context.log(f"session mode -> {mode}")

    @Slot()
    def _refresh_suggested_name(self) -> None:
        # Only re-suggest if the user hasn't hand-edited the name to something custom.
        self._name_edit.setText(self._suggest_name())

    # -- pre-flight panel lifecycle -------------------------------------------
    @Slot()
    def _toggle_preflight(self) -> None:
        expanded = self._preflight_toggle.isChecked()
        # Only honor the collapse toggle in the setup phase (pre-flight is hidden
        # entirely while recording / reviewing).
        self._preflight.setVisible(expanded and self._phase == "setup")
        self._preflight_toggle.setText(
            ("▾" if expanded else "▸") + "  Pre-flight (Section A)")

    _PHASE_STEP_TEXT = {
        "setup": "STEP 1 · SETUP — PRE-FLIGHT",
        "recording": "STEP 2 · RECORDING",
        "review": "STEP 3 · REVIEW",
    }

    def _apply_phase(self, phase: str) -> None:
        """Show exactly one rail panel + ONE action button for ``phase``."""
        self._phase = phase
        setup = phase == "setup"
        review = phase == "review"
        self._phase_step.setText(self._PHASE_STEP_TEXT.get(phase, phase.upper()))
        self._action_stack.setCurrentWidget(
            self._start_btn if setup
            else (self._stop_btn if phase == "recording" else self._new_btn))
        self._preflight.setVisible(setup and self._preflight_toggle.isChecked())
        self._monitor.setVisible(phase == "recording")
        # The review container AND its children must be shown — they're hidden by
        # default, so toggling only the container would leave the rail blank.
        self._post_stop.setVisible(review)
        self._metadata.setVisible(review)
        self._validate.setVisible(review)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Only (re)start the pre-flight sweeps when we're in the setup phase —
        # never revive them mid-recording (it would re-bind :5500 under the
        # collector). Always re-assert the phase's panel visibility.
        if self._phase == "setup":
            self._preflight.start()
        self._apply_phase(self._phase)

    def hideEvent(self, event) -> None:
        """Stop the pre-flight checks + release UDP :5500 when the page hides."""
        self._preflight.stop()
        super().hideEvent(event)

    # -- shared-context sync ---------------------------------------------------
    def sync_from_context(self) -> None:
        if self._source_edit.text() != self._context.camera_url:
            self._source_edit.setText(self._context.camera_url)

    def _sync_context_url(self) -> None:
        self._context.camera_url = self._source_edit.text().strip()

    # -- preview-fps control ---------------------------------------------------
    @Slot(int)
    def _on_fps_changed(self, _index: int) -> None:
        fps = self._fps_combo.currentData()
        if fps is not None:
            self._bridge.set_target_fps(float(fps))
            self._context.log(f"preview fps -> {fps}")

    # -- lifecycle -------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._controller is not None and self._controller.is_recording

    @Slot()
    def start_session(self) -> None:
        """Release :5500 (pause pre-flight), then start both session backends."""
        if self._controller is not None and self._controller.is_recording:
            return
        self._sync_context_url()
        camera = self._context.camera_url or self._source_edit.text().strip()
        name = self._name_edit.text().strip() or self._suggest_name()

        # Warn (but allow override) if pre-flight critical checks are not green.
        if not self._preflight.ready:
            self._context.log("Start session: pre-flight not fully green (override)")

        # Pause pre-flight BEFORE the backends spin up: it RELEASES :5500 (so the
        # CsiCollector can bind it) and stops the subprocess sweeps that would
        # otherwise collapse the tracker frame rate.
        self._preflight.pause()

        # Fresh monitor + hide the post-stop panels for this run. begin() builds
        # a NEW MonitorState, so we must bind the high-rate callbacks to it
        # AFTER this call (below) — never to a stale state object.
        empty = self._mode == "empty"
        self._monitor.begin(camera_enabled=not empty)
        self._apply_phase("recording")

        # on_csi (~100/s) + on_position (~22/s) go STRAIGHT to the thread-safe
        # MonitorState — no queued GUI signal per event. The monitor's repaint
        # timer samples the resulting counters at ~2 Hz. Only the low-rate
        # callbacks are relayed as queued GUI signals.
        controller = self._controller_factory(
            sessions_dir=os.path.join(self._context.root, "sessions"),
            on_board_stats=self._relay.boardStats.emit,
            on_clap=self._relay.clap.emit,
            on_csi=self._monitor.state.on_csi,
            on_position=self._monitor.state.on_position,
            on_frame=self._bridge.on_frame,
            on_log=self._context.logger,
            on_state=self._relay.stateChanged.emit,
            on_validated=self._relay.validated.emit,
            camera_enabled=not empty,
        )

        try:
            session_path = controller.start(name, camera)
        except Exception as exc:  # noqa: BLE001 — surface any setup error in the UI
            self._status.setText(f"start failed: {exc}")
            self._context.log(f"Start session FAILED ({name}, {camera}): {exc}")
            self._monitor.end()
            self._preflight.resume()
            self._apply_phase("setup")
            return

        self._controller = controller
        self._context.log(f"Start session: {name} ({camera}) -> {session_path}")
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._new_btn.setEnabled(False)  # can't reset mid-recording
        self._lock_session_fields(True)
        self._status.setText(f"recording: {name}")

    @Slot()
    def stop_session(self) -> None:
        """Stop + join both backends, resume pre-flight, reveal metadata+validate."""
        controller = self._controller
        if controller is None:
            return

        controller.stop()
        session_path = controller.session_path
        name = controller.session_name
        self._context.log(f"Stop session: {name}")

        self._monitor.end()

        # Recording finished -> bring pre-flight back (runs, but hidden during
        # review) or fully stop it (app shutdown / hidden page).
        if self.isVisible():
            self._preflight.resume()
        else:
            self._preflight.stop()

        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._new_btn.setEnabled(True)
        self._lock_session_fields(False)
        self._status.setText("stopped")

        # Review phase: show the metadata form + validate panel for this session
        # (pre-flight + monitor hidden). Fall back to setup if nothing was saved.
        if session_path is not None:
            self._metadata.set_session(session_path, purpose=self._purpose_edit.text().strip())
            self._apply_phase("review")
            # Auto-run validation once after Stop.
            self._run_validate()
        else:
            self._apply_phase("setup")

        # Suggest the next session name for the following capture.
        self._refresh_suggested_name()

    @Slot()
    def new_session(self) -> None:
        """Reset the page for a fresh capture — no app restart needed.

        Clears the purpose + metadata fields, regenerates the (time-stamped)
        name, resets the scorecard/monitor/status, and returns the rail to the
        pre-flight (setup) phase. A no-op while a recording is in progress.
        """
        if self._controller is not None and self._controller.is_recording:
            return
        self._controller = None
        self._purpose_edit.clear()
        self._refresh_suggested_name()        # fresh YYYYMMDD_HHMM_ name
        self._metadata.clear()                # wipe the last session's fields
        self._validate.set_idle()
        self._monitor.end()
        self._status.setText("idle")
        self._on_state_changed("ready")
        self._lock_session_fields(False)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._new_btn.setEnabled(True)
        # Bring the pre-flight checklist back (resume if it was paused/hidden).
        if self.isVisible():
            self._preflight.resume()
        self._apply_phase("setup")
        self._context.log("New session: form reset")

    def _lock_session_fields(self, locked: bool) -> None:
        for w in (self._source_edit, self._name_edit, self._purpose_edit):
            w.setEnabled(not locked)

    @Slot()
    def _run_validate(self) -> None:
        """Run the session report off-thread; render rows when it lands.

        Walk sessions run the full ``validate_session.build_report``; empty-room
        sessions run the CSI-only report (the full one would FAIL them on the
        deliberately-absent camera.csv).
        """
        controller = self._controller
        if controller is None or controller.session_path is None:
            return
        self._validate.set_running()
        if self._mode == "empty":
            from csi_gui.csi_report import build_csi_report
            controller.validate(build_report=build_csi_report)
        else:
            controller.validate()  # off-thread; -> _on_validated via the relay

    # -- relay slots (GUI thread) ---------------------------------------------
    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        self._state_label.setText(state)
        self._state_label.setProperty("st", state)
        self._state_label.style().unpolish(self._state_label)
        self._state_label.style().polish(self._state_label)

    @Slot(object)
    def _on_validated(self, report) -> None:
        self._validate.render_report(report)

    # -- shutdown hook + back-compat aliases -----------------------------------
    def start_tracker(self) -> None:
        """Back-compat alias for :meth:`start_session` (shell/tests)."""
        self.start_session()

    def stop_tracker(self) -> None:
        """Stop any running session (shell wires aboutToQuit -> here). Safe no-op.

        Doubles as the back-compat alias for :meth:`stop_session`.
        """
        if self._controller is not None and self._controller.is_recording:
            self.stop_session()
        else:
            # Page hidden / nothing recording: still ensure pre-flight is stopped
            # so :5500 is released at shutdown.
            if not self.isVisible():
                self._preflight.stop()

    # -- QML root + queued preview slots ---------------------------------------
    @property
    def _root(self) -> QObject | None:
        return self._quick.rootObject()

    @Slot(int)
    def _on_frame_ready(self, frame_id: int) -> None:
        root = self._root
        if root is not None:
            root.setProperty("frameCounter", frame_id)

    @Slot(object)
    def _on_position_updated(self, position) -> None:
        root = self._root
        if root is None or position is None:
            return
        root.setProperty("posDetected", bool(position.detected))
        root.setProperty("posX", float(position.x_cm) if position.x_cm is not None else 0.0)
        root.setProperty("posY", float(position.y_cm) if position.y_cm is not None else 0.0)
        root.setProperty("gridX", float(position.grid_x_cm) if position.grid_x_cm is not None else 0.0)
        root.setProperty("gridY", float(position.grid_y_cm) if position.grid_y_cm is not None else 0.0)
        root.setProperty("posMethod", str(position.method))

    @Slot(float)
    def _on_fps_measured(self, fps: float) -> None:
        root = self._root
        if root is not None:
            root.setProperty("posFps", float(fps))
