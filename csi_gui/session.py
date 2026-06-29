"""SessionController: the Stage-2 data-collection SESSION state machine.

This is the Qt-free *core* of the Record page's "record a session" flow
(SESSION_CHECKLIST.md section B). It owns the two backend producers — the
:class:`csi_collector.CsiCollector` (CSI + clapper -> ``csi.csv``/``clap.csv``)
and the :class:`aruco_track.ArucoTracker` (camera ground truth -> ``camera.csv``
+ ``corners.csv`` + keyframes) — each on its own daemon thread, mirroring the
``run_forever`` CLI path.

It is deliberately framework-light: callbacks are plain Python callables (never
Qt). The Record page wraps them as *queued* Qt signals so they hop to the GUI
thread, exactly like CameraBridge already does for the live preview.

States (a strict forward progression):

    IDLE/READY  ->  RECORDING  ->  STOPPED  ->  VALIDATED

  * ``start(name, camera_url)`` creates ``sessions/<name>/`` and starts both
    backends; -> RECORDING.
  * ``stop()`` stops + joins both backends; -> STOPPED.
  * ``validate()`` runs :func:`validate_session.build_report` on a worker
    thread and reports the Report via ``on_validated``; -> VALIDATED.

The backends are IMPORTED, never modified. The collectors/tracker can be
injected (``collector_factory`` / ``tracker_factory``) so tests drive the
controller with fakes — no real UDP :5500 bind, no real camera.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import threading
from typing import Callable, Optional

from csi_collector import CsiCollector
from aruco_track import ArucoTracker

# Session lifecycle states.
IDLE = "idle"
READY = "ready"
RECORDING = "recording"
STOPPED = "stopped"
VALIDATED = "validated"

# Match the ArUco tracker's default body-center offsets so the metadata records
# the same numbers the camera ground truth was computed with.
DEFAULT_OFFSETS = (20.0, -15.0)

def _slugify(purpose: str) -> str:
    """Lowercase, collapse non-alphanumerics to single underscores.

    "Walk grid (slow!)" -> "walk_grid_slow". An empty/blank purpose -> "session".
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (purpose or "").strip()).strip("_").lower()
    return slug or "session"


def next_session_name(purpose: str, sessions_dir: str = "sessions",
                      now: Optional["_dt.datetime"] = None) -> str:
    """Return a ``YYYYMMDD_HHMM_<slug>`` name from the current local time.

    The minute-resolution timestamp replaces the old per-day ``NN`` counter —
    captures are now identified by time of day, not 01/02/03. ``now`` is
    injectable (a ``datetime``) so tests are deterministic; it defaults to the
    local clock at call time. ``sessions_dir`` is accepted for call-site
    compatibility but no longer scanned.
    """
    if now is None:
        now = _dt.datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M')}_{_slugify(purpose)}"


class SessionController:
    """Drive the two-backend recording session lifecycle (Qt-free).

    Callbacks (all optional, plain callables, fired on backend worker threads):
      * ``on_board_stats(dict[int, BoardStats])`` — per-board CSI stats ~1 Hz.
      * ``on_clap(ClapEvent)`` — a de-duplicated clapper START/STOP/clap.
      * ``on_csi(CsiEvent)`` — one parsed CSI packet (cheap counters only).
      * ``on_position(PositionState)`` — one camera ground-truth position.
      * ``on_frame(FrameResult)`` — one camera preview frame (downscaled).
      * ``on_log(str)`` — a backend stdout line (routed off both backends).
      * ``on_state(str)`` — state transition (new state string).
      * ``on_validated(Report)`` — the validate() Report, on a worker thread.

    ``collector_factory`` / ``tracker_factory`` build the two backends; the
    defaults construct the real :class:`CsiCollector` / :class:`ArucoTracker`.
    Tests inject fakes so no socket/camera is touched.
    """

    def __init__(self, *, sessions_dir: str = "sessions",
                 on_board_stats: Optional[Callable] = None,
                 on_clap: Optional[Callable] = None,
                 on_csi: Optional[Callable] = None,
                 on_position: Optional[Callable] = None,
                 on_frame: Optional[Callable] = None,
                 on_log: Optional[Callable] = None,
                 on_state: Optional[Callable] = None,
                 on_validated: Optional[Callable] = None,
                 collector_factory: Optional[Callable] = None,
                 tracker_factory: Optional[Callable] = None,
                 display_scale: float = 0.25,
                 offsets: tuple = DEFAULT_OFFSETS,
                 camera_enabled: bool = True) -> None:
        self.sessions_dir = sessions_dir
        self.on_board_stats = on_board_stats
        self.on_clap = on_clap
        self.on_csi = on_csi
        self.on_position = on_position
        self.on_frame = on_frame
        self.on_log = on_log
        self.on_state = on_state
        self.on_validated = on_validated
        self._collector_factory = collector_factory or self._default_collector
        self._tracker_factory = tracker_factory or self._default_tracker
        self.display_scale = display_scale
        self.offsets = offsets
        # Empty-room (CSI-only) sessions skip the camera tracker entirely: only
        # csi.csv + clap.csv are written, which is exactly what the per-day
        # empty-room baseline capture needs (see the 50-min stability study).
        self.camera_enabled = camera_enabled

        self._state = IDLE
        self._session_name: Optional[str] = None
        self._session_path: Optional[str] = None

        self._collector = None
        self._tracker = None
        self._collector_thread: Optional[threading.Thread] = None
        self._tracker_thread: Optional[threading.Thread] = None
        self._validate_thread: Optional[threading.Thread] = None

    # -- default backend factories --------------------------------------------
    def _default_collector(self, name: str):
        # quiet=True suppresses the per-packet stdout logging (~100 lines/s ->
        # a 1.1 MB log + wasted IO/CPU). The on_csi/on_clap/on_board_stats
        # callbacks still fire for the live monitor; only the verbose logging is
        # silenced.
        return CsiCollector(
            session_name=name, write_csv=True, quiet=True,
            on_csi=self.on_csi, on_clap=self.on_clap,
            on_board_stats=self.on_board_stats, on_log=self.on_log)

    def _default_tracker(self, camera_url: str, camera_log: str):
        return ArucoTracker(
            camera=camera_url, log=camera_log,
            owns_window=False, display=False, emit_preview=True,
            display_scale=self.display_scale,
            on_frame=self.on_frame, on_position=self.on_position,
            on_log=self.on_log)

    # -- state -----------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def session_name(self) -> Optional[str]:
        return self._session_name

    @property
    def session_path(self) -> Optional[str]:
        return self._session_path

    @property
    def is_recording(self) -> bool:
        return self._state == RECORDING

    def _set_state(self, state: str) -> None:
        self._state = state
        if self.on_state is not None:
            self.on_state(state)

    # -- lifecycle -------------------------------------------------------------
    def start(self, name: str, camera_url: str) -> str:
        """Create ``sessions/<name>/`` and start both backends.

        Returns the session directory path. Idempotent guard: raises if already
        recording. The CsiCollector binds UDP :5500 here, so the caller MUST
        have released it first (the Record page pauses pre-flight to do so).
        """
        if self._state == RECORDING:
            raise RuntimeError("a session is already recording")

        name = (name or "").strip()
        if not name:
            raise ValueError("session name is required")

        session_path = os.path.join(self.sessions_dir, name)
        os.makedirs(session_path, exist_ok=True)
        self._session_name = name
        self._session_path = session_path

        camera_log = os.path.join(session_path, "camera.csv")

        # Build the backends BEFORE starting any thread, so a construction
        # error (e.g. missing floor_calibration.json) surfaces synchronously and
        # leaves nothing half-running. Empty-room (CSI-only) sessions skip the
        # camera tracker entirely.
        collector = self._collector_factory(name)
        tracker = (self._tracker_factory(camera_url, camera_log)
                   if self.camera_enabled else None)

        # start() on each opens the socket / frame source (still synchronous, so
        # a bind/open failure is raised to the caller); run_forever() then loops
        # on its own daemon thread, mirroring the CLI path.
        collector.start()
        if tracker is not None:
            tracker.start()

        collector_thread = threading.Thread(
            target=collector.run_forever, name="csi-collector", daemon=True)
        tracker_thread = (threading.Thread(
            target=tracker.run_forever, name="aruco-tracker", daemon=True)
            if tracker is not None else None)

        self._collector = collector
        self._tracker = tracker
        self._collector_thread = collector_thread
        self._tracker_thread = tracker_thread

        collector_thread.start()
        if tracker_thread is not None:
            tracker_thread.start()

        self._set_state(RECORDING)
        return session_path

    def stop(self, join_timeout: float = 3.0) -> None:
        """Stop + join both backends. Idempotent; -> STOPPED.

        Stops the camera tracker first (it can take a moment to drain its
        thread-pool frame pipeline), then the CSI collector (closes the CSVs +
        releases :5500). Each backend's ``stop`` is itself idempotent.
        """
        if self._state not in (RECORDING,):
            # Allow stop() from STOPPED/VALIDATED as a safe no-op (shutdown path).
            if self._state in (STOPPED, VALIDATED, IDLE, READY):
                return

        tracker = self._tracker
        collector = self._collector
        tracker_thread = self._tracker_thread
        collector_thread = self._collector_thread

        if tracker is not None:
            tracker.stop()
        if tracker_thread is not None and tracker_thread is not threading.current_thread():
            tracker_thread.join(timeout=join_timeout)

        if collector is not None:
            collector.stop()
        if collector_thread is not None and collector_thread is not threading.current_thread():
            collector_thread.join(timeout=join_timeout)

        self._set_state(STOPPED)

    def validate(self, build_report: Optional[Callable] = None,
                 blocking: bool = False):
        """Run validate_session.build_report on the session dir.

        Off the calling thread by default (``blocking=False``) so the GUI never
        stalls; the Report is delivered via ``on_validated`` and the state moves
        to VALIDATED when it completes. With ``blocking=True`` (tests) it runs
        inline and returns the Report. ``build_report`` is injectable for tests.
        """
        if self._session_path is None:
            raise RuntimeError("no session to validate")
        if build_report is None:
            from validate_session import build_report as _br
            build_report = _br

        session_path = self._session_path

        # Blocking path (tests): run inline; let build_report errors propagate and
        # return the Report, preserving the simple test contract.
        if blocking:
            report = build_report(session_path)
            self._set_state(VALIDATED)
            if self.on_validated is not None:
                self.on_validated(report)
            return report

        def _emit_safely(cb, *args) -> None:
            """Call a GUI callback, swallowing a deleted-Qt-object race.

            If the page (and its relay signals) was torn down while validation
            ran on this daemon thread, the queued ``emit`` raises ``RuntimeError:
            Signal source has been deleted``. That is benign here — the result
            simply has nowhere to go — so we ignore it instead of crashing the
            thread.
            """
            if cb is None:
                return
            try:
                cb(*args)
            except RuntimeError:
                pass

        def _run_threaded() -> None:
            try:
                report = build_report(session_path)
            except Exception as exc:  # noqa: BLE001 — surface, never crash the thread
                self._state = STOPPED
                _emit_safely(self.on_state, STOPPED)
                _emit_safely(self.on_log, f"validate failed: {exc}")
                return
            self._state = VALIDATED
            _emit_safely(self.on_state, VALIDATED)
            _emit_safely(self.on_validated, report)

        self._validate_thread = threading.Thread(
            target=_run_threaded, name="session-validate", daemon=True)
        self._validate_thread.start()
        return None
