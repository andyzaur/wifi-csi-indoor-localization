"""MonitorPanel: the live "is everything flowing?" view, shown while RECORDING.

The Record page wires the SessionController's backend callbacks (as queued Qt
signals) into this panel so it renders, at a cheap cadence:

  * Per-RX-board (1 / 4 / 5) row: Hz + age (s) + a green/red present dot, fed by
    ``on_board_stats`` (~1 Hz from the collector).
  * A CLAPPER banner: the last event (START=green / STOP=red) + elapsed-since-
    START, fed by ``on_clap``.
  * Camera detection % (rolling) + current grid cell, fed by ``on_position``
    (PositionState.detected / grid_x_cm / grid_y_cm).
  * Total CSI rows + elapsed session time, fed by ``on_csi`` + a 1 Hz tick.

The numeric/rollup logic lives in :class:`MonitorState` — a plain, Qt-free
object so it is unit-testable without a display. The QWidget is a thin renderer
over it.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

from csi_gui.ui import theme
from csi_gui.ui.theme import C

# Expected RX board ids (board 3 = TX, board 2 = clapper — neither sends CSI).
RX_BOARDS = (1, 4, 5)

# A board is "present" if its last stats sample is fresh AND moving.
PRESENT_AGE_S = 2.0
PRESENT_MIN_HZ = 1.0

# Rolling window (frames) for the camera detection percentage.
_DETECT_WINDOW = 300


class MonitorState:
    """Qt-free rollup of the live recording health metrics.

    Fed by the same plain-Python events the SessionController callbacks carry.
    Every getter is cheap and side-effect free so the renderer can poll it on a
    timer.

    THREADING: the high-rate ingest paths — :meth:`on_csi` (~100/s) and
    :meth:`on_position` (~22/s) — are called DIRECTLY on the backend worker
    threads (no per-event queued GUI signal, which would flood the event loop
    and starve the live preview). The GUI repaint timer samples the readouts
    (``csi_total`` / ``detection_pct`` / ``current_cell``) on its own (~2 Hz)
    thread. A single lock guards the mutable counters/rollups touched from both
    sides so a sampled read is consistent and never races a worker increment.
    The low-rate paths (``on_board_stats`` ~1/s, ``on_clap`` rare) still arrive
    as queued GUI signals but take the same lock so the data structures have one
    coherent owner.
    """

    def __init__(self, rx_boards=RX_BOARDS, detect_window: int = _DETECT_WINDOW,
                 clock=time.monotonic):
        self.rx_boards = tuple(rx_boards)
        self._clock = clock
        self._started_at: Optional[float] = None

        # Guards every mutable field below: the high-rate on_csi/on_position run
        # on worker threads while the GUI timer samples the readouts.
        self._lock = threading.Lock()

        # board_id -> (hz, age_s, last_seen_monotonic)
        self._board: dict[int, tuple] = {}

        self._csi_total = 0

        self._detect = deque(maxlen=int(detect_window))
        self._current_cell: Optional[tuple] = None

        # last clapper event
        self._last_clap_name: Optional[str] = None
        self._last_clap_wall: Optional[float] = None
        self._start_clap_wall: Optional[float] = None

    # -- ingest ----------------------------------------------------------------
    def mark_started(self) -> None:
        """Record the session start instant (drives elapsed session time)."""
        self._started_at = self._clock()

    def on_board_stats(self, stats: dict) -> None:
        """Ingest a ``{board_id: BoardStats}`` dict from the collector."""
        now = self._clock()
        with self._lock:
            for board_id, st in stats.items():
                self._board[int(board_id)] = (float(st.hz), float(st.age_s), now)

    def on_csi(self, event=None) -> None:
        """Count one CSI packet (cheap — total rows readout only).

        Called on the collector WORKER thread, ~100/s, with NO queued GUI
        signal. Just bumps a lock-guarded counter the GUI timer samples.
        """
        with self._lock:
            self._csi_total += 1

    def on_position(self, state) -> None:
        """Ingest a PositionState: detection roll + current grid cell.

        Called on the tracker WORKER thread, ~22/s, with NO queued GUI signal.
        """
        detected = bool(state.detected)
        cell = None
        if detected and state.grid_x_cm is not None and state.grid_y_cm is not None:
            cell = (float(state.grid_x_cm), float(state.grid_y_cm))
        with self._lock:
            self._detect.append(1 if detected else 0)
            if cell is not None:
                self._current_cell = cell

    def on_clap(self, event) -> None:
        """Ingest a ClapEvent: track last event + the START anchor for elapsed."""
        with self._lock:
            self._last_clap_name = event.event_name
            self._last_clap_wall = float(event.wall_time_s)
            if event.event_name == "start":
                self._start_clap_wall = float(event.wall_time_s)

    # -- readouts --------------------------------------------------------------
    def board_row(self, board_id: int) -> dict:
        """Return ``{hz, age_s, present}`` for one board (zeros if never seen)."""
        with self._lock:
            entry = self._board.get(int(board_id))
        if entry is None:
            return {"hz": 0.0, "age_s": float("inf"), "present": False}
        hz, age_s, _ = entry
        present = (age_s <= PRESENT_AGE_S) and (hz >= PRESENT_MIN_HZ)
        return {"hz": hz, "age_s": age_s, "present": present}

    def detection_pct(self) -> float:
        """Rolling-window camera detection percentage (0 if no frames yet)."""
        with self._lock:
            if not self._detect:
                return 0.0
            return 100.0 * sum(self._detect) / len(self._detect)

    @property
    def current_cell(self) -> Optional[tuple]:
        with self._lock:
            return self._current_cell

    @property
    def csi_total(self) -> int:
        with self._lock:
            return self._csi_total

    def elapsed_s(self) -> float:
        """Seconds since :meth:`mark_started` (0 before it / clap START)."""
        if self._started_at is None:
            return 0.0
        return max(0.0, self._clock() - self._started_at)

    @property
    def last_clap_name(self) -> Optional[str]:
        with self._lock:
            return self._last_clap_name

    def since_start_clap_s(self) -> Optional[float]:
        """Wall seconds since the START clap (None until a START is seen)."""
        with self._lock:
            start = self._start_clap_wall
        if start is None:
            return None
        return max(0.0, time.time() - start)


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# Qt widget (imported lazily by the Record page; the MonitorState above is the
# tested core and needs no Qt).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from PySide6.QtCore import Qt, QTimer, Slot
    from PySide6.QtWidgets import (
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QVBoxLayout,
        QWidget,
    )
    _HAVE_QT = True
except Exception:  # pragma: no cover
    _HAVE_QT = False


MONITOR_QSS = f"""
#monitorPanel {{ background: {C.SURFACE_1}; border: 1px solid {C.BORDER}; border-radius: {theme.RADIUS_CARD}px; }}
#monHeading {{ font-size: 15px; font-weight: bold; color: {C.TEXT}; }}
#monSub {{ color: {C.TEXT_DIM}; font-size: 11px; }}

#monBoardLabel {{ color: {C.TEXT}; font-size: 13px; }}
#monBoardStat {{ color: {C.TEXT_DIM}; font-family: {theme.MONO}; font-size: 12px; }}
#monDot {{ font-size: 16px; min-width: 16px; color: {C.TEXT_FAINT}; }}
#monDot[pf="ok"] {{ color: {C.OK}; }}
#monDot[pf="bad"] {{ color: {C.BAD}; }}

#monClap {{
    border-radius: 10px; padding: 10px; font-size: 14px; font-weight: bold;
    background: {C.SURFACE_2}; color: {C.TEXT_DIM}; border: 1px solid {C.BORDER};
}}
#monClap[clap="start"] {{ background: {C.OK_SOFT}; color: {C.OK}; border: 1px solid {C.ACCENT_LINE}; }}
#monClap[clap="stop"] {{ background: {C.BAD_SOFT}; color: {C.BAD_TEXT}; border: 1px solid {C.BAD_LINE}; }}

#monMetric {{ color: {C.TEXT_DIM}; font-family: {theme.MONO}; font-size: 12px; }}
#monMetricBig {{ color: {C.TEXT}; font-size: 13px; font-weight: bold; }}
"""


if _HAVE_QT:

    class MonitorPanel(QWidget):
        """Live recording-health view. Thin renderer over a :class:`MonitorState`."""

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.setObjectName("monitorPanel")
            self.setStyleSheet(MONITOR_QSS)
            self.state = MonitorState()
            self._camera_enabled = True

            self._board_rows: dict[int, dict] = {}
            self._build_ui()

            # A cheap 2 Hz repaint tick (board freshness + elapsed clocks tick
            # even when no new event arrives).
            self._tick = QTimer(self)
            self._tick.setInterval(500)
            self._tick.timeout.connect(self.refresh)

        # -- UI build ----------------------------------------------------------
        def _build_ui(self) -> None:
            heading = QLabel("Live monitor")
            heading.setObjectName("monHeading")
            sub = QLabel("Confirm CSI, camera + clapper are all flowing.")
            sub.setObjectName("monSub")
            sub.setWordWrap(True)

            # Per-board grid.
            board_grid = QGridLayout()
            board_grid.setHorizontalSpacing(10)
            board_grid.setVerticalSpacing(4)
            board_grid.setColumnStretch(2, 1)
            for r, bid in enumerate(self.state.rx_boards):
                dot = QLabel("●")
                dot.setObjectName("monDot")
                dot.setProperty("pf", "bad")
                dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label = QLabel(f"Board {bid}")
                label.setObjectName("monBoardLabel")
                stat = QLabel("— Hz   age —")
                stat.setObjectName("monBoardStat")
                board_grid.addWidget(dot, r, 0)
                board_grid.addWidget(label, r, 1)
                board_grid.addWidget(stat, r, 2)
                self._board_rows[bid] = {"dot": dot, "stat": stat}

            # Clapper banner.
            self._clap = QLabel("CLAPPER — waiting for START")
            self._clap.setObjectName("monClap")
            self._clap.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Camera + totals.
            self._camera = QLabel("Camera: — detected   cell —")
            self._camera.setObjectName("monMetric")
            self._totals = QLabel("CSI rows: 0    elapsed 00:00")
            self._totals.setObjectName("monMetricBig")

            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet(f"color: {C.HAIR};")

            layout = QVBoxLayout(self)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setSpacing(8)
            head = QVBoxLayout()
            head.setSpacing(2)
            head.addWidget(heading)
            head.addWidget(sub)
            layout.addLayout(head)
            layout.addWidget(self._clap)
            layout.addLayout(board_grid)
            layout.addWidget(sep)
            cam_row = QHBoxLayout()
            cam_row.addWidget(self._camera, 1)
            layout.addLayout(cam_row)
            layout.addWidget(self._totals)

        # -- lifecycle ---------------------------------------------------------
        def begin(self, camera_enabled: bool = True) -> None:
            """Reset the state for a fresh session + start the repaint tick.

            ``camera_enabled=False`` (empty-room / CSI-only session) hides the
            camera detection row — there is no tracker, so a permanent "0%
            detected" would read as a fault instead of a mode.
            """
            self.state = MonitorState(rx_boards=self.state.rx_boards)
            self.state.mark_started()
            self._camera_enabled = camera_enabled
            self._camera.setVisible(camera_enabled)
            self._tick.start()
            self.refresh()

        def end(self) -> None:
            """Stop the repaint tick (one final refresh keeps the last numbers)."""
            try:
                self._tick.stop()
            except RuntimeError:
                pass
            self.refresh()

        # -- ingest slots ------------------------------------------------------
        # board_stats (~1/s) + clap (rare) arrive as QUEUED GUI signals from the
        # Record page and trigger an immediate repaint — cheap at their rate.
        #
        # The high-rate paths (csi ~100/s, position ~22/s) are NOT wired through
        # these slots in production: the Record page points the collector/tracker
        # callbacks DIRECTLY at ``self.state.on_csi`` / ``self.state.on_position``
        # (thread-safe), so they never produce a GUI-thread event. The repaint
        # timer samples those readouts at ~2 Hz. The ingest_csi/ingest_position
        # slots are retained as thin, refresh-free pass-throughs for tests and
        # any caller that still routes them.
        @Slot(object)
        def ingest_board_stats(self, stats: dict) -> None:
            self.state.on_board_stats(stats)
            self.refresh()

        @Slot(object)
        def ingest_csi(self, event) -> None:
            self.state.on_csi(event)

        @Slot(object)
        def ingest_position(self, state) -> None:
            self.state.on_position(state)

        @Slot(object)
        def ingest_clap(self, event) -> None:
            self.state.on_clap(event)
            self.refresh()

        # -- render ------------------------------------------------------------
        @Slot()
        def refresh(self) -> None:
            for bid, widgets in self._board_rows.items():
                row = self.state.board_row(bid)
                dot = widgets["dot"]
                dot.setProperty("pf", "ok" if row["present"] else "bad")
                dot.style().unpolish(dot)
                dot.style().polish(dot)
                age = row["age_s"]
                age_txt = "—" if age == float("inf") else f"{age:.1f}s"
                widgets["stat"].setText(f"{row['hz']:5.1f} Hz   age {age_txt}")

            name = self.state.last_clap_name
            if name is None:
                self._clap.setText("CLAPPER — waiting for START")
                self._clap.setProperty("clap", "none")
            else:
                since = self.state.since_start_clap_s()
                tail = f"   +{_fmt_elapsed(since)} since START" if since is not None else ""
                self._clap.setText(f"CLAPPER — {name.upper()}{tail}")
                self._clap.setProperty("clap", name if name in ("start", "stop") else "none")
            self._clap.style().unpolish(self._clap)
            self._clap.style().polish(self._clap)

            cell = self.state.current_cell
            cell_txt = f"({cell[0]:.0f}, {cell[1]:.0f}) cm" if cell is not None else "—"
            self._camera.setText(
                f"Camera: {self.state.detection_pct():.0f}% detected   cell {cell_txt}")

            self._totals.setText(
                f"CSI rows: {self.state.csi_total:,}    "
                f"elapsed {_fmt_elapsed(self.state.elapsed_s())}")
