"""PreflightScheduler: the Qt driver that runs checks OFF the GUI thread.

A QTimer fires on a cadence (~every few seconds). On each tick it dispatches each
:class:`~csi_gui.preflight.engine.Check` to a QThreadPool worker so a slow probe
(ping/curl can take seconds) never blocks the UI thread. Each finished worker
emits :pysig:`checkUpdated(check_id, status, detail, hint)` back on the GUI
thread via a queued signal.

An **in-flight guard** keeps a slow check from stacking: a check that is still
running when the next tick arrives is simply skipped that round, so a hung curl
can never pile up dozens of workers.

The scheduler is deliberately thin — all logic lives in the Qt-free engine; this
just schedules it and marshals results back to the GUI thread.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)

from csi_gui.preflight.engine import Check, PreflightEngine

# Default cadence between full re-check sweeps (ms).
DEFAULT_INTERVAL_MS = 2500


class _CheckSignals(QObject):
    """Per-worker signal carrier (QRunnable can't own signals directly)."""

    done = Signal(str, str, str, str)  # check_id, status, detail, hint
    failed = Signal(str, str)          # check_id, error message


class _CheckRunner(QRunnable):
    """Runs ONE check's callable on a pool thread, emits its result."""

    def __init__(self, check: Check) -> None:
        super().__init__()
        self._check = check
        self.signals = _CheckSignals()

    def run(self) -> None:  # executed on a QThreadPool thread
        try:
            result = self._check.run()
        except Exception as exc:  # noqa: BLE001 — never let a probe kill the pool
            self.signals.failed.emit(self._check.id, str(exc))
            return
        self.signals.done.emit(
            self._check.id, result.status, result.detail, result.hint or "")


class PreflightScheduler(QObject):
    """Drives the engine's checks on a timer using a thread pool.

    Emits ``checkUpdated`` (id, status, detail, hint) for the panel to render.
    Call :meth:`start` when the panel is shown and :meth:`stop` on hide/close.
    """

    checkUpdated = Signal(str, str, str, str)

    def __init__(self, engine: PreflightEngine,
                 interval_ms: int = DEFAULT_INTERVAL_MS,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._pool = QThreadPool(self)
        # Bound the pool: at most a handful of probes ever run concurrently.
        self._pool.setMaxThreadCount(max(4, len(engine.checks)))

        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

        # In-flight guard: ids currently being evaluated (skip re-dispatch).
        self._inflight: set[str] = set()
        # Strong refs to live runners, keyed by check id. QThreadPool.start() does
        # NOT keep a Python QRunnable alive, so without this the _CheckRunner (and
        # its _CheckSignals) can be GC'd mid-run -> "Signal source has been deleted".
        # Held until the result is marshalled back (_on_done/_on_failed).
        self._runners: dict[str, _CheckRunner] = {}
        self._running = False

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        """Begin periodic checking and run one sweep immediately."""
        if self._running:
            return
        self._running = True
        self._timer.start()
        self._tick()  # don't wait a full interval for the first results

    def stop(self) -> None:
        """Stop the timer. In-flight workers finish but their signals are dropped.

        Defensive against interpreter/Qt teardown: at shutdown a late hideEvent
        can reach here after the underlying C++ QTimer is already deleted, which
        raises RuntimeError — swallow it (there is nothing left to stop).
        """
        self._running = False
        try:
            self._timer.stop()
        except RuntimeError:
            pass

    @property
    def is_running(self) -> bool:
        return self._running

    def recheck_now(self) -> None:
        """Force an immediate sweep (the 'Recheck all' button)."""
        self._tick()

    # -- dispatch --------------------------------------------------------------
    def _tick(self) -> None:
        for check in self._engine.checks:
            if check.id in self._inflight:
                continue  # still running from a previous tick — don't stack
            self._dispatch(check)

    def _dispatch(self, check: Check) -> None:
        self._inflight.add(check.id)
        runner = _CheckRunner(check)
        self._runners[check.id] = runner  # strong ref until the result returns
        runner.signals.done.connect(
            self._on_done, Qt.ConnectionType.QueuedConnection)
        runner.signals.failed.connect(
            self._on_failed, Qt.ConnectionType.QueuedConnection)
        self._pool.start(runner)

    # -- result marshalling (GUI thread) ---------------------------------------
    def _on_done(self, check_id: str, status: str, detail: str, hint: str) -> None:
        self._inflight.discard(check_id)
        self._runners.pop(check_id, None)  # release the strong ref
        if self._running:
            self.checkUpdated.emit(check_id, status, detail, hint)

    def _on_failed(self, check_id: str, message: str) -> None:
        self._inflight.discard(check_id)
        self._runners.pop(check_id, None)  # release the strong ref
        if self._running:
            from csi_gui.preflight.probes import YELLOW
            self.checkUpdated.emit(
                check_id, YELLOW, f"check error: {message}",
                "Transient error — will retry on the next sweep.")
