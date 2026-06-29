"""ADVERSARIAL VERIFIER test (independent): pause/resume wiring on Start/Stop.

Builds a real RecordPage and drives start_session()/stop_session() (and their
start_tracker()/stop_tracker() aliases), asserting the pre-flight scheduler is
STOPPED + the board listener RELEASED while recording, and both come back on
Stop. The SessionController's backends (ArucoTracker + CsiCollector) are stubbed
at the csi_gui.session seam so no real camera/socket is touched, and the
board-listener socket bind is stubbed so :5500 is never actually opened.
"""

from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

import csi_gui.session as session_mod
import csi_gui.ui.pages.record_page as record_page_mod
from csi_gui.app_context import AppContext


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _no_real_subprocess(monkeypatch):
    # Pre-flight probes must never hit the network/system in this test.
    from csi_gui.preflight import probes
    monkeypatch.setattr(probes, "_run", lambda *a, **k: (0, ""))


@pytest.fixture(autouse=True)
def _fake_board_listener(monkeypatch):
    # Avoid binding the real UDP :5500; track listener on/off via a flag instead.
    import csi_gui.preflight.engine as engine_mod

    class _FakeCollector:
        def __init__(self, *a, **k):
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def rate_hz(self, _bid):
            return 0.0

    monkeypatch.setattr(engine_mod, "CsiCollector", _FakeCollector)


class _StubTracker:
    """Stand-in for ArucoTracker: records start/stop, runs no thread work."""

    instances: list["_StubTracker"] = []

    def __init__(self, *args, **kwargs):
        self.on_log = kwargs.get("on_log")
        self.log = kwargs.get("log")
        self.started = False
        self.stopped = False
        _StubTracker.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def run_forever(self):
        return None


class _StubCollector:
    """Stand-in for CsiCollector inside the SessionController: binds nothing."""

    instances: list["_StubCollector"] = []

    def __init__(self, *args, **kwargs):
        self.started = False
        self.stopped = False
        _StubCollector.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def run_forever(self):
        return None


def _make_page(qapp, monkeypatch):
    # Stub the SessionController's backends at the session seam.
    monkeypatch.setattr(session_mod, "ArucoTracker", _StubTracker)
    monkeypatch.setattr(session_mod, "CsiCollector", _StubCollector)
    ctx = AppContext(camera_url="stub://cam")
    # Keep created session dirs out of the repo's sessions/ during the test.
    ctx.root = tempfile.mkdtemp()
    page = record_page_mod.RecordPage(ctx)
    page.show()  # showEvent -> preflight.start(): scheduler running, listener up
    qapp.processEvents()
    return page


def test_start_pauses_scheduler_and_listener_then_resume_on_stop(qapp, monkeypatch):
    _StubTracker.instances.clear()
    _StubCollector.instances.clear()
    page = _make_page(qapp, monkeypatch)
    panel = page._preflight

    # Baseline: pre-flight live.
    assert panel._scheduler.is_running is True
    assert panel._engine.board_listener_running is True
    assert panel.is_paused is False

    # Start recording -> pause must stop the scheduler AND release the listener.
    page.start_session()
    qapp.processEvents()
    assert page.is_running is True
    assert panel.is_paused is True
    assert panel._scheduler.is_running is False, "scheduler must be stopped while recording"
    assert panel._engine.board_listener_running is False, ":5500 must be released while recording"
    assert _StubTracker.instances[-1].started is True
    assert _StubCollector.instances[-1].started is True

    # Stop recording while visible -> resume brings pre-flight back live.
    page.stop_session()
    qapp.processEvents()
    assert page.is_running is False
    assert panel.is_paused is False
    assert panel._scheduler.is_running is True, "scheduler must resume after Stop"
    assert panel._engine.board_listener_running is True, "listener must rebind after Stop"
    assert _StubTracker.instances[-1].stopped is True
    assert _StubCollector.instances[-1].stopped is True

    page.hide()
    qapp.processEvents()


def test_start_tracker_alias_drives_session(qapp, monkeypatch):
    """The back-compat start_tracker()/stop_tracker() aliases still work."""
    _StubTracker.instances.clear()
    page = _make_page(qapp, monkeypatch)
    panel = page._preflight

    page.start_tracker()
    qapp.processEvents()
    assert page.is_running is True
    assert panel.is_paused is True

    page.stop_tracker()
    qapp.processEvents()
    assert page.is_running is False
    assert panel.is_paused is False

    page.hide()
    qapp.processEvents()


def test_pause_resume_idempotent(qapp, monkeypatch):
    page = _make_page(qapp, monkeypatch)
    panel = page._preflight

    panel.pause()
    panel.pause()  # second pause is a no-op
    assert panel.is_paused is True
    assert panel._scheduler.is_running is False
    assert panel._engine.board_listener_running is False

    panel.resume()
    panel.resume()  # second resume is a no-op
    assert panel.is_paused is False
    assert panel._scheduler.is_running is True
    assert panel._engine.board_listener_running is True

    # resume() with no prior pause is a no-op (does not crash / double-start).
    panel.resume()
    assert panel.is_paused is False

    page.hide()
    qapp.processEvents()


def test_start_while_paused_does_not_revive_sweeps(qapp, monkeypatch):
    # Page navigation mid-recording: a re-show (start()) must not revive sweeps.
    page = _make_page(qapp, monkeypatch)
    panel = page._preflight
    panel.pause()
    assert panel._scheduler.is_running is False

    panel.start()  # simulates showEvent firing again while paused
    assert panel.is_paused is True
    assert panel._scheduler.is_running is False, "re-show must NOT revive the scheduler while paused"
    assert panel._engine.board_listener_running is False

    # A full stop() supersedes pause.
    panel.stop()
    assert panel.is_paused is False
    assert panel._scheduler.is_running is False
    assert panel._engine.board_listener_running is False

    page.hide()
    qapp.processEvents()


def test_high_rate_callbacks_bypass_the_queued_relay(qapp, monkeypatch):
    """FIX B: on_csi/on_position go STRAIGHT to MonitorState, not a GUI signal.

    The per-event relay (which would flood the GUI event loop and starve the
    live preview) must carry only the LOW-rate callbacks. The high-rate ones are
    bound directly to the thread-safe MonitorState the repaint timer samples.
    """
    captured = {}

    class _CaptureController:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._recording = False

        def start(self, name, camera):
            self._recording = True
            return "/tmp/whatever"

        @property
        def is_recording(self):
            return self._recording

        @property
        def session_path(self):
            return "/tmp/whatever"

        @property
        def session_name(self):
            return "sess"

        def stop(self):
            self._recording = False

        def validate(self):
            pass

    ctx = AppContext(camera_url="stub://cam")
    ctx.root = tempfile.mkdtemp()
    page = record_page_mod.RecordPage(ctx, controller_factory=_CaptureController)
    page.show()
    qapp.processEvents()

    # The relay must NOT carry the high-rate signals anymore.
    assert not hasattr(page._relay, "csi"), "relay must not relay per-packet CSI"
    assert not hasattr(page._relay, "position"), "relay must not relay per-frame position"

    page.start_session()
    qapp.processEvents()

    # on_csi / on_position are bound directly to the (fresh) monitor state.
    state = page._monitor.state
    assert captured["on_csi"] == state.on_csi
    assert captured["on_position"] == state.on_position
    # Low-rate ones still go through the queued relay.
    assert captured["on_board_stats"] == page._relay.boardStats.emit
    assert captured["on_clap"] == page._relay.clap.emit

    # Driving the bound high-rate callbacks updates the state, no GUI signal.
    before = state.csi_total
    for _ in range(50):
        captured["on_csi"]()
    assert state.csi_total == before + 50

    page.stop_session()
    qapp.processEvents()
    page.hide()
    qapp.processEvents()


def test_stop_tracker_when_hidden_fully_stops_panel(qapp, monkeypatch):
    # The aboutToQuit path: page hidden -> stop_tracker() must STOP (release :5500),
    # not resume.
    _StubTracker.instances.clear()
    page = _make_page(qapp, monkeypatch)
    panel = page._preflight
    page.start_session()
    qapp.processEvents()
    assert panel._engine.board_listener_running is False

    page.hide()  # hideEvent -> panel.stop(); listener already down
    qapp.processEvents()
    page.stop_tracker()  # recording -> stop_session() resumes? No: hidden -> panel.stop()
    qapp.processEvents()
    assert panel.is_paused is False
    assert panel._scheduler.is_running is False
    assert panel._engine.board_listener_running is False
