"""Offscreen smoke tests for the PreflightPanel + RecordPage construction.

Runs with the Qt 'offscreen' platform plugin — no display needed. We patch the
board-listener CsiCollector so no real UDP socket is bound, and patch the
probes' subprocess so no real network command runs. We then assert the panel
constructs, renders a pushed check result, computes the READY gate, and toggles
the static-IP revert bar with the sentinel.
"""

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from csi_gui.app_context import AppContext  # noqa: E402
from csi_gui.preflight import engine as pf_engine  # noqa: E402
from csi_gui.preflight import netconfig  # noqa: E402
from csi_gui.preflight.probes import GREEN, RED  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _NoSocketCollector:
    """CsiCollector stand-in that binds nothing and reports steady rates."""

    def __init__(self, **kwargs):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def rate_hz(self, board_id):
        return 30.0


@pytest.fixture(autouse=True)
def _no_real_socket(monkeypatch):
    monkeypatch.setattr(pf_engine, "CsiCollector",
                        lambda **kw: _NoSocketCollector())
    yield


@pytest.fixture(autouse=True)
def _no_real_subprocess(monkeypatch):
    """Stub probes._run so the scheduler's sweeps never spawn ping/curl/etc.

    The scheduler's start() runs an immediate sweep on a thread pool; without
    this it would launch real ping/curl/networksetup subprocesses (and the
    workers could outlive a deleted panel). A fast stub keeps the suite hermetic.
    """
    from csi_gui.preflight import probes

    monkeypatch.setattr(probes, "_run", lambda argv, timeout=None: (-2, "stub"))
    yield


@pytest.fixture(autouse=True)
def _temp_sentinel(monkeypatch, tmp_path):
    monkeypatch.setattr(netconfig, "SENTINEL",
                        str(tmp_path / ".csi_static_ip_active"))
    yield


def _make_panel(context=None):
    from csi_gui.ui.preflight_panel import PreflightPanel
    return PreflightPanel(
        context or AppContext(camera_url="http://127.0.0.1:8080/video"))


def _drain(panel):
    """Wait for any scheduler thread-pool workers to finish before deletion.

    A worker still in flight when the panel is GC'd would emit through a deleted
    C++ signal object ("Signal source has been deleted"). Draining the pool keeps
    the test output clean.
    """
    QApplication.processEvents()
    panel._scheduler._pool.waitForDone(2000)
    QApplication.processEvents()


def test_panel_constructs_with_one_row_per_check():
    panel = _make_panel()
    # One row per engine check.
    assert set(panel._rows) == {c.id for c in panel._engine.checks}
    # Not ready before any check has run.
    assert panel.ready is False
    panel.deleteLater()


def test_panel_ready_gate_turns_green_when_all_critical_green():
    panel = _make_panel()
    # Push GREEN for every critical check, RED for a non-critical one.
    for c in panel._engine.checks:
        status = GREEN if c.critical else RED
        panel._on_check_updated(c.id, status, "ok", "")
    assert panel.ready is True
    assert "READY TO RECORD" in panel._ready.text()
    panel.deleteLater()


def test_panel_not_ready_when_a_critical_is_red():
    panel = _make_panel()
    for c in panel._engine.checks:
        panel._on_check_updated(c.id, GREEN, "ok", "")
    # Now flip one critical check to RED.
    crit = panel._engine.critical_ids()[0]
    panel._on_check_updated(crit, RED, "bad", "fix it")
    assert panel.ready is False
    assert "NOT READY" in panel._ready.text()
    panel.deleteLater()


def test_static_bar_follows_sentinel():
    panel = _make_panel()
    # isVisibleTo(parent) reflects the local visibility flag without needing the
    # top-level window to actually be shown on a display.
    assert panel._static_bar.isVisibleTo(panel) is False
    open(netconfig.SENTINEL, "w").close()
    panel._refresh_static_bar()
    assert panel._static_bar.isVisibleTo(panel) is True
    os.remove(netconfig.SENTINEL)
    panel._refresh_static_bar()
    assert panel._static_bar.isVisibleTo(panel) is False
    panel.deleteLater()


def test_panel_start_stop_drives_board_listener():
    panel = _make_panel()
    panel.start()
    assert panel._engine.board_listener_running is True
    panel.stop()
    assert panel._engine.board_listener_running is False
    _drain(panel)
    panel.deleteLater()


def test_panel_pause_stops_scheduler_and_listener():
    panel = _make_panel()
    panel.start()
    assert panel._engine.board_listener_running is True
    assert panel._scheduler.is_running is True

    panel.pause()
    assert panel.is_paused is True
    # Scheduler + board listener are stopped (frees the subprocess sweeps + :5500).
    assert panel._scheduler.is_running is False
    assert panel._engine.board_listener_running is False
    # Clear paused state on the banner.
    assert "paused" in panel._ready.text().lower()
    _drain(panel)
    panel.deleteLater()


def test_panel_pause_resume_is_idempotent():
    panel = _make_panel()
    panel.start()
    panel.pause()
    panel.pause()  # second pause is a no-op
    assert panel.is_paused is True

    panel.resume()
    assert panel.is_paused is False
    # Listener + scheduler are back.
    assert panel._engine.board_listener_running is True
    assert panel._scheduler.is_running is True
    panel.resume()  # second resume is a no-op
    assert panel.is_paused is False
    panel.stop()
    _drain(panel)
    panel.deleteLater()


def test_panel_resume_without_pause_is_safe():
    panel = _make_panel()
    # resume() before any pause() must not raise / flip state.
    panel.resume()
    assert panel.is_paused is False
    panel.deleteLater()


def test_panel_paused_ignores_late_check_results():
    panel = _make_panel()
    panel.start()
    panel.pause()
    # A late scheduler result arriving after pause must not overwrite the row.
    crit = panel._engine.critical_ids()[0]
    panel._on_check_updated(crit, GREEN, "ok", "")
    assert panel._statuses.get(crit) != GREEN  # dropped while paused
    assert "paused" in panel._ready.text().lower()
    _drain(panel)
    panel.deleteLater()


def test_panel_start_while_paused_does_not_revive_scheduler():
    panel = _make_panel()
    panel.start()
    panel.pause()
    # Re-showing the page (showEvent -> start) must NOT revive the sweeps.
    panel.start()
    assert panel._scheduler.is_running is False
    assert panel._engine.board_listener_running is False
    assert panel.is_paused is True
    _drain(panel)
    panel.deleteLater()


def test_record_page_constructs_with_preflight(monkeypatch):
    # Stub the SessionController backends so constructing RecordPage never touches
    # a camera/socket. RecordPage only *imports* them lazily via csi_gui.session.
    import csi_gui.session as session_mod
    from csi_gui.ui.pages import record_page as rp

    class _FakeTracker:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(session_mod, "ArucoTracker", _FakeTracker)
    page = rp.RecordPage(AppContext(camera_url="http://127.0.0.1:8080/video"))
    # The pre-flight panel is embedded.
    assert page._preflight is not None
    # Toggling the collapsible section does not raise.
    page._preflight_toggle.setChecked(False)
    page._toggle_preflight()
    assert page._preflight.isVisible() is False
    page.deleteLater()


class _FakeTracker:
    """ArucoTracker stand-in: records on_log + start/stop without a camera."""

    def __init__(self, *a, on_log=None, **k):
        self.on_log = on_log
        self.log = k.get("log")
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def run_forever(self):  # daemon worker target — return immediately
        pass

    def stop(self):
        self.stopped = True


class _FakeCollector:
    """CsiCollector stand-in inside the SessionController: binds nothing."""

    def __init__(self, *a, on_log=None, **k):
        self.on_log = on_log
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def run_forever(self):
        pass

    def stop(self):
        self.stopped = True


def test_record_page_start_pauses_preflight_and_passes_logger(monkeypatch, tmp_path):
    import csi_gui.session as session_mod
    from csi_gui.ui.pages import record_page as rp

    captured = {}

    def fake_tracker(*a, on_log=None, **k):
        t = _FakeTracker(on_log=on_log, **k)
        captured["tracker"] = t
        return t

    def fake_collector(*a, on_log=None, **k):
        c = _FakeCollector(on_log=on_log, **k)
        captured["collector"] = c
        return c

    monkeypatch.setattr(session_mod, "ArucoTracker", fake_tracker)
    monkeypatch.setattr(session_mod, "CsiCollector", fake_collector)

    logs = []
    ctx = AppContext(camera_url="http://127.0.0.1:8080/video",
                     logger=logs.append)
    # Keep session dirs out of the repo's sessions/ during the test.
    ctx.root = str(tmp_path)
    page = rp.RecordPage(ctx)
    page._preflight.start()  # listener + scheduler up before "recording"

    page.start_session()
    # Pre-flight is paused (the fps fix) and both backends got the file logger.
    assert page._preflight.is_paused is True
    assert page._preflight._engine.board_listener_running is False
    assert captured["tracker"].on_log is ctx.logger
    assert captured["collector"].on_log is ctx.logger
    assert any("Start session" in line for line in logs)

    # Stop while hidden (the page is never shown in offscreen tests) -> stop()
    # path; the listener stays released and the paused flag clears.
    page.stop_session()
    assert page._preflight.is_paused is False
    assert any(line.startswith("Stop session") for line in logs)
    _drain(page._preflight)
    page.deleteLater()
