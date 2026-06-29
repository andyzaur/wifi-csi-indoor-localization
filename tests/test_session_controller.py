"""Tests for csi_gui.session.SessionController (Qt-free core).

The controller owns two backends (CsiCollector + ArucoTracker), each on its own
daemon thread. We inject FAKES via collector_factory / tracker_factory so no
real UDP :5500 socket is bound and no real camera is opened:

  * start() creates sessions/<name>/, constructs + start()s both backends, and
    spawns a run_forever daemon thread for each; state -> RECORDING.
  * stop() stop()s + joins both; state -> STOPPED.
  * validate() runs an injected build_report and moves to VALIDATED.
"""

import os
import threading

import pytest

from csi_gui.session import (
    SessionController,
    IDLE,
    RECORDING,
    STOPPED,
    VALIDATED,
    DEFAULT_OFFSETS,
)


class FakeCollector:
    def __init__(self, name):
        self.name = name
        self.started = False
        self.stopped = False
        self.ran = threading.Event()

    def start(self):
        self.started = True

    def run_forever(self):
        self.ran.set()  # returns immediately; the daemon thread joins fast

    def stop(self):
        self.stopped = True


class FakeTracker:
    def __init__(self, camera_url, camera_log):
        self.camera_url = camera_url
        self.camera_log = camera_log
        self.started = False
        self.stopped = False
        self.ran = threading.Event()

    def start(self):
        self.started = True

    def run_forever(self):
        self.ran.set()

    def stop(self):
        self.stopped = True


def _make_controller(tmp_path, **kw):
    made = {}

    def collector_factory(name):
        c = FakeCollector(name)
        made["collector"] = c
        return c

    def tracker_factory(camera_url, camera_log):
        t = FakeTracker(camera_url, camera_log)
        made["tracker"] = t
        return t

    ctrl = SessionController(
        sessions_dir=str(tmp_path / "sessions"),
        collector_factory=collector_factory,
        tracker_factory=tracker_factory,
        **kw)
    return ctrl, made


def test_initial_state_is_idle(tmp_path):
    ctrl, _ = _make_controller(tmp_path)
    assert ctrl.state == IDLE
    assert ctrl.is_recording is False
    assert ctrl.session_path is None


def test_start_creates_dir_and_constructs_backends(tmp_path):
    states = []
    ctrl, made = _make_controller(tmp_path, on_state=states.append)
    path = ctrl.start("20260606_01_walk", "http://cam/video")

    # Directory created under sessions/.
    assert os.path.isdir(path)
    assert path.endswith(os.path.join("sessions", "20260606_01_walk"))

    # Both backends constructed + started.
    assert made["collector"].name == "20260606_01_walk"
    assert made["collector"].started is True
    assert made["tracker"].camera_url == "http://cam/video"
    # Tracker logs camera.csv inside the session dir.
    assert made["tracker"].camera_log == os.path.join(path, "camera.csv")
    assert made["tracker"].started is True

    # Both run_forever daemon threads were spawned (and returned immediately).
    assert made["collector"].ran.wait(2.0) is True
    assert made["tracker"].ran.wait(2.0) is True

    assert ctrl.state == RECORDING
    assert ctrl.is_recording is True
    assert states[-1] == RECORDING


def test_start_requires_name(tmp_path):
    ctrl, _ = _make_controller(tmp_path)
    with pytest.raises(ValueError):
        ctrl.start("", "http://cam/video")


def test_double_start_raises(tmp_path):
    ctrl, _ = _make_controller(tmp_path)
    ctrl.start("20260606_01_a", "http://cam/video")
    with pytest.raises(RuntimeError):
        ctrl.start("20260606_02_b", "http://cam/video")


def test_stop_joins_both_backends_and_sets_state(tmp_path):
    states = []
    ctrl, made = _make_controller(tmp_path, on_state=states.append)
    ctrl.start("20260606_01_walk", "http://cam/video")
    ctrl.stop()

    assert made["collector"].stopped is True
    assert made["tracker"].stopped is True
    # Threads joined (no longer alive).
    assert ctrl._collector_thread.is_alive() is False
    assert ctrl._tracker_thread.is_alive() is False
    assert ctrl.state == STOPPED
    assert states[-1] == STOPPED


def test_stop_is_idempotent_no_op_before_start(tmp_path):
    ctrl, _ = _make_controller(tmp_path)
    # Stop with nothing running is a safe no-op (state stays IDLE).
    ctrl.stop()
    assert ctrl.state == IDLE


def test_validate_blocking_runs_build_report_and_sets_state(tmp_path):
    ctrl, _ = _make_controller(tmp_path)
    ctrl.start("20260606_01_walk", "http://cam/video")
    ctrl.stop()

    captured = {}

    def fake_build_report(session_dir):
        captured["dir"] = session_dir
        return "REPORT"

    got = []
    ctrl.on_validated = got.append
    report = ctrl.validate(build_report=fake_build_report, blocking=True)

    assert report == "REPORT"
    assert captured["dir"] == ctrl.session_path
    assert got == ["REPORT"]
    assert ctrl.state == VALIDATED


def test_validate_offthread_delivers_via_callback(tmp_path):
    ctrl, _ = _make_controller(tmp_path)
    ctrl.start("20260606_01_walk", "http://cam/video")
    ctrl.stop()

    done = threading.Event()
    got = []

    def on_validated(rep):
        got.append(rep)
        done.set()

    ctrl.on_validated = on_validated
    ctrl.validate(build_report=lambda d: {"dir": d}, blocking=False)
    assert done.wait(2.0) is True
    assert got and got[0] == {"dir": ctrl.session_path}
    assert ctrl.state == VALIDATED


def test_default_offsets_match_aruco_defaults():
    assert DEFAULT_OFFSETS == (20.0, -15.0)


def test_default_collector_is_built_quiet(tmp_path, monkeypatch):
    """FIX A: the session CsiCollector must be constructed with quiet=True.

    Per-packet logging (~100 lines/s -> a 1.1 MB log + wasted IO/CPU) is
    suppressed; the on_csi/on_clap/on_board_stats callbacks still fire so the
    live monitor keeps working.
    """
    import csi_gui.session as session_mod

    captured = {}

    class FakeCsiCollector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # Patch the symbol the _default_collector factory references.
    monkeypatch.setattr(session_mod, "CsiCollector", FakeCsiCollector)

    ctrl = SessionController(sessions_dir=str(tmp_path / "sessions"))
    # Drive the REAL default factory (not an injected fake).
    ctrl._default_collector("20260606_01_walk")

    assert captured.get("quiet") is True
    # The monitor-feeding callbacks are still wired (nothing silenced but logs).
    assert "on_csi" in captured
    assert "on_clap" in captured
    assert "on_board_stats" in captured
