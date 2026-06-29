"""Empty-room (CSI-only) session mode: controller, pre-flight gate, report, UI.

The empty-room baseline capture (cf. the 50-min stability study) records only
csi.csv + clap.csv: no camera tracker, no camera pre-flight gating, and a
CSI-only validation report instead of the full build_report (which would FAIL
on the deliberately-missing camera.csv).
"""

import os

import pytest

from csi_gui.session import SessionController


# ---------------------------------------------------------------------------
# SessionController: camera_enabled=False never builds/starts the tracker.
# ---------------------------------------------------------------------------

class _FakeCollector:
    def __init__(self):
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def run_forever(self):
        pass

    def stop(self):
        self.stopped = True


def test_controller_skips_tracker_when_camera_disabled(tmp_path):
    tracker_calls = []
    collector = _FakeCollector()
    ctl = SessionController(
        sessions_dir=str(tmp_path),
        collector_factory=lambda name: collector,
        tracker_factory=lambda url, log: tracker_calls.append((url, log)),
        camera_enabled=False,
    )
    path = ctl.start("20260611_0000_empty_room", camera_url="")
    assert os.path.isdir(path)
    assert collector.started
    assert tracker_calls == [], "tracker must never be built in empty-room mode"
    ctl.stop()
    assert collector.stopped
    assert ctl.state == "stopped"


def test_controller_default_still_builds_tracker(tmp_path):
    built = []

    class _FakeTracker(_FakeCollector):
        pass

    ctl = SessionController(
        sessions_dir=str(tmp_path),
        collector_factory=lambda name: _FakeCollector(),
        tracker_factory=lambda url, log: built.append(_FakeTracker()) or built[-1],
    )
    ctl.start("20260611_0001_walk", camera_url="stub://cam")
    assert len(built) == 1
    ctl.stop()


# ---------------------------------------------------------------------------
# Pre-flight: camera check stops gating READY in empty-room mode.
# ---------------------------------------------------------------------------

def test_engine_camera_not_critical_when_not_required():
    from csi_gui.preflight.engine import CAMERA, PreflightEngine
    from csi_gui.preflight.probes import GREEN

    engine = PreflightEngine()
    all_green_but_camera = {c.id: GREEN for c in engine.checks if c.id != CAMERA}

    assert engine.all_critical_green(all_green_but_camera) is False
    engine.camera_required = False
    assert engine.all_critical_green(all_green_but_camera) is True
    assert CAMERA not in engine.critical_ids()
    # And back.
    engine.camera_required = True
    assert engine.all_critical_green(all_green_but_camera) is False


# ---------------------------------------------------------------------------
# CSI-only report: OK on a healthy camera-less session; no camera.csv FAIL.
# ---------------------------------------------------------------------------

def _write_csi_session(session_dir, *, boards=(1, 4, 5), hz=33, secs=10):
    os.makedirs(session_dir, exist_ok=True)
    t0 = 1000.0
    rows = ["wall_time_s,board_id,rssi,channel,rx_seq,esp_timestamp_us"
            + "".join(f",csi_{i}" for i in range(4))]
    seq = 0
    for k in range(int(hz * secs)):
        t = t0 + k / hz
        for b in boards:
            seq += 1
            rows.append(f"{t:.3f},{b},-40,6,{seq},{int(t * 1e6)},1,2,{k % 7},4")
    with open(os.path.join(session_dir, "csi.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(os.path.join(session_dir, "clap.csv"), "w") as f:
        f.write("wall_time_s,event_name,event_counter,esp_timestamp_us\n"
                f"{t0 + 0.5},start,1,1\n{t0 + secs - 0.5},stop,2,2\n")


def test_csi_report_passes_without_camera(tmp_path):
    from csi_gui.csi_report import build_csi_report

    session = tmp_path / "20260611_0002_empty_room"
    _write_csi_session(str(session))
    rep = build_csi_report(str(session))
    labels = [label for _lvl, label, _d in rep.rows]
    assert any("session mode" in lab for lab in labels)
    assert not any("camera" in lab.lower() for lab in labels)
    assert rep.worst() != "FAIL", rep.rows


def test_csi_report_fails_on_missing_csi(tmp_path):
    from csi_gui.csi_report import build_csi_report

    session = tmp_path / "20260611_0003_broken"
    os.makedirs(session)
    rep = build_csi_report(str(session))
    assert rep.worst() == "FAIL"


# ---------------------------------------------------------------------------
# RecordPage wiring (offscreen): mode switch drives gate + name + factory kwarg.
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from csi_gui.app_context import AppContext  # noqa: E402
import csi_gui.ui.pages.record_page as record_page_mod  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_record_page_empty_mode_wiring(qapp, tmp_path):
    captured = {}

    class _CaptureController:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._recording = False

        def start(self, name, camera):
            self._recording = True
            return str(tmp_path / name)

        @property
        def is_recording(self):
            return self._recording

        @property
        def session_path(self):
            return str(tmp_path)

        @property
        def session_name(self):
            return "sess"

        def stop(self):
            self._recording = False

        def validate(self, **kwargs):
            captured["validate_kwargs"] = kwargs

    ctx = AppContext(camera_url="stub://cam")
    ctx.root = str(tmp_path)
    page = record_page_mod.RecordPage(ctx, controller_factory=_CaptureController)

    # Default: walk mode, camera gates READY, camera fields enabled.
    assert page._mode == "walk"
    assert page._preflight._engine.camera_required is True

    page._mode_empty.setChecked(True)
    page._set_mode("empty")
    assert page._preflight._engine.camera_required is False
    assert not page._source_edit.isEnabled()
    assert "empty_room" in page._name_edit.text()

    page.start_session()
    assert captured.get("camera_enabled") is False

    page.stop_session()
    # Review-phase auto-validate must have used the CSI-only report.
    assert "build_report" in captured.get("validate_kwargs", {})

    # Switching back re-arms the camera gate.
    page._set_mode("walk")
    assert page._preflight._engine.camera_required is True
    assert page._source_edit.isEnabled()
    page._preflight.stop()
    page.deleteLater()
