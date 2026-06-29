"""Offscreen tests for the Phase-3 GUI live-video shell.

These run with the Qt 'offscreen' platform plugin — no window/display needed.
They cover the two thread-boundary adapters:

  * LiveFrameProvider: set_frame -> requestImage round-trip, size + pixel fidelity.
  * CameraBridge: the steady time-based throttle, the rolling-window fps
    measurement, and latest-frame-only delivery.
"""

import os

import numpy as np
import pytest

# PySide6 may be absent in a minimal env; skip the whole module if so.
pytest.importorskip("PySide6")

# MUST be set before any QtGui import / QApplication construction.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from csi_gui.adapters.frame_provider import LiveFrameProvider  # noqa: E402
from csi_gui.adapters.signal_bridge import CameraBridge  # noqa: E402


# ---------------------------------------------------------------------------
# A single process-wide QApplication (Qt allows only one).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Lightweight stand-ins for the real aruco_track dataclasses. The bridge only
# reads .preview_bgr and .position.fps, so a SimpleNamespace-style fake is fine.
# ---------------------------------------------------------------------------
class FakePosition:
    def __init__(self, fps):
        self.fps = fps
        self.detected = True
        self.x_cm = self.y_cm = self.grid_x_cm = self.grid_y_cm = 0.0
        self.n_markers = 1
        self.method = "fake"
        self.right_center = self.left_center = None


class FakeFrame:
    def __init__(self, preview_bgr, fps=0.0):
        self.frame_num = 0
        self.timestamp_s = 0.0
        self.preview_bgr = preview_bgr
        self.position = FakePosition(fps)


class FakeClock:
    """A deterministic monotonic clock the tests can advance by hand."""

    def __init__(self, start: float = 0.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += float(dt)


# ---------------------------------------------------------------------------
# (a) LiveFrameProvider: set_frame / requestImage round-trip.
# ---------------------------------------------------------------------------
def test_provider_roundtrip_size_and_pixel():
    provider = LiveFrameProvider()

    # A 4x3 RGB image with a known top-left pixel.
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[0, 0] = (10, 20, 30)  # R, G, B
    rgb[2, 3] = (200, 100, 50)

    provider.set_frame(rgb)

    img = provider.requestImage("frame/1", None, None)
    assert isinstance(img, QImage)
    assert img.width() == 4
    assert img.height() == 3

    px = img.pixelColor(0, 0)
    assert (px.red(), px.green(), px.blue()) == (10, 20, 30)
    px2 = img.pixelColor(3, 2)
    assert (px2.red(), px2.green(), px2.blue()) == (200, 100, 50)


def test_provider_placeholder_before_first_frame():
    provider = LiveFrameProvider()
    img = provider.requestImage("frame/0", None, None)
    assert isinstance(img, QImage)
    assert not img.isNull()
    assert img.width() > 0 and img.height() > 0


def test_provider_set_frame_copies_buffer():
    """Mutating the source ndarray after set_frame must not change the stored
    QImage — set_frame copies onto Qt-owned storage."""
    provider = LiveFrameProvider()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = (5, 6, 7)
    provider.set_frame(rgb)

    rgb[0, 0] = (250, 250, 250)  # clobber the source after handing it off

    img = provider.requestImage("frame/1", None, None)
    px = img.pixelColor(0, 0)
    assert (px.red(), px.green(), px.blue()) == (5, 6, 7)


# ---------------------------------------------------------------------------
# (b) CameraBridge: steady time-based throttle + fps measurement + latest-only.
#
# The throttle no longer derives a stride from the (broken) PositionState.fps.
# It forwards a frame only when >= 1/target_fps seconds of wall-clock have
# elapsed since the last forwarded frame. We inject a deterministic monotonic
# clock so the cadence is exactly controllable.
# ---------------------------------------------------------------------------
def test_bridge_time_throttle_forwards_on_target_cadence():
    """At 60 input fps with target 20 fps, ~1 of every 3 frames is forwarded."""
    clock = FakeClock()
    provider = LiveFrameProvider()
    bridge = CameraBridge(provider, target_fps=20.0, clock=clock)

    fired = []
    bridge.frameReady.connect(lambda fid: fired.append(fid))

    # 60 frames spaced 1/60 s apart (input ~60 fps). min_interval = 1/20 = 0.05s.
    for i in range(60):
        bgr = np.full((2, 2, 3), i % 256, dtype=np.uint8)
        bridge.on_frame(FakeFrame(bgr))
        clock.advance(1.0 / 60.0)

    QApplication.processEvents()

    # First frame forwards immediately; thereafter one per 0.05s window over a
    # ~0.983s total span -> ~20 forwarded. Allow a small off-by-one slack.
    assert 19 <= len(fired) <= 21, f"expected ~20 forwarded, got {len(fired)}"
    # Ids are monotonic starting at 1.
    assert fired == list(range(1, len(fired) + 1))


def test_bridge_first_frame_forwards_immediately():
    clock = FakeClock()
    provider = LiveFrameProvider()
    bridge = CameraBridge(provider, target_fps=20.0, clock=clock)

    fired = []
    bridge.frameReady.connect(lambda fid: fired.append(fid))

    bridge.on_frame(FakeFrame(np.zeros((2, 2, 3), dtype=np.uint8)))
    QApplication.processEvents()
    assert fired == [1]


def test_bridge_throttles_frames_within_one_interval():
    """Frames arriving faster than the target interval are dropped (not forwarded)."""
    clock = FakeClock()
    provider = LiveFrameProvider()
    bridge = CameraBridge(provider, target_fps=20.0, clock=clock)  # 0.05s interval

    fired = []
    bridge.frameReady.connect(lambda fid: fired.append(fid))

    bridge.on_frame(FakeFrame(np.zeros((2, 2, 3), dtype=np.uint8)))  # t=0 -> fwd
    clock.advance(0.01)
    bridge.on_frame(FakeFrame(np.zeros((2, 2, 3), dtype=np.uint8)))  # t=0.01 drop
    clock.advance(0.01)
    bridge.on_frame(FakeFrame(np.zeros((2, 2, 3), dtype=np.uint8)))  # t=0.02 drop
    clock.advance(0.04)
    bridge.on_frame(FakeFrame(np.zeros((2, 2, 3), dtype=np.uint8)))  # t=0.06 -> fwd

    QApplication.processEvents()
    assert fired == [1, 2]


def test_bridge_latest_only_and_bgr_to_rgb():
    clock = FakeClock()
    provider = LiveFrameProvider()
    bridge = CameraBridge(provider, target_fps=20.0, clock=clock)

    last_fwd_index = None
    for i in range(60):
        # BGR frame: B=i, G=0, R=255-i, so RGB top-left = (255-i, 0, i).
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[..., 0] = i          # B
        bgr[..., 2] = 255 - i    # R
        # Forward iff >= 0.05s since the last forward. Track the last forwarded i.
        forwarded = (i == 0) or (clock.now - (last_fwd_t if last_fwd_index is not None else -1.0)) >= 0.05
        if forwarded:
            last_fwd_index = i
            last_fwd_t = clock.now
        bridge.on_frame(FakeFrame(bgr))
        clock.advance(1.0 / 60.0)

    img = provider.requestImage("frame/x", None, None)
    px = img.pixelColor(0, 0)
    # BGR->RGB conversion must have flipped channels for the LAST forwarded frame.
    assert (px.red(), px.green(), px.blue()) == (255 - last_fwd_index, 0, last_fwd_index)


def test_bridge_handles_none_preview_without_exception():
    """A forwarded frame with no preview must still emit (position-only) and not raise."""
    clock = FakeClock()
    provider = LiveFrameProvider()
    bridge = CameraBridge(provider, target_fps=20.0, clock=clock)

    fired = []
    bridge.frameReady.connect(lambda fid: fired.append(fid))

    bridge.on_frame(FakeFrame(None))  # first frame -> forwarded, preview None
    QApplication.processEvents()
    assert fired == [1]


def test_bridge_measures_fps_over_rolling_window():
    """measured_fps reflects the TRUE arrival rate, independent of the throttle.

    Arrivals are recorded for EVERY frame (even throttled-away ones), so a 50 fps
    stream measures ~50 fps even though only ~20 fps is forwarded.
    """
    clock = FakeClock()
    provider = LiveFrameProvider()
    bridge = CameraBridge(provider, target_fps=20.0, fps_window=30, clock=clock)

    # Fewer than 2 arrivals -> 0.0 (no estimate yet).
    assert bridge.measured_fps() == 0.0
    bridge.on_frame(FakeFrame(np.zeros((2, 2, 3), dtype=np.uint8)))
    assert bridge.measured_fps() == 0.0

    # 50 frames spaced exactly 0.02s apart -> 50 fps.
    for _ in range(49):
        clock.advance(0.02)
        bridge.on_frame(FakeFrame(np.zeros((2, 2, 3), dtype=np.uint8)))

    assert abs(bridge.measured_fps() - 50.0) < 0.5


def test_bridge_emits_measured_fps_on_forwarded_frames():
    """fpsMeasured fires alongside each forwarded frame with the rolling estimate."""
    clock = FakeClock()
    provider = LiveFrameProvider()
    bridge = CameraBridge(provider, target_fps=20.0, clock=clock)

    fps_values = []
    bridge.fpsMeasured.connect(lambda f: fps_values.append(f))

    # 25 fps stream (0.04s spacing); target 20 fps -> some throttling.
    for _ in range(30):
        bridge.on_frame(FakeFrame(np.zeros((2, 2, 3), dtype=np.uint8)))
        clock.advance(0.04)

    QApplication.processEvents()
    assert fps_values, "fpsMeasured should fire at least once"
    # The last emitted value should be near the true 25 fps arrival rate.
    assert abs(fps_values[-1] - 25.0) < 1.0


# ---------------------------------------------------------------------------
# (c) Shell smoke: MainWindow builds with 4 sidebar rows, switching changes the
#     stacked page, and every page constructs without error. Offscreen.
# ---------------------------------------------------------------------------
from csi_gui.app_context import AppContext  # noqa: E402
from csi_gui.ui.main_window import MainWindow, _SECTIONS  # noqa: E402


def test_shell_has_four_sidebar_rows_and_pages():
    win = MainWindow(AppContext())
    assert win._sidebar.count() == 4
    assert win._stack.count() == 4
    assert [win._sidebar.item(i).text() for i in range(4)] == list(_SECTIONS)
    assert "Train" not in _SECTIONS


def test_shell_switching_changes_stacked_page():
    win = MainWindow(AppContext())
    for row in range(win._sidebar.count()):
        win._sidebar.setCurrentRow(row)
        assert win._stack.currentIndex() == row
        # The current page is the one registered at that stack index.
        assert win._stack.currentWidget() is win._pages[row]


def test_shell_record_page_tracker_not_running_at_construction():
    """The lifecycle change: no ArucoTracker is created until Start is pressed."""
    win = MainWindow(AppContext())
    assert win.record_page.is_running is False
    # stop_tracker must be a safe no-op when nothing is running (shutdown path).
    win.stop_tracker()
    assert win.record_page.is_running is False


def test_shell_shares_camera_url_via_context():
    """Calibrate and Record read the same shared camera URL from the context."""
    ctx = AppContext(camera_url="http://example/video")
    win = MainWindow(ctx)
    # Mutating the context and re-syncing reflects on the record page field.
    ctx.camera_url = "http://changed/video"
    win.record_page.sync_from_context()
    assert win.record_page._source_edit.text() == "http://changed/video"
