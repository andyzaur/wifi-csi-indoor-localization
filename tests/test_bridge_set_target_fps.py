"""Tests for CameraBridge.set_target_fps (the live preview-fps control).

set_target_fps recomputes the throttle's minimum inter-frame interval so the
user can trade preview smoothness for CPU mid-recording. Detection is unaffected
(it runs full-res on the tracker); only the GUI preview cadence changes.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from csi_gui.adapters.frame_provider import LiveFrameProvider  # noqa: E402
from csi_gui.adapters.signal_bridge import CameraBridge  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class FakeClock:
    def __init__(self, start=0.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += float(dt)


class FakeFrame:
    def __init__(self):
        self.preview_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        self.position = None


def test_set_target_fps_updates_min_interval():
    bridge = CameraBridge(LiveFrameProvider(), target_fps=30.0)
    assert bridge.target_fps == 30.0
    assert bridge._min_interval == pytest.approx(1.0 / 30.0)

    bridge.set_target_fps(10.0)
    assert bridge.target_fps == 10.0
    assert bridge._min_interval == pytest.approx(1.0 / 10.0)

    bridge.set_target_fps(15.0)
    assert bridge._min_interval == pytest.approx(1.0 / 15.0)


def test_set_target_fps_zero_disables_throttle():
    bridge = CameraBridge(LiveFrameProvider(), target_fps=30.0)
    bridge.set_target_fps(0.0)
    assert bridge._min_interval == 0.0


def test_new_target_fps_changes_forwarding_cadence():
    """Lowering the target fps mid-stream forwards fewer frames."""
    clock = FakeClock()
    bridge = CameraBridge(LiveFrameProvider(), target_fps=30.0, clock=clock)

    fired = []
    bridge.frameReady.connect(lambda fid: fired.append(fid))

    # At 30 fps target, a 0.05s gap (20 fps) forwards every frame.
    bridge.on_frame(FakeFrame())          # t=0 -> fwd (#1)
    clock.advance(0.05)
    bridge.on_frame(FakeFrame())          # t=0.05 -> fwd (#2, gap>=1/30)
    QApplication.processEvents()
    assert fired == [1, 2]

    # Drop the target to 10 fps (0.1s interval): the next 0.05s frame is dropped.
    bridge.set_target_fps(10.0)
    clock.advance(0.05)
    bridge.on_frame(FakeFrame())          # t=0.10, only 0.05s since last fwd -> drop
    QApplication.processEvents()
    assert fired == [1, 2]                 # no new forward

    clock.advance(0.06)
    bridge.on_frame(FakeFrame())          # t=0.16, 0.11s since last fwd -> fwd (#3)
    QApplication.processEvents()
    assert fired == [1, 2, 3]


def test_overlay_signals_are_throttled_below_frame_rate():
    """FIX B: positionUpdated + fpsMeasured emit at ~overlay_fps, not per frame.

    frameReady drives the image and stays at the preview cadence; the overlay
    TEXT signals are clamped much lower so they don't flood the GUI event loop.
    """
    clock = FakeClock()
    # target_fps high enough to forward every frame; overlay clamped to 5 Hz.
    bridge = CameraBridge(LiveFrameProvider(), target_fps=1000.0,
                          clock=clock, overlay_fps=5.0)

    frames = []
    positions = []
    fpses = []
    bridge.frameReady.connect(lambda fid: frames.append(fid))
    bridge.positionUpdated.connect(lambda p: positions.append(p))
    bridge.fpsMeasured.connect(lambda f: fpses.append(f))

    # Feed 100 frames over 1.0s of simulated time (100 fps input).
    n = 100
    for _ in range(n):
        bridge.on_frame(FakeFrame())
        clock.advance(0.01)
    QApplication.processEvents()

    # Every frame is forwarded as an image (preview cadence).
    assert len(frames) == n
    # Overlay text emits clamped to ~5 Hz over 1s -> far fewer than n.
    assert len(positions) == len(fpses)
    assert len(positions) <= 8, f"overlay not throttled: {len(positions)} emits"
    assert len(positions) < n
    # And we did get a handful (it's still "live"), not zero.
    assert len(positions) >= 4


def test_overlay_throttle_independent_of_frame_throttle():
    """Even when frames are dropped, overlay emits track the overlay cadence."""
    clock = FakeClock()
    bridge = CameraBridge(LiveFrameProvider(), target_fps=10.0,
                          clock=clock, overlay_fps=2.0)
    positions = []
    bridge.positionUpdated.connect(lambda p: positions.append(p))

    # 2.0s of frames at 10 fps input; overlay clamped to 2 Hz -> ~4-5 emits.
    for _ in range(20):
        bridge.on_frame(FakeFrame())
        clock.advance(0.1)
    QApplication.processEvents()
    assert 3 <= len(positions) <= 6, f"unexpected overlay emit count: {len(positions)}"
