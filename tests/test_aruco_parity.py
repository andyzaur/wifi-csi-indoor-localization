"""Byte-identical parity: ArucoTracker (default knobs) vs the frozen original
aruco_track.main() loop, over a deterministic synthetic --video fixture.

A generated DICT_4X4_50 fixture video makes detection deterministic (same input
bytes -> same corners), so the two code paths must produce byte-identical
stdout + camera.csv + corners.csv. The only clock-dependent output lines (the
0.5s "You:" throttle and the 2s "[camera] fps" counter) are made deterministic
by monkeypatching time.time to a fixed counter in BOTH runs.
"""

import csv as _csv
import os

import numpy as np
import cv2
from cv2 import aruco
import pytest

import aruco_track
from tests import _aruco_reference


# ─── Deterministic synthetic fixture ─────────────────────────────────────────

def _make_fixture_video(path, n_frames=30, w=900, h=700):
    """Render two DICT_4X4_50 foot markers (ID 0 right, ID 9 left) drifting
    across a white background; write an .mp4. Returns the file path. Marker
    positions vary per frame so the foot-fusion / orientation-EMA / ramp logic
    is actually exercised, and a few frames drop the left marker to hit the
    one-foot / re-acquire branches."""
    d = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    size = 110
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
    assert writer.isOpened(), "VideoWriter failed to open"
    marker_imgs = {mid: cv2.cvtColor(aruco.generateImageMarker(d, mid, size),
                                     cv2.COLOR_GRAY2BGR)
                   for mid in (0, 9)}
    for f in range(n_frames):
        frame = np.full((h, w, 3), 255, np.uint8)
        # Right foot (ID 0): drifts right + down.
        rx = 230 + 4 * f
        ry = 360 + 2 * f
        frame[ry:ry + size, rx:rx + size] = marker_imgs[0]
        # Left foot (ID 9): present except for a short dropout (frames 12-15),
        # which forces a re-acquire after a gap.
        if not (12 <= f < 16):
            lx = 470 + 3 * f
            ly = 380 + 1 * f
            frame[ly:ly + size, lx:lx + size] = marker_imgs[9]
        writer.write(frame)
    writer.release()
    return path


class _FakeClock:
    """Monotonic fake time.time(): advances a fixed step per call so the 0.5s /
    2s output throttles fire deterministically and identically in both runs."""

    def __init__(self, step=0.05, start=1000.0):
        self.t = start
        self.step = step

    def __call__(self):
        v = self.t
        self.t += self.step
        return v


def _read_lines(path):
    with open(path, "r", newline="") as f:
        return f.read()


@pytest.fixture
def fixture_video(tmp_path):
    return _make_fixture_video(str(tmp_path / "fixture.mp4"))


def test_parity_csv_and_corners_byte_identical(tmp_path, fixture_video, monkeypatch):
    # Golden: original loop.
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    gold_log = str(gold_dir / "camera.csv")
    monkeypatch.setattr(aruco_track.time, "time", _FakeClock())
    monkeypatch.setattr(_aruco_reference.time, "time", _FakeClock())
    gold_out = _aruco_reference.run_reference(fixture_video, gold_log,
                                              no_display=True)

    # New: ArucoTracker with default knobs (display off to match no_display
    # golden; owns_window irrelevant when display is off). on_log collects the
    # stdout lines instead of printing.
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    new_log = str(new_dir / "camera.csv")
    monkeypatch.setattr(aruco_track.time, "time", _FakeClock())
    new_out = []
    tracker = aruco_track.ArucoTracker(
        video=fixture_video, log=new_log, display=False, owns_window=True,
        on_log=new_out.append)
    tracker.run_forever()

    # Byte-identical CSVs.
    assert _read_lines(new_log) == _read_lines(gold_log)
    assert _read_lines(str(new_dir / "corners.csv")) == \
        _read_lines(str(gold_dir / "corners.csv"))

    # The new run's log paths differ by design; normalize the one line that
    # embeds the absolute log path, then assert byte-identical stdout.
    def _norm(lines):
        return ["Position log saved to <LOG>" if s.startswith("Position log saved to")
                else s for s in lines]
    assert _norm(new_out) == _norm(gold_out)

    # Sanity: the fixture actually produced detections (non-trivial golden).
    with open(gold_log) as f:
        rows = list(_csv.DictReader(f))
    assert any(r["detected"] == "1" for r in rows)
    assert len(rows) == 30


def test_parity_display_on_owns_window_headless_monkeypatch(tmp_path, fixture_video, monkeypatch):
    """display=True must still produce byte-identical CSV/stdout vs the original
    show-path golden. cv2 GUI calls are stubbed so the test stays headless."""
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)

    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    gold_log = str(gold_dir / "camera.csv")
    monkeypatch.setattr(aruco_track.time, "time", _FakeClock())
    monkeypatch.setattr(_aruco_reference.time, "time", _FakeClock())
    gold_out = _aruco_reference.run_reference(fixture_video, gold_log,
                                              no_display=False)

    new_dir = tmp_path / "new"
    new_dir.mkdir()
    new_log = str(new_dir / "camera.csv")
    monkeypatch.setattr(aruco_track.time, "time", _FakeClock())
    new_out = []
    tracker = aruco_track.ArucoTracker(
        video=fixture_video, log=new_log, display=True, owns_window=True,
        on_log=new_out.append)
    tracker.run_forever()

    assert _read_lines(new_log) == _read_lines(gold_log)
    assert _read_lines(str(new_dir / "corners.csv")) == \
        _read_lines(str(gold_dir / "corners.csv"))

    def _norm(lines):
        return ["Position log saved to <LOG>" if s.startswith("Position log saved to")
                else s for s in lines]
    assert _norm(new_out) == _norm(gold_out)


def test_undistort_corners_raises(tmp_path):
    with pytest.raises(NotImplementedError):
        aruco_track.ArucoTracker(video="x.mp4", undistort="corners")


def test_constructor_raises_on_marker_collision(tmp_path):
    # A wearable marker ID that collides with a floor anchor must raise ValueError
    # (not sys.exit). Derive a real anchor ID from the loaded calibration so this
    # stays valid regardless of which anchor IDs the current floor_calibration.json
    # uses (it is a mutable, recalibrated-per-room data file).
    ok = aruco_track.ArucoTracker(video="x.mp4")
    anchor_ids = sorted(ok.floor_marker_ids)
    assert anchor_ids, "floor_calibration.json has no anchor markers to test against"
    with pytest.raises(ValueError):
        aruco_track.ArucoTracker(video="x.mp4", marker_right=anchor_ids[0])


def test_constructor_raises_on_both_disabled():
    with pytest.raises(ValueError):
        aruco_track.ArucoTracker(video="x.mp4", marker_right=-1, marker_left=-1)
