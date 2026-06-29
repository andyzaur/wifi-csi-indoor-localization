"""Headless-host contract: with owns_window=False the tracker NEVER touches the
cv2 GUI (imshow/waitKey/destroyAllWindows), and the on_position / on_frame
callback stream matches the camera.csv rows it writes.
"""

import csv as _csv

import numpy as np
import cv2
from cv2 import aruco
import pytest

import aruco_track
from tests.test_aruco_parity import _make_fixture_video, _FakeClock


@pytest.fixture
def fixture_video(tmp_path):
    return _make_fixture_video(str(tmp_path / "fixture.mp4"))


def test_owns_window_false_never_calls_gui(tmp_path, fixture_video, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("cv2 GUI must not be called when owns_window=False")

    monkeypatch.setattr(cv2, "imshow", _boom)
    monkeypatch.setattr(cv2, "waitKey", _boom)
    monkeypatch.setattr(cv2, "destroyAllWindows", _boom)
    monkeypatch.setattr(aruco_track.time, "time", _FakeClock())

    log = str(tmp_path / "camera.csv")
    states = []
    frames = []
    tracker = aruco_track.ArucoTracker(
        video=fixture_video, log=log,
        display=True, owns_window=False,        # display on, but host owns window
        emit_preview=True,
        on_position=states.append,
        on_frame=frames.append)
    tracker.run_forever()

    # 30 frames -> 30 callbacks each.
    assert len(states) == 30
    assert len(frames) == 30

    # emit_preview=True -> a downscaled BGR preview, never the full-res frame.
    for fr in frames:
        assert fr.preview_bgr is not None
        assert fr.preview_bgr.ndim == 3 and fr.preview_bgr.shape[2] == 3


def test_callback_stream_matches_csv(tmp_path, fixture_video, monkeypatch):
    monkeypatch.setattr(aruco_track.time, "time", _FakeClock())

    log = str(tmp_path / "camera.csv")
    results = []
    tracker = aruco_track.ArucoTracker(
        video=fixture_video, log=log,
        display=False, owns_window=False,
        on_frame=results.append)
    tracker.run_forever()

    with open(log) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == len(results) == 30

    for row, fr in zip(rows, results):
        assert int(row["frame"]) == fr.frame_num
        assert f"{fr.timestamp_s:.4f}" == row["timestamp_s"]
        pos = fr.position
        assert row["method"] == pos.method
        assert int(row["n_markers"]) == pos.n_markers
        if row["detected"] == "1":
            assert pos.detected is True
            assert f"{pos.x_cm:.1f}" == row["x_cm"]
            assert f"{pos.y_cm:.1f}" == row["y_cm"]
            assert f"{pos.grid_x_cm:.0f}" == row["grid_x_cm"]
            assert f"{pos.grid_y_cm:.0f}" == row["grid_y_cm"]
        else:
            assert pos.detected is False
            assert pos.x_cm is None and pos.y_cm is None
            assert row["x_cm"] == ""
        # right/left center parity with the CSV's right_x / left_x columns.
        if pos.right_center is not None:
            assert f"{pos.right_center[0]:.1f}" == row["right_x"]
            assert f"{pos.right_center[1]:.1f}" == row["right_y"]
        else:
            assert row["right_x"] == ""
        if pos.left_center is not None:
            assert f"{pos.left_center[0]:.1f}" == row["left_x"]
            assert f"{pos.left_center[1]:.1f}" == row["left_y"]
        else:
            assert row["left_x"] == ""


def test_no_preview_when_emit_preview_false(tmp_path, fixture_video, monkeypatch):
    monkeypatch.setattr(aruco_track.time, "time", _FakeClock())
    frames = []
    tracker = aruco_track.ArucoTracker(
        video=fixture_video, log=str(tmp_path / "camera.csv"),
        display=False, owns_window=False, emit_preview=False,
        on_frame=frames.append)
    tracker.run_forever()
    assert frames and all(fr.preview_bgr is None for fr in frames)
