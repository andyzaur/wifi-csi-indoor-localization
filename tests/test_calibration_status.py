"""Tests for csi_gui.calibration_status — pure read-only JSON status.

No Qt needed (these never construct a widget): we write temp JSON files into a
fake ROOT and assert the status helper reports present/missing + key fields, and
that a missing file => "not calibrated".
"""

import json

import pytest

# The helper itself has no Qt dependency, but it lives under csi_gui which the
# rest of the package shares; importorskip keeps this honest in a minimal env.
calibration_status = pytest.importorskip("csi_gui.calibration_status")
cs = calibration_status


# ---------------------------------------------------------------------------
# Sample on-disk contents mirroring the real tool outputs.
# ---------------------------------------------------------------------------
_FLOOR = {
    "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "homography_inv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "marker_positions_cm": {"2": [0, 0], "3": [50, 0], "4": [0, 50]},
    "grid_spacing_cm": 50.0,
    "grid_bounds_cm": [0.0, 400.0, 0.0, 400.0],
    "camera_resolution": [3264, 2448],
    "aruco_dict": "DICT_4X4_50",
}
_LENS = {
    "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "dist_coeffs": [0, 0, 0, 0, 0],
    "resolution": [3264, 2448],
    "reprojection_error": 0.0777,
    "num_captures": 31,
}
_MARKER = {
    "positions_cm": {"1": [0.0, 0.0], "2": [0.0, 148.0], "5": [200.0, 0.0]},
    "anchors": {"1": [0.0, 0.0]},
    "distance_measurements_cm": {},
}


def _write(root, name, data):
    path = root / name
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# Present + key fields.
# ---------------------------------------------------------------------------
def test_floor_present_reports_key_fields(tmp_path):
    _write(tmp_path, cs.FLOOR_CALIBRATION, _FLOOR)
    st = cs.floor_status(str(tmp_path))

    assert st.present is True
    assert st.error is None
    assert st.summary == "calibrated"
    assert st.fields["grid_spacing_cm"] == 50.0
    assert st.fields["grid_bounds_cm"] == [0.0, 400.0, 0.0, 400.0]
    assert st.fields["camera_resolution"] == [3264, 2448]
    assert st.fields["n_marker_positions"] == 3
    assert st.fields["aruco_dict"] == "DICT_4X4_50"
    assert st.mtime is not None
    assert st.mtime_str != "never"


def test_lens_present_reports_key_fields(tmp_path):
    _write(tmp_path, cs.LENS_PROFILE, _LENS)
    st = cs.lens_status(str(tmp_path))

    assert st.present is True
    assert st.summary == "calibrated"
    assert st.fields["resolution"] == [3264, 2448]
    assert st.fields["reprojection_error"] == pytest.approx(0.0777)
    assert st.fields["num_captures"] == 31


def test_marker_present_reports_position_count(tmp_path):
    _write(tmp_path, cs.MARKER_LAYOUT, _MARKER)
    st = cs.marker_status(str(tmp_path))

    assert st.present is True
    assert st.summary == "calibrated"
    assert st.fields["n_positions"] == 3


# ---------------------------------------------------------------------------
# Missing => "not calibrated".
# ---------------------------------------------------------------------------
def test_missing_floor_is_not_calibrated(tmp_path):
    st = cs.floor_status(str(tmp_path))
    assert st.present is False
    assert st.summary == "not calibrated"
    assert st.mtime is None
    assert st.mtime_str == "never"
    assert st.fields == {}


def test_missing_lens_and_marker_are_not_calibrated(tmp_path):
    assert cs.lens_status(str(tmp_path)).summary == "not calibrated"
    assert cs.marker_status(str(tmp_path)).summary == "not calibrated"


# ---------------------------------------------------------------------------
# Present-but-unreadable => present=True, error set, "unreadable" summary.
# ---------------------------------------------------------------------------
def test_corrupt_json_is_unreadable_not_calibrated(tmp_path):
    (tmp_path / cs.FLOOR_CALIBRATION).write_text("{ this is not json ]")
    st = cs.floor_status(str(tmp_path))
    # The file exists on disk (present) but failed to parse -> "unreadable",
    # distinct from a genuinely-absent file ("not calibrated").
    assert st.present is True
    assert st.error is not None
    assert "unreadable" in st.summary
    assert st.summary != "not calibrated"


# ---------------------------------------------------------------------------
# all_status aggregates the three by name.
# ---------------------------------------------------------------------------
def test_all_status_keys_and_mixed_presence(tmp_path):
    _write(tmp_path, cs.FLOOR_CALIBRATION, _FLOOR)
    _write(tmp_path, cs.LENS_PROFILE, _LENS)
    # marker intentionally missing

    statuses = cs.all_status(str(tmp_path))
    assert set(statuses) == {"floor", "lens", "marker"}
    assert statuses["floor"].present is True
    assert statuses["lens"].present is True
    assert statuses["marker"].present is False
