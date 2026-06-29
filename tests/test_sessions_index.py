"""Tests for csi_gui.sessions_index — session discovery + name parsing + counts.

Qt-free: builds a temp ``sessions/`` with a few fixture session dirs (and some
decoys) and asserts the discovery, the ``YYYYMMDD_NN_purpose`` parsing, the
wc-style row counts, and newest-first ordering.
"""

import os
from datetime import date

from csi_gui.sessions_index import (
    SessionInfo,
    build_session_info,
    list_sessions,
    parse_session_name,
)
from csi_gui.session_labels import Label, save_label


def _write(path: str, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def _make_session(root: str, name: str, *, csi_rows=3, cam_rows=2, clap_rows=2,
                  camera=True, clap=True, metadata=False):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    _write(os.path.join(d, "csi.csv"),
           ["wall_time_s,board_id,rssi"] + [f"{i}.0,1,-40" for i in range(csi_rows)])
    if camera:
        _write(os.path.join(d, "camera.csv"),
               ["frame,timestamp_s,x_cm,y_cm,grid_x_cm,grid_y_cm,detected"]
               + [f"{i},{i}.0,10,10,0,0,1" for i in range(cam_rows)])
    if clap:
        _write(os.path.join(d, "clap.csv"),
               ["wall_time_s,event,event_name,seq,timestamp_us"]
               + [f"{i}.0,{i},start,0,0" for i in range(clap_rows)])
    if metadata:
        _write(os.path.join(d, "metadata.json"), ['{"room": "lab"}'])
    return d


def test_parse_session_name_conforming():
    d, nn, purpose = parse_session_name("20260604_0025_RealSession2")
    assert d == date(2026, 6, 4)
    assert nn == 25
    assert purpose == "RealSession2"


def test_parse_session_name_no_purpose():
    d, nn, purpose = parse_session_name("20260519_01")
    assert d == date(2026, 5, 19)
    assert nn == 1
    assert purpose == ""


def test_parse_session_name_non_conforming_tolerated():
    assert parse_session_name("combined_01_02_03") == (None, None, "")
    assert parse_session_name("drift_full") == (None, None, "")
    # Invalid calendar date degrades date to None but is still tolerated.
    d, nn, purpose = parse_session_name("20261345_07_weird")
    assert d is None
    assert nn == 7
    assert purpose == "weird"


def test_list_sessions_discovers_only_csi_dirs(tmp_path):
    root = str(tmp_path)
    _make_session(root, "20260601_01_alpha")
    _make_session(root, "20260602_02_beta")
    # decoy: a directory with no csi.csv must be ignored.
    os.makedirs(os.path.join(root, "not_a_session"))
    _write(os.path.join(root, "not_a_session", "camera.csv"), ["frame"])
    # decoy: a loose file at top level.
    _write(os.path.join(root, "loose.txt"), ["x"])

    infos = list_sessions(root)
    names = [i.name for i in infos]
    assert names == ["20260602_02_beta", "20260601_01_alpha"]  # newest first
    assert all(isinstance(i, SessionInfo) for i in infos)
    assert "not_a_session" not in names


def test_list_sessions_missing_dir_returns_empty(tmp_path):
    assert list_sessions(str(tmp_path / "does_not_exist")) == []


def test_row_counts_are_wc_style(tmp_path):
    root = str(tmp_path)
    _make_session(root, "20260601_01_counts", csi_rows=5, cam_rows=3, clap_rows=2)
    info = list_sessions(root)[0]
    assert info.csi_rows == 5
    assert info.camera_rows == 3
    assert info.clap_rows == 2
    assert info.has_camera and info.has_clap
    assert not info.has_metadata
    assert "5 csi" in info.row_summary


def test_header_only_files_count_zero(tmp_path):
    root = str(tmp_path)
    d = _make_session(root, "20260601_01_empty", csi_rows=0, cam_rows=0, clap_rows=0)
    # csi.csv has only a header now.
    info = build_session_info(d)
    assert info.csi_rows == 0
    assert info.camera_rows == 0


def test_missing_camera_and_clap_flags(tmp_path):
    root = str(tmp_path)
    _make_session(root, "20260601_01_csionly", camera=False, clap=False)
    info = list_sessions(root)[0]
    assert info.has_camera is False
    assert info.has_clap is False
    assert info.camera_rows == 0
    assert "—" in info.row_summary


def test_metadata_flag_detected(tmp_path):
    root = str(tmp_path)
    _make_session(root, "20260601_01_withmeta", metadata=True)
    info = list_sessions(root)[0]
    assert info.has_metadata is True


def test_label_is_loaded_into_info(tmp_path):
    root = str(tmp_path)
    d = _make_session(root, "20260601_01_rated")
    save_label(d, Label(rating="best", tags=["clean"], notes="the good one"))
    info = list_sessions(root)[0]
    assert info.rating == "best"
    assert info.label.tags == ["clean"]


def test_undated_sessions_sort_last(tmp_path):
    root = str(tmp_path)
    _make_session(root, "20260601_01_dated")
    _make_session(root, "combined_01_02_03")  # undated
    infos = list_sessions(root)
    assert infos[0].name == "20260601_01_dated"
    assert infos[-1].name == "combined_01_02_03"


def test_same_day_sorted_by_nn_desc(tmp_path):
    root = str(tmp_path)
    _make_session(root, "20260601_01_first")
    _make_session(root, "20260601_03_third")
    _make_session(root, "20260601_02_second")
    infos = list_sessions(root)
    assert [i.nn for i in infos] == [3, 2, 1]
