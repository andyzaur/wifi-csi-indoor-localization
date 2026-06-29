"""Parity + contract tests for validate_session.build_report.

build_report() is the library entry point extracted out of main(). It must:
  * return a Report whose .rows reproduce exactly what the CLI prints, and
  * NEVER call sys.exit (process control stays in main()).

Where a real recorded session exists under sessions/, we pin its check rows to
a golden captured from the current code. The session dir is gitignored, so the
golden test self-skips when it is absent; the contract tests (no sys.exit, rows
structure, verdict math) always run via a synthetic fixture.
"""

import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

import validate_session as vs
from validate_session import build_report, Report, OK, WARN, FAIL


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_SESSION = os.path.join(REPO_ROOT, "sessions", "20260603_2029_QuickTest5min")

# Golden rows captured from the CURRENT code path (CLI main()) on REAL_SESSION.
# Each row is (level, label, detail). These are byte-for-byte the strings the
# CLI prints for this session.
GOLDEN_ROWS = [
    ('OK', 'csi.csv non-empty', '24,418 data rows'),
    ('OK', 'camera.csv non-empty', '6,221 data rows'),
    ('OK', 'clap.csv non-empty', '2 data rows'),
    ('WARN', 'session window', '190 s between START and STOP'),
    ('OK', 'clapper START/STOP count', 'exactly 1 each'),
    ('OK', '3 boards present', 'boards [1, 4, 5] all sent CSI'),
    ('OK', 'per-board CSI rate', 'b1=33.3/s (6,329), b4=33.3/s (6,334), b5=33.3/s (6,335)'),
    ('OK', 'TX MAC purity', 'single TX MAC 48:f6:ee:c2:ce:75'),
    ('OK', 'csi_len purity', 'csi_len uniform 128'),
    ('OK', 'camera detection rate', '99.7% frames detected'),
    ('WARN', 'camera frame flow', '~28.3 fps (median frame interval), max gap 2.13 s — frame gap >1 s leaves CSI unlabeled'),
    ('OK', 'CSI<->camera time gap', 'median 10 ms, max 71 ms, 18,998 aligned samples'),
    ('WARN', 'two-foot label mix', 'both=62% (both=62%, right_only=26%, left_only=11%, none=0%) — both<75%, single-foot fallback heavy'),
    ('OK', 'duplicate-CSI fraction', '0.2% (5,215->5,202 unique); >20% = CSI delivery capped (broadcast/rate)'),
    ('OK', 'per-board CSI age (median/p90)', 'b1=15/28ms, b4=16/28ms, b5=16/28ms'),
    ('OK', 'grid coverage', '23 cells; samples/cell min=17 median=193 max=625'),
    ('OK', 'session metadata', 'present'),
]


def _copy_session(src, dst):
    # Copy only the CSVs + metadata.json the validator reads; skip heavy
    # keyframes/corners so the fixture is light.
    os.makedirs(dst, exist_ok=True)
    for name in ("csi.csv", "camera.csv", "clap.csv", "metadata.json"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst, name))


@pytest.mark.skipif(not os.path.isdir(REAL_SESSION),
                    reason="real recorded session (gitignored) not present")
def test_build_report_rows_match_golden(tmp_path):
    sd = tmp_path / "sess"
    _copy_session(REAL_SESSION, str(sd))
    rep = build_report(str(sd))
    assert rep.rows == GOLDEN_ROWS


@pytest.mark.skipif(not os.path.isdir(REAL_SESSION),
                    reason="real recorded session (gitignored) not present")
def test_cli_output_matches_build_report_render(tmp_path):
    # The CLI must print exactly build_report(...).print(); prove parity by
    # comparing the rendered report text to a captured run of main().
    sd = tmp_path / "sess"
    _copy_session(REAL_SESSION, str(sd))

    rep = build_report(str(sd))
    # Render the report the way main() does (capture Report.print()).
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rep.print()
    rendered = buf.getvalue()

    # Run the actual CLI; strip the leading "Validating session:" header line
    # that main() prints before the report body.
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "validate_session.py"), str(sd)],
        cwd=REPO_ROOT, capture_output=True, text=True)
    cli_out = proc.stdout
    header = f"Validating session: {sd}\n"
    assert cli_out.startswith(header)
    assert cli_out[len(header):] == rendered
    # exit code parity
    assert proc.returncode == rep.exit_code


def test_build_report_does_not_call_sys_exit(monkeypatch, tmp_path):
    # Even on a totally broken session (missing CSVs -> load failure), the
    # extracted library function must NOT terminate the process.
    called = {"exit": False}

    def boom(*a, **k):
        called["exit"] = True
        raise AssertionError("build_report must not call sys.exit")

    monkeypatch.setattr(sys, "exit", boom)

    broken = tmp_path / "broken"
    broken.mkdir()
    rep = build_report(str(broken))

    assert isinstance(rep, Report)
    assert called["exit"] is False
    # Missing files + load failure are all FAIL rows.
    assert any(level == FAIL and label == "load session CSVs" for level, label, _ in rep.rows)
    assert rep.worst() == FAIL
    assert rep.exit_code == 1
    assert rep.ok is False


def test_build_report_returns_report_with_check_rows(tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    rep = build_report(str(broken))
    # Each missing CSV yields a FAIL "<name> exists" row before the load fails.
    labels = [label for _, label, _ in rep.rows]
    assert "csi.csv exists" in labels
    assert "camera.csv exists" in labels
    assert "clap.csv exists" in labels


def test_report_verdict_and_exit_code_math():
    rep = Report()
    rep.add(OK, "a")
    assert rep.worst() == OK and rep.exit_code == 0 and rep.ok is True
    rep.add(WARN, "b")
    assert rep.worst() == WARN and rep.exit_code == 0 and rep.ok is True
    rep.add(FAIL, "c")
    assert rep.worst() == FAIL and rep.exit_code == 1 and rep.ok is False


# ---------------------------------------------------------------------------
# Synthetic-fixture unit tests for individual checks (always run, no session).
# ---------------------------------------------------------------------------

def _row(rep, label):
    """The single (level, detail) for a labelled row; fails if 0 or >1 match."""
    matches = [(lvl, det) for lvl, lab, det in rep.rows if lab == label]
    assert len(matches) == 1, f"expected exactly one {label!r} row in {rep.rows}"
    return matches[0]


def _clap(dur_s):
    return pd.DataFrame({"event_name": ["start", "stop"],
                         "wall_time_s": [100.0, 100.0 + dur_s]})


def test_session_window_warns_below_300s():
    # Real sessions are 15-20 min; a 190 s window now warns (old bar was 30 s).
    rep = Report()
    vs.check_clap(_clap(190.0), rep)
    assert _row(rep, "session window") == (WARN, "190 s between START and STOP")


def test_session_window_ok_above_300s():
    rep = Report()
    vs.check_clap(_clap(900.0), rep)
    assert _row(rep, "session window") == (OK, "900 s between START and STOP")


def test_stream_purity_ok_single_mac_uniform_len():
    csi = pd.DataFrame({"mac": ["aa:bb:cc:dd:ee:ff"] * 5, "csi_len": [128] * 5})
    rep = Report()
    vs.check_stream_purity(csi, rep)
    assert _row(rep, "TX MAC purity") == (OK, "single TX MAC aa:bb:cc:dd:ee:ff")
    assert _row(rep, "csi_len purity") == (OK, "csi_len uniform 128")


def test_stream_purity_warns_on_two_macs():
    csi = pd.DataFrame({"mac": ["aa:aa"] * 4 + ["bb:bb"] * 2, "csi_len": [128] * 6})
    rep = Report()
    vs.check_stream_purity(csi, rep)
    level, detail = _row(rep, "TX MAC purity")
    assert level == WARN
    assert "aa:aa=4" in detail and "bb:bb=2" in detail
    assert "unexpected source frames" in detail
    assert _row(rep, "csi_len purity")[0] == OK  # length still pure


def test_stream_purity_warns_on_mixed_csi_len():
    csi = pd.DataFrame({"mac": ["aa:aa"] * 6, "csi_len": [128] * 4 + [384] * 2})
    rep = Report()
    vs.check_stream_purity(csi, rep)
    assert _row(rep, "TX MAC purity")[0] == OK  # MAC still pure
    level, detail = _row(rep, "csi_len purity")
    assert level == WARN
    assert "mixed CSI formats" in detail
    assert "len=128: 4" in detail and "len=384: 2" in detail


def _camera_times(times):
    return pd.DataFrame({"wall_time_s": times})


def test_camera_flow_ok_at_30fps_no_gaps():
    rep = Report()
    vs.check_camera_flow(_camera_times([i / 30.0 for i in range(300)]), rep)
    level, detail = _row(rep, "camera frame flow")
    assert level == OK
    assert "~30.0 fps" in detail


def test_camera_flow_warns_below_20fps_mentions_gui_stall():
    rep = Report()
    vs.check_camera_flow(_camera_times([i / 10.0 for i in range(100)]), rep)
    level, detail = _row(rep, "camera frame flow")
    assert level == WARN
    assert "GUI preview stall" in detail


def test_camera_flow_warns_on_gap_over_1s():
    # 30 fps with a single ~2 s hole mid-recording.
    times = ([i / 30.0 for i in range(100)]
             + [100 / 30.0 + 2.0 + i / 30.0 for i in range(100)])
    rep = Report()
    vs.check_camera_flow(_camera_times(times), rep)
    level, detail = _row(rep, "camera frame flow")
    assert level == WARN
    assert "frame gap >1 s" in detail


def test_camera_flow_fails_on_gap_over_5s():
    times = ([i / 30.0 for i in range(100)]
             + [100 / 30.0 + 6.0 + i / 30.0 for i in range(100)])
    rep = Report()
    vs.check_camera_flow(_camera_times(times), rep)
    level, detail = _row(rep, "camera frame flow")
    assert level == FAIL
    assert "stalled" in detail


def test_camera_flow_skips_tiny_capture():
    rep = Report()
    vs.check_camera_flow(_camera_times([0.0, 0.033, 0.066]), rep)
    assert rep.rows == []  # guarded: <10 frames says nothing about flow


def _camera_methods(counts):
    methods = [m for m, n in counts.items() for _ in range(n)]
    return pd.DataFrame({"method": methods})


def test_two_foot_mix_fails_below_50pct():
    cam = _camera_methods({"both": 40, "right_only": 35, "left_only": 25})
    rep = Report()
    vs.check_quality(pd.DataFrame(), cam, pd.DataFrame(), rep)
    level, detail = _row(rep, "two-foot label mix")
    assert level == FAIL
    assert "both=40%" in detail
    assert "inflate the LOSO fold" in detail and "RealSession2" in detail


def test_two_foot_mix_warn_between_50_and_75pct():
    cam = _camera_methods({"both": 62, "right_only": 38})
    rep = Report()
    vs.check_quality(pd.DataFrame(), cam, pd.DataFrame(), rep)
    level, detail = _row(rep, "two-foot label mix")
    assert level == WARN
    assert "single-foot fallback heavy" in detail


def test_two_foot_mix_ok_at_75pct_or_more():
    cam = _camera_methods({"both": 80, "right_only": 20})
    rep = Report()
    vs.check_quality(pd.DataFrame(), cam, pd.DataFrame(), rep)
    level, detail = _row(rep, "two-foot label mix")
    assert level == OK
    assert "both=80%" in detail
