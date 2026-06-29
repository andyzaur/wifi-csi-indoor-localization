"""Tests for csi_gui.applog.AppLogger — the diagnostic file logger.

The logger appends TIMESTAMPED lines to a file, is itself a callable (so it can
be passed straight as the ArucoTracker's ``on_log``), is line-flushed (readable
mid-run), and is safe to use after close (no-op). No network / Qt involved.
"""

import re

from csi_gui.applog import AppLogger

_TS = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\s{2}")


def test_logger_appends_timestamped_lines(tmp_path):
    path = tmp_path / "csi.log"
    log = AppLogger(str(path))
    log("Start tracker: http://127.0.0.1:8080/video")
    log("check wifi: GREEN joined CSI_TX")
    log.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    # Banner + the two messages.
    assert any("log opened" in ln for ln in lines)
    assert lines[-2].endswith("Start tracker: http://127.0.0.1:8080/video")
    assert lines[-1].endswith("check wifi: GREEN joined CSI_TX")
    # Each line carries a millisecond timestamp prefix.
    assert all(_TS.match(ln) for ln in lines)


def test_logger_is_callable_for_on_log(tmp_path):
    path = tmp_path / "csi.log"
    log = AppLogger(str(path))
    # Used exactly as ArucoTracker(on_log=...) would call it.
    on_log = log
    on_log("Tracking foot markers: right=0 left=9")
    log.close()
    assert "Tracking foot markers" in path.read_text(encoding="utf-8")


def test_logger_flushes_so_file_is_readable_mid_run(tmp_path):
    path = tmp_path / "csi.log"
    log = AppLogger(str(path))
    log("midrun line")
    # No close() yet — the line must already be on disk (line-buffered + flush).
    assert "midrun line" in path.read_text(encoding="utf-8")
    log.close()


def test_logger_safe_after_close(tmp_path):
    path = tmp_path / "csi.log"
    log = AppLogger(str(path))
    log.close()
    log.close()        # idempotent
    log("after close")  # no-op, must not raise
    assert "after close" not in path.read_text(encoding="utf-8")
