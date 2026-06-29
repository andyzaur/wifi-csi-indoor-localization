"""Regression tests for the Sessions-page CRASH fixes.

Two distinct fixes are covered:

  1. ``session_worker`` tolerance — a capture with ``csi.csv`` but no
     ``camera.csv`` / ``clap.csv`` (the ``diag_*`` / no-camera runs) appears in
     the explorer; selecting it used to raise ``FileNotFoundError`` in the worker.
     It must now render the CSI plots and a graceful "empty" placeholder for the
     camera-only plots. The lazy read (camera-only plots never touch ``csi.csv``)
     is checked by handing it a deliberately unparseable ``csi.csv``.

  2. ``SessionsPage`` pool hardening — every ``executor.submit`` runs on the GUI
     thread inside a Qt slot, so a ``BrokenProcessPool`` escaping it would
     terminate the app under PySide6. ``_submit`` must swallow it and surface an
     inline error instead of raising.
"""

import os
from concurrent.futures.process import BrokenProcessPool

import pytest

from csi_gui import session_worker


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _make_csi_only_session(root, name="diag_nocam"):
    """A session with a valid csi.csv but NO camera.csv / clap.csv."""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    header = (["wall_time_s", "board_id", "mac", "rssi", "channel",
               "timestamp_us", "rx_seq", "csi_len"]
              + [f"csi_{i}" for i in range(128)])
    rows = [",".join(header)]
    t = 0.0
    for k in range(30):
        for b in (1, 4, 5):
            iq = [str((k * 7 + i * 3 + b) % 50 - 25) for i in range(128)]
            vals = [f"{t:.3f}", str(b), "aa:bb", str(-40 - b), "6",
                    str(int(t * 1e6)), str(k), "128"] + iq
            rows.append(",".join(vals))
            t += 0.005
    _write(os.path.join(d, "csi.csv"), "\n".join(rows) + "\n")
    return d


# ---------------------------------------------------------------------------
# (1) worker tolerance — the actual crash
# ---------------------------------------------------------------------------

def test_render_all_plots_tolerates_missing_camera(tmp_path):
    """All 4 plots render for a csi-only session — no FileNotFoundError."""
    d = _make_csi_only_session(str(tmp_path))
    # CSI plots (heatmap, rate timeline) render real images.
    for idx in (0, 3):
        r = session_worker.render_session_plot(d, idx)
        assert r["empty"] is False
        assert r["width"] > 0 and len(r["buffer"]) > 0
    # Camera-only plots (coverage, walked path) fall back to a graceful empty
    # placeholder instead of raising.
    for idx in (1, 2):
        r = session_worker.render_session_plot(d, idx)
        assert r["empty"] is True
        assert len(r["buffer"]) > 0  # still a valid (placeholder) image


def test_camera_only_plot_does_not_read_csi(tmp_path):
    """A camera-only plot must NOT parse csi.csv (lazy read) — even a broken one."""
    d = os.path.join(str(tmp_path), "weird")
    os.makedirs(d)
    # csi.csv that pandas would choke on if it were read.
    _write(os.path.join(d, "csi.csv"), "\x00 not really csv \x00\n,,,\n")
    _write(os.path.join(d, "camera.csv"),
           "frame,timestamp_s,x_cm,y_cm,grid_x_cm,grid_y_cm,detected\n"
           "0,0.0,100,100,100,100,1\n1,0.1,150,150,150,150,1\n")
    # Coverage (idx 1) is camera-only: renders without touching the bad csi.csv.
    r = session_worker.render_session_plot(d, 1)
    assert isinstance(r, dict) and len(r["buffer"]) > 0


def test_load_session_tolerant_missing_files(tmp_path):
    """The tolerant loader returns empty camera/clap frames when files are absent."""
    d = _make_csi_only_session(str(tmp_path), "x")
    csi, camera, clap = session_worker._load_session_tolerant(d, need_csi=True)
    assert len(csi) > 0
    assert len(camera) == 0   # missing camera.csv -> empty frame, not a raise
    assert len(clap) == 0
    # need_csi=False skips the csi read entirely.
    csi2, _, _ = session_worker._load_session_tolerant(d, need_csi=False)
    assert len(csi2) == 0


# ---------------------------------------------------------------------------
# (2) SessionsPage pool hardening — broken pool must not crash the app
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from csi_gui.ui.pages.sessions_page import SessionsPage  # noqa: E402


class _BrokenExecutor:
    """An injected executor whose submit always poisons the pool."""

    def submit(self, *args, **kwargs):
        raise BrokenProcessPool("simulated dead pool")

    def shutdown(self, wait=True, cancel_futures=False):
        pass


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_render_tab_survives_broken_pool(tmp_path, _qapp):
    """A BrokenProcessPool on submit() shows an inline error, never raises."""
    page = SessionsPage(sessions_dir=str(tmp_path), executor=_BrokenExecutor())
    page._selected_path = "/fake/path"
    # Must NOT raise (would otherwise escape the Qt slot and kill the app).
    page._render_tab("/fake/path", 0)
    assert "unavailable" in page._viz_tabs[0]._status.text().lower()
    # The dead injected pool is never rebuilt (we only rebuild pools we own).
    assert page._executor is not None
    page.deleteLater()


def test_analyze_survives_broken_pool(tmp_path, _qapp):
    """Analyze on a broken pool resets the scorecard instead of crashing."""
    page = SessionsPage(sessions_dir=str(tmp_path), executor=_BrokenExecutor())
    page._selected_path = "/fake/path"
    page._on_analyze()  # must not raise
    # No report was recorded as in-flight (submit failed cleanly).
    assert "/fake/path" not in page._report_inflight
    page.deleteLater()


def test_process_pool_creation_failure_falls_back_to_thread(
        tmp_path, _qapp, monkeypatch):
    """Pool construction failures happen inside Qt slots and must not escape."""
    import csi_gui.ui.pages.sessions_page as sessions_page

    def _raise_permission(*_args, **_kwargs):
        raise PermissionError("simulated semaphore denial")

    monkeypatch.setattr(sessions_page, "ProcessPoolExecutor", _raise_permission)

    page = SessionsPage(sessions_dir=str(tmp_path))
    future = page._submit(lambda: "ok")

    assert future is not None
    assert future.result(timeout=2) == "ok"
    assert page._executor_mode == "thread"
    assert "PermissionError" in page._worker_error
    page.shutdown()
    page.deleteLater()
