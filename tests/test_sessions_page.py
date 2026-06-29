"""Offscreen smoke tests for csi_gui.ui.pages.sessions_page.SessionsPage.

Verifies the page constructs with the PROCESS-POOL wiring, lists a fixture
sessions/ dir, and that selecting a session populates the scorecard +
visualization tabs without raising. The heavy work normally runs in a SEPARATE
PROCESS; to keep these tests in-process and deterministic we inject a SYNCHRONOUS
fake "executor" whose ``submit`` runs the function immediately and returns a done
future (so the future's done-callback fires inline). Qt is importorskip +
offscreen so this stays headless.
"""

import os
from concurrent.futures import Future

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QDeadlineTimer, QEventLoop  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from csi_gui.session_labels import load_label  # noqa: E402
from csi_gui.ui.pages.sessions_page import SessionsPage  # noqa: E402


class _SyncExecutor:
    """A drop-in for ProcessPoolExecutor that runs everything in-process, now.

    ``submit(fn, *args)`` calls ``fn(*args)`` synchronously and returns an
    already-resolved :class:`concurrent.futures.Future`, so the page's
    ``future.add_done_callback`` fires immediately on the calling (GUI) thread.
    The page then emits a QUEUED signal, which we pump via ``processEvents``.
    """

    def __init__(self):
        self.shutdown_called = False
        self.submitted = []

    def submit(self, fn, *args, **kwargs):
        self.submitted.append((fn, args, kwargs))
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_called = True


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _make_full_session(root, name):
    """A small but *valid* session: 3 boards, detected camera path, a clap."""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(1)

    # csi.csv
    header = (["wall_time_s", "board_id", "mac", "rssi", "channel",
               "timestamp_us", "rx_seq", "csi_len"]
              + [f"csi_{i}" for i in range(128)])
    rows = [",".join(header)]
    t = 0.0
    for k in range(40):
        for b in (1, 4, 5):
            iq = rng.integers(-30, 30, size=128)
            vals = [f"{t:.3f}", str(b), "aa:bb", str(-40 - b), "6",
                    str(int(t * 1e6)), str(k), "128"] + [str(int(v)) for v in iq]
            rows.append(",".join(vals))
            t += 0.005
    _write(os.path.join(d, "csi.csv"), rows)

    # camera.csv (7-col), a circular detected path inside the trim window.
    n = 80
    ct = np.linspace(0.0, 0.55, n)
    x = 100 + 60 * np.sin(np.linspace(0, 6.28, n))
    y = 100 + 60 * np.cos(np.linspace(0, 6.28, n))
    cam = ["frame,timestamp_s,x_cm,y_cm,grid_x_cm,grid_y_cm,detected"]
    for i in range(n):
        cam.append(f"{i},{ct[i]:.3f},{x[i]:.1f},{y[i]:.1f},"
                   f"{int(x[i] // 50 * 50)},{int(y[i] // 50 * 50)},1")
    _write(os.path.join(d, "camera.csv"), cam)

    # clap.csv: a START and a STOP bracketing the data.
    _write(os.path.join(d, "clap.csv"),
           ["wall_time_s,event,event_name,seq,timestamp_us",
            "0.0,0,start,0,0",
            "0.60,1,stop,1,600000"])
    return d


def _make_page(root, _qapp):
    """A SessionsPage backed by the synchronous in-process fake executor."""
    fake = _SyncExecutor()
    page = SessionsPage(sessions_dir=root, executor=fake)
    return page, fake


def _spin_until(app, predicate, timeout_ms=8000):
    """Pump the event loop until predicate() is true or the timeout elapses."""
    deadline = QDeadlineTimer(timeout_ms)
    while not predicate() and not deadline.hasExpired():
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
    return predicate()


def test_page_constructs_and_lists(tmp_path, _qapp):
    root = str(tmp_path)
    _make_full_session(root, "20260601_01_alpha")
    _make_full_session(root, "20260601_02_beta")
    page, _ = _make_page(root, _qapp)
    assert page._list.count() == 2
    # Newest-first: beta (NN=02) before alpha (NN=01).
    assert "beta" in page._list.item(0).text()
    page.deleteLater()


def test_constructs_without_spawning_pool(tmp_path, _qapp):
    """The process pool is lazy: constructing the page must not create one."""
    root = str(tmp_path)
    _make_full_session(root, "20260601_01_alpha")
    # No injected executor -> the page should still construct without spawning.
    page = SessionsPage(sessions_dir=root)
    assert page._executor is None
    page.shutdown()  # idempotent, never spawned -> nothing to kill
    page.deleteLater()


def test_selecting_session_loads_detail(tmp_path, _qapp):
    root = str(tmp_path)
    d = _make_full_session(root, "20260601_01_alpha")
    page, fake = _make_page(root, _qapp)

    loaded = []
    page.detailLoaded.connect(lambda p: loaded.append(p))

    # Select the only session: dispatches the active-tab render to the (sync)
    # executor; the queued vizReady signal lands after we pump events.
    page.select_session(os.path.abspath(d))
    assert _spin_until(_qapp, lambda: bool(loaded)), "detail never loaded"
    assert loaded[0] == os.path.abspath(d)

    # Selecting does NOT run the report (opt-in); scorecard has no rows yet.
    assert len(page._scorecard._row_widgets) == 0
    # The ACTIVE viz tab renders on select (lazy: only that one).
    active = page._tabs.currentIndex()
    assert page._viz_tabs[active]._pixmap is not None
    # render_session_plot was submitted for the active plot only.
    assert any(args[1] == active for _fn, args, _kw in fake.submitted)
    page.deleteLater()


def test_select_does_not_run_report_until_analyze(tmp_path, _qapp):
    """Selecting a session must NOT compute the report; Analyze runs it once."""
    from csi_gui import session_worker

    root = str(tmp_path)
    d = _make_full_session(root, "20260601_01_alpha")
    page, fake = _make_page(root, _qapp)

    page.select_session(os.path.abspath(d))
    _qapp.processEvents()
    # No compute_report submitted on select.
    assert not any(fn is session_worker.compute_report
                   for fn, _args, _kw in fake.submitted)
    assert len(page._scorecard._row_widgets) == 0

    # Press Analyze -> compute_report runs once, scorecard fills in.
    page._on_analyze()
    assert _spin_until(_qapp, lambda: len(page._scorecard._row_widgets) > 0), \
        "scorecard never rendered after Analyze"
    report_calls = [fn for fn, _a, _k in fake.submitted
                    if fn is session_worker.compute_report]
    assert len(report_calls) == 1, f"expected one compute_report, got {len(report_calls)}"
    page.deleteLater()


def test_only_active_tab_renders_on_select_and_switch(tmp_path, _qapp):
    """Only the active plot renders on select; switching renders the new one."""
    from csi_gui import session_worker

    root = str(tmp_path)
    d = _make_full_session(root, "20260601_01_alpha")
    page, fake = _make_page(root, _qapp)

    def rendered_indices():
        return sorted(args[1] for fn, args, _kw in fake.submitted
                      if fn is session_worker.render_session_plot)

    page.select_session(os.path.abspath(d))
    _qapp.processEvents()
    active = page._tabs.currentIndex()
    # Exactly the active plot was dispatched.
    assert rendered_indices() == [active]

    # Switch to a different tab -> that plot renders now (and only now).
    other = (active + 1) % len(page._viz_tabs)
    page._tabs.setCurrentIndex(other)
    _qapp.processEvents()
    assert rendered_indices() == sorted([active, other])
    page.deleteLater()


def test_label_save_round_trip_through_ui(tmp_path, _qapp):
    root = str(tmp_path)
    d = _make_full_session(root, "20260601_01_alpha")
    page, _ = _make_page(root, _qapp)
    page.select_session(os.path.abspath(d))

    page._rating_buttons["best"].setChecked(True)
    page._tags_input.setText("clean, 33hz")
    page._notes_input.setPlainText("the keeper")
    page._on_save_label()

    saved = load_label(d)
    assert saved.rating == "best"
    assert saved.tags == ["clean", "33hz"]
    assert saved.notes == "the keeper"
    # The list cell now carries the BEST badge.
    assert "BEST" in page._list.item(0).text()
    page.deleteLater()


def test_label_save_new_ratings_show_badge(tmp_path, _qapp):
    """The extended ratings (test/useless/ignore) save + badge in the list."""
    root = str(tmp_path)
    d = _make_full_session(root, "20260601_01_alpha")
    page, _ = _make_page(root, _qapp)
    page.select_session(os.path.abspath(d))

    page._rating_buttons["useless"].setChecked(True)
    page._on_save_label()
    assert load_label(d).rating == "useless"
    assert "USELESS" in page._list.item(0).text()
    page.deleteLater()


def test_filter_by_rating(tmp_path, _qapp):
    root = str(tmp_path)
    da = _make_full_session(root, "20260601_01_alpha")
    _make_full_session(root, "20260601_02_beta")
    page, _ = _make_page(root, _qapp)

    # Mark alpha as best via the UI, then filter to "Best".
    page.select_session(os.path.abspath(da))
    page._rating_buttons["best"].setChecked(True)
    page._on_save_label()

    page._filter.setCurrentText("Best")
    assert page._list.count() == 1
    assert "alpha" in page._list.item(0).text()
    page.deleteLater()


def test_filter_new_rating_test(tmp_path, _qapp):
    """The new 'Test' filter keeps only test-rated sessions."""
    root = str(tmp_path)
    da = _make_full_session(root, "20260601_01_alpha")
    _make_full_session(root, "20260601_02_beta")
    page, _ = _make_page(root, _qapp)

    page.select_session(os.path.abspath(da))
    page._rating_buttons["test"].setChecked(True)
    page._on_save_label()

    page._filter.setCurrentText("Test")
    assert page._list.count() == 1
    assert "alpha" in page._list.item(0).text()
    page.deleteLater()


def test_shutdown_shuts_executor(tmp_path, _qapp):
    root = str(tmp_path)
    _make_full_session(root, "20260601_01_alpha")
    # When WE own the executor, shutdown() shuts it down; an injected fake is
    # left alone (owns_executor=False).
    page = SessionsPage(sessions_dir=root)
    page._ensure_executor()  # force-create the real pool
    real = page._executor
    assert real is not None
    page.shutdown()
    assert page._executor is None
    real.shutdown(wait=False)  # idempotent; already shut down
    page.deleteLater()


def test_empty_sessions_dir_lists_nothing(tmp_path, _qapp):
    page = SessionsPage(sessions_dir=str(tmp_path / "nope"), executor=_SyncExecutor())
    assert page._list.count() == 0
    page.deleteLater()
