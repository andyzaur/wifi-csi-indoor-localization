"""Tests for csi_gui.ui.monitor_panel.

The numeric core (MonitorState) is Qt-free and tested directly with injected
BoardStats / ClapEvent / PositionState-shaped objects + a fake clock. A separate
offscreen smoke test constructs the QWidget and feeds it events.
"""

import os
import time
from types import SimpleNamespace

import pytest

from csi_gui.ui.monitor_panel import (
    MonitorState,
    PRESENT_AGE_S,
    RX_BOARDS,
    _fmt_elapsed,
)


class FakeClock:
    def __init__(self, start=0.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += float(dt)


def _board_stats(board_id, hz, age_s):
    # Mirrors csi_collector.BoardStats fields the monitor reads.
    return SimpleNamespace(board_id=board_id, hz=hz, age_s=age_s, rssi=-50,
                           last_rx_seq=0, seq_gaps=0, total=0)


def _position(detected, gx=None, gy=None):
    return SimpleNamespace(detected=detected, x_cm=gx, y_cm=gy,
                           grid_x_cm=gx, grid_y_cm=gy, n_markers=1,
                           method="both", right_center=None, left_center=None,
                           fps=0.0)


def _clap(name, wall_time_s):
    return SimpleNamespace(wall_time_s=wall_time_s, event=0 if name == "start" else 1,
                           event_name=name, seq=1, timestamp_us=0)


def test_board_row_present_when_fresh_and_moving():
    clock = FakeClock()
    st = MonitorState(clock=clock)
    st.on_board_stats({1: _board_stats(1, hz=33.0, age_s=0.02)})
    row = st.board_row(1)
    assert row["present"] is True
    assert row["hz"] == 33.0
    assert row["age_s"] == 0.02


def test_board_row_absent_when_never_seen():
    st = MonitorState()
    row = st.board_row(4)
    assert row["present"] is False
    assert row["hz"] == 0.0
    assert row["age_s"] == float("inf")


def test_board_row_absent_when_stale_age():
    # A high reported age_s (board dropped out) -> not present even if hz>0.
    st = MonitorState()
    st.on_board_stats({5: _board_stats(5, hz=30.0, age_s=PRESENT_AGE_S + 1.0)})
    assert st.board_row(5)["present"] is False


def test_detection_pct_rolling():
    st = MonitorState()
    assert st.detection_pct() == 0.0  # no frames yet
    for _ in range(3):
        st.on_position(_position(True, 10.0, 20.0))
    st.on_position(_position(False))
    # 3 of 4 detected = 75%.
    assert st.detection_pct() == pytest.approx(75.0)


def test_current_cell_tracks_last_detected():
    st = MonitorState()
    st.on_position(_position(True, 10.0, 20.0))
    st.on_position(_position(True, 30.0, 40.0))
    st.on_position(_position(False))  # non-detection must not clobber the cell
    assert st.current_cell == (30.0, 40.0)


def test_csi_total_counts():
    st = MonitorState()
    for _ in range(5):
        st.on_csi(object())
    assert st.csi_total == 5


def test_on_csi_accepts_no_arg():
    # The worker callback may be bound directly with no event payload.
    st = MonitorState()
    st.on_csi()
    st.on_csi()
    assert st.csi_total == 2


def test_high_rate_ingest_is_threadsafe_and_sampled():
    """FIX B: many on_csi/on_position from worker threads accumulate correctly.

    Production wires on_csi (~100/s) + on_position (~22/s) STRAIGHT to the
    thread-safe MonitorState (no per-event GUI signal). Hammering them from
    several threads must leave a consistent total + detection roll that a later
    sampled read reflects.
    """
    import threading as _t

    st = MonitorState(detect_window=100_000)
    n_threads = 4
    per_thread = 500

    def pump():
        for _ in range(per_thread):
            st.on_csi()
            st.on_position(_position(True, 50.0, 50.0))

    threads = [_t.Thread(target=pump) for _ in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    total = n_threads * per_thread
    # Sampled reads (what the GUI timer does) reflect every event, no loss.
    assert st.csi_total == total
    assert st.detection_pct() == pytest.approx(100.0)
    assert st.current_cell == (50.0, 50.0)


def test_elapsed_uses_injected_clock():
    clock = FakeClock(start=100.0)
    st = MonitorState(clock=clock)
    assert st.elapsed_s() == 0.0  # before mark_started
    st.mark_started()
    clock.advance(12.5)
    assert st.elapsed_s() == pytest.approx(12.5)


def test_clap_tracks_last_event_and_start_anchor(monkeypatch):
    st = MonitorState()
    # Freeze wall clock for since_start_clap_s.
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    assert st.since_start_clap_s() is None
    st.on_clap(_clap("start", wall_time_s=950.0))
    assert st.last_clap_name == "start"
    assert st.since_start_clap_s() == pytest.approx(50.0)
    st.on_clap(_clap("stop", wall_time_s=990.0))
    assert st.last_clap_name == "stop"
    # since_start still measured from the START anchor.
    assert st.since_start_clap_s() == pytest.approx(50.0)


def test_fmt_elapsed():
    assert _fmt_elapsed(0) == "00:00"
    assert _fmt_elapsed(65) == "01:05"
    assert _fmt_elapsed(-3) == "00:00"


def test_rx_boards_default():
    assert RX_BOARDS == (1, 4, 5)


# ---------------------------------------------------------------------------
# Offscreen QWidget smoke test.
# ---------------------------------------------------------------------------
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_monitor_panel_widget_ingests_and_renders():
    from csi_gui.ui.monitor_panel import MonitorPanel

    panel = MonitorPanel()
    panel.begin()  # fresh state + repaint tick

    panel.ingest_board_stats({1: _board_stats(1, hz=33.0, age_s=0.02)})
    panel.ingest_csi(object())
    panel.ingest_position(_position(True, 10.0, 20.0))
    panel.ingest_clap(_clap("start", wall_time_s=time.time()))
    panel.refresh()

    # Board 1 dot should read present.
    assert panel._board_rows[1]["dot"].property("pf") == "ok"
    # Clapper banner reflects START.
    assert "START" in panel._clap.text()
    assert panel._clap.property("clap") == "start"
    # CSI total + cell rendered.
    assert "CSI rows: 1" in panel._totals.text()
    assert "(10, 20)" in panel._camera.text()

    panel.end()
    panel.deleteLater()
