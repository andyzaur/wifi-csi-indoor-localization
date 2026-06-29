"""Tests for csi_gui.preflight.engine — the Qt-free Checks + board-rate gate.

The board listener wraps a real CsiCollector; here we stub it so no UDP socket is
bound. We assert: the check ordering/criticality, the board-rate status mapping
(GREEN all>5Hz, YELLOW at 0, RED for a silent board), EADDRINUSE degrading to a
YELLOW 'port :5500 busy', and the all_critical_green gate.
"""

import pytest

engine = pytest.importorskip("csi_gui.preflight.engine")
from csi_gui.preflight.probes import GREEN, RED, YELLOW  # noqa: E402


class _FakeCollector:
    """Stand-in for CsiCollector: records start/stop, serves canned rates."""

    def __init__(self, rates=None, raise_on_start=None, **kwargs):
        self._rates = rates or {}
        self._raise = raise_on_start
        self.started = False
        self.stopped = False

    def start(self):
        if self._raise is not None:
            raise self._raise
        self.started = True

    def stop(self):
        self.stopped = True

    def rate_hz(self, board_id):
        return self._rates.get(board_id, 0.0)


def _engine_with(monkeypatch, rates=None, raise_on_start=None):
    fake = _FakeCollector(rates=rates, raise_on_start=raise_on_start)
    monkeypatch.setattr(engine, "CsiCollector", lambda **kw: fake)
    eng = engine.PreflightEngine(camera_url_getter=lambda: "http://x/video")
    return eng, fake


# ---------------------------------------------------------------------------
# Check list shape.
# ---------------------------------------------------------------------------
def test_checks_have_expected_ids_and_criticality():
    eng = engine.PreflightEngine()
    ids = [c.id for c in eng.checks]
    assert ids == [
        engine.ETHERNET, engine.WIFI, engine.STATIC_IP, engine.TX,
        engine.BOARDS, engine.IPROXY, engine.CAMERA, engine.FLOOR,
    ]
    crit = set(eng.critical_ids())
    assert engine.ETHERNET in crit and engine.CAMERA in crit
    # iproxy + floor are intentionally non-critical.
    assert engine.IPROXY not in crit
    assert engine.FLOOR not in crit


# ---------------------------------------------------------------------------
# Board-rate gate.
# ---------------------------------------------------------------------------
def test_board_rate_green_when_all_above_threshold(monkeypatch):
    eng, fake = _engine_with(monkeypatch, rates={1: 30.0, 4: 28.0, 5: 31.0})
    assert eng.start_board_listener() is True
    res = eng.board_rate_result()
    assert res.status == GREEN


def test_board_rate_yellow_when_all_zero(monkeypatch):
    eng, fake = _engine_with(monkeypatch, rates={1: 0.0, 4: 0.0, 5: 0.0})
    eng.start_board_listener()
    res = eng.board_rate_result()
    assert res.status == YELLOW


def test_board_rate_red_when_one_board_silent(monkeypatch):
    eng, fake = _engine_with(monkeypatch, rates={1: 30.0, 4: 0.0, 5: 31.0})
    eng.start_board_listener()
    res = eng.board_rate_result()
    assert res.status == RED
    assert "4" in res.detail


def test_board_rate_yellow_before_listener_started(monkeypatch):
    eng, _ = _engine_with(monkeypatch, rates={})
    # Not started yet.
    res = eng.board_rate_result()
    assert res.status == YELLOW
    assert "not started" in res.detail


def test_eaddrinuse_degrades_to_yellow_port_busy(monkeypatch):
    err = OSError()
    err.errno = 48  # EADDRINUSE
    eng, fake = _engine_with(monkeypatch, raise_on_start=err)
    assert eng.start_board_listener() is False
    res = eng.board_rate_result()
    assert res.status == YELLOW
    assert ":5500" in res.detail


def test_stop_board_listener_calls_collector_stop(monkeypatch):
    eng, fake = _engine_with(monkeypatch, rates={1: 10.0, 4: 10.0, 5: 10.0})
    eng.start_board_listener()
    assert eng.board_listener_running is True
    eng.stop_board_listener()
    assert fake.stopped is True
    assert eng.board_listener_running is False


# ---------------------------------------------------------------------------
# READY gate.
# ---------------------------------------------------------------------------
def test_all_critical_green_requires_every_critical(monkeypatch):
    eng = engine.PreflightEngine()
    # All critical green, non-critical missing -> READY.
    statuses = {c.id: GREEN for c in eng.checks if c.critical}
    assert eng.all_critical_green(statuses) is True

    # One critical RED -> not ready.
    statuses[engine.TX] = RED
    assert eng.all_critical_green(statuses) is False

    # Missing critical entry -> not ready.
    del statuses[engine.TX]
    assert eng.all_critical_green(statuses) is False
