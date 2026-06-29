"""Callback-surface tests for csi_collector.CsiCollector.

These assert the additive (GUI-facing) behavior layered on top of the verbatim
recv loop: on_csi / on_clap per-packet fire, on_board_stats hz + seq_gaps
(including the uint16 65535->0 wrap and per-board independence), and the
thread-safe rate_hz / last_clap_ts readers.

The recv loop runs on the daemon thread spawned by start(); it is the SOLE
consumer of the patched clock, so the values are deterministic. The fake socket
sets a threading.Event once the scripted scenario is exhausted, after which it
blocks (raising socket.timeout) until the test calls stop().
"""
import socket as socket_mod
import struct
import threading

import pytest

import csi_collector
from csi_collector import (CsiCollector, CsiEvent, ClapEvent, BoardStats,
                           CSI_HDR_FMT, CLAP_FMT, CLAP_MAGIC)


def make_csi(board_id, rssi, channel, rx_seq, ts_us, csi_len=128, fill=0):
    hdr = struct.pack(CSI_HDR_FMT, board_id, 0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6,
                      rssi, channel, ts_us, rx_seq, csi_len)
    payload = bytes((i + fill) % 256 for i in range(csi_len))
    return hdr + payload


def make_clap(event, ts_us, seq):
    return struct.pack(CLAP_FMT, CLAP_MAGIC, event, ts_us, seq)


class ScriptedClock:
    """Deterministic clock from a fixed list, then holds the last value."""
    def __init__(self, times):
        self._times = times
        self._i = 0

    def __call__(self):
        if self._i < len(self._times):
            v = self._times[self._i]
            self._i += 1
            return v
        return self._times[-1] if self._times else 0.0


class ScriptedSocket:
    """Replays datagrams; after the script is exhausted, sets `done` and then
    raises socket.timeout on every further recvfrom so the loop idles until the
    test calls stop()."""
    def __init__(self, scenario, done_event):
        self._scenario = list(scenario)
        self._i = 0
        self._done = done_event
        self.timeout = None

    def setsockopt(self, *a, **k):
        pass

    def bind(self, *a, **k):
        pass

    def settimeout(self, t):
        self.timeout = t

    def recvfrom(self, bufsize):
        if self._i >= len(self._scenario):
            self._done.set()
            raise socket_mod.timeout()
        dgram, addr = self._scenario[self._i]
        self._i += 1
        if dgram is None:
            raise socket_mod.timeout()
        return dgram, addr

    def close(self):
        pass


def _drive(monkeypatch, scenario, times, **kwargs):
    """Run a CsiCollector over the scripted scenario; return the collector
    after the scenario is consumed and the thread is stopped."""
    import time as time_mod
    done = threading.Event()
    sock = ScriptedSocket(scenario, done)
    monkeypatch.setattr(time_mod, "time", ScriptedClock(times))
    monkeypatch.setattr(socket_mod, "socket", lambda *a, **k: sock)

    c = CsiCollector(write_csv=False, **kwargs)
    c.start()
    assert done.wait(timeout=5.0), "scenario was not consumed in time"
    c.stop(join_timeout=5.0)
    return c


def _times_for(scenario, step=0.01, base=1000.0, fire_after=True):
    """Build the time.time() sequence: 2 pre-loop calls, then per iteration a
    wall_t (only on recv) + a `now`.

    With the default small `step`, every scripted packet lands inside ONE
    stats window (so accumulated seq_gaps land in a single on_board_stats
    batch). When `fire_after` is set, one extra post-scenario idle iteration
    jumps the clock past stats_interval (+10s) so exactly that one batch fires
    with the whole scenario accumulated, then the clock holds.
    """
    times = [base, base]
    t = base
    for (d, _a) in scenario:
        if d is not None:
            t += 0.001
            times.append(t)
        t += step
        times.append(t)
    if fire_after:
        # one idle loop whose `now` crosses any reasonable stats_interval
        t += 10.0
        times.append(t)        # now (fires on_board_stats once)
    # remaining idle loops hold the last value (harmless, below threshold)
    times.extend([t] * 50)
    return times


# --------------------------------------------------------------------------
def test_on_csi_and_on_clap_counts(monkeypatch):
    addr = ("10.0.0.1", 5500)
    scenario = [
        (make_csi(1, -40, 11, 1, 100), addr),
        (make_csi(1, -41, 11, 2, 200), addr),
        (make_clap(0, 5000, 1), addr),
        (make_clap(0, 5000, 1), addr),   # burst dup -> NOT a second on_clap
        (make_csi(4, -50, 6, 1, 300), addr),
        (make_clap(1, 6000, 2), addr),
    ]
    csi_events = []
    clap_events = []
    _drive(monkeypatch, scenario, _times_for(scenario, step=0.01),
           on_csi=csi_events.append, on_clap=clap_events.append)

    assert len(csi_events) == 3
    assert all(isinstance(e, CsiEvent) for e in csi_events)
    assert [e.board_id for e in csi_events] == [1, 1, 4]
    assert csi_events[0].mac == "a1:b2:c3:d4:e5:f6"
    assert len(csi_events[0].csi) == 128

    # only 2 distinct claps despite the 3-packet burst (start x2 + stop)
    assert len(clap_events) == 2
    assert [e.event_name for e in clap_events] == ["start", "stop"]
    assert all(isinstance(e, ClapEvent) for e in clap_events)


def test_csi_signed_int8_payload(monkeypatch):
    addr = ("10.0.0.1", 5500)
    # fill so payload[0]=200 -> signed -56; verify CsiEvent.csi matches CSV math
    scenario = [(make_csi(1, -40, 11, 1, 100, csi_len=128, fill=200), addr)]
    got = []
    _drive(monkeypatch, scenario, _times_for(scenario, step=0.01),
           on_csi=got.append)
    assert len(got) == 1
    raw0 = 200  # (0 + 200) % 256
    assert got[0].csi[0] == raw0 - 256  # -56


def test_board_stats_hz_and_total(monkeypatch):
    addr = ("10.0.0.1", 5500)
    # 3 packets for board 1, all within one stats window; the next `now` step
    # crosses stats_interval and fires on_board_stats.
    scenario = [
        (make_csi(1, -40, 11, 1, 100), addr),
        (make_csi(1, -42, 11, 2, 200), addr),
        (make_csi(1, -44, 11, 3, 300), addr),
    ]
    stats_batches = []
    _drive(monkeypatch, scenario, _times_for(scenario),
           on_board_stats=stats_batches.append, stats_interval=1.0)

    assert stats_batches, "on_board_stats never fired"
    first = stats_batches[0]
    assert 1 in first
    bs = first[1]
    assert isinstance(bs, BoardStats)
    assert bs.board_id == 1
    assert bs.total == 3
    assert bs.last_rx_seq == 3
    assert bs.rssi == -44       # last rssi seen
    # contiguous seqs 1,2,3 -> no gaps
    assert bs.seq_gaps == 0
    assert bs.hz > 0


def test_seq_gaps_counts_missing_frames(monkeypatch):
    addr = ("10.0.0.1", 5500)
    # seqs 1, 5 -> gap = (5 - 1 - 1) % 65536 = 3 missing frames
    scenario = [
        (make_csi(1, -40, 11, 1, 100), addr),
        (make_csi(1, -40, 11, 5, 200), addr),
    ]
    batches = []
    _drive(monkeypatch, scenario, _times_for(scenario),
           on_board_stats=batches.append, stats_interval=1.0)
    assert batches
    assert batches[0][1].seq_gaps == 3


def test_seq_gaps_uint16_wrap(monkeypatch):
    addr = ("10.0.0.1", 5500)
    # 65535 -> 0 is the contiguous next frame: gap = (0 - 65535 - 1) % 65536 = 0
    scenario = [
        (make_csi(1, -40, 11, 65535, 100), addr),
        (make_csi(1, -40, 11, 0, 200), addr),
    ]
    batches = []
    _drive(monkeypatch, scenario, _times_for(scenario),
           on_board_stats=batches.append, stats_interval=1.0)
    assert batches
    assert batches[0][1].seq_gaps == 0      # wrap, not a 65535-frame jump

    # 65534 -> 1 across the wrap drops {65535, 0} -> 2 missing frames
    scenario2 = [
        (make_csi(2, -40, 11, 65534, 100), addr),
        (make_csi(2, -40, 11, 1, 200), addr),
    ]
    batches2 = []
    _drive(monkeypatch, scenario2, _times_for(scenario2),
           on_board_stats=batches2.append, stats_interval=1.0)
    assert batches2
    assert batches2[0][2].seq_gaps == 2


def test_seq_gaps_per_board_independent(monkeypatch):
    addr = ("10.0.0.1", 5500)
    # interleave two boards; gaps must NOT bleed across board ids.
    scenario = [
        (make_csi(1, -40, 11, 1, 100), addr),
        (make_csi(4, -50, 6, 10, 100), addr),
        (make_csi(1, -40, 11, 2, 200), addr),     # board 1 contiguous: gap 0
        (make_csi(4, -50, 6, 14, 200), addr),     # board 4: 10->14 gap 3
    ]
    batches = []
    _drive(monkeypatch, scenario, _times_for(scenario),
           on_board_stats=batches.append, stats_interval=1.0)
    assert batches
    # find a batch that has both boards' final accumulated gaps
    last = batches[-1]
    # accumulate seq_gaps across all batches per board (windows reset gaps)
    total_gaps = {}
    for b in batches:
        for bid, bs in b.items():
            total_gaps[bid] = total_gaps.get(bid, 0) + bs.seq_gaps
    assert total_gaps.get(1, 0) == 0
    assert total_gaps.get(4, 0) == 3


def test_rate_hz_thread_safe_read(monkeypatch):
    addr = ("10.0.0.1", 5500)
    scenario = [
        (make_csi(1, -40, 11, 1, 100), addr),
        (make_csi(1, -40, 11, 2, 200), addr),
    ]
    c = _drive(monkeypatch, scenario, _times_for(scenario),
               stats_interval=1.0)
    # rate_hz is O(1) and returns a float for a seen board, 0.0 otherwise
    assert isinstance(c.rate_hz(1), float)
    assert c.rate_hz(1) > 0.0
    assert c.rate_hz(999) == 0.0


def test_last_clap_ts_thread_safe(monkeypatch):
    addr = ("10.0.0.1", 5500)
    assert CsiCollector(write_csv=False).last_clap_ts() is None
    scenario = [
        (make_clap(0, 5000, 1), addr),
        (make_csi(1, -40, 11, 1, 100), addr),
        (make_clap(1, 6000, 2), addr),
    ]
    c = _drive(monkeypatch, scenario, _times_for(scenario, step=0.01))
    ts = c.last_clap_ts()
    assert ts is not None
    assert isinstance(ts, float)


def test_on_log_none_prints(monkeypatch, capsys):
    # on_log=None -> the verbatim string goes to stdout (parity preserved).
    addr = ("10.0.0.1", 5500)
    scenario = [(make_csi(1, -40, 11, 7, 100), addr)]
    _drive(monkeypatch, scenario, _times_for(scenario, step=0.01))
    out = capsys.readouterr().out
    assert "Board 1 | RSSI  -40 | ch 11 | seq     7" in out


def test_on_log_callback_intercepts(monkeypatch):
    # on_log set -> nothing printed; the callback receives the SAME strings.
    addr = ("10.0.0.1", 5500)
    scenario = [(make_csi(1, -40, 11, 7, 100), addr)]
    logs = []
    _drive(monkeypatch, scenario, _times_for(scenario, step=0.01),
           on_log=logs.append)
    assert any("Board 1 | RSSI  -40 | ch 11 | seq     7" in m for m in logs)


def test_start_stop_idempotent(monkeypatch):
    addr = ("10.0.0.1", 5500)
    scenario = [(make_csi(1, -40, 11, 1, 100), addr)]
    import time as time_mod
    done = threading.Event()
    sock = ScriptedSocket(scenario, done)
    monkeypatch.setattr(time_mod, "time", ScriptedClock(_times_for(scenario)))
    monkeypatch.setattr(socket_mod, "socket", lambda *a, **k: sock)

    c = CsiCollector(write_csv=False)
    c.start()
    c.start()  # second start is a no-op
    assert done.wait(timeout=5.0)
    c.stop()
    c.stop()   # second stop is a no-op (must not raise)
