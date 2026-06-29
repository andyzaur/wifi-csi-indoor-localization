"""CLI byte-parity test for csi_collector.CsiCollector.

Strategy (per the architecture plan):
  * Drive the PRE-REFACTOR original main() (committed verbatim as the fixture
    tests/_csi_collector_golden_ref.py) with a scripted sequence of synthetic
    UDP datagrams in the real wire format and a deterministic clock -> this is
    the GOLDEN (stdout bytes + csi.csv bytes + clap.csv bytes).
  * Drive the refactored CsiCollector (on_log=None, default CLI behavior) with
    the SAME datagrams and the SAME clock.
  * Assert the two are byte-identical.

The clock is controlled (time.time monkeypatched to a fixed counter) in BOTH
runs so the wall_time_s columns and the "t=..." clap lines compare exactly.
"""
import contextlib
import importlib
import io
import os
import socket as socket_mod
import struct
import sys

import pytest

# --- wire-format fixture builders ---------------------------------------
PORT = 5500
CSI_HDR_FMT = "<B6BbBIHH"
CLAP_FMT = "<BBIH"
CLAP_MAGIC = 0xCA


def make_csi(board_id, rssi, channel, rx_seq, ts_us, csi_len=128, fill=0):
    hdr = struct.pack(CSI_HDR_FMT, board_id, 0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6,
                      rssi, channel, ts_us, rx_seq, csi_len)
    payload = bytes((i + fill) % 256 for i in range(csi_len))
    return hdr + payload


def make_clap(event, ts_us, seq):
    return struct.pack(CLAP_FMT, CLAP_MAGIC, event, ts_us, seq)


def build_scenario():
    """A scripted list of (datagram_or_None, addr).

    None => the fake socket raises socket.timeout (the silent-iteration path).
    Exercises: multi-board CSI, clap + burst dedup, short legacy csi_len, the
    uint16 seq wrap (65535 -> 0), a malformed packet, an unknown packet shape,
    a timeout, the 5s summary, the 2s flush, and the MALFORMED summary suffix.
    """
    addr = ("192.168.4.50", 5500)
    s = [
        (make_csi(1, -40, 11, 100, 1000, 128, fill=1), addr),
        (make_csi(4, -55, 6, 200, 2000, 128, fill=2), addr),
        (make_clap(0, 5000, 1), addr),
        (make_clap(0, 5000, 1), addr),                      # dup -> ignored
        (make_csi(5, -60, 1, 65535, 3000, 64, fill=3), addr),
        (make_csi(5, -61, 1, 0, 3100, 64, fill=4), addr),   # 65535 -> 0 wrap
        (struct.pack(CSI_HDR_FMT, 7, 1, 2, 3, 4, 5, 6, -30, 11, 99, 1, 200)
         + bytes(128), addr),                               # malformed
        (b"\xff\x00\x01\x02", addr),                        # unknown shape
        (None, addr),                                       # timeout
        (make_clap(1, 6000, 2), addr),
        (make_csi(1, -41, 11, 101, 1100, 128, fill=5), addr),
    ]
    return s


class FakeClock:
    """Deterministic clock: returns a fixed pre-computed list of values, then
    holds the last value. Shared sequence guarantees both runs see the exact
    same wall_time_s for every packet."""
    def __init__(self, times):
        self._times = times
        self._i = 0

    def __call__(self):
        if self._i < len(self._times):
            v = self._times[self._i]
            self._i += 1
            return v
        return self._times[-1] if self._times else 0.0


def build_time_sequence(scenario):
    """Exact ordered list of time.time() return values.

    Call order in the loop:
      1. last_report = time.time()
      2. last_flush  = time.time()
      then per iteration:
        - if recvfrom succeeds: wall_t = time.time()   (1 call)
        - now = time.time()                            (1 call)
    The `now` value advances by 3s/iteration so the 5s summary and 2s flush
    boundaries are crossed repeatedly and print deterministically.
    """
    times = []
    base = 1000.0
    times.append(base)   # last_report
    times.append(base)   # last_flush
    t = base
    for (dgram, _addr) in scenario:
        if dgram is not None:
            t += 0.001
            times.append(t)   # wall_t
        t += 3.0
        times.append(t)       # now
    return times


class FakeSocket:
    """Replays the scenario through recvfrom; raises KeyboardInterrupt at the
    end to terminate the loop exactly as a Ctrl-C did in the original CLI."""
    def __init__(self, scenario):
        self._scenario = list(scenario)
        self._i = 0
        self.timeout = None

    def setsockopt(self, *a, **k):
        pass

    def bind(self, *a, **k):
        pass

    def settimeout(self, t):
        self.timeout = t

    def recvfrom(self, bufsize):
        if self._i >= len(self._scenario):
            raise KeyboardInterrupt
        dgram, addr = self._scenario[self._i]
        self._i += 1
        if dgram is None:
            raise socket_mod.timeout()
        return dgram, addr

    def close(self):
        pass


def _run_original(monkeypatch, tmp_path, session_name):
    """Run the committed pre-refactor main() and return (stdout, csi_bytes, clap_bytes)."""
    golden = importlib.import_module("tests._csi_collector_golden_ref")
    import time as time_mod

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    scenario = build_scenario()
    times = build_time_sequence(scenario)
    clock = FakeClock(times)
    fake_sock = FakeSocket(scenario)

    monkeypatch.setattr(time_mod, "time", clock)
    monkeypatch.setattr(socket_mod, "socket", lambda *a, **k: fake_sock)
    monkeypatch.setattr(sys, "argv", ["csi_collector.py", "--session", session_name])

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        golden.main()

    sess = tmp_path / "sessions" / session_name
    csi = (sess / "csi.csv").read_bytes()
    clap = (sess / "clap.csv").read_bytes()
    return captured.getvalue(), csi, clap


def _run_refactored(monkeypatch, tmp_path, session_name):
    """Run the refactored CsiCollector via run_forever() (CLI path, on_log=None)."""
    import csi_collector
    import time as time_mod

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    scenario = build_scenario()
    times = build_time_sequence(scenario)
    clock = FakeClock(times)
    fake_sock = FakeSocket(scenario)

    monkeypatch.setattr(time_mod, "time", clock)
    monkeypatch.setattr(socket_mod, "socket", lambda *a, **k: fake_sock)
    monkeypatch.setattr(sys, "argv", ["csi_collector.py", "--session", session_name])

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        csi_collector.main()

    sess = tmp_path / "sessions" / session_name
    csi = (sess / "csi.csv").read_bytes()
    clap = (sess / "clap.csv").read_bytes()
    return captured.getvalue(), csi, clap


def test_cli_byte_identical_with_csv(monkeypatch, tmp_path):
    g_out, g_csi, g_clap = _run_original(monkeypatch, tmp_path / "g", "sess")
    r_out, r_csi, r_clap = _run_refactored(monkeypatch, tmp_path / "r", "sess")

    assert r_out == g_out, "stdout differs from pre-refactor golden"
    assert r_csi == g_csi, "csi.csv bytes differ from pre-refactor golden"
    assert r_clap == g_clap, "clap.csv bytes differ from pre-refactor golden"


def test_cli_stdout_exercises_summary_and_clap(monkeypatch, tmp_path):
    """Sanity: the golden actually drove the rich paths we care about, so the
    byte-parity assertion above is meaningful (not comparing two empty runs)."""
    g_out, _, _ = _run_original(monkeypatch, tmp_path / "g", "sess")
    assert "Session: sess" in g_out
    assert "summary:" in g_out
    assert "CLAP [START]" in g_out and "CLAP [STOP]" in g_out
    assert "MALFORMED: 1" in g_out
    assert "Malformed CSI packet" in g_out
    assert "Unknown packet" in g_out
    assert "NO CSI PACKETS RECEIVED" in g_out
    assert g_out.rstrip().endswith("Session saved to: sessions/sess")


def test_cli_quiet_flag_parity(monkeypatch, tmp_path):
    """--quiet must remain byte-identical between original and refactored."""
    import time as time_mod
    import csi_collector
    golden = importlib.import_module("tests._csi_collector_golden_ref")

    def run(entry_main, base):
        base.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(base)
        scenario = build_scenario()
        times = build_time_sequence(scenario)
        monkeypatch.setattr(time_mod, "time", FakeClock(times))
        monkeypatch.setattr(socket_mod, "socket",
                            lambda *a, **k: FakeSocket(scenario))
        monkeypatch.setattr(sys, "argv",
                            ["csi_collector.py", "--session", "q", "--quiet"])
        cap = io.StringIO()
        with contextlib.redirect_stdout(cap):
            entry_main()
        sess = base / "sessions" / "q"
        return cap.getvalue(), (sess / "csi.csv").read_bytes()

    g_out, g_csi = run(golden.main, tmp_path / "g")
    r_out, r_csi = run(csi_collector.main, tmp_path / "r")
    assert r_out == g_out
    assert r_csi == g_csi
    # quiet really suppressed the per-packet print:
    assert "Board 1 | RSSI" not in g_out


def test_cli_no_csv_flag_parity(monkeypatch, tmp_path):
    """--no-csv must remain byte-identical (no CSVs written, print-only)."""
    import time as time_mod
    import csi_collector
    golden = importlib.import_module("tests._csi_collector_golden_ref")

    def run(entry_main, base):
        base.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(base)
        scenario = build_scenario()
        times = build_time_sequence(scenario)
        monkeypatch.setattr(time_mod, "time", FakeClock(times))
        monkeypatch.setattr(socket_mod, "socket",
                            lambda *a, **k: FakeSocket(scenario))
        monkeypatch.setattr(sys, "argv", ["csi_collector.py", "--no-csv"])
        cap = io.StringIO()
        with contextlib.redirect_stdout(cap):
            entry_main()
        return cap.getvalue(), (base / "sessions").exists()

    g_out, g_sessions = run(golden.main, tmp_path / "g")
    r_out, r_sessions = run(csi_collector.main, tmp_path / "r")
    assert r_out == g_out
    assert g_sessions is False and r_sessions is False  # no session dir created
