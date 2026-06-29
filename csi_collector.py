#!/usr/bin/env python3
"""CSI UDP Collector — receives CSI packets from ESP32-C6 RX boards and CLAP
events from the spare ESP32-C6 "clapperboard" for time-syncing with the camera.

Usage:
    python3 csi_collector.py                       # default: timestamped session
    python3 csi_collector.py --session walk_01     # custom session name
    python3 csi_collector.py --no-csv              # print only, no CSV

Outputs (in ./sessions/<session_name>/):
    csi.csv     One row per CSI packet (header + 128 csi_data columns)
    clap.csv    One row per clapper button press (start/stop boundaries)

Both CSVs use wall_time_s (Mac wall clock at receipt) which matches what
aruco_track.py writes for the camera CSV. Join the three by wall_time_s.

Packet formats (both on UDP :5500):

    CSI packet (145 bytes on the wire, first byte = board_id 1..200):
        uint8  board_id, uint8 mac[6], int8 rssi, uint8 channel,
        uint32 timestamp_us, uint16 rx_seq, uint16 csi_len, int8 csi_data[128]

    Clapper packet (8 bytes, first byte = 0xCA magic):
        uint8  magic = 0xCA, uint8 event (0=start, 1=stop, 2=clap),
        uint32 timestamp_us, uint16 seq
    The clapper sends a short BURST of identical packets per press (same
    event+seq) so a single UDP loss doesn't drop a session boundary; the
    collector de-duplicates by (event, seq).

Programmatic use:
    The recv/parse/write loop lives in :class:`CsiCollector`, an importable,
    GUI-agnostic backend. Construct it with optional plain-Python callbacks
    (``on_csi``/``on_clap``/``on_board_stats``/``on_log``) and call
    :meth:`CsiCollector.start` / :meth:`CsiCollector.stop`, or
    :meth:`CsiCollector.run_forever` for the blocking CLI path. With all
    callbacks left ``None`` the behavior is byte-identical to the historical
    CLI (every log line prints the same verbatim string to stdout).
"""

import argparse
import csv
import datetime as dt
import os
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass


PORT = 5500

# CSI packet
CSI_HDR_FMT = "<B6BbBIHH"
CSI_HDR_SIZE = struct.calcsize(CSI_HDR_FMT)  # 17 bytes
CSI_DATA_LEN = 128
CSI_PKT_SIZE = CSI_HDR_SIZE + CSI_DATA_LEN   # 145 bytes

# Clapper packet
CLAP_MAGIC = 0xCA
CLAP_FMT = "<BBIH"
CLAP_SIZE = struct.calcsize(CLAP_FMT)        # 8 bytes
CLAP_EVENT_LABELS = {0: "start", 1: "stop", 2: "clap"}


@dataclass(frozen=True)
class CsiEvent:
    """One parsed CSI packet, handed to ``on_csi`` on the recv thread."""
    wall_time_s: float
    board_id: int
    mac: str
    rssi: int
    channel: int
    timestamp_us: int
    rx_seq: int
    csi_len: int
    csi: list  # the 128 signed int8 values (padded/truncated to CSI_DATA_LEN)


@dataclass(frozen=True)
class ClapEvent:
    """One de-duplicated clapper button press, handed to ``on_clap``."""
    wall_time_s: float
    event: int
    event_name: str
    seq: int
    timestamp_us: int


@dataclass
class BoardStats:
    """Per-board rolling stats, handed to ``on_board_stats`` ~every second."""
    board_id: int
    hz: float
    age_s: float
    rssi: int
    last_rx_seq: int
    seq_gaps: int
    total: int


def parse_csi_packet(data):
    """Validate and parse a CSI packet body.

    The caller has already checked the first byte is a plausible board id
    (1..200, and not the clapper magic 0xCA=202). This guards the *rest* of the
    packet so a corrupt/truncated datagram can't write a garbage CSI row:

    Returns (board_id, mac, rssi, channel, timestamp_us, rx_seq, csi_len,
    csi_bytes) on success, or None if malformed — too short for the header, a
    csi_len that exceeds the 128-byte field, or fewer bytes than csi_len claims.
    """
    if len(data) < CSI_HDR_SIZE:
        return None
    fields = struct.unpack_from(CSI_HDR_FMT, data, 0)
    csi_len = fields[11]
    if csi_len > CSI_DATA_LEN:                       # implausible length claim
        return None
    if len(data) < CSI_HDR_SIZE + csi_len:           # truncated vs claimed len
        return None
    csi_bytes = data[CSI_HDR_SIZE:CSI_HDR_SIZE + csi_len]
    return (fields[0], fields[1:7], fields[7], fields[8],
            fields[9], fields[10], csi_len, bytes(csi_bytes))


def make_session_dir(name=None):
    if name is None:
        name = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("sessions", name)
    os.makedirs(path, exist_ok=True)
    return path, name


class CsiCollector:
    """Importable, controllable CSI/CLAP UDP collector.

    Owns the UDP socket, optional CSV writers, and a daemon recv thread. The
    recv/parse/write loop body is transplanted verbatim from the historical
    CLI ``main()`` so that, with all callbacks left ``None``, output is
    byte-identical to the original script.

    Callbacks are plain Python callables (never Qt). When ``on_log`` is ``None``
    every log line is printed to stdout exactly as the original code did.
    """

    def __init__(self, session_name=None, write_csv=True, quiet=False,
                 on_csi=None, on_clap=None, on_board_stats=None, on_log=None,
                 port=PORT, stats_interval=1.0):
        self._session_name_arg = session_name
        self.write_csv = write_csv
        self.quiet = quiet
        self.on_csi = on_csi
        self.on_clap = on_clap
        self.on_board_stats = on_board_stats
        self.on_log = on_log
        self.port = port
        self.stats_interval = stats_interval

        self._csi_file = None
        self._clap_file = None
        self._csi_writer = None
        self._clap_writer = None
        self._session_path = None
        self._session_name = None
        self._sock = None

        self._thread = None
        self._stop_event = threading.Event()
        self._started = False
        self._stopped = False

        # Thread-safe shared state for rate_hz()/last_clap_ts() readers.
        self._lock = threading.Lock()
        self._rates = {}           # board_id -> hz (latest stats_interval window)
        self._last_clap_ts = None  # wall_time_s of the most recent clap

        # Stats-window bookkeeping (recv thread only).
        self._stat_counts = {}     # board_id -> packets this stats window
        self._stat_last_seq = {}   # board_id -> last rx_seq seen
        self._stat_gaps = {}       # board_id -> seq_gaps this stats window
        self._stat_last_wall = {}  # board_id -> last wall_time_s
        self._stat_last_rssi = {}  # board_id -> last rssi
        self._stat_total = {}      # board_id -> cumulative total packets
        self._stat_window_start = None

    # ── logging ──────────────────────────────────────────────────────────
    def _log(self, msg):
        """Route a log line through ``on_log`` if set, else print verbatim.

        When ``on_log`` is ``None`` this prints the SAME string the original
        CLI printed (preserving byte-identical stdout).
        """
        if self.on_log is not None:
            self.on_log(msg)
        else:
            print(msg)

    # ── lifecycle ────────────────────────────────────────────────────────
    def _setup(self):
        if not self.write_csv:
            return

        self._session_path, self._session_name = make_session_dir(self._session_name_arg)
        csi_path = os.path.join(self._session_path, "csi.csv")
        clap_path = os.path.join(self._session_path, "clap.csv")

        self._csi_file = open(csi_path, "w", newline="")
        self._csi_writer = csv.writer(self._csi_file)
        csi_header = (
            ["wall_time_s", "board_id", "mac", "rssi", "channel",
             "timestamp_us", "rx_seq", "csi_len"]
            + [f"csi_{i}" for i in range(CSI_DATA_LEN)]
        )
        self._csi_writer.writerow(csi_header)

        self._clap_file = open(clap_path, "w", newline="")
        self._clap_writer = csv.writer(self._clap_file)
        self._clap_writer.writerow(
            ["wall_time_s", "event", "event_name", "seq", "timestamp_us"]
        )

        self._log(f"Session: {self._session_name}")
        self._log(f"  CSI  -> {csi_path}")
        self._log(f"  CLAP -> {clap_path}")

    def _open_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # macOS quirk: receiving 255.255.255.255 limited-broadcast UDP into a
        # user-space socket requires SO_BROADCAST and SO_REUSEADDR to be set on the
        # listener too (not just on the sender).
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(("", self.port))
        self._sock = sock
        self._log(f"Listening on UDP port {self.port}...")

        # Timeout so the loop wakes even when no packets arrive — this is what lets
        # the periodic summary fire (and warn) during a silent session instead of
        # blocking forever inside recvfrom.
        sock.settimeout(1.0)

    def start(self):
        """Spawn a daemon recv thread. Idempotent and non-blocking."""
        if self._started:
            return
        self._started = True
        self._setup()
        self._open_socket()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self, join_timeout=2.0):
        """Signal the recv thread to stop, join it, and close CSVs/socket.

        Idempotent.
        """
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=join_timeout)
        self._teardown()

    def _teardown(self):
        if self._csi_file:
            self._csi_file.close()
        if self._clap_file:
            self._clap_file.close()
        if self._sock:
            self._sock.close()
        if self._session_path:
            self._log(f"Session saved to: {self._session_path}")

    def run_forever(self):
        """CLI path: start(), block until KeyboardInterrupt, then stop().

        The recv loop runs on the calling thread here so that a Ctrl-C
        interrupts ``recvfrom`` exactly as it did in the historical CLI, and
        the final ``"\\nStopping..."`` / ``"Session saved to:"`` lines are
        emitted in the same order and from the same place.
        """
        if self._started:
            return
        self._started = True
        self._setup()
        self._open_socket()
        try:
            self._recv_loop()
        except KeyboardInterrupt:
            self._log("\nStopping...")
        finally:
            self.stop()

    # ── thread-safe readers ──────────────────────────────────────────────
    def rate_hz(self, board_id):
        """Latest per-board packet rate (Hz). Thread-safe O(1) read."""
        with self._lock:
            return self._rates.get(board_id, 0.0)

    def last_clap_ts(self):
        """wall_time_s of the most recent clap, or None. Thread-safe."""
        with self._lock:
            return self._last_clap_ts

    @property
    def session_path(self):
        return self._session_path

    @property
    def session_name(self):
        return self._session_name

    # ── recv loop (verbatim transplant of the original main() loop) ──────
    def _recv_loop(self):
        sock = self._sock
        quiet = self.quiet
        csi_writer = self._csi_writer
        clap_writer = self._clap_writer
        csi_file = self._csi_file
        clap_file = self._clap_file

        counts = {}
        clap_count = 0
        malformed = 0
        seen_claps = set()          # (event, seq) already logged — de-dups clap bursts
        last_report = time.time()
        last_flush = time.time()

        # Reuse the already-fetched clock value (no extra time.time() call, so
        # the call ORDER/COUNT — and thus every wall_t — stays byte-identical
        # to the original main()).
        self._stat_window_start = last_flush

        while not self._stop_event.is_set():
            try:
                data, addr = sock.recvfrom(2048)
                wall_t = time.time()
            except socket.timeout:
                data = None

            # Clapper packet? De-dup the burst: only the first (event, seq) is logged.
            if data is not None and len(data) >= CLAP_SIZE and data[0] == CLAP_MAGIC:
                magic, event, ts_us, seq = struct.unpack_from(CLAP_FMT, data, 0)
                if (event, seq) not in seen_claps:
                    seen_claps.add((event, seq))
                    event_name = CLAP_EVENT_LABELS.get(event, f"unknown_{event}")
                    clap_count += 1
                    marker = "=" * 20
                    self._log(f"\n{marker} CLAP [{event_name.upper()}] seq={seq} t={wall_t:.3f} {marker}\n")
                    if clap_writer:
                        clap_writer.writerow([f"{wall_t:.4f}", event, event_name, seq, ts_us])
                        clap_file.flush()
                    with self._lock:
                        self._last_clap_ts = wall_t
                    if self.on_clap is not None:
                        self.on_clap(ClapEvent(
                            wall_time_s=wall_t, event=event, event_name=event_name,
                            seq=seq, timestamp_us=ts_us))
                # else: a duplicate from the same burst — silently ignore

            # CSI packet? Accept any plausible board id (1..200). The clapper
            # magic (0xCA = 202) is handled above, so it can't collide. This is
            # what lets boards with real IDs like 4 and 5 through instead of
            # being dropped as "unknown".
            elif data is not None and 1 <= data[0] <= 200:
                parsed = parse_csi_packet(data)
                if parsed is None:
                    malformed += 1
                    if not quiet:
                        self._log(f"Malformed CSI packet from {addr}: {len(data)} bytes "
                                  f"(board {data[0]}) — dropped")
                else:
                    board_id, mac, rssi, channel, timestamp_us, rx_seq, csi_len, csi_bytes = parsed
                    mac_str = ":".join(f"{b:02x}" for b in mac)

                    counts[board_id] = counts.get(board_id, 0) + 1

                    # Pad/truncate to 128 bytes; store as signed ints
                    padded = list(csi_bytes) + [0] * (CSI_DATA_LEN - len(csi_bytes))
                    padded = padded[:CSI_DATA_LEN]
                    # Convert unsigned bytes to signed int8
                    signed = [b if b < 128 else b - 256 for b in padded]

                    if csi_writer:
                        row = [f"{wall_t:.4f}", board_id, mac_str, rssi, channel,
                               timestamp_us, rx_seq, csi_len] + signed
                        csi_writer.writerow(row)

                    self._update_board_stats(board_id, rx_seq, rssi, wall_t)

                    if self.on_csi is not None:
                        self.on_csi(CsiEvent(
                            wall_time_s=wall_t, board_id=board_id, mac=mac_str,
                            rssi=rssi, channel=channel, timestamp_us=timestamp_us,
                            rx_seq=rx_seq, csi_len=csi_len, csi=signed))

                    if not quiet:
                        mid = csi_len // 2 - 4
                        sample = list(csi_bytes[mid:mid + 8]) if csi_len >= mid + 8 else list(csi_bytes[:8])
                        self._log(f"Board {board_id} | RSSI {rssi:+4d} | ch {channel} | "
                                  f"seq {rx_seq:5d} | len {csi_len:3d} | sample {sample}")

            # Unknown packet shape
            elif data is not None:
                if not quiet:
                    self._log(f"Unknown packet from {addr}: {len(data)} bytes, first byte 0x{data[0]:02x}")

            # ── Periodic maintenance: runs EVERY iteration (and on timeout when no
            # packet arrived), not trapped under the unknown-packet branch. This is
            # the fix for the summary/flush previously never firing during normal
            # CSI flow, which is how the first session went silent unnoticed.
            now = time.time()
            if now - last_report >= 5.0:
                elapsed = now - last_report
                parts = [f"Board {bid}: {cnt} ({cnt/elapsed:.1f}/s)"
                         for bid, cnt in sorted(counts.items())]
                bad = f" | MALFORMED: {malformed}" if malformed else ""
                if parts:
                    self._log(f"\n--- {elapsed:.0f}s summary: {', '.join(parts)} | CLAP: {clap_count}{bad} ---\n")
                else:
                    self._log(f"\n--- {elapsed:.0f}s summary: !! NO CSI PACKETS RECEIVED !! "
                              f"(ethernet unplugged? WiFi associated to CSI_TX? boards powered?) "
                              f"| CLAP: {clap_count}{bad} ---\n")
                counts.clear()
                clap_count = 0
                malformed = 0
                last_report = now

            # Periodic CSV flush
            if csi_file and now - last_flush >= 2.0:
                csi_file.flush()
                last_flush = now

            # Periodic per-board stats callback (~stats_interval, independent of
            # the 5s human-readable summary above).
            self._maybe_emit_board_stats(now)

    # ── per-board stats (callback machinery, additive to original logic) ─
    def _update_board_stats(self, board_id, rx_seq, rssi, wall_t):
        """Accumulate per-board counters for the on_board_stats callback.

        seq_gaps uses rx_seq with uint16 wraparound: gap=(rx_seq-last-1)%65536,
        counting only 0<gap<1000, per board.
        """
        self._stat_counts[board_id] = self._stat_counts.get(board_id, 0) + 1
        self._stat_total[board_id] = self._stat_total.get(board_id, 0) + 1
        last = self._stat_last_seq.get(board_id)
        if last is not None:
            gap = (rx_seq - last - 1) % 65536
            if 0 < gap < 1000:
                self._stat_gaps[board_id] = self._stat_gaps.get(board_id, 0) + gap
        self._stat_last_seq[board_id] = rx_seq
        self._stat_last_rssi[board_id] = rssi
        self._stat_last_wall[board_id] = wall_t

    def _maybe_emit_board_stats(self, now):
        if self._stat_window_start is None:
            self._stat_window_start = now
            return
        elapsed = now - self._stat_window_start
        if elapsed < self.stats_interval:
            return

        stats = {}
        rates = {}
        for board_id, cnt in self._stat_counts.items():
            hz = cnt / elapsed if elapsed > 0 else 0.0
            last_wall = self._stat_last_wall.get(board_id, now)
            age_s = now - last_wall
            stats[board_id] = BoardStats(
                board_id=board_id,
                hz=hz,
                age_s=age_s,
                rssi=self._stat_last_rssi.get(board_id, 0),
                last_rx_seq=self._stat_last_seq.get(board_id, 0),
                seq_gaps=self._stat_gaps.get(board_id, 0),
                total=self._stat_total.get(board_id, 0),
            )
            rates[board_id] = hz

        with self._lock:
            self._rates = rates

        if self.on_board_stats is not None and stats:
            self.on_board_stats(stats)

        # Reset the rolling window (counts + gaps), keep cumulative totals.
        self._stat_counts = {}
        self._stat_gaps = {}
        self._stat_window_start = now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", "-s", default=None,
                        help="Session name (default: timestamp)")
    parser.add_argument("--no-csv", action="store_true",
                        help="Do not write CSV files (print only)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress per-packet print (still prints summary)")
    args = parser.parse_args()

    collector = CsiCollector(
        session_name=args.session,
        write_csv=not args.no_csv,
        quiet=args.quiet,
    )
    collector.run_forever()


if __name__ == "__main__":
    main()
