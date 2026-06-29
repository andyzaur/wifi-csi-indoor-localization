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
"""

import argparse
import csv
import datetime as dt
import os
import socket
import struct
import sys
import time


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", "-s", default=None,
                        help="Session name (default: timestamp)")
    parser.add_argument("--no-csv", action="store_true",
                        help="Do not write CSV files (print only)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress per-packet print (still prints summary)")
    args = parser.parse_args()

    csi_file = clap_file = csi_writer = clap_writer = None
    session_path = session_name = None

    if not args.no_csv:
        session_path, session_name = make_session_dir(args.session)
        csi_path = os.path.join(session_path, "csi.csv")
        clap_path = os.path.join(session_path, "clap.csv")

        csi_file = open(csi_path, "w", newline="")
        csi_writer = csv.writer(csi_file)
        csi_header = (
            ["wall_time_s", "board_id", "mac", "rssi", "channel",
             "timestamp_us", "rx_seq", "csi_len"]
            + [f"csi_{i}" for i in range(CSI_DATA_LEN)]
        )
        csi_writer.writerow(csi_header)

        clap_file = open(clap_path, "w", newline="")
        clap_writer = csv.writer(clap_file)
        clap_writer.writerow(
            ["wall_time_s", "event", "event_name", "seq", "timestamp_us"]
        )

        print(f"Session: {session_name}")
        print(f"  CSI  -> {csi_path}")
        print(f"  CLAP -> {clap_path}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # macOS quirk: receiving 255.255.255.255 limited-broadcast UDP into a
    # user-space socket requires SO_BROADCAST and SO_REUSEADDR to be set on the
    # listener too (not just on the sender).
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("", PORT))
    print(f"Listening on UDP port {PORT}...")

    counts = {}
    clap_count = 0
    malformed = 0
    seen_claps = set()          # (event, seq) already logged — de-dups clap bursts
    last_report = time.time()
    last_flush = time.time()

    # Timeout so the loop wakes even when no packets arrive — this is what lets
    # the periodic summary fire (and warn) during a silent session instead of
    # blocking forever inside recvfrom.
    sock.settimeout(1.0)

    try:
        while True:
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
                    print(f"\n{marker} CLAP [{event_name.upper()}] seq={seq} t={wall_t:.3f} {marker}\n")
                    if clap_writer:
                        clap_writer.writerow([f"{wall_t:.4f}", event, event_name, seq, ts_us])
                        clap_file.flush()
                # else: a duplicate from the same burst — silently ignore

            # CSI packet? Accept any plausible board id (1..200). The clapper
            # magic (0xCA = 202) is handled above, so it can't collide. This is
            # what lets boards with real IDs like 4 and 5 through instead of
            # being dropped as "unknown".
            elif data is not None and 1 <= data[0] <= 200:
                parsed = parse_csi_packet(data)
                if parsed is None:
                    malformed += 1
                    if not args.quiet:
                        print(f"Malformed CSI packet from {addr}: {len(data)} bytes "
                              f"(board {data[0]}) — dropped")
                else:
                    board_id, mac, rssi, channel, timestamp_us, rx_seq, csi_len, csi_bytes = parsed
                    mac_str = ":".join(f"{b:02x}" for b in mac)

                    counts[board_id] = counts.get(board_id, 0) + 1

                    if csi_writer:
                        # Pad/truncate to 128 bytes; store as signed ints
                        padded = list(csi_bytes) + [0] * (CSI_DATA_LEN - len(csi_bytes))
                        padded = padded[:CSI_DATA_LEN]
                        # Convert unsigned bytes to signed int8
                        signed = [b if b < 128 else b - 256 for b in padded]
                        row = [f"{wall_t:.4f}", board_id, mac_str, rssi, channel,
                               timestamp_us, rx_seq, csi_len] + signed
                        csi_writer.writerow(row)

                    if not args.quiet:
                        mid = csi_len // 2 - 4
                        sample = list(csi_bytes[mid:mid + 8]) if csi_len >= mid + 8 else list(csi_bytes[:8])
                        print(f"Board {board_id} | RSSI {rssi:+4d} | ch {channel} | "
                              f"seq {rx_seq:5d} | len {csi_len:3d} | sample {sample}")

            # Unknown packet shape
            elif data is not None:
                if not args.quiet:
                    print(f"Unknown packet from {addr}: {len(data)} bytes, first byte 0x{data[0]:02x}")

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
                    print(f"\n--- {elapsed:.0f}s summary: {', '.join(parts)} | CLAP: {clap_count}{bad} ---\n")
                else:
                    print(f"\n--- {elapsed:.0f}s summary: !! NO CSI PACKETS RECEIVED !! "
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

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if csi_file:
            csi_file.close()
        if clap_file:
            clap_file.close()
        sock.close()
        if session_path:
            print(f"Session saved to: {session_path}")


if __name__ == "__main__":
    main()
