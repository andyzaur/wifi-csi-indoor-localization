import struct
from csi_collector import (parse_csi_packet, CSI_HDR_FMT, CSI_HDR_SIZE,
                           CSI_DATA_LEN, CSI_PKT_SIZE)


def make_csi(board_id=1, rssi=-40, channel=11, rx_seq=7, csi_len=128,
             payload_len=None):
    """Build a raw CSI datagram. payload_len lets us simulate truncation
    (fewer bytes than the header's csi_len claims)."""
    if payload_len is None:
        payload_len = csi_len
    hdr = struct.pack(CSI_HDR_FMT, board_id, 1, 2, 3, 4, 5, 6,
                      rssi, channel, 123456, rx_seq, csi_len)
    return hdr + bytes(payload_len)


def test_valid_full_packet_parses():
    pkt = make_csi(board_id=4, rssi=-55, csi_len=128)
    assert len(pkt) == CSI_PKT_SIZE                 # 145 bytes on the wire
    out = parse_csi_packet(pkt)
    assert out is not None
    board_id, mac, rssi, channel, ts, rx_seq, csi_len, csi_bytes = out
    assert board_id == 4 and rssi == -55 and csi_len == 128
    assert mac == (1, 2, 3, 4, 5, 6)
    assert len(csi_bytes) == 128


def test_valid_short_csi_len_parses():
    # a legitimately shorter CSI (e.g. legacy 64) is still valid
    out = parse_csi_packet(make_csi(csi_len=64))
    assert out is not None and out[6] == 64 and len(out[7]) == 64


def test_short_header_rejected():
    assert parse_csi_packet(b"\x01\x02\x03") is None


def test_implausible_csi_len_rejected():
    # header claims 200 bytes of CSI (> the 128-byte field) → garbage, drop it
    assert parse_csi_packet(make_csi(csi_len=200, payload_len=128)) is None


def test_truncated_payload_rejected():
    # header claims 128 but only 10 payload bytes arrived → drop, don't write junk
    assert parse_csi_packet(make_csi(csi_len=128, payload_len=10)) is None


def test_csi_len_bounds_constant():
    assert CSI_HDR_SIZE == 17 and CSI_DATA_LEN == 128 and CSI_PKT_SIZE == 145
