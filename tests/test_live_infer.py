"""Hermetic tests for the live-inference core (csi_gui.live_infer).

No hardware and no real session data: synthetic CSI packets, an in-memory model
bundle, and a tiny generated csi.csv exercise the whole path
(parse -> per-board state -> feature vector -> model -> render).
"""
from __future__ import annotations

import struct
import warnings

import numpy as np
import pandas as pd
import pytest

from csi_gui import live_infer as li


def _make_packet(board_id: int, rssi: int = -55, amps_seed: int = 0) -> bytes:
    """Build one on-the-wire CSI datagram matching csi_collector.py's format."""
    csi = bytes(((np.arange(li.CSI_DATA_LEN) + amps_seed) % 9 + 1).astype(np.int8).tobytes())
    header = struct.pack(li.CSI_HDR_FMT, board_id, 0, 0, 0, 0, 0, 0,
                         rssi, 6, 1234, 1, li.CSI_DATA_LEN)
    return header + csi


def test_parse_csi_packet_roundtrip():
    pkt = _make_packet(4, rssi=-61)
    board, rssi, amps = li.parse_csi_packet(pkt)
    assert board == 4
    assert rssi == -61
    assert amps.shape == (li.AMPS_PER_BOARD,)
    assert np.all(amps >= 0)


def test_parse_rejects_clap_and_short():
    assert li.parse_csi_packet(bytes([li.CLAP_MAGIC, 0, 0, 0])) is None
    assert li.parse_csi_packet(b"\x01\x02") is None


def test_boardstate_snapshot_order_and_shape():
    state = li.BoardState()
    now = 1000.0
    # three boards, deterministic amps so we can check column order
    for b in (1, 4, 5):
        state.update(b, now, rssi=-50 - b, amps=np.full(64, float(b)))
    feat, ages = state.snapshot(n_boards=3, max_age_s=1.0, now=now)
    assert feat.shape == (1, 3 * li.FEATS_PER_BOARD)  # 195
    # board order is ascending: [amps(1)*64, rssi(1), amps(4)*64, rssi(4), ...]
    assert feat[0, 0] == 1.0 and feat[0, 64] == -51.0
    assert feat[0, 65] == 4.0 and feat[0, 129] == -54.0
    assert feat[0, 130] == 5.0 and feat[0, 194] == -55.0


def test_boardstate_none_until_enough_fresh():
    state = li.BoardState()
    now = 500.0
    state.update(1, now, -50, np.ones(64))
    state.update(4, now, -50, np.ones(64))
    feat, _ = state.snapshot(n_boards=3, max_age_s=1.0, now=now)
    assert feat is None  # only two boards
    state.update(5, now - 5.0, -50, np.ones(64))  # stale
    feat, ages = state.snapshot(n_boards=3, max_age_s=1.0, now=now)
    assert feat is None and ages[5] == pytest.approx(5.0)


def _synthetic_bundle(n_features: int = 195):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.random((60, n_features)).astype(np.float32)
    y = rng.random((60, 2)).astype(np.float32) * 200.0
    scaler = StandardScaler().fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = MLPRegressor(hidden_layer_sizes=(16,), max_iter=40,
                           random_state=0).fit(scaler.transform(X), y)
    return {"scaler": scaler, "regressor": reg, "n_features": n_features}


def test_predictor_predicts_two_floats():
    pred = li.Predictor(_synthetic_bundle())
    assert pred.n_boards == 3
    feat = np.random.default_rng(1).random((1, 195)).astype(np.float32)
    x, y = pred.predict_xy(feat)
    assert isinstance(x, float) and isinstance(y, float)


def test_predictor_rejects_bad_bundle():
    with pytest.raises(ValueError):
        li.Predictor({"regressor": object()})  # missing scaler


def test_render_floor_map_returns_image():
    anchors = {0: (0.0, 0.0), 1: (200.0, 0.0), 9: (100.0, 150.0)}
    img = li.render_floor_map(anchors, [(50, 50), (60, 60)], (70, 70), ["live"])
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8
    # also tolerates empty anchors
    blank = li.render_floor_map({}, [], None, ["no layout"])
    assert blank.shape[2] == 3


def test_end_to_end_state_to_prediction():
    pred = li.Predictor(_synthetic_bundle())
    state = li.BoardState()
    now = 10.0
    for b in (1, 4, 5):
        state.update(b, now, -55, np.abs(np.random.default_rng(b).normal(8, 2, 64)))
    feat, _ = state.snapshot(pred.n_boards, max_age_s=1.0, now=now)
    assert feat is not None
    x, y = pred.predict_xy(feat)
    assert np.isfinite(x) and np.isfinite(y)


def _write_session_csv(path, boards=(1, 4, 5), rows_per_board=20):
    rng = np.random.default_rng(2)
    recs = []
    t = 0.0
    for r in range(rows_per_board):
        for b in boards:
            t += 0.01
            rec = {"board_id": b, "rssi": -50 - b, "wall_time_s": t}
            csi = rng.integers(-8, 8, size=li.CSI_DATA_LEN)
            for i in range(li.CSI_DATA_LEN):
                rec[f"csi_{i}"] = int(csi[i])
            recs.append(rec)
    pd.DataFrame(recs).to_csv(path, index=False)


def test_replay_source_feeds_state(tmp_path):
    csv = tmp_path / "csi.csv"
    _write_session_csv(csv)
    state = li.BoardState()
    replay = li.ReplaySource(state, str(csv), loop=False)
    assert replay.n_rows == 60
    assert replay._amps.shape == (60, 64)
    for i in range(replay.n_rows):
        replay.feed_row(i)
    feat, _ = state.snapshot(n_boards=3, max_age_s=10.0)
    assert feat is not None and feat.shape == (1, 195)


def test_discovery_helpers(tmp_path):
    sess = tmp_path / "20260101_01_demo"
    sess.mkdir()
    _write_session_csv(sess / "csi.csv")
    import joblib
    joblib.dump(_synthetic_bundle(), sess / "model_final.joblib")

    models = li.find_model_bundles(str(tmp_path))
    assert any(p.endswith("model_final.joblib") for _, p in models)
    replays = li.find_replay_sessions(str(tmp_path))
    assert any(p.endswith("csi.csv") for _, p in replays)
    # missing dir is tolerated
    assert li.find_model_bundles(str(tmp_path / "nope")) == []
