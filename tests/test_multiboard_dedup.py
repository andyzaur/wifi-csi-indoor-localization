import numpy as np
import pandas as pd
from dataset import drop_duplicate_csi_rows, build_multiboard_dataset


def _row(amp_val, x):
    return {"b1_amp_0": amp_val, "b1_amp_1": amp_val, "x_cm": x, "y_cm": 0.0}


def test_drops_consecutive_identical_csi():
    df = pd.DataFrame([_row(1.0, 0), _row(1.0, 5), _row(2.0, 10)])
    out = drop_duplicate_csi_rows(df)
    assert len(out) == 2                       # middle duplicate dropped
    assert list(out["b1_amp_0"]) == [1.0, 2.0]
    assert list(out["x_cm"]) == [0, 10]        # first of the run kept


def test_keeps_all_when_unique():
    df = pd.DataFrame([_row(1.0, 0), _row(2.0, 5), _row(3.0, 10)])
    assert len(drop_duplicate_csi_rows(df)) == 3


def test_empty_df_returns_empty():
    df = pd.DataFrame(columns=["b1_amp_0", "x_cm"])
    assert len(drop_duplicate_csi_rows(df)) == 0


def test_non_consecutive_identical_kept():
    # same CSI value recurs later but not consecutively → kept (genuine re-observation)
    df = pd.DataFrame([_row(1.0, 0), _row(2.0, 5), _row(1.0, 10)])
    assert len(drop_duplicate_csi_rows(df)) == 3


def _make_session():
    # 3 camera frames at 25 fps; each board has ONE packet at t=0 → all 3 frames
    # reuse it → 3 byte-identical multiboard rows.
    cam = pd.DataFrame({
        "wall_time_s": [0.0, 0.04, 0.08],
        "x_cm": [10.0, 11.0, 12.0], "y_cm": [0.0, 0.0, 0.0],
        "grid_x_cm": [0, 0, 0], "grid_y_cm": [0, 0, 0],
        "detected": [1, 1, 1],
    })
    rows = []
    for b in (1, 2, 3):
        rows.append({"wall_time_s": 0.0, "board_id": b, "rssi": -40, "channel": 6,
                     "timestamp_us": 0, "rx_seq": 0, "csi_len": 128,
                     **{f"csi_{i}": (b if i % 2 == 0 else 0) for i in range(128)}})
    csi = pd.DataFrame(rows)
    return csi, cam


def test_build_multiboard_dedup_default_on():
    csi, cam = _make_session()
    deduped = build_multiboard_dataset(csi, cam)            # dedup defaults True
    assert len(deduped) == 1


def test_build_multiboard_dedup_off_keeps_duplicates():
    csi, cam = _make_session()
    full = build_multiboard_dataset(csi, cam, dedup=False)
    assert len(full) == 3
