import pandas as pd
from train_final import per_cell_temporal_split


def test_per_cell_temporal_split_holds_out_latest_per_cell():
    df = pd.DataFrame({
        "cell_id": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "wall_time_s": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "x_cm": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })
    tr, te = per_cell_temporal_split(df, train_frac=0.8)
    # 5 frames/cell, 80% -> 4 train (earliest), 1 test (latest), per cell
    assert len(tr) == 8
    assert len(te) == 2
    # every cell appears in BOTH sets (full coverage)
    assert set(tr["cell_id"]) == {"A", "B"}
    assert set(te["cell_id"]) == {"A", "B"}
    # test frames are the latest-in-time (t=5) for each cell
    assert set(te["wall_time_s"]) == {5}
    # train frames are the earliest (t=1..4)
    assert set(tr["wall_time_s"]) == {1, 2, 3, 4}


def test_per_cell_temporal_split_singleton_cell_goes_to_train():
    df = pd.DataFrame({
        "cell_id": ["A", "A", "C"],
        "wall_time_s": [1, 2, 9],
        "x_cm": [0, 0, 2],
    })
    tr, te = per_cell_temporal_split(df, train_frac=0.5)
    # cell C has 1 frame -> goes to train, never test
    assert 9 in set(tr["wall_time_s"])
    assert 9 not in set(te["wall_time_s"])
