"""Unit tests for mlpipe.py — v2 feature transforms, drift splits, adaptation."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpipe import (moving_average_detrend, remove_static_path, window_stats,
                    leave_one_session_out, time_gap_bins,
                    online_adapt_forgetting, inner_percell_temporal_folds,
                    coral_transform, kalman_smooth, causal_session_standardize)


# ── moving_average_detrend ──────────────────────────────────────────────

def test_detrend_constant_signal_goes_to_zero():
    X = np.full((10, 3), 7.0)
    out = moving_average_detrend(X, groups=None, window=4)
    assert np.allclose(out, 0.0)


def test_detrend_window_le_1_is_identity():
    X = np.random.default_rng(0).normal(size=(8, 2))
    assert np.allclose(moving_average_detrend(X, None, 1), X)


def test_detrend_first_row_is_zero():
    # Row 0 has only itself in the causal window → mean == value → residual 0.
    X = np.array([[5.0], [9.0], [2.0]])
    out = moving_average_detrend(X, None, window=3)
    assert out[0, 0] == pytest.approx(0.0)


def test_detrend_does_not_bleed_across_sessions():
    # Session A constant 5, session B constant 10 → each detrends to ~0.
    X = np.array([[5.0]] * 4 + [[10.0]] * 4)
    groups = np.array(["A"] * 4 + ["B"] * 4)
    out = moving_average_detrend(X, groups, window=3)
    assert np.allclose(out, 0.0)  # would be non-zero at the boundary if it bled


# ── remove_static_path ──────────────────────────────────────────────────

def test_remove_static_path_subtracts_baseline():
    X = np.array([[3.0, 4.0], [5.0, 6.0]])
    base = np.array([1.0, 2.0])
    out = remove_static_path(X, base)
    assert np.allclose(out, [[2.0, 2.0], [4.0, 4.0]])


def test_remove_static_path_shape_mismatch_raises():
    with pytest.raises(ValueError):
        remove_static_path(np.zeros((2, 3)), np.zeros(2))


# ── window_stats ──────────────────────────────────────────────────────────

def test_window_stats_shape_is_doubled():
    X = np.random.default_rng(1).normal(size=(6, 4))
    out = window_stats(X, None, window=3)
    assert out.shape == (6, 8)


def test_window_stats_known_values():
    X = np.array([[0.0], [2.0], [4.0]])
    out = window_stats(X, None, window=2)
    # means: 0, 1, 3 ; vars: 0, 1, 1
    assert np.allclose(out[:, 0], [0.0, 1.0, 3.0])
    assert np.allclose(out[:, 1], [0.0, 1.0, 1.0])


def test_window_stats_variance_nonnegative():
    X = np.random.default_rng(2).normal(size=(20, 5))
    out = window_stats(X, None, window=4)
    assert (out[:, 5:] >= 0).all()


# ── leave_one_session_out ─────────────────────────────────────────────────

def test_loso_single_session_yields_nothing():
    df = pd.DataFrame({"session": ["s1"] * 5, "v": range(5)})
    assert list(leave_one_session_out(df)) == []


def test_loso_three_sessions_partition():
    df = pd.DataFrame({"session": ["a", "a", "b", "c", "c", "c"], "v": range(6)})
    splits = list(leave_one_session_out(df))
    assert len(splits) == 3
    for held, train, test in splits:
        assert set(test["session"]) == {held}
        assert held not in set(train["session"])
        assert len(train) + len(test) == len(df)


# ── time_gap_bins ──────────────────────────────────────────────────────────

def test_time_gap_bins_digitizes_elapsed_time():
    times = np.array([105.0, 115.0, 130.0])
    bins = time_gap_bins(times, t_ref=100.0, edges_s=[10.0, 20.0])
    # gaps 5,15,30 → bins 0,1,2
    assert list(bins) == [0, 1, 2]


# ── inner_percell_temporal_folds ──────────────────────────────────────────

def test_inner_folds_time_ordered_within_cell():
    df = pd.DataFrame({
        "cell_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "wall_time_s": [1.0, 2, 3, 4, 5, 6, 7, 8],
    })
    f = inner_percell_temporal_folds(df, k=2)
    # each cell: earliest 2 -> fold 0, latest 2 -> fold 1
    assert list(f) == [0, 0, 1, 1, 0, 0, 1, 1]


def test_inner_folds_respects_actual_time_not_row_order():
    df = pd.DataFrame({"cell_id": ["A"] * 4,
                       "wall_time_s": [9.0, 1.0, 8.0, 2.0]})  # unsorted
    f = inner_percell_temporal_folds(df, k=2)
    # times 9,1,8,2 -> ranks 3,0,2,1 -> folds 1,0,1,0
    assert list(f) == [1, 0, 1, 0]


def test_inner_folds_length_and_range():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"cell_id": rng.integers(0, 6, 200),
                       "wall_time_s": rng.uniform(0, 100, 200)})
    f = inner_percell_temporal_folds(df, k=4)
    assert len(f) == 200 and set(np.unique(f)).issubset({0, 1, 2, 3})


# ── torch_net.TorchLocalizer ──────────────────────────────────────────────

def test_torch_localizer_shapes_and_learns():
    from torch_net import TorchLocalizer
    rng = np.random.default_rng(6)
    X = rng.normal(size=(800, 10)).astype(np.float32)
    W = rng.normal(size=(10, 2))
    y_xy = (X @ W) * 10.0 + rng.normal(scale=1.0, size=(800, 2))  # cm-scale, learnable
    y_cls = np.digitize(y_xy[:, 0], np.quantile(y_xy[:, 0], [0.25, 0.5, 0.75]))
    Xtr, Xva, Xte = X[:500], X[500:640], X[640:]
    ytr, yva, yte = y_xy[:500], y_xy[500:640], y_xy[640:]
    ctr = y_cls[:500]
    m = TorchLocalizer(n_cls=4, width=64, depth=2, max_epochs=60, patience=20,
                       batch_size=128, device="cpu", seed=0)
    m.fit(Xtr, ytr, ctr, X_val=Xva, y_val_xy=yva)
    pred = m.predict(Xte)
    proba = m.predict_proba(Xte)
    assert pred.shape == (len(Xte), 2)
    assert proba.shape == (len(Xte), 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    # learns: beats predicting the train mean
    model_err = np.median(np.linalg.norm(yte - pred, axis=1))
    mean_err = np.median(np.linalg.norm(yte - ytr.mean(0), axis=1))
    assert model_err < 0.6 * mean_err


# ── online_adapt_forgetting ───────────────────────────────────────────────

def test_online_adapt_trajectory_mechanics():
    from sklearn.neural_network import MLPRegressor
    rng = np.random.default_rng(3)
    X = rng.normal(size=(700, 4))
    y = np.column_stack([X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]])
    model = MLPRegressor(hidden_layer_sizes=(16,), random_state=0, max_iter=1)
    traj = online_adapt_forgetting(model, X, y, init_frac=0.3, block=100)
    assert len(traj) >= 1
    idxs = [t[0] for t in traj]
    assert idxs == sorted(idxs) and idxs[-1] == len(X)


def test_online_adapt_beats_frozen_on_drift():
    from sklearn.linear_model import SGDRegressor
    rng = np.random.default_rng(4)
    n = 1500
    x = rng.uniform(-1, 1, size=(n, 1))
    slope = np.linspace(1.0, 6.0, n)          # the mapping drifts over time
    y = (slope * x[:, 0])
    cut = int(n * 0.25)

    adapt = SGDRegressor(random_state=0)
    traj = online_adapt_forgetting(adapt, x, y, init_frac=0.25, block=150)
    adapt_last = traj[-1][1]

    frozen = SGDRegressor(random_state=0)
    frozen.partial_fit(x[:cut], y[:cut])
    frozen_last = np.median(np.abs(y[-150:] - frozen.predict(x[-150:])))

    # Online adaptation tracks the drifting mapping; a frozen model fit only on
    # the early window lags badly. (Deterministic seeds → stable margin.)
    assert adapt_last < 0.8 * frozen_last


# ── coral_transform (unsupervised domain adaptation) ─────────────────────

def test_coral_aligns_source_to_target_stats():
    rng = np.random.default_rng(0)
    d, n = 6, 4000
    # source: mean 5, anisotropic covariance A
    A = rng.normal(size=(d, d))
    Xs = rng.normal(size=(n, d)) @ A + 5.0
    # target: different mean and covariance B
    B = rng.normal(size=(d, d))
    Xt = rng.normal(size=(n, d)) @ B - 2.0
    out = coral_transform(Xs, Xt, eps=1e-6)
    # aligned source now matches target mean and covariance closely
    assert np.allclose(out.mean(0), Xt.mean(0), atol=0.1)
    rel = np.linalg.norm(np.cov(out, rowvar=False) - np.cov(Xt, rowvar=False))
    rel /= np.linalg.norm(np.cov(Xt, rowvar=False))
    assert rel < 0.05


def test_coral_self_alignment_is_identity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 4)) @ rng.normal(size=(4, 4)) + 3.0
    out = coral_transform(X, X, eps=1e-8)
    assert np.allclose(out, X, atol=1e-4)


def test_coral_uses_no_labels_and_checks_shape():
    with pytest.raises(ValueError):
        coral_transform(np.zeros((10, 4)), np.zeros((10, 3)))


# ── multiboard_static_baseline (empty-room) ──────────────────────────────

def test_multiboard_static_baseline_aligns_to_names():
    import pandas as pd
    from dataset import multiboard_static_baseline, CSI_FEATURE_COLS
    # two boards, constant I/Q -> known amplitude (3,4)->5 for board 1; (6,8)->10 for board 4
    rows = []
    for b, (i, q) in [(1, (3, 4)), (4, (6, 8))]:
        for _ in range(5):
            r = {"board_id": b, "rssi": -40.0 if b == 1 else -50.0}
            for k in range(64):
                r[f"csi_{2*k}"] = i
                r[f"csi_{2*k+1}"] = q
            rows.append(r)
    csi = pd.DataFrame(rows)
    names = [f"b1_amp_{k}" for k in range(64)] + ["b1_rssi"] \
        + [f"b4_amp_{k}" for k in range(64)] + ["b4_rssi"]
    base = multiboard_static_baseline(csi, names, board_ids=(1, 4))
    assert base.shape == (130,)
    assert np.allclose(base[:64], 5.0)        # board 1 amps
    assert base[64] == -40.0                  # board 1 rssi
    assert np.allclose(base[65:129], 10.0)    # board 4 amps
    assert base[129] == -50.0                 # board 4 rssi


def test_multiboard_static_baseline_phase_features_stay_zero():
    import pandas as pd
    from dataset import multiboard_static_baseline
    rows = [{"board_id": 1, "rssi": -40.0, **{f"csi_{k}": 1 for k in range(128)}}]
    csi = pd.DataFrame(rows)
    names = ["b1_amp_0", "b1_phase_0", "b1_rssi"]
    base = multiboard_static_baseline(csi, names, board_ids=(1,))
    assert base[1] == 0.0                      # phase untouched


# ── TorchLocalizer warm-start (fine-tune / calibration) ──────────────────

def test_torch_warm_start_loads_pretrained_weights():
    from torch_net import TorchLocalizer
    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    W = rng.normal(size=(8, 2))
    y = (X @ W) * 10.0
    cls = np.zeros(len(X), dtype=int)
    base = TorchLocalizer(n_cls=1, width=32, depth=1, max_epochs=30, device="cpu", seed=0)
    base.fit(X, y, cls, X_val=X, y_val_xy=y)
    state = base.get_state()
    # a fresh net warm-started from `state`, 0 epochs, inheriting the base y-stats,
    # must reproduce the base net exactly (identical weights + de-standardization)
    ft = TorchLocalizer(n_cls=1, width=32, depth=1, max_epochs=0, device="cpu", seed=1)
    ft.fit(X[:50], y[:50], cls[:50], init_state=state, y_stats=(base.ymean_, base.ystd_))
    assert np.allclose(base.predict(X[:20]), ft.predict(X[:20]), atol=1e-4)


# ── kalman_smooth (causal constant-velocity track filter) ─────────────────

def test_kalman_constant_position_converges_near_truth():
    rng = np.random.default_rng(10)
    n = 300
    truth = np.tile([120.0, 80.0], (n, 1))
    noisy = truth + rng.normal(scale=25.0, size=(n, 2))
    out = kalman_smooth(noisy, dt=0.1)
    raw_err = np.median(np.linalg.norm(noisy - truth, axis=1))
    kf_err = np.median(np.linalg.norm(out - truth, axis=1))
    assert kf_err < raw_err  # averages the measurement noise down


def test_kalman_walk_beats_raw_and_causal_rolling_mean():
    rng = np.random.default_rng(11)
    n, dt = 400, 0.1
    t = np.arange(n) * dt
    truth = np.column_stack([50.0 + 90.0 * t, 200.0 - 60.0 * t])  # ~108 cm/s walk
    noisy = truth + rng.normal(scale=25.0, size=(n, 2))
    out = kalman_smooth(noisy, dt=dt)
    # inline causal rolling mean w=9 — the smoothing train_final sweeps
    # (kept local: importing train_final drags in the full training stack)
    w = 9
    roll = np.stack([noisy[max(0, i - w + 1):i + 1].mean(axis=0) for i in range(n)])

    def med(p):
        return np.median(np.linalg.norm(p - truth, axis=1))
    # on a moving target the rolling mean lags ((w-1)/2 frames behind); the CV
    # filter's velocity state removes that bias — strictly better than both
    assert med(out) < med(noisy)
    assert med(out) < med(roll)


def test_kalman_is_causal():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(50, 2)) * 30.0
    b = a.copy()
    b[30:] += 500.0  # perturb ONLY the tail
    out_a = kalman_smooth(a, dt=0.1)
    out_b = kalman_smooth(b, dt=0.1)
    assert np.array_equal(out_a[:30], out_b[:30])  # past blind to the future
    assert not np.allclose(out_a[30:], out_b[30:])  # tail did change


def test_kalman_shape_and_edge_cases():
    assert kalman_smooth(np.zeros((0, 2)), dt=0.1).shape == (0, 2)
    one = kalman_smooth(np.array([[3.0, 4.0]]), dt=0.1)
    assert one.shape == (1, 2) and np.allclose(one, [[3.0, 4.0]])
    out = kalman_smooth(np.ones((7, 2)), dt=0.1)
    assert out.shape == (7, 2)
    assert np.allclose(out[0], [1.0, 1.0])  # init = first measurement
    with pytest.raises(ValueError):
        kalman_smooth(np.zeros((5, 3)), dt=0.1)


# ── causal_session_standardize (online per-session normalizer) ────────────

def test_causal_sess_std_last_row_matches_batch_zscore():
    rng = np.random.default_rng(20)
    X = np.vstack([rng.normal(5.0, 3.0, size=(40, 4)),
                   rng.normal(-2.0, 0.5, size=(60, 4))])
    groups = np.array(["A"] * 40 + ["B"] * 60)
    out = causal_session_standardize(X, groups)
    for g in ("A", "B"):
        m = groups == g
        Xg = X[m]
        # at the last row the expanding window IS the whole session
        assert np.allclose(out[m][-1], (Xg[-1] - Xg.mean(0)) / Xg.std(0))


def test_causal_sess_std_is_causal_within_group():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(50, 3))
    groups = np.array(["A"] * 30 + ["B"] * 20)
    Y = X.copy()
    Y[20:30] += 100.0                      # perturb ONLY session A's tail
    out_x = causal_session_standardize(X, groups)
    out_y = causal_session_standardize(Y, groups)
    assert np.array_equal(out_x[:20], out_y[:20])   # past blind to the future
    assert not np.allclose(out_x[20:30], out_y[20:30])
    assert np.array_equal(out_x[30:], out_y[30:])   # session B untouched


def test_causal_sess_std_groups_do_not_contaminate():
    rng = np.random.default_rng(22)
    Xa = rng.normal(50.0, 1.0, size=(30, 2))
    Xb = rng.normal(-50.0, 4.0, size=(20, 2))
    groups = np.array(["A"] * 30 + ["B"] * 20)
    joint = causal_session_standardize(np.vstack([Xa, Xb]), groups)
    # each group standardized alone == its slice of the joint call
    assert np.array_equal(joint[:30], causal_session_standardize(Xa, None))
    assert np.array_equal(joint[30:], causal_session_standardize(Xb, None))


def test_causal_sess_std_constant_feature_is_zero_no_nan():
    X = np.column_stack([np.full(8, 7.0), np.arange(8, dtype=float)])
    groups = np.array(["A"] * 4 + ["B"] * 4)
    out = causal_session_standardize(X, groups)
    assert np.allclose(out[:, 0], 0.0)     # var 0 -> eps floor, no blow-up
    assert np.isfinite(out).all()
