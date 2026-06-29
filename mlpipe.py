"""v2 ML-pipeline helpers for device-free CSI localization.

Pure, unit-tested transforms / splits / adaptation that augment the baseline
`train_final.py`. Each is grounded in the 2026-06-03 literature synthesis:

- `moving_average_detrend` — causal high-pass per feature stream. The survey's
  named real-time drift tactic (COMST22, Table V [111]); also the device-free
  fundamental of suppressing the slow static-channel component.
- `remove_static_path` — subtract an empty-room baseline (the overnight
  calibration capture), leaving only the human-induced channel perturbation.
- `window_stats` — per-feature causal rolling mean + variance over an N-frame
  window (Chaudhari 2026); a small-data-friendly temporal feature.
- `causal_session_standardize` — causal/online variant of per-session
  standardization (the diagonal-alignment drift method): expanding
  stats-so-far per session, deployable on a live stream.
- `leave_one_session_out` / `time_gap_bins` — the cross-session / error-vs-
  time-gap drift evaluation (Zhang TMC 2025; BSWCLoc TNSM 2024).
- `online_adapt_forgetting` — incremental partial-fit on fresh labelled
  windows, recent data dominating (Zhang TMC 2025 forgetting factor).

All transforms operate on a feature matrix X (n_samples, n_features) plus a
`groups` array of session labels, and assume rows are TIME-ORDERED within each
group. They are causal (a row only ever sees its own past), so applying them
before a train/test split introduces no future/label leakage.
"""
from __future__ import annotations

import numpy as np


def _causal_rolling_mean(X: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean: row i = mean of rows [max(0, i-window+1) .. i].

    Early rows (fewer than `window` predecessors) average over what exists.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n == 0:
        return X.copy()
    cs = np.cumsum(X, axis=0)
    out = np.empty_like(X)
    for i in range(n):
        lo = i - window + 1
        if lo <= 0:
            out[i] = cs[i] / (i + 1)
        else:
            out[i] = (cs[i] - cs[lo - 1]) / window
    return out


def moving_average_detrend(X: np.ndarray, groups: np.ndarray | None,
                           window: int) -> np.ndarray:
    """Subtract a causal rolling mean from each feature, per session group.

    Removes slow drift (the static channel + environmental trend), leaving the
    fast, human-induced fluctuation. `groups` labels each row's session so the
    rolling mean never spans two sessions; pass None to treat all rows as one
    time-ordered stream.
    """
    X = np.asarray(X, dtype=np.float64)
    if window <= 1:
        return X.copy()
    out = X.copy()
    if groups is None:
        return X - _causal_rolling_mean(X, window)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        m = groups == g
        out[m] = X[m] - _causal_rolling_mean(X[m], window)
    return out


def remove_static_path(X: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Subtract a per-feature empty-room baseline (h_static) from X.

    `baseline` is a length-n_features vector — e.g. the mean amplitude per
    (board, subcarrier) measured in the empty-room calibration capture. The
    result is how much the present human perturbs each channel coefficient.
    """
    X = np.asarray(X, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(1, -1)
    if baseline.shape[1] != X.shape[1]:
        raise ValueError(
            f"baseline has {baseline.shape[1]} features, X has {X.shape[1]}")
    return X - baseline


def _sym_sqrt(C: np.ndarray, inverse: bool = False, floor: float = 1e-12) -> np.ndarray:
    """Symmetric (inverse) square root of a symmetric PSD matrix via eigh."""
    vals, vecs = np.linalg.eigh((C + C.T) / 2.0)
    vals = np.clip(vals, floor, None)
    s = 1.0 / np.sqrt(vals) if inverse else np.sqrt(vals)
    return (vecs * s) @ vecs.T


def coral_transform(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """CORAL unsupervised domain adaptation: recolour source features `Xs` to the
    second-order statistics (mean + covariance) of target features `Xt`.

    Sun & Saenko 2016, "Return of Frustratingly Easy Domain Adaptation." Uses ONLY
    feature statistics — no labels — so `Xt` is the *unlabelled* target-session CSI
    available at deployment (transductive). Train a model on the returned aligned
    source features, then test on the raw target features: the train/test feature
    distributions now match, which is the drift the cross-session gap is made of.

    Whitens the (centred) source by Cs^-1/2, recolours by Ct^1/2, shifts to the
    target mean. `eps` ridge-regularises both covariances (features ≫ rank-stable).
    Aligning a matrix to itself is the identity (up to `eps`).
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    Xt = np.asarray(Xt, dtype=np.float64)
    if Xs.shape[1] != Xt.shape[1]:
        raise ValueError(f"feature mismatch: source {Xs.shape[1]} vs target {Xt.shape[1]}")
    d = Xs.shape[1]
    ms, mt = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(d)
    A = _sym_sqrt(Cs, inverse=True) @ _sym_sqrt(Ct)
    return (Xs - ms) @ A + mt


def window_stats(X: np.ndarray, groups: np.ndarray | None,
                 window: int) -> np.ndarray:
    """Per-feature causal rolling mean + variance over `window` frames.

    Returns an (n_samples, 2*n_features) matrix = [rolling_mean | rolling_var],
    computed per session group. Variance uses E[x^2] - E[x]^2, clipped at 0.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    means = np.empty_like(X)
    varis = np.empty_like(X)
    if groups is None:
        groups = np.zeros(n, dtype=int)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        m = groups == g
        Xg = X[m]
        mean_g = _causal_rolling_mean(Xg, window)
        mean_sq_g = _causal_rolling_mean(Xg ** 2, window)
        means[m] = mean_g
        varis[m] = np.clip(mean_sq_g - mean_g ** 2, 0.0, None)
    return np.hstack([means, varis])


def causal_session_standardize(X: np.ndarray, groups: np.ndarray | None,
                               eps: float = 1e-8) -> np.ndarray:
    """Causal/online variant of per-session standardization (the diagonal-
    alignment drift method): row i of a session is standardized by the
    EXPANDING mean/std of that session's rows [0..i] — the stats so far, row
    i included. This mirrors a deployable streaming normalizer: unlike the
    batch (transductive) variant it needs no end-of-session knowledge, so it
    runs on a live stream; the first seconds are intentionally noisy (stats
    from few frames), which is the honest deployment reality. Vectorized via
    per-group cumsum / cumsum-of-squares (no per-row loop); variance floored
    at `eps`. `groups` labels each row's session (None = one stream); rows
    are assumed time-ordered within each group. Returns a same-shape array.
    """
    X = np.asarray(X, dtype=np.float64)
    out = np.empty_like(X)
    if groups is None:
        groups = np.zeros(len(X), dtype=int)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        m = groups == g
        Xg = X[m]
        n = np.arange(1, len(Xg) + 1, dtype=np.float64)[:, None]
        mean = np.cumsum(Xg, axis=0) / n
        var = np.maximum(np.cumsum(Xg ** 2, axis=0) / n - mean ** 2, eps)
        out[m] = (Xg - mean) / np.sqrt(var)
    return out


def leave_one_session_out(df, session_col: str = "session"):
    """Yield (held_out_session, train_df, test_df) for each session.

    Yields NOTHING when fewer than two distinct sessions are present — so the
    caller can simply iterate and the drift evaluation auto-disables on a
    single-session dataset.
    """
    sessions = list(dict.fromkeys(df[session_col].tolist()))  # stable order
    if len(sessions) < 2:
        return
    for s in sessions:
        test = df[df[session_col] == s]
        train = df[df[session_col] != s]
        yield s, train.copy(), test.copy()


class MultiAxis:
    """Picklable wrapper turning a single-output regressor class into a 2-output
    (x, y) regressor: fits one estimator per axis. Lives here (an importable
    module) so models saved by train_v3 unpickle anywhere, not only when train_v3
    was the __main__ script.
    """
    def __init__(self, cls, params):
        self.cls, self.params = cls, params

    def fit(self, X, Y):
        self.m = [self.cls(**self.params).fit(X, Y[:, k]) for k in (0, 1)]
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.m])


def inner_percell_temporal_folds(df, k: int = 5, time_col: str = "wall_time_s",
                                 cell_col: str = "cell_id") -> np.ndarray:
    """Assign each row to one of k TIME-ORDERED folds within its grid cell.

    Generalizes the per-cell-temporal split to k ordered chunks per cell: fold 0
    is the earliest 1/k of a cell's frames by time, fold k-1 the latest. Used for
    honest out-of-fold stacking — a meta-learner can then train on base-model
    predictions for rows the base model did NOT fit, with no temporal
    near-duplicate leakage (unlike random KFold, which re-injects it).

    Positional: returns an int array of length len(df) aligned to the df's
    current row order (pass a reset-index df). Cells with fewer than k rows fill
    only the lowest fold ids.
    """
    n = len(df)
    fold = np.zeros(n, dtype=int)
    times = np.asarray(df[time_col].to_numpy(), dtype=np.float64)
    cells = df[cell_col].to_numpy()
    for cell in np.unique(cells):
        rows = np.where(cells == cell)[0]
        order = rows[np.argsort(times[rows], kind="stable")]
        m = len(order)
        if m:
            fold[order] = np.minimum((np.arange(m) * k) // m, k - 1)
    return fold


def time_gap_bins(times: np.ndarray, t_ref: float,
                  edges_s) -> np.ndarray:
    """Bin each row by elapsed time since `t_ref` (e.g. end of the train window).

    Returns an integer bin index per row using np.digitize on `edges_s`
    (in seconds). Bin 0 = gap below the first edge, etc. Used to plot median
    localization error vs how far in the future the test sample is — the
    drift curve.
    """
    times = np.asarray(times, dtype=np.float64)
    gap = times - t_ref
    return np.digitize(gap, np.asarray(edges_s, dtype=np.float64))


def online_adapt_forgetting(model, X: np.ndarray, y: np.ndarray,
                            init_frac: float = 0.3, block: int = 200):
    """Walk a fitted-from-scratch regressor forward in time, adapting online.

    Trains on the first `init_frac` of (time-ordered) rows, then for each
    subsequent block of `block` rows: first PREDICT it (record error before the
    model has seen it — honest deployment error), then `partial_fit` on it so
    recent data dominates (SGD recency = the forgetting factor of Zhang 2025).

    `model` must support `partial_fit` (e.g. sklearn MLPRegressor). Returns a
    list of (block_end_time_index, median_error) measured BEFORE each update —
    compare against a frozen model to show adaptation closes the drift gap.
    Rows are assumed time-ordered.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(X)
    cut = max(1, int(n * init_frac))
    model.partial_fit(X[:cut], y[:cut])
    trajectory = []
    i = cut
    while i < n:
        j = min(i + block, n)
        pred = model.predict(X[i:j])
        err = np.linalg.norm(y[i:j] - pred, axis=1) if y.ndim > 1 else np.abs(y[i:j] - pred)
        trajectory.append((j, float(np.median(err))))
        model.partial_fit(X[i:j], y[i:j])
        i = j
    return trajectory


def kalman_smooth(pred_xy: np.ndarray, dt: float, process_std: float = 60.0,
                  meas_std: float = 25.0) -> np.ndarray:
    """Causal constant-velocity Kalman filter over a per-frame (x, y) track.

    Post-prediction smoothing: the regressor's frame-wise position estimates
    are near-independent draws around the true position, while a walking
    person moves with roughly piecewise-constant velocity. Fusing the two with
    the classic CV tracker (Kalman 1960; discrete white-noise-acceleration
    process model, Bar-Shalom et al. 2001 §6.3.2) averages the measurement
    noise away WITHOUT the lag bias a plain causal rolling mean pays on a
    moving target — the velocity state extrapolates between updates.

    State [x, y, vx, vy]; `dt` is the scalar seconds between rows;
    `process_std` is the white-acceleration std in cm/s^2 (how hard the walker
    may accelerate — higher tracks turns faster, smooths less); `meas_std` is
    the regressor's per-axis position noise in cm. Forward pass ONLY (no RTS
    backward smoothing), so row i of the output sees rows [0..i] of the input
    — causal, deployable live, and directly comparable to the causal
    rolling-mean smoothing sweep in train_final. Initialised at the first
    measurement with zero velocity and a large covariance. Returns (n, 2);
    n of 0 or 1 passes through unchanged.
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    if pred_xy.size == 0:
        return pred_xy.reshape(0, 2).copy()
    if pred_xy.ndim != 2 or pred_xy.shape[1] != 2:
        raise ValueError(f"pred_xy must be (n, 2), got {pred_xy.shape}")
    n = len(pred_xy)
    if n == 1:
        return pred_xy.copy()
    dt = float(dt)
    F = np.eye(4)                       # constant-velocity transition
    F[0, 2] = F[1, 3] = dt
    q = float(process_std) ** 2         # white-acceleration noise, per axis
    Q = np.zeros((4, 4))
    Q[0, 0] = Q[1, 1] = q * dt ** 4 / 4.0
    Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = q * dt ** 3 / 2.0
    Q[2, 2] = Q[3, 3] = q * dt ** 2
    H = np.zeros((2, 4))                # we only measure position
    H[0, 0] = H[1, 1] = 1.0
    R = float(meas_std) ** 2 * np.eye(2)
    x = np.array([pred_xy[0, 0], pred_xy[0, 1], 0.0, 0.0])
    P = np.diag([float(meas_std) ** 2, float(meas_std) ** 2, 1e6, 1e6])
    out = np.empty((n, 2))
    out[0] = x[:2]
    eye4 = np.eye(4)
    for i in range(1, n):
        x = F @ x                       # predict
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R             # update on measurement i
        K = np.linalg.solve(S, H @ P).T   # gain P Hᵀ S⁻¹ (S symmetric)
        x = x + K @ (pred_xy[i] - H @ x)
        P = (eye4 - K @ H) @ P
        out[i] = x[:2]
    return out
