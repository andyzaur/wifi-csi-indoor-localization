"""Load and align CSI + camera + clap data from a session.

A session lives at sessions/<name>/ and contains:
    csi.csv     — one row per CSI packet (board_id, mac, rssi, channel,
                  timestamp_us, rx_seq, csi_len, csi_0..csi_127)
    camera.csv  — one row per camera frame (frame, timestamp_s, x_cm, y_cm,
                  grid_x_cm, grid_y_cm, detected)
    clap.csv    — clap events (event 0=start, 1=stop)

Both csi.csv and camera.csv carry the SAME laptop wall clock as `wall_time_s`
(camera CSV uses `timestamp_s`). This is what allows joining.

Use:
    from dataset import load_session
    csi, camera, clap = load_session("sessions/20260519_02_firstWorkingTest")
    df = build_aligned_dataset(csi, camera, clap)   # CSI rows labelled with x,y
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd


CSI_FEATURE_COLS = [f"csi_{i}" for i in range(128)]

# ESP32-C6 HT20 CSI: these 8 subcarriers (DC + guard bands) never carry signal —
# their I/Q bytes are always 0, so amplitude/phase there is meaningless.
NULL_SUBCARRIERS = (0, 1, 2, 3, 32, 61, 62, 63)
VALID_SUBCARRIERS = tuple(i for i in range(64) if i not in NULL_SUBCARRIERS)


def load_session(session_dir: str):
    csi = pd.read_csv(os.path.join(session_dir, "csi.csv"))
    camera = pd.read_csv(os.path.join(session_dir, "camera.csv"))
    clap = pd.read_csv(os.path.join(session_dir, "clap.csv"))
    # Normalize column names — camera writes timestamp_s, csi writes wall_time_s
    if "timestamp_s" in camera.columns and "wall_time_s" not in camera.columns:
        camera = camera.rename(columns={"timestamp_s": "wall_time_s"})
    return csi, camera, clap


def trim_to_session(df: pd.DataFrame, clap: pd.DataFrame,
                    time_col: str = "wall_time_s") -> pd.DataFrame:
    """Keep rows between the START clap and the STOP clap."""
    starts = clap[clap["event_name"] == "start"]["wall_time_s"]
    stops = clap[clap["event_name"] == "stop"]["wall_time_s"]
    if len(starts) == 0 or len(stops) == 0:
        return df
    t_start = starts.iloc[0]
    t_stop = stops.iloc[-1]
    return df[(df[time_col] >= t_start) & (df[time_col] <= t_stop)].reset_index(drop=True)


def amplitudes_from_csi(row_csi: np.ndarray) -> np.ndarray:
    """Convert 128 interleaved I/Q bytes to 64 amplitudes."""
    # bytes are signed int8 (csi_collector wrote them as signed already)
    iq = row_csi.reshape(-1, 2).astype(np.float32)
    return np.sqrt(iq[:, 0] ** 2 + iq[:, 1] ** 2)


def phase_from_csi(row_csi: np.ndarray) -> np.ndarray:
    """Raw phase per subcarrier from 128 interleaved I/Q bytes (radians)."""
    iq = row_csi.reshape(-1, 2).astype(np.float32)
    return np.arctan2(iq[:, 1], iq[:, 0])


def sanitized_phase(phase_row: np.ndarray, amp_row: np.ndarray | None = None) -> np.ndarray:
    """Remove linear trend from per-packet phase (cancels CFO + sampling offset).

    Reference: Sen et al. "PinLoc" / Ma et al. "WiFi sensing with channel state
    information" survey. After this transform, what's left is the multipath
    pattern unique to the propagation environment.

    The linear trend `alpha*k + beta` absorbs:
        beta  — per-packet random phase offset (packet-detection timing)
        alpha — sampling frequency offset slope across subcarriers
    """
    n = len(phase_row)
    k = np.arange(n, dtype=np.float32)
    # Unwrap to avoid 2pi jumps
    phi = np.unwrap(phase_row).astype(np.float32)
    # If amplitudes are given, weight the fit by amplitude (ignore null subcarriers)
    if amp_row is not None:
        w = amp_row + 1e-3
    else:
        w = np.ones_like(phi)
    # Weighted least-squares fit: phi ≈ alpha*k + beta
    sw = w.sum()
    swk = (w * k).sum()
    swkk = (w * k * k).sum()
    swp = (w * phi).sum()
    swkp = (w * k * phi).sum()
    det = sw * swkk - swk * swk
    if det < 1e-6:
        return phi - phi.mean()
    alpha = (sw * swkp - swk * swp) / det
    beta = (swkk * swp - swk * swkp) / det
    return phi - (alpha * k + beta)


def sanitized_phase_masked(phase_row: np.ndarray, amp_row: np.ndarray | None = None) -> np.ndarray:
    """Like sanitized_phase, but the 8 null subcarriers are excluded entirely.

    The ESP32-C6 reports I/Q = 0 on NULL_SUBCARRIERS, so their phase is
    meaningless — yet sanitized_phase still unwraps through them and lets them
    perturb the linear fit (measured: valid-subcarrier residual std 0.08 rad
    with nulls in vs 0.02 rad with them excluded). Here both the unwrap and
    the weighted fit see only VALID_SUBCARRIERS; the fit keeps the ORIGINAL
    subcarrier indices as k, so `alpha` stays in per-subcarrier units. Null
    positions in the (64,) output are set to 0.0.
    """
    valid = np.asarray(VALID_SUBCARRIERS)
    k = valid.astype(np.float32)
    # Unwrap over the 56 valid subcarriers only — no 2pi jumps from the zeros
    phi = np.unwrap(np.asarray(phase_row)[valid]).astype(np.float32)
    if amp_row is not None:
        w = np.asarray(amp_row)[valid] + 1e-3
    else:
        w = np.ones_like(phi)
    out = np.zeros(64, dtype=np.float32)
    # Weighted least-squares fit: phi ≈ alpha*k + beta (same as sanitized_phase)
    sw = w.sum()
    swk = (w * k).sum()
    swkk = (w * k * k).sum()
    swp = (w * phi).sum()
    swkp = (w * k * phi).sum()
    det = sw * swkk - swk * swk
    if det < 1e-6:
        out[valid] = phi - phi.mean()
        return out
    alpha = (sw * swkp - swk * swp) / det
    beta = (swkk * swp - swk * swkp) / det
    out[valid] = phi - (alpha * k + beta)
    return out


def build_aligned_dataset(csi: pd.DataFrame, camera: pd.DataFrame,
                          clap: pd.DataFrame | None = None,
                          max_time_gap_s: float = 0.5) -> pd.DataFrame:
    """For each CSI packet, attach the camera position nearest in time.

    Returns a DataFrame with columns:
        wall_time_s, board_id, rssi, csi_0..csi_127,
        amp_0..amp_63,              # computed amplitudes
        x_cm, y_cm, grid_x_cm, grid_y_cm,
        time_gap_s                  # how stale the camera label is
    """
    if clap is not None and not clap.empty:
        csi = trim_to_session(csi, clap, "wall_time_s")
        camera = trim_to_session(camera, clap, "wall_time_s")

    # Keep only camera frames with a detection
    camera = camera[camera["detected"] == 1].reset_index(drop=True)
    if camera.empty:
        raise RuntimeError("No detected camera frames in this session.")

    csi = csi.sort_values("wall_time_s").reset_index(drop=True)
    camera = camera.sort_values("wall_time_s").reset_index(drop=True)

    # Nearest-time join (camera carries the labels)
    merged = pd.merge_asof(
        csi,
        camera[["wall_time_s", "x_cm", "y_cm", "grid_x_cm", "grid_y_cm"]],
        on="wall_time_s",
        direction="nearest",
    )
    # Compute the gap against the matched camera ts
    cam_ts = camera["wall_time_s"].to_numpy()
    csi_ts = merged["wall_time_s"].to_numpy()
    idx = np.searchsorted(cam_ts, csi_ts)
    idx = np.clip(idx, 1, len(cam_ts) - 1)
    left = cam_ts[idx - 1]
    right = cam_ts[idx]
    pick_right = (np.abs(right - csi_ts) < np.abs(csi_ts - left))
    nearest = np.where(pick_right, right, left)
    merged["time_gap_s"] = np.abs(merged["wall_time_s"] - nearest)

    # Drop rows where camera position is too stale
    merged = merged[merged["time_gap_s"] <= max_time_gap_s].reset_index(drop=True)

    # Compute amplitudes from the 128 raw I/Q bytes
    csi_data = merged[CSI_FEATURE_COLS].to_numpy()
    amps = np.empty((csi_data.shape[0], 64), dtype=np.float32)
    for i, row in enumerate(csi_data):
        amps[i] = amplitudes_from_csi(row)
    for i in range(64):
        merged[f"amp_{i}"] = amps[:, i]

    return merged


def feature_matrix(df: pd.DataFrame, use_amps: bool = True,
                   include_rssi: bool = True,
                   include_board_onehot: bool = True) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix from an aligned DataFrame."""
    feats = []
    names = []

    if use_amps:
        amp_cols = [f"amp_{i}" for i in range(64)]
        feats.append(df[amp_cols].to_numpy(dtype=np.float32))
        names.extend(amp_cols)
    else:
        feats.append(df[CSI_FEATURE_COLS].to_numpy(dtype=np.float32))
        names.extend(CSI_FEATURE_COLS)

    if include_rssi:
        feats.append(df[["rssi"]].to_numpy(dtype=np.float32))
        names.append("rssi")

    if include_board_onehot:
        for bid in detect_board_ids(df):
            feats.append((df["board_id"].to_numpy() == bid).astype(np.float32).reshape(-1, 1))
            names.append(f"board_{bid}")

    X = np.hstack(feats)
    return X, names


def grid_cell_id(df: pd.DataFrame) -> pd.Series:
    """Map (grid_x_cm, grid_y_cm) to a single integer cell id."""
    return df["grid_x_cm"].astype(int).astype(str) + "_" + df["grid_y_cm"].astype(int).astype(str)


def detect_board_ids(csi: pd.DataFrame) -> tuple[int, ...]:
    """RX board IDs present in a raw CSI frame, sorted ascending.

    Replaces hardcoded (1, 2, 3) so the pipeline works with any board labels
    (e.g. 1, 4, 5) and stays backward-compatible with older sessions.
    """
    return tuple(int(b) for b in sorted(pd.unique(csi["board_id"])))


def boards_in_multiboard_df(df: pd.DataFrame) -> tuple[int, ...]:
    """Board IDs present as b{N}_rssi columns in a multiboard dataframe, sorted."""
    ids = []
    for c in df.columns:
        if c.startswith("b") and c.endswith("_rssi"):
            try:
                ids.append(int(c[1:-len("_rssi")]))
            except ValueError:
                pass
    return tuple(sorted(ids))


def drop_duplicate_csi_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop frames whose joint multi-board CSI is identical to the previous frame.

    These occur when the camera advanced but no board received a new packet, so
    the same CSI was reused. They carry no new signal and, under a random
    train/test split, leak an identical feature vector across train and test —
    inflating accuracy. Keeps the first frame of each identical run.

    Amplitudes alone are a sufficient signature: a reused packet yields
    bit-identical amps (and phase), and distinct packets practically never
    collide across all subcarriers/boards — so phase columns add nothing here.
    """
    if df.empty:
        return df
    amp_cols = [c for c in df.columns if "_amp_" in c]
    if not amp_cols:
        return df
    sig = df[amp_cols]
    is_dup = (sig == sig.shift()).all(axis=1)
    return df.loc[~is_dup].reset_index(drop=True)


def build_multiboard_dataset(csi: pd.DataFrame, camera: pd.DataFrame,
                             clap: pd.DataFrame | None = None,
                             max_age_s: float = 0.5,
                             board_ids: tuple[int, ...] | None = None,
                             dedup: bool = True,
                             phase_mode: str = "legacy") -> pd.DataFrame:
    """One sample per camera frame: the most recent CSI from each RX board.

    A single CSI packet only carries one RX's view of the TX. The localization
    signal really lives in the differences across the RX boards. This function
    groups packets by board, sorts by time, and for each camera frame picks the
    latest CSI from each board that arrived within `max_age_s` seconds. Samples
    missing any board are dropped.

    `board_ids` defaults to whatever IDs are present in the data (auto-detected
    and sorted), so this works for boards labeled (1, 2, 3), (1, 4, 5), etc.

    Output columns (one b{ID}_* group per board, in sorted ID order):
        wall_time_s, x_cm, y_cm, grid_x_cm, grid_y_cm,
        b{ID}_amp_0..b{ID}_amp_63, b{ID}_phase_0..b{ID}_phase_63,
        b{ID}_rssi, b{ID}_age_s

    `dedup=True` (default) removes consecutive frames whose joint multi-board
    CSI is identical to the previous frame (the camera advanced but no board got
    a new packet); `dedup=False` keeps them, reproducing the naive/leaky numbers.

    `phase_mode` picks the per-packet phase sanitizer: "legacy" (default)
    unwraps + detrends all 64 subcarriers (sanitized_phase); "masked" excludes
    the 8 NULL_SUBCARRIERS from the unwrap and the fit and writes 0.0 at their
    positions (sanitized_phase_masked) — less noisy on the valid subcarriers.
    """
    if phase_mode not in ("legacy", "masked"):
        raise ValueError(f"phase_mode must be 'legacy' or 'masked', got {phase_mode!r}")
    if clap is not None and not clap.empty:
        csi = trim_to_session(csi, clap, "wall_time_s")
        camera = trim_to_session(camera, clap, "wall_time_s")

    camera = camera[camera["detected"] == 1].sort_values("wall_time_s").reset_index(drop=True)
    if camera.empty:
        raise RuntimeError("No detected camera frames in this session.")

    csi = csi.sort_values("wall_time_s").reset_index(drop=True)

    if board_ids is None:
        board_ids = detect_board_ids(csi)

    # Precompute amplitudes + sanitized phase per packet
    phase_fn = sanitized_phase if phase_mode == "legacy" else sanitized_phase_masked
    csi_data = csi[CSI_FEATURE_COLS].to_numpy()
    amps_all = np.empty((csi_data.shape[0], 64), dtype=np.float32)
    phase_all = np.empty((csi_data.shape[0], 64), dtype=np.float32)
    for i, row in enumerate(csi_data):
        amps_all[i] = amplitudes_from_csi(row)
        raw_phase = phase_from_csi(row)
        phase_all[i] = phase_fn(raw_phase, amps_all[i])

    # Per-board frame: time-sorted CSI for board b
    out_rows = {
        "wall_time_s": camera["wall_time_s"].to_numpy(),
        "x_cm": camera["x_cm"].to_numpy(),
        "y_cm": camera["y_cm"].to_numpy(),
        "grid_x_cm": camera["grid_x_cm"].to_numpy(),
        "grid_y_cm": camera["grid_y_cm"].to_numpy(),
    }
    valid_mask = np.ones(len(camera), dtype=bool)

    for b in board_ids:
        mask = (csi["board_id"].to_numpy() == b)
        b_times = csi["wall_time_s"].to_numpy()[mask]
        b_rssi = csi["rssi"].to_numpy()[mask].astype(np.float32)
        b_amps = amps_all[mask]
        if len(b_times) == 0:
            raise RuntimeError(f"No CSI from board {b}.")

        # For each camera frame, find the last b_times index <= frame_time
        idx = np.searchsorted(b_times, camera["wall_time_s"].to_numpy(),
                              side="right") - 1
        # Frames before the first packet from board b are invalid
        in_range = idx >= 0
        age = np.full(len(camera), np.nan, dtype=np.float32)
        age[in_range] = (camera["wall_time_s"].to_numpy()[in_range]
                         - b_times[idx[in_range]])
        too_old = ~in_range | (age > max_age_s)
        valid_mask &= ~too_old

        b_phase = phase_all[mask]
        picked = np.where(in_range, idx, 0)  # safe index for fancy indexing
        for i in range(64):
            out_rows[f"b{b}_amp_{i}"] = b_amps[picked, i]
        for i in range(64):
            out_rows[f"b{b}_phase_{i}"] = b_phase[picked, i]
        out_rows[f"b{b}_rssi"] = b_rssi[picked]
        out_rows[f"b{b}_age_s"] = age

    df = pd.DataFrame(out_rows)
    df = df[valid_mask].reset_index(drop=True)
    if dedup:
        df = drop_duplicate_csi_rows(df)
    return df


def multiboard_static_baseline(empty_csi: pd.DataFrame, names: list[str],
                               board_ids: tuple[int, ...] | None = None
                               ) -> np.ndarray:
    """Per-feature empty-room static-channel baseline aligned to `names`.

    `empty_csi` is the raw CSI of an empty-room capture (no camera). For every
    feature name produced by `multiboard_feature_matrix` (e.g. `b1_amp_3`,
    `b4_rssi`, `b5_phase_7`) this returns the matching empty-room statistic:

        b{B}_amp_{i}   -> mean amplitude of board B, subcarrier i over the capture
        b{B}_rssi      -> mean RSSI of board B
        b{B}_phase_{i} -> 0.0  (per-packet sanitised phase has ~zero mean by
                                 construction; a static phase baseline is not
                                 meaningful, so leave those features untouched)

    Subtracting this vector from a walk feature matrix (mlpipe.remove_static_path)
    removes the room's static path, leaving the human-induced perturbation.
    """
    if board_ids is None:
        board_ids = detect_board_ids(empty_csi)
    iq = empty_csi[CSI_FEATURE_COLS].to_numpy().reshape(-1, 64, 2).astype(np.float32)
    amp_all = np.sqrt(iq[:, :, 0] ** 2 + iq[:, :, 1] ** 2)          # (N, 64)
    bid = empty_csi["board_id"].to_numpy()
    rssi = empty_csi["rssi"].to_numpy().astype(np.float64)
    mean_amp, mean_rssi = {}, {}
    for b in board_ids:
        m = bid == b
        if not m.any():
            raise ValueError(f"empty-room capture has no CSI from board {b}")
        mean_amp[b] = amp_all[m].mean(0)
        mean_rssi[b] = float(rssi[m].mean())
    base = np.zeros(len(names), dtype=np.float64)
    for j, nm in enumerate(names):
        b = int(nm[1:].split("_", 1)[0])           # 'b4_amp_3' -> 4
        if "_amp_" in nm:
            base[j] = mean_amp[b][int(nm.rsplit("_", 1)[1])]
        elif nm.endswith("_rssi"):
            base[j] = mean_rssi[b]
        # phase features stay 0.0
    return base


def multiboard_feature_matrix(df: pd.DataFrame,
                              board_ids: tuple[int, ...] | None = None,
                              include_rssi: bool = True,
                              include_phase: bool = False,
                              drop_null_subcarriers: bool = False
                              ) -> tuple[np.ndarray, list[str]]:
    """Flat feature matrix from build_multiboard_dataset output.

    `board_ids` defaults to whatever b{N}_* groups are present, in sorted order.
    `drop_null_subcarriers=True` omits the 8 NULL_SUBCARRIERS from the amp and
    phase groups (56 instead of 64 columns per group; e.g. 3 boards with phase
    + rssi -> 3*(56+56+1) = 339 features instead of 387).
    """
    if board_ids is None:
        board_ids = boards_in_multiboard_df(df)
    subcarriers = VALID_SUBCARRIERS if drop_null_subcarriers else tuple(range(64))
    feats = []
    names = []
    for b in board_ids:
        cols = [f"b{b}_amp_{i}" for i in subcarriers]
        feats.append(df[cols].to_numpy(dtype=np.float32))
        names.extend(cols)
        if include_phase:
            pcols = [f"b{b}_phase_{i}" for i in subcarriers]
            feats.append(df[pcols].to_numpy(dtype=np.float32))
            names.extend(pcols)
        if include_rssi:
            feats.append(df[[f"b{b}_rssi"]].to_numpy(dtype=np.float32))
            names.append(f"b{b}_rssi")
    return np.hstack(feats), names
