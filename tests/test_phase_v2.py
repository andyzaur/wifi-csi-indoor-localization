"""Phase v2 — null-subcarrier-aware sanitized phase (masked mode).

ESP32-C6 HT20 CSI has 8 always-null subcarriers (DC + guards, I/Q = 0).
`sanitized_phase_masked` excludes them from the unwrap + linear fit;
`build_multiboard_dataset(phase_mode=...)` selects the sanitizer and
`multiboard_feature_matrix(drop_null_subcarriers=True)` drops their columns.
Defaults must keep legacy behavior bit-for-bit.
"""
import numpy as np
import pandas as pd
import pytest

from dataset import (CSI_FEATURE_COLS, NULL_SUBCARRIERS, VALID_SUBCARRIERS,
                     amplitudes_from_csi, phase_from_csi,
                     sanitized_phase, sanitized_phase_masked,
                     build_multiboard_dataset, multiboard_feature_matrix)


def _make_session(n_frames=6, boards=(1, 4, 5), seed=0):
    # One distinct packet per board just before each camera frame (age 5 ms),
    # random I/Q with the 8 null subcarriers zeroed like the real hardware.
    rng = np.random.default_rng(seed)
    times = 0.04 * np.arange(n_frames)
    cam = pd.DataFrame({
        "wall_time_s": times,
        "x_cm": 10.0 + np.arange(n_frames), "y_cm": np.zeros(n_frames),
        "grid_x_cm": np.zeros(n_frames, dtype=int),
        "grid_y_cm": np.zeros(n_frames, dtype=int),
        "detected": np.ones(n_frames, dtype=int),
    })
    rows = []
    for b in boards:
        for j, t in enumerate(times):
            iq = rng.integers(-30, 30, size=128)
            for sc in NULL_SUBCARRIERS:
                iq[2 * sc] = 0
                iq[2 * sc + 1] = 0
            rows.append({"wall_time_s": t - 0.005, "board_id": b,
                         "rssi": -40 - b, "channel": 6,
                         "timestamp_us": int(t * 1e6), "rx_seq": j,
                         "csi_len": 128,
                         **{f"csi_{i}": int(iq[i]) for i in range(128)}})
    csi = pd.DataFrame(rows)
    return csi, cam


# ── constants ─────────────────────────────────────────────────────────────

def test_subcarrier_constants_partition_64():
    assert NULL_SUBCARRIERS == (0, 1, 2, 3, 32, 61, 62, 63)
    assert len(VALID_SUBCARRIERS) == 56
    assert sorted(NULL_SUBCARRIERS + VALID_SUBCARRIERS) == list(range(64))


# ── (a) legacy default unchanged ──────────────────────────────────────────

def test_default_phase_mode_is_legacy_bit_for_bit():
    csi, cam = _make_session()
    default = build_multiboard_dataset(csi, cam)
    legacy = build_multiboard_dataset(csi, cam, phase_mode="legacy")
    pd.testing.assert_frame_equal(default, legacy, check_exact=True)
    # and the default really is sanitized_phase, recomputed by hand for the
    # first frame's board-1 packet
    raw = csi[csi["board_id"] == 1][CSI_FEATURE_COLS].to_numpy()[0]
    expect = sanitized_phase(phase_from_csi(raw), amplitudes_from_csi(raw))
    got = default[[f"b1_phase_{i}" for i in range(64)]].iloc[0].to_numpy(dtype=np.float32)
    assert np.array_equal(got, expect)


def test_phase_mode_validated():
    csi, cam = _make_session(n_frames=2, boards=(1,))
    with pytest.raises(ValueError, match="phase_mode"):
        build_multiboard_dataset(csi, cam, phase_mode="bogus")


# ── (b) masked mode ───────────────────────────────────────────────────────

def test_masked_zeroes_nulls_and_keeps_valids_finite():
    csi, cam = _make_session()
    df = build_multiboard_dataset(csi, cam, phase_mode="masked")
    assert len(df) == 6
    for b in (1, 4, 5):
        for sc in NULL_SUBCARRIERS:
            assert (df[f"b{b}_phase_{sc}"] == 0.0).all()
        valid_cols = [f"b{b}_phase_{i}" for i in VALID_SUBCARRIERS]
        assert np.isfinite(df[valid_cols].to_numpy()).all()


def test_masked_recovers_clean_linear_phase_where_legacy_does_not():
    # Clean ramp alpha*k + beta whose +/-pi wrap crossing lands on the null at
    # k=32: legacy unwrap misses the 2pi correction there (the embedded zero
    # hides the jump), corrupting every later subcarrier; masked unwraps over
    # the valid 56 only and recovers the ramp exactly.
    k = np.arange(64, dtype=np.float32)
    alpha = 0.2
    beta = float(np.pi) - alpha * 32 + 2 * float(np.pi)
    phase = ((alpha * k + beta + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)
    amp = np.full(64, 10.0, dtype=np.float32)
    nulls = list(NULL_SUBCARRIERS)
    valid = list(VALID_SUBCARRIERS)
    phase[nulls] = 0.0      # hardware: I/Q = 0 -> atan2(0, 0) = 0
    amp[nulls] = 0.0

    masked = sanitized_phase_masked(phase, amp)
    legacy = sanitized_phase(phase, amp)
    assert masked.shape == (64,)
    assert np.abs(masked[valid]).max() < 1e-3          # pure trend -> ~0 residual
    assert np.abs(legacy[valid]).max() > 0.5           # legacy is corrupted
    assert (masked[nulls] == 0.0).all()
    assert np.isfinite(masked).all()


# ── (c) drop_null_subcarriers ─────────────────────────────────────────────

def test_drop_null_subcarriers_339_features_for_3_boards():
    csi, cam = _make_session()
    df = build_multiboard_dataset(csi, cam, phase_mode="masked")
    X, names = multiboard_feature_matrix(df, include_phase=True,
                                         drop_null_subcarriers=True)
    assert X.shape == (len(df), 339)                   # 3*(56+56+1)
    assert len(names) == 339
    for nm in names:
        if "_amp_" in nm or "_phase_" in nm:
            assert int(nm.rsplit("_", 1)[1]) not in NULL_SUBCARRIERS
    # names line up with the matrix columns
    assert np.array_equal(X, df[names].to_numpy(dtype=np.float32))


def test_drop_null_subcarriers_default_keeps_387():
    csi, cam = _make_session()
    df = build_multiboard_dataset(csi, cam)
    X, names = multiboard_feature_matrix(df, include_phase=True)
    assert X.shape == (len(df), 387)                   # 3*(64+64+1)
    assert len(names) == 387


# ── (d) degenerate fallback ───────────────────────────────────────────────

def test_masked_degenerate_det_falls_back_without_nan():
    # amp = -1e-3 makes every fit weight exactly 0 -> det 0 -> mean-removal
    # fallback (same logic as sanitized_phase), never NaN.
    rng = np.random.default_rng(3)
    phase = rng.uniform(-np.pi, np.pi, 64).astype(np.float32)
    amp = np.full(64, -1e-3, dtype=np.float32)
    out = sanitized_phase_masked(phase, amp)
    assert out.shape == (64,)
    assert np.isfinite(out).all()
    assert (out[list(NULL_SUBCARRIERS)] == 0.0).all()
    assert abs(out[list(VALID_SUBCARRIERS)].mean()) < 1e-5   # mean-removed
