import numpy as np
from aruco_track import (estimate_body_center, combine_foot_estimates,
                         marker_axes, body_center_from_axes, ema_unit,
                         ramp_weight, fuse_feet)

# Axis-aligned marker, 10 cm half-size, top edge toward +y (toes).
# OpenCV ArUco corner order: TL, TR, BR, BL.
SQ = np.array([[-10.0, 10.0], [10.0, 10.0], [10.0, -10.0], [-10.0, -10.0]])


def test_right_foot_offset_left_and_back():
    # right foot: body center 20 cm to marker's LEFT (-x), 15 cm back (-y)
    est = estimate_body_center(SQ, "right", 20.0, -15.0)
    assert np.allclose(est, [-20.0, -15.0])


def test_left_foot_offset_right_and_back():
    # left foot: body center 20 cm to marker's RIGHT (+x), 15 cm back (-y)
    est = estimate_body_center(SQ, "left", 20.0, -15.0)
    assert np.allclose(est, [20.0, -15.0])


def test_translation_preserved():
    shifted = SQ + np.array([100.0, 50.0])
    est = estimate_body_center(shifted, "right", 20.0, -15.0)
    assert np.allclose(est, [80.0, 35.0])  # (100-20, 50-15)


def test_degenerate_marker_returns_none():
    zero = np.zeros((4, 2))
    assert estimate_body_center(zero, "right", 20.0, -15.0) is None


def test_combine_both_averages():
    pos, n, method = combine_foot_estimates(np.array([-20.0, -15.0]),
                                            np.array([60.0, -15.0]))
    assert np.allclose(pos, [20.0, -15.0])  # error cancels to true midpoint
    assert n == 2
    assert method == "both"


def test_combine_right_only():
    pos, n, method = combine_foot_estimates(np.array([-20.0, -15.0]), None)
    assert np.allclose(pos, [-20.0, -15.0])
    assert n == 1
    assert method == "right_only"


def test_combine_left_only():
    pos, n, method = combine_foot_estimates(None, np.array([20.0, -15.0]))
    assert np.allclose(pos, [20.0, -15.0])
    assert n == 1
    assert method == "left_only"


def test_combine_none():
    pos, n, method = combine_foot_estimates(None, None)
    assert pos is None
    assert n == 0
    assert method == "none"


def test_full_chain_both_feet_average_to_body_center():
    # Full pipeline: two foot markers 40 cm apart on the x axis, both axis-aligned.
    # Right at origin, left at (40, 0). Each per-foot estimate over/undershoots,
    # but averaging recovers the true center x=20 and keeps the 15 cm heel offset.
    est_r = estimate_body_center(SQ, "right", 20.0, -15.0)               # (-20, -15)
    est_l = estimate_body_center(SQ + np.array([40.0, 0.0]),
                                 "left", 20.0, -15.0)                    # (60, -15)
    pos, n, method = combine_foot_estimates(est_r, est_l)
    assert np.allclose(pos, [20.0, -15.0])
    assert n == 2
    assert method == "both"


def test_marker_axes_axis_aligned():
    center, fwd, rt = marker_axes(SQ)
    assert np.allclose(center, [0.0, 0.0])
    assert np.allclose(fwd, [0.0, 1.0])   # top edge toward +y
    assert np.allclose(rt, [1.0, 0.0])    # right edge toward +x


def test_marker_axes_degenerate_none():
    assert marker_axes(np.zeros((4, 2))) is None


def test_body_center_from_axes_matches_offsets():
    center = np.array([0.0, 0.0]); fwd = np.array([0.0, 1.0]); rt = np.array([1.0, 0.0])
    assert np.allclose(body_center_from_axes(center, fwd, rt, "right", 20.0, -15.0), [-20.0, -15.0])
    assert np.allclose(body_center_from_axes(center, fwd, rt, "left", 20.0, -15.0), [20.0, -15.0])


def test_ema_unit_none_prev_returns_new():
    new = np.array([0.0, 1.0])
    assert np.allclose(ema_unit(None, new, 0.3), [0.0, 1.0])


def test_ema_unit_alpha_one_returns_new():
    prev = np.array([1.0, 0.0]); new = np.array([0.0, 1.0])
    assert np.allclose(ema_unit(prev, new, 1.0), [0.0, 1.0])


def test_ema_unit_blends_and_renormalizes():
    prev = np.array([1.0, 0.0]); new = np.array([0.0, 1.0])
    out = ema_unit(prev, new, 0.5)
    assert np.allclose(out, [np.sqrt(0.5), np.sqrt(0.5)])  # 45 deg, unit length
    assert np.isclose(np.linalg.norm(out), 1.0)


# ── ramp_weight / fuse_feet: confidence-weighted two-foot fusion ───────────

def test_ramp_weight_up_down_and_clamp():
    assert abs(ramp_weight(0.0, True, 8, 8) - 0.125) < 1e-9    # ramps up
    assert abs(ramp_weight(1.0, False, 8, 8) - 0.875) < 1e-9   # decays down
    assert ramp_weight(1.0, True, 8, 8) == 1.0                 # clamped at 1
    assert ramp_weight(0.0, False, 8, 8) == 0.0                # clamped at 0


def test_ramp_weight_one_frame_dropout_stays_high():
    # hysteresis: a single missed frame from full weight barely changes it
    assert ramp_weight(1.0, False, 8, 8) > 0.8


def test_fuse_feet_both_equal_is_midpoint():
    pos, n, method = fuse_feet(np.array([0.0, 0.0]), 1.0, np.array([10.0, 4.0]), 1.0)
    assert np.allclose(pos, [5.0, 2.0]) and n == 2 and method == "both"


def test_fuse_feet_below_threshold_is_transitional_not_a_jump():
    # left foot ramping in (w=0.2 < 0.5): method stays right_only, but the
    # position is only slightly blended — no instant jump to the midpoint.
    pos, n, method = fuse_feet(np.array([0.0, 0.0]), 1.0, np.array([10.0, 0.0]), 0.2)
    assert method == "right_only" and n == 1
    assert 0.0 < pos[0] < 5.0


def test_fuse_feet_single_foot():
    pos, n, method = fuse_feet(np.array([3.0, 7.0]), 1.0, None, 0.0)
    assert np.allclose(pos, [3.0, 7.0]) and n == 1 and method == "right_only"


def test_fuse_feet_none():
    assert fuse_feet(None, 0.0, None, 0.0) == (None, 0, "none")
