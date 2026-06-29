import numpy as np
from repro_v3 import bootstrap_ci, per_sample_error, room_half_spread


def test_per_sample_error_euclidean():
    y = np.array([[0.0, 0.0], [3.0, 0.0]])
    pred = np.array([[0.0, 4.0], [0.0, 0.0]])
    assert np.allclose(per_sample_error(y, pred), [4.0, 3.0])


def test_bootstrap_ci_point_is_exact_statistic():
    errors = np.arange(1.0, 101.0)          # median = 50.5
    point, lo, hi = bootstrap_ci(errors, B=500, seed=0)
    assert abs(point - 50.5) < 1e-9         # point estimate is the true statistic
    assert lo < point < hi                  # interval brackets the point


def test_bootstrap_ci_is_deterministic_given_seed():
    errors = np.random.default_rng(1).normal(40, 10, 200)
    a = bootstrap_ci(errors, B=300, seed=7)
    b = bootstrap_ci(errors, B=300, seed=7)
    assert a == b                           # same seed -> identical interval


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(2)
    small = bootstrap_ci(rng.normal(40, 10, 50), B=400, seed=0)
    big = bootstrap_ci(rng.normal(40, 10, 5000), B=400, seed=0)
    assert (big[2] - big[1]) < (small[2] - small[1])   # CI width shrinks with n


def test_room_half_spread_is_median_distance_from_centroid():
    # four points at +/-10 on each axis: centroid origin, each 10 away
    y = np.array([[10.0, 0.0], [-10.0, 0.0], [0.0, 10.0], [0.0, -10.0]])
    assert abs(room_half_spread(y) - 10.0) < 1e-9


def test_room_half_spread_zero_when_all_identical():
    y = np.full((20, 2), 7.0)
    assert room_half_spread(y) == 0.0
