"""Pure-part tests for the ml_drift v2 extensions (no training, no sessions):
--methods slugging/filtering, --groups parsing/remap, per-session
standardization, board-slice derivation, --dump-preds npz round-trip, and the
causal sess-std / twin build_raw_features spec wiring (synthetic names_cache).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_drift import (ROSTER, BASELINE_NAME, AUG_DROP_P, AUG_GAIN_STD,
                      apply_session_groups, board_slices_from_names,
                      build_raw_features, dump_fold_preds, file_slug,
                      method_slug, parse_groups, per_session_standardize,
                      select_methods, spec_feature_key)
from mlpipe import causal_session_standardize, window_stats


# ── method slugging ───────────────────────────────────────────────────────

def test_method_slug_is_lowercase_alnum_only():
    assert method_slug("+phase_v2") == "phasev2"
    assert method_slug("baseline (amp+rssi)") == "baselineamprssi"
    assert method_slug("per-sess std +phase_v2") == "persessstdphasev2"
    assert method_slug("CORAL align") == "coralalign"


def test_file_slug_keeps_dashes():
    assert file_slug("+phase_v2 +aug") == "phase-v2-aug"
    assert file_slug("20260606_02_session") == "20260606-02-session"
    assert file_slug("baseline (amp+rssi)") == "baseline-amp-rssi"


def test_roster_slugs_are_unique():
    slugs = [method_slug(n) for n, _ in ROSTER]
    assert len(slugs) == len(set(slugs))


# ── --methods filtering ───────────────────────────────────────────────────

def test_select_methods_default_is_everything():
    assert select_methods(ROSTER, None) == list(ROSTER)
    assert select_methods(ROSTER, "") == list(ROSTER)


def test_select_methods_by_exact_name_and_slug():
    out = select_methods(ROSTER, "phasev2,baseline (amp+rssi)")
    assert [n for n, _ in out] == ["+phase_v2", BASELINE_NAME]
    # exact slug match wins over the longer 'phasev2aug' (no prefix confusion)
    assert out[0][1] == dict(phase=True, phase_v2=True)


def test_select_methods_unambiguous_prefix():
    out = select_methods(ROSTER, "baseline")
    assert [n for n, _ in out] == [BASELINE_NAME]


def test_select_methods_ambiguous_prefix_aborts():
    with pytest.raises(SystemExit, match="ambiguous"):
        select_methods(ROSTER, "coral")     # CORAL align vs CORAL +phase


def test_select_methods_unknown_aborts():
    with pytest.raises(SystemExit, match="matches nothing"):
        select_methods(ROSTER, "nosuchmethod")


def test_select_methods_dedups_repeats():
    out = select_methods(ROSTER, "baseline,baseline")
    assert len(out) == 1


def test_new_roster_entries_present_with_expected_specs():
    d = dict(ROSTER)
    assert d["+phase_v2"] == dict(phase=True, phase_v2=True)
    assert d["per-session std"] == dict(sess_std=True)
    assert d["per-sess std +phase_v2"] == dict(sess_std=True, phase=True, phase_v2=True)
    assert d["+phase_v2 +aug"] == dict(phase=True, phase_v2=True, aug=True)
    assert AUG_GAIN_STD == 0.05 and AUG_DROP_P == 0.1


def test_spec_feature_key_routing():
    assert spec_feature_key(dict()) == "amp"
    assert spec_feature_key(dict(phase=True)) == "phase"
    assert spec_feature_key(dict(phase=True, phase_v2=True)) == "phase_v2"


# ── --groups parsing + remap ──────────────────────────────────────────────

def test_parse_groups_maps_sessions_to_groups():
    m = parse_groups("g1=sessA;g2=sessB,sessC")
    assert m == {"sessA": "g1", "sessB": "g2", "sessC": "g2"}


def test_parse_groups_rejects_bad_part_and_duplicates():
    with pytest.raises(SystemExit):
        parse_groups("g1sessA")                    # no '='
    with pytest.raises(SystemExit, match="twice"):
        parse_groups("g1=sessA;g2=sessA")          # one session in two groups


def test_apply_session_groups_remaps_and_keeps_order():
    df = pd.DataFrame({"session": ["sessA", "sessB", "sessA", "sessC"],
                       "v": [1, 2, 3, 4]})
    out = apply_session_groups(df, parse_groups("d1=sessA;d2=sessB,sessC"))
    assert list(out["session"]) == ["d1", "d2", "d1", "d2"]
    assert list(out["v"]) == [1, 2, 3, 4]          # rows untouched
    assert list(df["session"]) == ["sessA", "sessB", "sessA", "sessC"]  # copy


def test_apply_session_groups_errors_on_unmapped():
    df = pd.DataFrame({"session": ["sessA", "sessX"]})
    with pytest.raises(SystemExit, match="unmapped"):
        apply_session_groups(df, {"sessA": "d1"})


# ── per-session standardization (sess_std) ────────────────────────────────

def test_per_session_standardize_zero_mean_unit_std_per_session():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(5.0, 3.0, size=(50, 4)),     # session A: shifted/scaled
                   rng.normal(-2.0, 0.5, size=(70, 4))])   # session B: different stats
    groups = np.array(["A"] * 50 + ["B"] * 70)
    out = per_session_standardize(X, groups)
    for g in ("A", "B"):
        m = groups == g
        assert np.allclose(out[m].mean(0), 0.0, atol=1e-9)
        assert np.allclose(out[m].std(0), 1.0, atol=1e-6)


def test_per_session_standardize_constant_feature_and_no_mutation():
    X = np.column_stack([np.full(6, 7.0), np.arange(6, dtype=float)])
    groups = np.array(["A"] * 3 + ["B"] * 3)
    X_orig = X.copy()
    out = per_session_standardize(X, groups)
    assert np.allclose(out[:, 0], 0.0)             # std 0 -> eps floor, no NaN/inf
    assert np.isfinite(out).all()
    assert np.array_equal(X, X_orig)               # input not modified in place


# ── board_slices_from_names (aug wiring) ──────────────────────────────────

def test_board_slices_group_by_board_prefix_including_rssi():
    names = ["b1_amp_0", "b1_amp_5", "b1_phase_4", "b1_rssi",
             "b4_amp_0", "b4_rssi",
             "b5_amp_0", "b5_phase_9", "b5_rssi"]
    slices = board_slices_from_names(names)
    assert len(slices) == 3                        # sorted board order 1, 4, 5
    assert slices[0].tolist() == [0, 1, 2, 3]
    assert slices[1].tolist() == [4, 5]
    assert slices[2].tolist() == [6, 7, 8]
    # disjoint and complete over board-prefixed names
    allidx = np.concatenate(slices)
    assert sorted(allidx.tolist()) == list(range(len(names)))


def test_board_slices_ignore_non_board_columns():
    slices = board_slices_from_names(["b2_amp_0", "bias", "rssi", "b2_rssi"])
    assert len(slices) == 1
    assert slices[0].tolist() == [0, 3]


# ── --dump-preds npz round-trip ───────────────────────────────────────────

def test_dump_fold_preds_npz_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    te = np.arange(100, 110)
    y_xy = rng.normal(size=(10, 2)) * 50.0
    pred = y_xy + rng.normal(size=(10, 2))
    wt = 1000.0 + 0.04 * np.arange(10)
    gx = np.repeat([0.0, 60.0], 5)
    gy = np.repeat([60.0, 120.0], 5)
    path = dump_fold_preds(str(tmp_path), "+phase_v2 +aug", "20260606_02_session",
                           te, y_xy, pred, wt, gx, gy)
    assert os.path.basename(path) == "preds__phase-v2-aug__20260606-02-session.npz"
    assert os.path.isfile(path)
    z = np.load(path)
    assert set(z.files) == {"idx", "y_xy", "pred_xy", "wall_time_s",
                            "grid_x_cm", "grid_y_cm"}
    assert np.array_equal(z["idx"], te)
    assert np.allclose(z["y_xy"], y_xy)
    assert np.allclose(z["pred_xy"], pred)
    assert np.allclose(z["wall_time_s"], wt)
    assert np.allclose(z["grid_x_cm"], gx)
    assert np.allclose(z["grid_y_cm"], gy)


# ── causal sess-std + twin spec wiring (build_raw_features, synthetic) ────

def _v2_cache_and_df(n_a=30, n_b=20, seed=0):
    """Synthetic phase_v2-layout names_cache entry + session frame: a
    pre-populated cache lets build_raw_features run without dataset/sessions
    (it only builds features for keys it has not seen)."""
    rng = np.random.default_rng(seed)
    n = n_a + n_b
    names = [f"b1_amp_{k}" for k in range(4)] + ["b1_phase_0", "b1_rssi"]
    X = rng.normal(5.0, 3.0, size=(n, len(names)))
    df = pd.DataFrame({"session": ["A"] * n_a + ["B"] * n_b})
    return df, {"phase_v2": (X, names)}


def test_sess_std_causal_wiring_differs_from_batch_same_shape():
    df, cache = _v2_cache_and_df()
    X0 = cache["phase_v2"][0]
    Xb = build_raw_features(df, dict(sess_std=True, phase=True, phase_v2=True),
                            {}, cache)
    Xc = build_raw_features(df, dict(sess_std_causal=True, phase=True,
                                     phase_v2=True), {}, cache)
    assert Xb.shape == Xc.shape == X0.shape
    assert not np.allclose(Xb, Xc)     # expanding stats != whole-session stats
    # and each flag routes to the right normalizer, grouped by session
    groups = df["session"].to_numpy()
    assert np.allclose(Xb, per_session_standardize(X0, groups))
    assert np.allclose(Xc, causal_session_standardize(X0, groups))


def test_sess_std_and_causal_are_mutually_exclusive():
    df, cache = _v2_cache_and_df()
    with pytest.raises(AssertionError, match="mutually exclusive"):
        build_raw_features(df, dict(sess_std=True, sess_std_causal=True,
                                    phase=True, phase_v2=True), {}, cache)


def test_twin_appends_2x_amp_window_stats_after_standardization():
    df, cache = _v2_cache_and_df()
    names = cache["phase_v2"][1]
    X = build_raw_features(df, dict(sess_std=True, phase=True, phase_v2=True,
                                    twin=20), {}, cache)
    base = build_raw_features(df, dict(sess_std=True, phase=True, phase_v2=True),
                              {}, cache)
    n_amp = sum("_amp_" in n for n in names)
    # +2 cols per amp col (rolling mean|var) — 339 + 2*168 = 675 on the real v2 layout
    assert X.shape == (len(base), len(names) + 2 * n_amp)
    assert np.allclose(X[:, :len(names)], base)    # originals untouched, twin on the right
    amp_idx = [i for i, n in enumerate(names) if "_amp_" in n]
    expect = window_stats(base[:, amp_idx], df["session"].to_numpy(), 20)
    assert np.allclose(X[:, len(names):], expect)  # stats of the STANDARDIZED amps


def test_twin_block_is_causal_across_group_boundary():
    df, cache = _v2_cache_and_df(seed=3)
    X0, names = cache["phase_v2"]
    spec = dict(phase=True, phase_v2=True, twin=5)  # no std: isolate the twin block
    Xa = build_raw_features(df, spec, {}, cache)
    X1 = X0.copy()
    X1[25:30] += 100.0                              # session A's tail only
    Xb = build_raw_features(df, spec, {}, {"phase_v2": (X1, names)})
    assert np.array_equal(Xa[:25], Xb[:25])         # past frames unchanged
    assert not np.allclose(Xa[25:30], Xb[25:30])
    assert np.array_equal(Xa[30:], Xb[30:])         # session B sealed off


def test_new_causal_and_twin_roster_entries_slug_and_select():
    d = dict(ROSTER)
    assert d["per-sess std CAUSAL +phase_v2"] == dict(sess_std_causal=True,
                                                      phase=True, phase_v2=True)
    assert d["winner +twin20"] == dict(sess_std=True, phase=True, phase_v2=True,
                                       twin=20)
    assert method_slug("per-sess std CAUSAL +phase_v2") == "persessstdcausalphasev2"
    assert method_slug("winner +twin20") == "winnertwin20"
    out = select_methods(ROSTER, "baseline,persessstdcausalphasev2,winnertwin20")
    assert [n for n, _ in out] == [BASELINE_NAME, "per-sess std CAUSAL +phase_v2",
                                   "winner +twin20"]
    # exact slug still uniquely selects the batch variant (no causal collision)
    assert [n for n, _ in select_methods(ROSTER, "persessstdphasev2")] == \
        ["per-sess std +phase_v2"]
