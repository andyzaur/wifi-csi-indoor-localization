#!/usr/bin/env python3
"""ml_drift.py — honest cross-session (drift) experiment harness.

The in-domain accuracy is already good (~25 cm pooled). The open frontier is the
LEAVE-ONE-SESSION-OUT gap: train on the days you have, predict an unseen day. This
script measures that gap correctly and A/Bs the approaches that might close it,
under one fixed, strong backbone so the *method* is what varies, not the search.

What it does (all on dedup'd multiboard data, LOSO = the only headline):

  ZERO-SHOT matrix — for each method, for each held-out session: scaler fit on the
  TRAINING sessions only, torch backbone (fixed config, seed-averaged) trained on
  the training sessions, scored ONCE on the held-out session. Per-fold + mean.
    methods: baseline (amp+rssi) · +phase · +detrend · +empty-room subtraction ·
             CORAL domain alignment (transductive, unlabelled target) · combos
  CALIBRATION sweep — the practical "calibrate to today" story: add the first K s
  of LABELLED target data to the training set, test on the rest (fixed test window
  so every K is comparable). Error-vs-calibration-time curve.
  LABEL-SHUFFLE control — shuffle train labels; LOSO must collapse to ~room
  half-spread. Cheap proof the numbers are signal, not leak.

Honesty caveat (disclosed in the card): the backbone hyperparameters are fixed
from a prior pooled search, so the ABSOLUTE LOSO number carries a mild optimistic
bias; but every method/fold uses the SAME config, so the A/B deltas are unbiased.
Pass --tune to additionally re-tune the winner on training-sessions-only (no leak).

Usage:
    source venv/bin/activate
    python3 -u ml_drift.py sessions/A sessions/B sessions/C \
        --empty sessions/20260603_2047_empty_camera_1 \
        --out sessions/drift_<date> --seeds 3 > sessions/drift_<date>/run.log 2>&1 &
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import subprocess
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dataset import (boards_in_multiboard_df, build_multiboard_dataset, grid_cell_id,
                     load_session, multiboard_feature_matrix, multiboard_static_baseline)
from mlpipe import (causal_session_standardize, coral_transform,
                    inner_percell_temporal_folds, moving_average_detrend,
                    remove_static_path, window_stats)
from train_final import per_cell_temporal_split
from torch_net import TorchLocalizer

# Fixed strong backbone — the tuned torch config from the 3-session pooled card
# (sessions/loso_3session_20260604/model_card_v3.json). Same config for every
# method/fold so method deltas are apples-to-apples.
BACKBONE = dict(depth=3, width=256, dropout=0.10, norm="layer", residual=False,
                lr=2.66e-3, weight_decay=1.1e-5, noise_std=0.037, feat_drop=0.114,
                label_smoothing=0.07, w_cls=0.29)

# Zero-shot method roster. Each flag composes a feature/transform pipeline.
# NOTE on empty-room subtraction: subtracting a constant per-feature baseline is
# ABSORBED by StandardScaler's centering (proven no-op; confirmed empirically — it
# reproduces the baseline to the decimal). One entry is kept to show the tie in the
# artifact; complex-domain subtraction is undefined here (random per-packet ESP32
# phase offsets), so amplitude-domain subtraction is the only option and it cannot
# help a standardized model. The lead is CORAL (second-order alignment), not a shift.
# v2 additions: phase_v2 = null-subcarrier-aware phase (masked sanitizer, null
# columns dropped); sess_std = transductive per-session feature standardization
# (uses the held-out session's own unlabeled stats, like CORAL — disclosed);
# aug = per-board train-time augmentation in TorchLocalizer (gain jitter +
# board dropout, mimicking inter-session per-board gain shifts / a dead RX).
# v3 additions: sess_std_causal = causal/online variant of sess_std (row i
# standardized by the session's EXPANDING stats over rows [0..i] only —
# deployable on a live stream, uses strictly less target information than the
# batch transductive variant); twin=k = append causal rolling mean+var
# (mlpipe.window_stats, window k) over the amplitude columns, computed AFTER
# standardization.
ROSTER = [
    ("baseline (amp+rssi)",      dict()),
    ("+phase",                   dict(phase=True)),
    ("+detrend20",               dict(detrend=20)),
    ("+empty-room sub (=no-op)", dict(empty_sub=True)),
    ("CORAL align",              dict(coral=True)),
    ("CORAL +phase",             dict(coral=True, phase=True)),
    ("+phase_v2",                dict(phase=True, phase_v2=True)),
    ("per-session std",          dict(sess_std=True)),
    ("per-sess std +phase_v2",   dict(sess_std=True, phase=True, phase_v2=True)),
    ("+phase_v2 +aug",           dict(phase=True, phase_v2=True, aug=True)),
    ("per-sess std CAUSAL +phase_v2", dict(sess_std_causal=True, phase=True, phase_v2=True)),
    ("winner +twin20",           dict(sess_std=True, phase=True, phase_v2=True, twin=20)),
]

BASELINE_NAME = "baseline (amp+rssi)"

# Per-board augmentation strengths for the 'aug' spec flag (TorchLocalizer):
# per-sample per-board gain g ~ N(1, AUG_GAIN_STD); AUG_DROP_P chance per sample
# to zero one uniformly-chosen board's slice.
AUG_GAIN_STD = 0.05
AUG_DROP_P = 0.1


def method_slug(name):
    """Lowercase alnum-only slug of a roster name ('+phase_v2' -> 'phasev2')."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def file_slug(name):
    """Lowercase filename slug: alnum kept, any other run collapses to one dash."""
    out = "".join(ch if ch.isalnum() else "-" for ch in name.lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def select_methods(roster, methods_arg):
    """Filter the roster by --methods: comma list of exact names or slugs; a slug
    may also be an unambiguous prefix ('baseline' -> 'baseline (amp+rssi)').
    None/'' selects everything; unknown or ambiguous entries abort."""
    if not methods_arg:
        return list(roster)
    out = []
    for want in (w.strip() for w in methods_arg.split(",")):
        if not want:
            continue
        ws = method_slug(want)
        hits = [(n, sp) for n, sp in roster if n == want or method_slug(n) == ws]
        if not hits:
            hits = [(n, sp) for n, sp in roster if method_slug(n).startswith(ws)]
        if not hits:
            raise SystemExit(f"--methods: {want!r} matches nothing; slugs: "
                             + ", ".join(method_slug(n) for n, _ in roster))
        if len(hits) > 1:
            raise SystemExit(f"--methods: {want!r} is ambiguous between: "
                             + ", ".join(n for n, _ in hits))
        if hits[0][0] not in [n for n, _ in out]:
            out.append(hits[0])
    return out


def parse_groups(spec_str):
    """Parse --groups 'g1=sessA;g2=sessB,sessC' -> {session_basename: group}."""
    mapping = {}
    for part in spec_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--groups: {part!r} is not name=sessA,sessB")
        gname, members = part.split("=", 1)
        gname = gname.strip()
        sess = [s.strip() for s in members.split(",") if s.strip()]
        if not gname or not sess:
            raise SystemExit(f"--groups: empty group name or member list in {part!r}")
        for s in sess:
            if s in mapping:
                raise SystemExit(f"--groups: session {s!r} listed twice")
            mapping[s] = gname
    return mapping


def apply_session_groups(df, mapping):
    """Remap df['session'] to group names (turns LOSO into leave-one-GROUP-out,
    e.g. leave-one-DAY-out). Every session present must be mapped; returns a copy."""
    present = list(dict.fromkeys(df["session"].tolist()))
    unmapped = [s for s in present if s not in mapping]
    if unmapped:
        raise SystemExit(f"--groups: unmapped sessions {unmapped}")
    out = df.copy()
    out["session"] = out["session"].map(mapping)
    return out


def per_session_standardize(X, groups):
    """Standardize each session's rows by that session's OWN per-feature mean/std
    (std floored at 1e-8). Transductive on a held-out session — it uses the
    target's unlabeled feature statistics, like CORAL; disclosed in the card."""
    X = np.asarray(X, dtype=np.float64).copy()
    groups = np.asarray(groups)
    for s in np.unique(groups):
        m = groups == s
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return X


def board_slices_from_names(names):
    """Per-board feature-index arrays from a names list: every column whose name
    starts 'b<ID>_' (amp/phase/rssi alike) joins board <ID>'s slice; sorted by
    board id. Feeds TorchLocalizer(board_slices=...) for the 'aug' spec flag."""
    groups = {}
    for j, nm in enumerate(names):
        head = nm.split("_", 1)[0]
        if len(head) > 1 and head[0] == "b" and head[1:].isdigit():
            groups.setdefault(int(head[1:]), []).append(j)
    return [np.asarray(groups[b], dtype=int) for b in sorted(groups)]


def spec_feature_key(spec):
    """names_cache key for a method spec's feature layout."""
    if spec.get("phase_v2"):
        return "phase_v2"
    return "phase" if spec.get("phase") else "amp"


def dump_fold_preds(dump_dir, method_name, fold_name, idx, y_xy, pred_xy,
                    wall_time_s, grid_x_cm, grid_y_cm):
    """Save one LOSO fold's per-frame predictions (--dump-preds) for offline
    analysis (error maps, smoothing replays). Returns the .npz path."""
    path = os.path.join(dump_dir,
                        f"preds__{file_slug(method_name)}__{file_slug(fold_name)}.npz")
    np.savez(path, idx=np.asarray(idx), y_xy=np.asarray(y_xy),
             pred_xy=np.asarray(pred_xy), wall_time_s=np.asarray(wall_time_s),
             grid_x_cm=np.asarray(grid_x_cm), grid_y_cm=np.asarray(grid_y_cm))
    return path


def reg_metrics(y, pred):
    err = np.linalg.norm(np.asarray(y) - np.asarray(pred), axis=1)
    return dict(median=float(np.median(err)), mean=float(err.mean()),
                p90=float(np.percentile(err, 90)))


def room_half_spread(y):
    y = np.asarray(y, float)
    return float(np.median(np.linalg.norm(y - y.mean(0), axis=1)))


def build_raw_features(df, spec, baseline_vec, names_cache, df_v2=None):
    """Causal raw feature matrix for a method spec (pre-scaling). Returns X.
    (Feature names live in names_cache[spec_feature_key(spec)][1] after the call.)

    phase    -> include sanitised phase columns
    phase_v2 -> null-subcarrier-aware layout from the phase_mode="masked" frame
                `df_v2` (amp+phase+rssi, null subcarriers dropped) — supersedes
                the legacy amp/phase layouts
    detrend  -> causal per-session moving-average high-pass on amplitudes
    empty_sub-> subtract the empty-room static baseline (amp/rssi only)
    sess_std -> per-session standardization by each session's OWN stats
                (transductive on the held-out session; disclosed in the card)
    sess_std_causal -> causal/online per-session standardization: row i uses
                only that session's expanding stats over rows [0..i]
                (deployable live; mutually exclusive with sess_std)
    twin     -> append mlpipe.window_stats (causal per-session rolling
                mean+var, window=twin) over the amplitude columns of the
                CURRENT matrix, AFTER the standardization step
    """
    key = spec_feature_key(spec)
    if key not in names_cache:
        if key == "phase_v2":
            assert df_v2 is not None, "phase_v2 spec needs the masked-mode dataframe"
            X, names = multiboard_feature_matrix(df_v2, include_phase=True,
                                                 include_rssi=True,
                                                 drop_null_subcarriers=True)
        else:
            X, names = multiboard_feature_matrix(df, include_phase=(key == "phase"),
                                                 include_rssi=True)
        names_cache[key] = (X.astype(np.float64), names)
    X, names = names_cache[key]
    X = X.copy()
    if spec.get("empty_sub"):
        base = baseline_vec[key]                       # aligned to these names
        X = remove_static_path(X, base)
    w = spec.get("detrend", 0)
    if w and w > 1:
        groups = df["session"].to_numpy()
        amp_idx = [i for i, n in enumerate(names) if "_amp_" in n]
        X[:, amp_idx] = moving_average_detrend(X[:, amp_idx], groups, w)
    assert not (spec.get("sess_std") and spec.get("sess_std_causal")), \
        "sess_std and sess_std_causal are mutually exclusive"
    if spec.get("sess_std"):
        X = per_session_standardize(X, df["session"].to_numpy())
    elif spec.get("sess_std_causal"):
        X = causal_session_standardize(X, df["session"].to_numpy())
    k = spec.get("twin", 0)
    if k and k > 1:
        amp_idx = [i for i, n in enumerate(names) if "_amp_" in n]
        X = np.hstack([X, window_stats(X[:, amp_idx], df["session"].to_numpy(), k)])
    return X.astype(np.float64)


def seed_predictions(Xtr, ytr, ctr, Xva, yva, Xte, n_cls, seeds, max_epochs,
                     aug_kw=None):
    """Train `seeds` torch backbones (early-stop on inner-val). Return the LIST of
    per-seed test predictions so callers can report both the seed-ENSEMBLE (their
    mean) AND the per-seed spread (so method deltas can be tested against seed
    noise, not just point-estimated). `aug_kw` forwards optional per-board
    augmentation params (aug_board_gain_std/aug_board_drop_p/board_slices)."""
    preds = []
    for s in range(seeds):
        m = TorchLocalizer(n_cls=n_cls, seed=s, batch_size=2048, max_epochs=max_epochs,
                           patience=30, **BACKBONE,
                           **(aug_kw or {})).fit(Xtr, ytr, ctr, X_val=Xva, y_val_xy=yva)
        preds.append(m.predict(Xte))
    return preds


def aug_kwargs_for_spec(spec, names):
    """TorchLocalizer augmentation kwargs for a spec, or None when aug is off."""
    if not spec.get("aug"):
        return None
    return dict(aug_board_gain_std=AUG_GAIN_STD, aug_board_drop_p=AUG_DROP_P,
                board_slices=board_slices_from_names(names))


def loso_eval_method(df, spec, baseline_vec, names_cache, y, c, n_cls, seeds,
                     max_epochs, shuffle=False, df_v2=None, dump_dir=None,
                     method_name=""):
    """Proper LOSO for one method: scaler + model fit on TRAINING sessions only,
    held-out session scored once. Per fold records: the seed-ensemble median, the
    per-seed median spread (seed_std), the unseen-cell fraction, and the
    label-destroyed floor (median dist of held-out y to the TRAINING centroid —
    the correct per-fold ceiling for the shuffle control). 'mean' aggregates across
    folds; 'seed_band' = mean per-fold std of the per-seed medians.

    shuffle=True permutes labels over the WHOLE training pool BEFORE the inner
    split, so BOTH the model and its early-stopping val see scrambled targets — a
    strict negative control (no real-label signal anywhere in training).

    df_v2 = the phase_mode="masked" twin frame (only read by phase_v2 specs).
    dump_dir + method_name: when given, each fold's seed-ensemble predictions are
    saved as preds__<method>__<fold>.npz (see dump_fold_preds)."""
    X = build_raw_features(df, spec, baseline_vec, names_cache, df_v2=df_v2)
    aug_kw = aug_kwargs_for_spec(spec, names_cache[spec_feature_key(spec)][1])
    sessions = list(dict.fromkeys(df["session"].tolist()))
    per_fold = {}
    for held in sessions:
        te = np.where((df["session"] == held).to_numpy())[0]
        tr_idx = np.where((df["session"] != held).to_numpy())[0]
        ys, cs = y.copy(), c.copy()
        if shuffle:                       # scramble the entire training pool's labels
            rng = np.random.default_rng(0)
            perm = rng.permutation(len(tr_idx))
            ys[tr_idx], cs[tr_idx] = y[tr_idx][perm], c[tr_idx][perm]
        # inner per-cell-temporal split of the TRAINING sessions (early stop only)
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        itr_df, iva_df = per_cell_temporal_split(tr_df, 0.85)
        i_tr = tr_idx[itr_df.index.to_numpy()]
        i_va = tr_idx[iva_df.index.to_numpy()]
        sc = StandardScaler().fit(X[i_tr])
        Xtr, Xva, Xte = sc.transform(X[i_tr]), sc.transform(X[i_va]), sc.transform(X[te])
        if spec.get("coral"):
            # align (scaled) training source to (scaled, UNLABELLED) target features
            Xt_all = sc.transform(X[te])
            Xtr = coral_transform(Xtr, Xt_all, eps=1.0)
            Xva = coral_transform(Xva, Xt_all, eps=1.0)
        preds = seed_predictions(Xtr, ys[i_tr], cs[i_tr], Xva, ys[i_va], Xte,
                                 n_cls, seeds, max_epochs, aug_kw=aug_kw)
        ens = np.mean(preds, axis=0)
        if dump_dir:
            dump_fold_preds(dump_dir, method_name, held, te, y[te], ens,
                            df["wall_time_s"].to_numpy()[te],
                            df["grid_x_cm"].to_numpy()[te],
                            df["grid_y_cm"].to_numpy()[te])
        seed_meds = [reg_metrics(y[te], p)["median"] for p in preds]   # true labels for scoring
        m = reg_metrics(y[te], ens)
        train_cells = set(np.unique(c[tr_idx]).tolist())
        m["seed_std"] = float(np.std(seed_meds))
        m["unseen_cell_frac"] = float(np.mean(~np.isin(c[te], list(train_cells))))
        m["floor"] = float(np.median(np.linalg.norm(y[te] - y[tr_idx].mean(0), axis=1)))
        per_fold[held] = m
    meds = [per_fold[h]["median"] for h in sessions]
    per_fold["mean"] = dict(
        median=float(np.mean(meds)),
        mean=float(np.mean([per_fold[h]["mean"] for h in sessions])),
        p90=float(np.mean([per_fold[h]["p90"] for h in sessions])),
        seed_band=float(np.mean([per_fold[h]["seed_std"] for h in sessions])),
        floor=float(np.mean([per_fold[h]["floor"] for h in sessions])),
        max_unseen_frac=float(np.max([per_fold[h]["unseen_cell_frac"] for h in sessions])))
    return per_fold


def calibration_sweep(df, baseline_vec, names_cache, y, c, n_cls, seeds, max_epochs,
                      fracs=(0.0, 0.1, 0.25, 0.5, 1.0), spec=None, df_v2=None):
    """Calibration-pass curve: how much LABELLED target data (spread over the room)
    do you need today to localize well today?

    Naive 'first K seconds' calibration FAILS (the start of a walk is spatially
    concentrated → fine-tuning collapses onto a few cells). The realistic demo is a
    *calibration pass over the room*. So per held-out target session: per-cell
    temporal split → the LAST half of each cell's frames is the FIXED test set; the
    FIRST half is the calibration pool. For fraction f, take the earliest f of EACH
    cell's pool (representative, temporally-before-test), pool with the source
    sessions, retrain, score the fixed test. f=0 = zero-shot LOSO on the same test
    frames; f=1 = a full first-half labelled pass. Returns {held: {frac: median}}."""
    spec = spec or dict()
    X = build_raw_features(df, spec, baseline_vec, names_cache, df_v2=df_v2)
    aug_kw = aug_kwargs_for_spec(spec, names_cache[spec_feature_key(spec)][1])
    sessions = list(dict.fromkeys(df["session"].tolist()))
    out, n_calib = {}, {}
    for held in sessions:
        tgt = np.where((df["session"] == held).to_numpy())[0]
        tgt_df = df.iloc[tgt].reset_index(drop=True)
        early_df, late_df = per_cell_temporal_split(tgt_df, 0.5)   # early=calib pool, late=test
        test_idx = tgt[late_df.index.to_numpy()]
        if len(test_idx) < 50:
            continue
        pool_local = early_df.index.to_numpy()                     # tgt-local indices
        pool_cells = early_df["cell_id"].to_numpy()
        pool_times = early_df["wall_time_s"].to_numpy()
        other = np.where((df["session"] != held).to_numpy())[0]
        out[held], n_calib[held] = {}, {}
        for f in fracs:
            if f <= 0:
                calib = np.array([], dtype=int)
            else:
                sel = []
                for cell in np.unique(pool_cells):
                    pos = np.where(pool_cells == cell)[0]
                    pos = pos[np.argsort(pool_times[pos])]
                    sel.extend(pos[:max(1, int(round(len(pos) * f)))].tolist())
                calib = tgt[pool_local[np.array(sel, dtype=int)]]
            tr_idx = np.concatenate([other, calib]).astype(int)
            tr_df = df.iloc[tr_idx].reset_index(drop=True)
            itr_df, iva_df = per_cell_temporal_split(tr_df, 0.85)
            i_tr = tr_idx[itr_df.index.to_numpy()]
            i_va = tr_idx[iva_df.index.to_numpy()]
            sc = StandardScaler().fit(X[i_tr])
            Xtr, Xva, Xte = sc.transform(X[i_tr]), sc.transform(X[i_va]), sc.transform(X[test_idx])
            preds = seed_predictions(Xtr, y[i_tr], c[i_tr], Xva, y[i_va], Xte,
                                     n_cls, seeds, max_epochs, aug_kw=aug_kw)
            out[held][f] = reg_metrics(y[test_idx], np.mean(preds, axis=0))["median"]
            n_calib[held][f] = int(len(calib))
    return out, n_calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--empty", default=None, help="empty-room CSI-only session dir")
    ap.add_argument("--out", "-o", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=250)
    ap.add_argument("--calib-fracs", default="0.0,0.1,0.25,0.5,1.0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--methods", default=None,
                    help="comma list of ROSTER names or slugs (e.g. baseline,phasev2); default all")
    ap.add_argument("--groups", default=None,
                    help='remap sessions to groups, e.g. "d1=sessA;d2=sessB,sessC" '
                         "-> leave-one-GROUP-out (leave-one-day-out)")
    ap.add_argument("--dump-preds", default=None,
                    help="dir for per-fold prediction .npz dumps (zero-shot matrix only)")
    ap.add_argument("--skip-calib", action="store_true",
                    help="skip the calibration sweep section")
    ap.add_argument("--backbone-json", default=None,
                    help="override the fixed BACKBONE from a JSON file: either a raw "
                         "backbone dict or a train_v3 model_card_v3.json (uses its "
                         "model_params.torch). Non-destructive — for re-tuned-backbone runs.")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.max_epochs = 1, 40

    # Optional re-tuned backbone (keeps the frozen default reproducible; only the
    # searched arch/regularization keys are overridden — batch_size/max_epochs/
    # patience stay as seed_predictions sets them).
    if args.backbone_json:
        global BACKBONE
        with open(args.backbone_json) as _f:
            _bb = json.load(_f)
        if isinstance(_bb, dict) and "model_params" in _bb and "torch" in _bb["model_params"]:
            _bb = _bb["model_params"]["torch"]
        _bb = {k: v for k, v in _bb.items() if k in BACKBONE}
        BACKBONE = {**BACKBONE, **_bb}
        print(f"Backbone overridden from {args.backbone_json}: {BACKBONE}")

    out_dir = args.out or ("sessions/drift_" + args.sessions[0].rstrip("/").split("/")[-1])
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    # ── method roster (--methods filter; empty-room method needs --empty) ──
    have_empty = bool(args.empty)
    roster = [(n, sp) for n, sp in select_methods(ROSTER, args.methods)
              if have_empty or not sp.get("empty_sub")]
    if not roster:
        raise SystemExit("--methods selected nothing runnable "
                         "(empty-room method without --empty?)")
    need_v2 = any(sp.get("phase_v2") for _, sp in roster)

    # ── load + build (dedup; sort by session,time; freeze cell ids) ──
    dfs, dfs_v2 = [], []
    for s in args.sessions:
        csi, cam, clap = load_session(s.rstrip("/"))
        d = build_multiboard_dataset(csi, cam, clap)
        d["session"] = os.path.basename(s.rstrip("/"))
        dfs.append(d)
        if need_v2:                       # masked-phase twin, same rows/order
            d2 = build_multiboard_dataset(csi, cam, clap, phase_mode="masked")
            d2["session"] = os.path.basename(s.rstrip("/"))
            dfs_v2.append(d2)
    df = pd.concat(dfs, ignore_index=True).sort_values(["session", "wall_time_s"]).reset_index(drop=True)
    df_v2 = None
    if need_v2:
        df_v2 = pd.concat(dfs_v2, ignore_index=True).sort_values(
            ["session", "wall_time_s"]).reset_index(drop=True)
        assert len(df_v2) == len(df), "phase_v2 (masked) frame count differs from legacy build"
        assert np.allclose(df_v2["wall_time_s"].to_numpy(), df["wall_time_s"].to_numpy()), \
            "phase_v2 (masked) frame times differ from legacy build"
    if args.groups:                       # leave-one-DAY/GROUP-out remap
        mapping = parse_groups(args.groups)
        df = apply_session_groups(df, mapping)
        if df_v2 is not None:
            df_v2 = apply_session_groups(df_v2, mapping)
        print("Session groups (LOSO becomes leave-one-GROUP-out):")
        for g in dict.fromkeys(mapping.values()):
            print(f"    {g}: " + ", ".join(s for s, gg in mapping.items() if gg == g))
    df["cell_id"] = grid_cell_id(df)
    enc = LabelEncoder().fit(df["cell_id"])
    df["cls"] = enc.transform(df["cell_id"])
    n_cls = len(enc.classes_)
    y = df[["x_cm", "y_cm"]].to_numpy(np.float64)
    c = df["cls"].to_numpy()
    sessions = list(dict.fromkeys(df["session"].tolist()))
    print(f"Loaded {len(df):,} samples · {n_cls} cells · {len(sessions)} sessions "
          f"· boards {list(boards_in_multiboard_df(df))}")
    for s in sessions:
        print(f"    {s:34s} {int((df['session']==s).sum()):>7,} samples")
    if len(sessions) < 2:
        raise SystemExit("Need >=2 sessions for a LOSO drift study.")

    # ── empty-room baseline vectors (aligned to amp and amp+phase layouts) ──
    baseline_vec = {}
    if have_empty:
        empty_csi = pd.read_csv(os.path.join(args.empty.rstrip("/"), "csi.csv"))
        for key, phase in [("amp", False), ("phase", True)]:
            _, names = multiboard_feature_matrix(df.head(2), include_phase=phase, include_rssi=True)
            baseline_vec[key] = multiboard_static_baseline(empty_csi, names)
        print(f"Empty-room baseline from {args.empty} ({len(empty_csi):,} packets)")
    names_cache = {}

    dump_dir = args.dump_preds
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)

    # ── ZERO-SHOT LOSO MATRIX ──
    print("\n" + "=" * 78)
    print("ZERO-SHOT LOSO  (median cm; train on other sessions, predict the held-out one)")
    print("=" * 78)
    hdr = f"{'method':26s} " + " ".join(f"{h[:14]:>14s}" for h in sessions) + f" {'MEAN':>8s} {'±seed':>6s}"
    print(hdr)
    results = {}
    for name, spec in roster:
        t0 = time.time()
        pf = loso_eval_method(df, spec, baseline_vec, names_cache, y, c, n_cls,
                              args.seeds, args.max_epochs, df_v2=df_v2,
                              dump_dir=dump_dir, method_name=name)
        results[name] = pf
        row = f"{name:26s} " + " ".join(f"{pf[h]['median']:>14.1f}" for h in sessions)
        row += f" {pf['mean']['median']:>8.1f} {pf['mean']['seed_band']:>6.1f}"
        print(row + f"   ({time.time()-t0:.0f}s)", flush=True)
    # baseline is the A/B reference; with a --methods subset that omits it, the
    # first selected method stands in (deltas/robust-wins are then skipped).
    have_base = BASELINE_NAME in results
    ref_name = BASELINE_NAME if have_base else roster[0][0]
    base = results[ref_name]
    if base["mean"]["max_unseen_frac"] > 0.01:
        print(f"  NOTE: up to {base['mean']['max_unseen_frac']*100:.1f}% of a held-out session's "
              f"frames fall in cells no training session covered (extrapolation).")
    # honest A/B: a method only 'beats baseline' if its mean gain exceeds the seed
    # band AND it helps on every fold (sign-consistent). Selection is NOT on test.
    if have_base and len(roster) > 1:
        print("\n  delta vs baseline (negative = better); robust only if |Δ| > seed band AND helps all folds:")
        for name, spec in roster:
            if name == BASELINE_NAME:
                continue
            pf = results[name]
            d_mean = pf["mean"]["median"] - base["mean"]["median"]
            per_fold_d = [pf[h]["median"] - base[h]["median"] for h in sessions]
            band = max(pf["mean"]["seed_band"], base["mean"]["seed_band"])
            robust = (d_mean < -band) and all(x < 0 for x in per_fold_d)
            tag = "ROBUST WIN" if robust else ("worse" if d_mean > band else "within noise")
            print(f"    {name:26s} Δmean {d_mean:+5.1f}  per-fold {['%+.0f'%x for x in per_fold_d]}  -> {tag}")

    # ── LABEL-SHUFFLE CONTROL (baseline method) ──
    print("\n" + "-" * 78)
    print("LABEL-SHUFFLE CONTROL (baseline; scrambled labels must collapse to the floor)")
    # Same seed count as the headline; floor = per-fold median dist of held-out y to
    # the TRAINING centroid (what a label-destroyed model can achieve on THAT fold),
    # averaged over folds — the correct reference, not the pooled global half-spread.
    shuf = loso_eval_method(df, dict(), baseline_vec, names_cache, y, c, n_cls,
                            args.seeds, args.max_epochs, shuffle=True)
    floor = base["mean"]["floor"]
    real_mean = base["mean"]["median"]
    near_floor = abs(shuf["mean"]["median"] - floor) <= 0.15 * floor
    real_below = real_mean < 0.9 * floor
    passes = near_floor and real_below
    print(f"  shuffled LOSO mean {shuf['mean']['median']:.1f} cm   "
          f"per-fold train-centroid floor {floor:.1f} cm   real '{ref_name}' {real_mean:.1f} cm")
    print(f"  -> {'PASS' if passes else 'CHECK'}: scrambled collapses to the floor "
          f"({shuf['mean']['median']:.0f}≈{floor:.0f}) and real ({real_mean:.0f}) sits below it")

    # ── CALIBRATION SWEEP (a labelled calibration pass over the room, today) ──
    fracs = tuple(float(x) for x in args.calib_fracs.split(","))
    calib, n_calib = {}, {}
    if not args.skip_calib:
        print("\n" + "=" * 78)
        print("CALIBRATION SWEEP  (label a fraction of each cell's first half today; "
              "test on its second half)")
        print("=" * 78)
        calib, n_calib = calibration_sweep(df, baseline_vec, names_cache, y, c, n_cls,
                                           args.seeds, args.max_epochs, fracs=fracs)
        print(f"{'held-out':26s} " + " ".join(f"{'f='+str(f):>9s}" for f in fracs))
        for held, curve in calib.items():
            print(f"{held:26s} " + " ".join(f"{curve.get(f, float('nan')):>9.1f}" for f in fracs))

    # ── honest A/B summary: NO test-fold crowning (winner's-curse). Baseline is
    #    the reference; a method is a 'robust win' only if it beats baseline by more
    #    than the seed band on the mean AND on every fold (sign-consistent). ──
    base_mean = base["mean"]["median"]

    def ab(name):
        pf = results[name]
        d = pf["mean"]["median"] - base_mean
        per_fold_d = [pf[h]["median"] - base[h]["median"] for h in sessions]
        band = max(pf["mean"]["seed_band"], base["mean"]["seed_band"])
        return dict(delta_mean=d, per_fold_delta=per_fold_d, band=band,
                    robust_win=bool(d < -band and all(x < 0 for x in per_fold_d)))

    ab_summary = {n: ab(n) for n, _ in roster if have_base and n != BASELINE_NAME}
    robust_wins = [n for n, a in ab_summary.items() if a["robust_win"]]

    def git_commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            return "unknown"

    card = {
        "kind": "ml_drift LOSO study",
        "sessions": [s.rstrip("/") for s in args.sessions],
        "session_groups": (parse_groups(args.groups) if args.groups else None),
        "methods": [n for n, _ in roster],
        "empty_room": args.empty, "n_samples": int(len(df)), "n_cells": int(n_cls),
        "git_commit": git_commit(), "backbone": BACKBONE, "seeds": args.seeds,
        "honesty_note": ("LOSO is the only headline. Backbone hyperparameters are FIXED from a "
                         "prior pooled search (same config for every method/fold), so absolute "
                         "LOSO carries a mild optimistic bias but A/B deltas are unbiased. No "
                         "method is crowned on the test folds: a 'robust win' must beat baseline "
                         "by more than the seed band on the mean AND on every fold. CORAL is "
                         "transductive (unlabelled target features, no target labels). Calibration "
                         "is a DISTINCT same-day result, not zero-shot. Per-session "
                         "standardization is transductive too (it uses the held-out session's own "
                         "unlabeled feature statistics), like CORAL was. The CAUSAL per-session-std "
                         "variant instead uses only past frames of the held-out session (deployable "
                         "on a live stream — strictly less information than the batch transductive "
                         "variant)."),
        "zero_shot_loso": {n: {**{h: results[n][h]["median"] for h in sessions},
                               "mean": results[n]["mean"]["median"],
                               "seed_band": results[n]["mean"]["seed_band"]} for n in results},
        "ab_vs_baseline": {n: ab_summary[n] for n in ab_summary},
        "robust_wins_over_baseline": robust_wins,
        "max_unseen_cell_frac": base["mean"]["max_unseen_frac"],
        "label_shuffle_control": {"shuffled_mean_cm": shuf["mean"]["median"],
                                  "per_fold_train_centroid_floor_cm": floor,
                                  "real_baseline_cm": real_mean, "passes": bool(passes)},
        "calibration_sweep_median_cm": {h: {str(k): v for k, v in calib[h].items()} for h in calib},
        "calibration_n_frames": {h: {str(k): v for k, v in n_calib[h].items()} for h in n_calib},
        "lib_versions": {m: __import__(m).__version__ for m in ["numpy", "torch", "sklearn", "pandas"]},
        "runtime_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(out_dir, "drift_card.json"), "w") as f:
        json.dump(card, f, indent=2, default=str)

    # ── verdict ──
    print("\n" + "=" * 78)
    print("VERDICT (honest, LOSO — baseline is the reference; wins must clear seed noise)")
    print("=" * 78)
    print(f"  zero-shot LOSO reference  {base_mean:5.1f} cm  "
          f"(±{base['mean']['seed_band']:.1f} seed band)  [{ref_name}]")
    if robust_wins:
        for n in robust_wins:
            print(f"  ROBUST zero-shot win      {n:26s} {results[n]['mean']['median']:5.1f} cm "
                  f"({ab_summary[n]['delta_mean']:+.1f})")
    elif have_base and len(roster) > 1:
        print("  no zero-shot method beats the baseline beyond seed noise "
              "(phase/detrend/empty-room/CORAL all within noise or worse).")
    if calib:
        def cmean(f):
            vals = [calib[h][f] for h in calib if f in calib[h]]
            return float(np.mean(vals)) if vals else float("nan")
        fmin = min(f for f in fracs if f > 0) if any(f > 0 for f in fracs) else None
        fmax = max(fracs)
        print(f"  calibration pass (same-day, test-half held out) — the real lever:")
        print(f"      zero-shot (f=0)        {cmean(0.0):5.1f} cm")
        if fmin is not None:
            print(f"      light pass (f={fmin})     {cmean(fmin):5.1f} cm  ({cmean(fmin)-cmean(0.0):+.1f})")
        print(f"      full pass  (f={fmax})     {cmean(fmax):5.1f} cm  ({cmean(fmax)-cmean(0.0):+.1f})")
    print(f"\n  card: {out_dir}/drift_card.json   ({card['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
