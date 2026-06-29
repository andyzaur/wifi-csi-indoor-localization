#!/usr/bin/env python3
"""train_v3.py — hardware-saturating, LEAK-PROOF search for the best honest model.

Synthesises the 2026-06-03 design workflow (six specs). The whole
point vs train_v2 is EVALUATION DISCIPLINE — the nested protocol:

    OUTER  per-cell-temporal split  -> (tr_outer, TE_OUTER)   TE_OUTER is sacred
    INNER  per-cell-temporal split  -> (inner_tr, inner_val)  ALL selection here

Every feature/model/hyperparameter/smoothing choice is made on inner_val (or via
out-of-fold predictions inside tr_outer). TE_OUTER is scored EXACTLY ONCE, after
all choices are frozen. No selection ever reads TE_OUTER.

Hardware: CPU model search (lightgbm/xgboost/HGB/RF) via Optuna across the
performance cores (each model single-threaded → no oversubscription); the neural
search runs on the MPS GPU (torch_net.TorchLocalizer). The leaky stratified split
is printed only as a flagged upper bound — never cited.

Usage:
    python3 train_v3.py sessions/20260603_01_real30hztest            # full search
    python3 train_v3.py sessions/<s> --smoke                          # fast sanity
    python3 train_v3.py sessions/a sessions/b -o sessions/combined    # multi-session
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")     # single-thread inner libs;
os.environ.setdefault("MKL_NUM_THREADS", "1")     # parallelism is at the trial level
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import subprocess
import threading
import time
import warnings

import joblib
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb

from dataset import (boards_in_multiboard_df, build_multiboard_dataset,
                     grid_cell_id, load_session, multiboard_feature_matrix)
from mlpipe import (MultiAxis, inner_percell_temporal_folds, leave_one_session_out,
                    moving_average_detrend, window_stats)
from train_final import (per_cell_temporal_split, smooth_classification,
                         smooth_regression, stratified_cell_split, temporal_split)
from torch_net import TorchLocalizer, fit_seed_ensemble, ensemble_predict, ensemble_proba

optuna.logging.set_verbosity(optuna.logging.WARNING)
SMOOTH = (1, 3, 5, 9, 15)


# ── metrics ────────────────────────────────────────────────────────────────

def reg_metrics(y, pred):
    err = np.linalg.norm(np.asarray(y) - np.asarray(pred), axis=1)
    return {"median": float(np.median(err)), "mean": float(err.mean()),
            "p90": float(np.percentile(err, 90))}


def cls_report(y_true_cells, pred_cells, train_classes):
    """Honest classification accuracy: report all three numbers."""
    known = np.isin(y_true_cells, train_classes)
    n = len(y_true_cells)
    n_correct_known = int((pred_cells[known] == y_true_cells[known]).sum())
    return {
        "unknown_cell_fraction": float((~known).mean()),
        "known_cell_accuracy": float(n_correct_known / max(1, known.sum())),
        "full_test_accuracy": float(n_correct_known / n),  # unknowns count as wrong
    }


# ── features (causal; built once on the full time-ordered df) ───────────────

def build_features(df, cfg):
    """Causal feature matrix aligned to df rows. Fitted transforms (scaler/PCA)
    happen LATER, per fold. cfg keys: phase, rssi, detrend, winstats, amp_norm, age."""
    groups = df["session"].to_numpy() if "session" in df.columns else None
    boards = boards_in_multiboard_df(df)
    X, names = multiboard_feature_matrix(df, include_phase=cfg.get("phase", False),
                                         include_rssi=cfg.get("rssi", True))
    X = X.astype(np.float64)
    amp_idx = [i for i, n in enumerate(names) if "_amp_" in n]
    if cfg.get("amp_norm") == "l2":            # per-board, per-frame L2 (row-wise, causal)
        for b in boards:
            cols = [i for i, n in enumerate(names) if n.startswith(f"b{b}_amp_")]
            blk = X[:, cols]
            X[:, cols] = blk / (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-8)
    w = cfg.get("detrend", 0)
    if w and w > 1:
        X[:, amp_idx] = moving_average_detrend(X[:, amp_idx], groups, w)
    ws = cfg.get("winstats", 0)
    if ws and ws > 1:
        X = np.hstack([X, window_stats(X[:, amp_idx], groups, ws)])
    if cfg.get("age", False):
        age = df[[f"b{b}_age_s" for b in boards]].to_numpy(np.float64)
        X = np.hstack([X, np.nan_to_num(age)])
    return X.astype(np.float32)


# ── regressors (uniform fit(X,Y2d)/predict->(n,2) interface) ────────────────

def make_regressor(kind, params):
    if kind == "lgbm":
        return MultiAxis(lgb.LGBMRegressor, dict(n_jobs=1, verbosity=-1, **params))
    if kind == "xgb":
        return MultiAxis(xgb.XGBRegressor, dict(n_jobs=1, tree_method="hist", **params))
    if kind == "hgb":
        return MultiOutputRegressor(HistGradientBoostingRegressor(**params))
    if kind == "rf":
        return RandomForestRegressor(n_jobs=1, random_state=42, **params)
    raise ValueError(kind)


def suggest_params(kind, trial):
    # NOTE: Optuna param names MUST match the estimator kwargs exactly, because
    # study.best_params is fed straight back into the estimator at refit time.
    if kind == "lgbm":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 1200),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 15, 255),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    if kind == "xgb":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 1200),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_depth=trial.suggest_int("max_depth", 3, 12),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    if kind == "hgb":
        return dict(max_iter=trial.suggest_int("max_iter", 200, 1000),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 80),
                    l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True))
    if kind == "rf":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 800),
                    max_depth=trial.suggest_int("max_depth", 6, 40),
                    max_features=trial.suggest_float("max_features", 0.1, 1.0),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20))
    raise ValueError(kind)


# ── nested-protocol search ──────────────────────────────────────────────────

def scale(Xtr, *others):
    sc = StandardScaler().fit(Xtr)
    return (sc, sc.transform(Xtr), *[sc.transform(o) for o in others])


def search_cpu(kind, Xtr_s, ytr, Xval_s, yval, n_trials, timeout):
    def obj(trial):
        m = make_regressor(kind, suggest_params(kind, trial)).fit(Xtr_s, ytr)
        return reg_metrics(yval, m.predict(Xval_s))["median"]
    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=42))
    st.optimize(obj, n_trials=n_trials, n_jobs=5, timeout=timeout,
                show_progress_bar=False)
    return st.best_params, st.best_value


def search_torch(n_cls, Xtr_s, ytr, ctr, Xval_s, yval, n_trials, timeout):
    def obj(trial):
        nl = trial.suggest_int("depth", 1, 3)
        params = dict(width=trial.suggest_categorical("width", [128, 256, 512]),
                      depth=nl,
                      dropout=trial.suggest_float("dropout", 0.0, 0.5),
                      norm=trial.suggest_categorical("norm", ["batch", "layer"]),
                      residual=trial.suggest_categorical("residual", [False, True]) if nl >= 2 else False,
                      lr=trial.suggest_float("lr", 3e-4, 5e-3, log=True),
                      weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                      noise_std=trial.suggest_float("noise_std", 0.0, 0.1),
                      feat_drop=trial.suggest_float("feat_drop", 0.0, 0.2),
                      label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.1),
                      w_cls=trial.suggest_float("w_cls", 0.2, 0.8),
                      batch_size=2048, max_epochs=300, patience=30)

        def report(epoch, val):
            trial.report(val, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        m = TorchLocalizer(n_cls=n_cls, seed=0, **params)
        m.fit(Xtr_s, ytr, ctr, X_val=Xval_s, y_val_xy=yval, report_cb=report)
        return m.best_val_
    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=1),
                             pruner=optuna.pruners.HyperbandPruner(min_resource=20, max_resource=300, reduction_factor=3))
    st.optimize(obj, n_trials=n_trials, n_jobs=1, timeout=timeout, show_progress_bar=False)
    return st.best_params, st.best_value


def oof_regressor_preds(kind_or_torch, params, X, Y, C, df_tr, k, n_cls):
    """Out-of-fold predictions on tr_outer via per-cell-temporal inner folds (no
    leakage — each row predicted by a model that did not fit it). Stacking."""
    folds = inner_percell_temporal_folds(df_tr.reset_index(drop=True), k=k)
    oof = np.zeros((len(X), 2))
    for f in range(k):
        te = folds == f
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        sc, Xtr_s, Xte_s = scale(X[tr], X[te])
        if kind_or_torch == "torch":
            m = TorchLocalizer(n_cls=n_cls, seed=0, batch_size=2048, max_epochs=200,
                               patience=25, **params).fit(Xtr_s, Y[tr], C[tr],
                                                           X_val=Xte_s, y_val_xy=Y[te])
            oof[te] = m.predict(Xte_s)
        else:
            m = make_regressor(kind_or_torch, params).fit(Xtr_s, Y[tr])
            oof[te] = m.predict(Xte_s)
    return oof


def fit_convex_weights(oof_list, y):
    """Non-negative weights summing to 1 minimizing OOF MSE (honest blend)."""
    P = np.stack(oof_list, axis=0)                  # (M, n, 2)
    M = len(oof_list)

    def loss(w):
        w = np.clip(w, 0, None); w = w / (w.sum() + 1e-12)
        pred = np.tensordot(w, P, axes=(0, 0))
        return float(((y - pred) ** 2).sum(1).mean())
    res = minimize(loss, np.full(M, 1 / M), method="SLSQP",
                   bounds=[(0, 1)] * M, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w = np.clip(res.x, 0, None); w = w / (w.sum() + 1e-12)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", "-o", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trials-cpu", type=int, default=150)
    ap.add_argument("--trials-torch", type=int, default=80)
    ap.add_argument("--timeout-cpu", type=int, default=420)
    ap.add_argument("--timeout-torch", type=int, default=600)
    ap.add_argument("--no-loso-tail", action="store_true",
                    help="skip the diagnostic end-of-run context/lgbm-LOSO splits "
                         "(slow at many-session scale; ml_drift.py is the real LOSO). "
                         "The card is written right after the nested search either way.")
    args = ap.parse_args()
    if args.smoke:
        args.trials_cpu, args.trials_torch = 6, 5
        args.timeout_cpu, args.timeout_torch = 45, 45

    out_dir = args.out or args.sessions[0].rstrip("/")
    os.makedirs(out_dir, exist_ok=True)

    # ── load + build (dedup BEFORE split; sort by session,time; freeze cell_id) ──
    dfs = []
    for s in args.sessions:
        csi, cam, clap = load_session(s.rstrip("/"))
        d = build_multiboard_dataset(csi, cam, clap)
        d["session"] = os.path.basename(s.rstrip("/"))
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).sort_values(["session", "wall_time_s"]).reset_index(drop=True)
    df["cell_id"] = grid_cell_id(df)
    n_sessions = df["session"].nunique()
    print(f"Loaded {len(df):,} samples, {df['cell_id'].nunique()} cells, {n_sessions} session(s)")

    enc = LabelEncoder().fit(df["cell_id"])
    df["cls"] = enc.transform(df["cell_id"])
    n_cls = len(enc.classes_)

    # ── OUTER split (TE_OUTER is sacred), INNER split for all selection ──
    tr_outer, te_outer = per_cell_temporal_split(df, 0.8)
    inner_tr, inner_val = per_cell_temporal_split(tr_outer, 0.8)
    iTR, iVA = inner_tr.index.to_numpy(), inner_val.index.to_numpy()
    print(f"Protocol: outer train {len(tr_outer):,} / TEST {len(te_outer):,} (untouched until final); "
          f"inner train {len(inner_tr):,} / val {len(inner_val):,}")

    y = df[["x_cm", "y_cm"]].to_numpy(np.float64)
    c = df["cls"].to_numpy()

    # ── STAGE 1: feature-config selection on INNER-VAL (model = lgbm default) ──
    feat_cfgs = {
        "amp+rssi (baseline)": dict(),
        "+detrend20": dict(detrend=20),
        "+l2norm": dict(amp_norm="l2"),
        "+detrend20+l2norm": dict(detrend=20, amp_norm="l2"),
        "+winstats20": dict(winstats=20),
        "+phase (expect loses)": dict(phase=True),
    }
    if args.smoke:
        feat_cfgs = {k: feat_cfgs[k] for k in ["amp+rssi (baseline)", "+detrend20", "+detrend20+l2norm"]}
    print("\n── STAGE 1: feature config (selected on inner-val, lgbm) ──")
    feat_scores = {}
    feat_X = {}
    lgbm_probe = dict(n_estimators=400, learning_rate=0.05, num_leaves=63)
    for name, cfg in feat_cfgs.items():
        X = build_features(df, cfg)
        feat_X[name] = X
        sc, Xtr_s, Xva_s = scale(X[iTR], X[iVA])
        m = make_regressor("lgbm", lgbm_probe).fit(Xtr_s, y[iTR])
        med = reg_metrics(y[iVA], m.predict(Xva_s))["median"]
        feat_scores[name] = med
        print(f"  {name:26s} inner-val reg median {med:6.1f} cm  ({X.shape[1]} feats)")
    best_cfg_name = min(feat_scores, key=feat_scores.get)
    X = feat_X[best_cfg_name]
    print(f"  -> best feature config: {best_cfg_name}")

    # Pre-scale the inner split once (scaler fit on inner-train ONLY)
    sc_in, Xtr_s, Xva_s = scale(X[iTR], X[iVA])

    # ── STAGE 2: model search — CPU studies (n_jobs across cores) run CONCURRENTLY
    #    with the MPS torch study (its own thread) so cores AND GPU stay busy at
    #    once. RF dropped (slowest + weakest in testing). All selection on inner-val.
    print("\n── STAGE 2: model search (inner-val reg median, cm) — CPU+GPU concurrent ──")
    cpu_families = ["lgbm", "hgb"]   # xgb dropped: ~43 min/run, blend weight 0.00, no gain
    results = {}

    def run_cpu():
        for kind in cpu_families:
            t0 = time.time()
            p, v = search_cpu(kind, Xtr_s, y[iTR], Xva_s, y[iVA], args.trials_cpu, args.timeout_cpu)
            results[kind] = {"params": p, "inner_val": v}
            print(f"  {kind:6s} inner-val {v:6.1f} cm   ({time.time()-t0:.0f}s, CPU)", flush=True)

    def run_gpu():
        t0 = time.time()
        p, v = search_torch(n_cls, Xtr_s, y[iTR], c[iTR], Xva_s, y[iVA],
                            args.trials_torch, args.timeout_torch)
        results["torch"] = {"params": p, "inner_val": v}
        print(f"  {'torch':6s} inner-val {v:6.1f} cm   ({time.time()-t0:.0f}s, MPS/GPU)", flush=True)

    tg = threading.Thread(target=run_gpu, daemon=True)
    tg.start()
    run_cpu()
    tg.join()
    fam_order = [k for k in cpu_families + ["torch"] if k in results]

    # ── STAGE 3: convex blend weights, fit on INNER-VAL (honest; never the test) ──
    print("\n── STAGE 3: blend weights (fit on inner-val) ──")
    Xtr_outer = X[tr_outer.index.to_numpy()]
    ytr_outer = y[tr_outer.index.to_numpy()]
    ctr_outer = c[tr_outer.index.to_numpy()]
    inner_preds = []
    for kind in fam_order:
        p = results[kind]["params"]
        if kind == "torch":
            mi = TorchLocalizer(n_cls=n_cls, seed=0, batch_size=2048, max_epochs=300,
                                patience=30, **p).fit(Xtr_s, y[iTR], c[iTR],
                                                      X_val=Xva_s, y_val_xy=y[iVA])
        else:
            mi = make_regressor(kind, p).fit(Xtr_s, y[iTR])
        inner_preds.append(mi.predict(Xva_s))
    w = fit_convex_weights(inner_preds, y[iVA])
    print("  blend weights: " + ", ".join(f"{f}={wi:.2f}" for f, wi in zip(fam_order, w)))

    # ── STAGE 4: FINAL — refit on full tr_outer, score TE_OUTER exactly once ──
    print("\n" + "=" * 70)
    print("FINAL — scored ONCE on the untouched per-cell-temporal test fold")
    print("=" * 70)
    Xte = X[te_outer.index.to_numpy()]
    yte = y[te_outer.index.to_numpy()]
    te_cells = df.loc[te_outer.index, "cell_id"].to_numpy()
    # time-order the test stream for honest causal smoothing
    order = np.argsort(df.loc[te_outer.index, "wall_time_s"].to_numpy(), kind="stable")
    sc_f = StandardScaler().fit(Xtr_outer)
    Xtr_f, Xte_f, Xval_f = sc_f.transform(Xtr_outer), sc_f.transform(Xte), sc_f.transform(X[iVA])

    # refit each family on full tr_outer; collect test predictions. The torch
    # refit uses inner_val (scaled by the SAME outer scaler) only for early-stop.
    fam_models, fam_pred = {}, {}
    for kind in fam_order:
        p = results[kind]["params"]
        if kind == "torch":
            m = TorchLocalizer(n_cls=n_cls, seed=0, batch_size=2048, max_epochs=300,
                               patience=30, **p).fit(Xtr_f, ytr_outer, ctr_outer,
                                                     X_val=Xval_f, y_val_xy=y[iVA])
        else:
            m = make_regressor(kind, p).fit(Xtr_f, ytr_outer)
        fam_models[kind] = m
        fam_pred[kind] = m.predict(Xte_f)
    blend_pred = np.tensordot(w, np.stack([fam_pred[k] for k in fam_order]), axes=(0, 0))
    best_single = min(fam_order, key=lambda k: results[k]["inner_val"])

    # torch seed-ensemble (cheap GPU variance reduction) as an extra candidate
    torch_ens_pred = None
    if "torch" in fam_order:
        tp = results["torch"]["params"]
        seeds = fit_seed_ensemble(
            lambda s: TorchLocalizer(n_cls=n_cls, seed=s, batch_size=2048,
                                     max_epochs=300, patience=30, **tp),
            Xtr_f, ytr_outer, ctr_outer, Xval_f, y[iVA], n_seeds=2 if args.smoke else 3)
        torch_ens_pred = ensemble_predict(seeds, Xte_f)

    # smoothing window chosen HONESTLY on inner-val: a model trained on inner_tr
    # predicts inner_val OUT-OF-SAMPLE (time-ordered). Window in samples + seconds.
    if best_single == "torch":
        wm = TorchLocalizer(n_cls=n_cls, seed=0, batch_size=2048, max_epochs=300,
                            patience=30, **results["torch"]["params"]).fit(
                                Xtr_s, y[iTR], c[iTR], X_val=Xva_s, y_val_xy=y[iVA])
    else:
        wm = make_regressor(best_single, results[best_single]["params"]).fit(Xtr_s, y[iTR])
    vorder = np.argsort(df.loc[inner_val.index, "wall_time_s"].to_numpy(), kind="stable")
    vpred, vy = wm.predict(Xva_s)[vorder], y[iVA][vorder]
    swin, swin_med = 1, reg_metrics(vy, vpred)["median"]
    for ww in SMOOTH:
        md = reg_metrics(vy, smooth_regression(vpred, ww))["median"]
        if md < swin_med:
            swin, swin_med = ww, md
    dt = float(np.median(np.diff(np.sort(df.loc[te_outer.index, "wall_time_s"].to_numpy()))))

    print("\nREGRESSION (median / mean / p90 cm) — RAW first:")
    cands = {**{k: fam_pred[k] for k in fam_order}, "ENSEMBLE": blend_pred}
    if torch_ens_pred is not None:
        cands["TORCH-ENS"] = torch_ens_pred
    final_metrics = {}
    for name, pred in cands.items():
        raw = reg_metrics(yte, pred)
        sm = reg_metrics(yte[order], smooth_regression(pred[order], swin))
        final_metrics[name] = {"raw": raw, "smoothed": sm}
        print(f"  {name:9s} raw {raw['median']:5.1f}/{raw['mean']:5.1f}/{raw['p90']:5.1f}   "
              f"smoothed(w={swin}, ~{swin*dt:.2f}s) {sm['median']:5.1f}/{sm['mean']:5.1f}/{sm['p90']:5.1f}")

    # classification (honest 3 numbers) from the torch head
    proba = fam_models["torch"].predict_proba(Xte_f)
    pred_cls_idx = proba.argmax(1)
    pred_cells = enc.inverse_transform(pred_cls_idx)
    train_classes = enc.inverse_transform(np.unique(ctr_outer))
    crep = cls_report(te_cells, pred_cells, train_classes)
    sm_idx = smooth_classification(proba[order], swin)
    crep_sm = cls_report(te_cells[order], enc.inverse_transform(sm_idx), train_classes)
    print("\nCLASSIFICATION (torch head):")
    print(f"  RAW      known-cell {crep['known_cell_accuracy']*100:5.1f}%  "
          f"full-test {crep['full_test_accuracy']*100:5.1f}%  "
          f"unknown-cell frac {crep['unknown_cell_fraction']*100:.1f}%")
    print(f"  SMOOTHED known-cell {crep_sm['known_cell_accuracy']*100:5.1f}%  "
          f"full-test {crep_sm['full_test_accuracy']*100:5.1f}%")

    # context-only leaky splits + end-of-run lgbm LOSO. Pure diagnostics, NOT used
    # to build the card. SKIP with --no-loso-tail: the tuned-lgbm refit per session
    # is pathologically slow at many-session scale (a 20-session run spends hours
    # here) and ml_drift.py is the proper LOSO harness anyway.
    if args.no_loso_tail:
        print("\nCONTEXT + LEAVE-ONE-SESSION-OUT: skipped (--no-loso-tail); "
              "use ml_drift.py for the honest LOSO.")
    else:
        # context-only leaky splits (NEVER cited) using best single family regressor
        print("\nCONTEXT (NOT the result — flagged splits):")
        for label, (a, b) in [("stratified (LEAKY upper bound)", stratified_cell_split(df, 0.8)),
                              ("global-temporal (coverage-limited)", temporal_split(df, 0.8))]:
            sc_c, Xa, Xb = scale(X[a.index.to_numpy()], X[b.index.to_numpy()])
            mc = make_regressor(best_single if best_single != "torch" else "lgbm",
                                results.get(best_single, {"params": lgbm_probe})["params"]
                                if best_single != "torch" else lgbm_probe).fit(Xa, y[a.index.to_numpy()])
            med = reg_metrics(y[b.index.to_numpy()], mc.predict(Xb))["median"]
            print(f"  {label:36s} reg median {med:5.1f} cm")

        # leave-one-session-out (auto-skips on 1 session) — the real generalization
        loso = list(leave_one_session_out(df))
        if loso:
            print("\nLEAVE-ONE-SESSION-OUT (headline once >=2 sessions):")
            for held, a, b in loso:
                sc_l, Xa, Xb = scale(X[a.index.to_numpy()], X[b.index.to_numpy()])
                m = make_regressor("lgbm", results["lgbm"]["params"]).fit(Xa, y[a.index.to_numpy()])
                print(f"  hold {held:30s} reg median {reg_metrics(y[b.index.to_numpy()], m.predict(Xb))['median']:5.1f} cm")
        else:
            print("\nLEAVE-ONE-SESSION-OUT: skipped (1 session). Collect >=2 to unlock the real drift test.")

    # ── model card + artifacts ──
    def git_commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            return "unknown"
    best_name = min(final_metrics, key=lambda n: final_metrics[n]["smoothed"]["median"])
    card = {
        "sessions": [s.rstrip("/") for s in args.sessions],
        "n_samples": int(len(df)), "n_cells": int(n_cls), "n_sessions": int(n_sessions),
        "git_commit": git_commit(),
        "split_policy": "nested per-cell-temporal: outer test untouched; selection on inner-val/OOF",
        "feature_config": best_cfg_name, "feature_params": feat_cfgs[best_cfg_name],
        "board_ids": list(boards_in_multiboard_df(df)),
        "ensemble_weights": {f: float(wi) for f, wi in zip(fam_order, w)},
        "smoothing_window_samples": int(swin), "smoothing_window_seconds": round(swin * dt, 3),
        "best_by_smoothed_median": best_name,
        "final_test_metrics_cm": {k: v for k, v in final_metrics.items()},
        "classification": {"raw": crep, "smoothed": crep_sm},
        "per_family_inner_val_cm": {k: results[k]["inner_val"] for k in results},
        "model_params": {k: results[k]["params"] for k in results},
        "honest_note": "per-cell-temporal test scored once; stratified shown only as flagged leaky upper bound",
        "load_note": "to load model_v3.joblib, import lightgbm and xgboost BEFORE torch (or just `import train_v3`) — torch's OpenMP runtime loaded first segfaults the lightgbm Booster deserialization. torch model is saved on CPU.",
        "lib_versions": {m: __import__(m).__version__ for m in ["numpy", "sklearn", "torch", "optuna", "lightgbm", "xgboost"]},
    }
    with open(os.path.join(out_dir, "model_card_v3.json"), "w") as f:
        json.dump(card, f, indent=2, default=str)
    for _m in fam_models.values():       # CPU tensors so the bundle loads anywhere
        if hasattr(_m, "to_cpu"):
            _m.to_cpu()
    try:
        joblib.dump({"models": fam_models, "weights": w, "fam_order": fam_order,
                     "encoder": enc, "scaler": sc_f, "feature_config": feat_cfgs[best_cfg_name],
                     "smoothing_window": swin}, os.path.join(out_dir, "model_v3.joblib"))
    except Exception as e:
        print(f"  (model_v3.joblib not saved: {type(e).__name__}: {e}; model_card_v3.json still written)")

    # plot: per-family + ensemble test median (raw vs smoothed)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(final_metrics)
    xr = np.arange(len(labels))
    ax.bar(xr - 0.2, [final_metrics[l]["raw"]["median"] for l in labels], 0.4, label="raw")
    ax.bar(xr + 0.2, [final_metrics[l]["smoothed"]["median"] for l in labels], 0.4, label=f"smoothed w={swin}")
    ax.axhline(57.7, ls="--", color="grey", label="v1 baseline 57.7cm")
    ax.set_xticks(xr); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("test reg median (cm)"); ax.set_title("train_v3 — honest per-cell-temporal test")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_v3_results.png"), dpi=110); plt.close()

    print("\n" + "=" * 70)
    print(f"HEADLINE (honest, scored once): best = {best_name}")
    bm = final_metrics[best_name]
    print(f"  regression: {bm['smoothed']['median']:.1f} cm median (smoothed) / "
          f"{bm['raw']['median']:.1f} cm raw   [v1 baseline 57.7 cm]")
    print(f"  classification (torch): {crep_sm['full_test_accuracy']*100:.1f}% full-test "
          f"/ {crep_sm['known_cell_accuracy']*100:.1f}% known-cell   [v1 baseline 39.1%]")
    print(f"  saved model_card_v3.json + model_v3.joblib + plot_v3_results.png in {out_dir}")


if __name__ == "__main__":
    main()
