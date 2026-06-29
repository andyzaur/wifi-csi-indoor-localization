#!/usr/bin/env python3
"""Final training script: multi-board MLP + temporal smoothing.

Combines every winning ingredient from this session:
- Multi-board CSI features (one sample per camera frame, 195 features from 3 RX)
- MLP classifier + MLP regressor
- Stratified-by-cell split (model capacity bound)
- Temporal split (realistic deployment evaluation)
- Rolling-window temporal smoothing on predictions

Produces the headline numbers for the thesis: raw vs smoothed, model capacity
vs deployment accuracy.

Usage:
    python3 train_final.py sessions/20260519_02_firstWorkingTest
    python3 train_final.py sessions/s1 sessions/s2 -o sessions/combined
"""

import argparse
import os
import time
import warnings
import joblib
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dataset import (load_session, build_multiboard_dataset,
                     multiboard_feature_matrix, grid_cell_id)


def stratified_cell_split(df, train_frac=0.8, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cell, grp in df.groupby("cell_id"):
        idx = grp.index.to_numpy().copy()
        if len(idx) < 2:
            train_idx.extend(idx)
            continue
        rng.shuffle(idx)
        cut = int(len(idx) * train_frac)
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def temporal_split(df, train_frac=0.8):
    df_sorted = df.sort_values("wall_time_s").reset_index(drop=True)
    n = len(df_sorted)
    cut = int(n * train_frac)
    return df_sorted.iloc[:cut].copy(), df_sorted.iloc[cut:].copy()


def per_cell_temporal_split(df, train_frac=0.8):
    """Per-cell temporal block split (the honest split).

    Within each cell, the earliest `train_frac` of frames (by wall_time_s) go to
    train, the latest go to test. Keeps every cell in both sets (full room
    coverage) while separating test frames in time from their train neighbours,
    so it defeats near-duplicate leakage WITHOUT the global-temporal split's
    coverage collapse. Cells with <2 frames go entirely to train.
    """
    train_idx, test_idx = [], []
    for cell, grp in df.groupby("cell_id"):
        idx = grp.sort_values("wall_time_s").index.to_numpy()
        if len(idx) < 2:
            train_idx.extend(idx)
            continue
        cut = int(len(idx) * train_frac)
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def smooth_regression(pred_xy, window):
    """Rolling-mean smooth (causal, last N predictions). pred_xy is (N, 2) time-ordered."""
    if window <= 1:
        return pred_xy
    out = np.empty_like(pred_xy)
    for i in range(len(pred_xy)):
        lo = max(0, i - window + 1)
        out[i] = pred_xy[lo:i + 1].mean(axis=0)
    return out


def smooth_classification(prob, window):
    """Rolling-mean over predicted probabilities (causal). Returns smoothed argmax."""
    if window <= 1:
        return prob.argmax(axis=1)
    out_prob = np.empty_like(prob)
    for i in range(len(prob)):
        lo = max(0, i - window + 1)
        out_prob[i] = prob[lo:i + 1].mean(axis=0)
    return out_prob.argmax(axis=1)


def evaluate_split(df, df_train, df_test, label, smoothing_windows=(1, 3, 5, 9)):
    """Train MLP on (df_train) and evaluate (raw + smoothed) on time-ordered df_test."""
    print(f"\n────── {label} ──────")
    print(f"Train: {len(df_train):,}  Test: {len(df_test):,}")

    # Time-order test set so smoothing is meaningful
    df_test = df_test.sort_values("wall_time_s").reset_index(drop=True)

    X_tr, _ = multiboard_feature_matrix(df_train)
    X_te, _ = multiboard_feature_matrix(df_test)
    enc = LabelEncoder()
    y_tr_cls = enc.fit_transform(df_train["cell_id"].to_numpy())

    # Some test cells may be missing from train; handle by computing accuracy only on
    # known classes (or skipping unknown). For temporal split this can matter.
    known = np.isin(df_test["cell_id"].to_numpy(), enc.classes_)
    n_unknown = (~known).sum()
    if n_unknown:
        print(f"  ({n_unknown} test rows have cells never seen in train — excluded from cls)")

    y_te_cls = np.full(len(df_test), -1, dtype=int)
    y_te_cls[known] = enc.transform(df_test["cell_id"].to_numpy()[known])

    y_tr_xy = df_train[["x_cm", "y_cm"]].to_numpy(dtype=np.float32)
    y_te_xy = df_test[["x_cm", "y_cm"]].to_numpy(dtype=np.float32)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    n_classes = len(enc.classes_)

    # ── Classification ──
    t0 = time.time()
    mlpc = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400, random_state=42)
    mlpc.fit(X_tr_s, y_tr_cls)
    prob = mlpc.predict_proba(X_te_s)
    dt_c = time.time() - t0

    cls_results = []
    for w in smoothing_windows:
        pred_cls = smooth_classification(prob, w)
        acc_all = accuracy_score(y_te_cls[known], pred_cls[known])
        cls_results.append((w, acc_all))

    # ── Regression ──
    t0 = time.time()
    mlpr = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=400, random_state=42, early_stopping=True)
    mlpr.fit(X_tr_s, y_tr_xy)
    pred_xy_raw = mlpr.predict(X_te_s)
    dt_r = time.time() - t0

    reg_results = []
    for w in smoothing_windows:
        pred = smooth_regression(pred_xy_raw, w)
        err = np.linalg.norm(y_te_xy - pred, axis=1)
        reg_results.append((w, np.median(err), err.mean(), np.percentile(err, 90)))

    print(f"  classification ({dt_c:.1f}s, {n_classes} classes):")
    for w, acc in cls_results:
        print(f"    window {w:>2}:  {acc*100:6.2f}%")
    print(f"  regression ({dt_r:.1f}s):")
    for w, med, mean, p90 in reg_results:
        print(f"    window {w:>2}:  median {med:6.2f}cm  mean {mean:6.2f}cm  p90 {p90:6.2f}cm")

    return {
        "label": label,
        "n_classes": n_classes,
        "cls_results": cls_results,
        "reg_results": reg_results,
        "pred_xy_raw": pred_xy_raw,
        "y_te_xy": y_te_xy,
        "df_test": df_test,
        "model_cls": mlpc,
        "model_reg": mlpr,
        "scaler": scaler,
        "encoder": enc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--out", "-o", default=None)
    args = parser.parse_args()

    out_dir = args.out or args.sessions[0].rstrip("/")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {len(args.sessions)} session(s)...")
    dfs = []
    for s in args.sessions:
        csi, camera, clap = load_session(s.rstrip("/"))
        # dedup=True by default: drops camera frames that reused the same CSI
        # (camera faster than CSI rate). Honest numbers; pass dedup=False to
        # reproduce the older, leakage-inflated figures.
        df = build_multiboard_dataset(csi, camera, clap)
        df["session"] = os.path.basename(s.rstrip("/"))
        dfs.append(df)
        print(f"  {s}: {len(df):,} multiboard samples")
    df = pd.concat(dfs, ignore_index=True)
    df["cell_id"] = grid_cell_id(df)
    print(f"Total: {len(df):,} samples, {df['cell_id'].nunique()} cells")

    # ─── Two splits, side-by-side ─────────────────────────────────────
    df_str_tr, df_str_te = stratified_cell_split(df, train_frac=0.8)
    df_str_tr = df_str_tr.reset_index(drop=True)
    df_str_te = df_str_te.reset_index(drop=True)
    res_strat = evaluate_split(df, df_str_tr, df_str_te,
                               "STRATIFIED-BY-CELL split (model capacity)")

    df_temp_tr, df_temp_te = temporal_split(df, train_frac=0.8)
    df_temp_tr = df_temp_tr.reset_index(drop=True)
    df_temp_te = df_temp_te.reset_index(drop=True)
    res_temp = evaluate_split(df, df_temp_tr, df_temp_te,
                              "TEMPORAL split (deployment realism)")

    df_pct_tr, df_pct_te = per_cell_temporal_split(df, train_frac=0.8)
    df_pct_tr = df_pct_tr.reset_index(drop=True)
    df_pct_te = df_pct_te.reset_index(drop=True)
    res_pct = evaluate_split(df, df_pct_tr, df_pct_te,
                             "PER-CELL TEMPORAL split (honest)")

    # ─── Summary table + plot ─────────────────────────────────────────
    rows = []
    for r in (res_strat, res_temp, res_pct):
        for w, acc in r["cls_results"]:
            rows.append({"split": r["label"], "metric": "cls_acc_%",
                         "window": w, "value": acc * 100})
        for w, med, mean, p90 in r["reg_results"]:
            rows.append({"split": r["label"], "metric": "reg_med_cm",
                         "window": w, "value": med})
            rows.append({"split": r["label"], "metric": "reg_mean_cm",
                         "window": w, "value": mean})
            rows.append({"split": r["label"], "metric": "reg_p90_cm",
                         "window": w, "value": p90})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "final_results.csv"), index=False)

    # Plot: 2x3 — top row classification, bottom regression; one column per
    # split (stratified, global-temporal, per-cell-temporal).
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    ws = [w for w, _ in res_strat["cls_results"]]

    for i, r in enumerate((res_strat, res_temp, res_pct)):
        accs = [a * 100 for _, a in r["cls_results"]]
        axes[0, i].plot(ws, accs, "o-", lw=2)
        axes[0, i].set_xlabel("smoothing window (samples)")
        axes[0, i].set_ylabel("classification accuracy (%)")
        axes[0, i].set_title(f"{r['label']}\nclassification (45 cells)")
        axes[0, i].grid(alpha=0.3)
        axes[0, i].axhline(100 / r["n_classes"], ls="--", color="grey",
                            lw=0.5, label=f"random ({100/r['n_classes']:.1f}%)")
        for w, a in zip(ws, accs):
            axes[0, i].text(w, a + 0.5, f"{a:.1f}%", ha="center", fontsize=9)
        axes[0, i].legend()

        meds = [m for _, m, _, _ in r["reg_results"]]
        p90s = [p for _, _, _, p in r["reg_results"]]
        axes[1, i].plot(ws, meds, "o-", lw=2, label="median")
        axes[1, i].plot(ws, p90s, "s-", lw=2, label="p90")
        axes[1, i].set_xlabel("smoothing window (samples)")
        axes[1, i].set_ylabel("regression error (cm)")
        axes[1, i].set_title(f"{r['label']}\nregression error")
        axes[1, i].grid(alpha=0.3)
        for w, m in zip(ws, meds):
            axes[1, i].text(w, m + 1, f"{m:.0f}", ha="center", fontsize=9)
        axes[1, i].legend()

    plt.tight_layout()
    p = os.path.join(out_dir, "plot_final_results.png")
    plt.savefig(p, dpi=110)
    plt.close()
    print(f"\nSaved {p}")

    # CDF plot: best smoothed regression vs raw, both splits
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for i, r in enumerate((res_strat, res_temp, res_pct)):
        for w in (1, 5):
            pred = smooth_regression(r["pred_xy_raw"], w)
            err = np.linalg.norm(r["y_te_xy"] - pred, axis=1)
            errs_sorted = np.sort(err)
            cdf = np.arange(1, len(errs_sorted) + 1) / len(errs_sorted)
            axes[i].plot(errs_sorted, cdf, lw=2,
                         label=f"window={w} (median {np.median(err):.0f}cm)")
        axes[i].set_xlabel("error (cm)")
        axes[i].set_ylabel("CDF")
        axes[i].set_title(r["label"])
        axes[i].grid(alpha=0.3)
        axes[i].legend()
        axes[i].axhline(0.5, ls="--", color="grey", lw=0.5)
        axes[i].axhline(0.9, ls="--", color="grey", lw=0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, "plot_cdf_smoothed.png")
    plt.savefig(p, dpi=110)
    plt.close()
    print(f"Saved {p}")

    # ─── Save the production model (per-cell-temporal MLP — the honest split) ──
    model_path = os.path.join(out_dir, "model_final.joblib")
    joblib.dump({
        "classifier": res_pct["model_cls"],
        "regressor": res_pct["model_reg"],
        "scaler": res_pct["scaler"],
        "encoder": res_pct["encoder"],
        "feature_set": "multiboard_amp+rssi",
        "n_features": 195,
        "n_classes": res_pct["n_classes"],
    }, model_path)
    print(f"Saved {model_path}")

    # Headline summary
    best_str = max(res_strat["cls_results"], key=lambda x: x[1])
    best_str_reg = min(res_strat["reg_results"], key=lambda x: x[1])
    best_pct = max(res_pct["cls_results"], key=lambda x: x[1])
    best_pct_reg = min(res_pct["reg_results"], key=lambda x: x[1])
    best_temp = max(res_temp["cls_results"], key=lambda x: x[1])
    best_temp_reg = min(res_temp["reg_results"], key=lambda x: x[1])
    print()
    print("=" * 60)
    print("HEADLINE NUMBERS")
    print("=" * 60)
    print("PER-CELL TEMPORAL split (HONEST -- cite this one):")
    print(f"  best classification: {best_pct[1]*100:.1f}%  (window {best_pct[0]})")
    print(f"  best regression:     {best_pct_reg[1]:.1f}cm median  (window {best_pct_reg[0]})")
    print("Stratified split (optimistic upper bound; near-duplicate leakage):")
    print(f"  best classification: {best_str[1]*100:.1f}%  (window {best_str[0]})")
    print(f"  best regression:     {best_str_reg[1]:.1f}cm median  (window {best_str_reg[0]})")
    print("Global temporal split (coverage-limited; collapses on raster walks):")
    print(f"  best classification: {best_temp[1]*100:.1f}%  (window {best_temp[0]})")
    print(f"  best regression:     {best_temp_reg[1]:.1f}cm median  (window {best_temp_reg[0]})")


if __name__ == "__main__":
    main()
