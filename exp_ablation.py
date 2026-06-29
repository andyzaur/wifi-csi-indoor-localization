#!/usr/bin/env python3
"""exp_ablation.py — RX-count ablation: how much accuracy does each receiver buy?

Same honest protocol and the same fixed torch backbone as ml_drift (BACKBONE is
imported from there), applied to every non-empty subset of the RX boards: all
singles, all pairs, the full triple (7 subsets for 3 boards).

FAIRNESS: the multiboard frame is built ONCE with ALL boards (masked-phase v2
build), so the row set — fixed by all-boards CSI freshness + dedup — is identical
for every subset; only the FEATURE COLUMNS vary. Comparing subsets on different
row sets would conflate "fewer boards" with "different/easier frames".

Two evals per subset, both seed-ensembled under the fixed BACKBONE:
  IN-DOMAIN  pooled per-cell-temporal 0.8 split (median / p90)
  ZERO-SHOT  LOSO over sessions — scaler fit on TRAINING sessions only, inner
             0.85 per-cell-temporal split for early stop, seed ensemble (the
             ml_drift.loso_eval_method recipe, implemented locally because that
             function hardwires the all-board feature specs)

Writes ablation_card.json + ablation_bars.png into --out.

Usage:
    source venv/bin/activate
    python3 -u exp_ablation.py sessions/A sessions/B ... --out sessions/ablation_<date>
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, itertools, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dataset import (load_session, build_multiboard_dataset, grid_cell_id,
                     boards_in_multiboard_df, multiboard_feature_matrix)
from train_final import per_cell_temporal_split
import ml_drift as M


def subset_label(subset):
    """(1, 4) -> 'b1+b4' — bar/json label for a board subset."""
    return "+".join(f"b{b}" for b in subset)


def board_subsets(boards):
    """All non-empty subsets, singles -> pairs -> full set (7 for 3 boards)."""
    return [s for r in range(1, len(boards) + 1)
            for s in itertools.combinations(boards, r)]


def fit_eval(X, df, y, c, n_cls, tr_idx, te_idx, seeds, max_epochs):
    """One scaler+seed-ensemble fit on tr_idx, scored once on te_idx.

    The ml_drift recipe: inner 0.85 per-cell-temporal split of the train rows
    for early stopping, StandardScaler fit on the inner-train only, fixed
    BACKBONE via M.seed_predictions. Returns reg metrics + the per-seed median
    spread (seed_std) so subset deltas can be read against seed noise."""
    tr_df = df.iloc[tr_idx].reset_index(drop=True)
    itr_df, iva_df = per_cell_temporal_split(tr_df, 0.85)
    i_tr = tr_idx[itr_df.index.to_numpy()]
    i_va = tr_idx[iva_df.index.to_numpy()]
    sc = StandardScaler().fit(X[i_tr])
    Xtr, Xva, Xte = sc.transform(X[i_tr]), sc.transform(X[i_va]), sc.transform(X[te_idx])
    preds = M.seed_predictions(Xtr, y[i_tr], c[i_tr], Xva, y[i_va], Xte,
                               n_cls, seeds, max_epochs)
    m = M.reg_metrics(y[te_idx], np.mean(preds, axis=0))
    m["seed_std"] = float(np.std([M.reg_metrics(y[te_idx], p)["median"] for p in preds]))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", "-o", default=None)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=250)
    ap.add_argument("--smoke", action="store_true",
                    help="wiring check: 1 seed, 40 epochs, every 5th row per session")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.max_epochs = 1, 40
    out_dir = args.out or ("sessions/ablation_" + args.sessions[0].rstrip("/").split("/")[-1])
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    # ── load ONCE with ALL boards (masked-phase build) — the fixed row set ──
    dfs = []
    for s in args.sessions:
        csi, cam, clap = load_session(s.rstrip("/"))
        d = build_multiboard_dataset(csi, cam, clap, phase_mode="masked")
        d["session"] = os.path.basename(s.rstrip("/"))
        if args.smoke:
            d = d.iloc[::5].reset_index(drop=True)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).sort_values(["session", "wall_time_s"]).reset_index(drop=True)
    df["cell_id"] = grid_cell_id(df)
    enc = LabelEncoder().fit(df["cell_id"])
    c = enc.transform(df["cell_id"])
    n_cls = len(enc.classes_)
    y = df[["x_cm", "y_cm"]].to_numpy(np.float64)
    sessions = list(dict.fromkeys(df["session"].tolist()))
    boards = boards_in_multiboard_df(df)
    subsets = board_subsets(boards)
    print(f"Loaded {len(df):,} samples · {n_cls} cells · {len(sessions)} sessions · "
          f"boards {list(boards)} -> {len(subsets)} subsets", flush=True)
    if len(sessions) < 2:
        print("NOTE: <2 sessions — LOSO skipped, in-domain only.", flush=True)

    # outer pooled split is the SAME for every subset (rows fixed by design)
    tr_pool, te_pool = per_cell_temporal_split(df, 0.8)
    pool_tr, pool_te = tr_pool.index.to_numpy(), te_pool.index.to_numpy()

    results = {}
    for subset in subsets:
        lab = subset_label(subset)
        t0 = time.time()
        # only the feature columns change per subset — same rows, same labels
        X, names = multiboard_feature_matrix(df, board_ids=subset, include_phase=True,
                                             include_rssi=True, drop_null_subcarriers=True)
        X = X.astype(np.float64)
        indom = fit_eval(X, df, y, c, n_cls, pool_tr, pool_te, args.seeds, args.max_epochs)
        loso = None
        if len(sessions) >= 2:
            per_fold = {}
            for held in sessions:
                te = np.where((df["session"] == held).to_numpy())[0]
                tr = np.where((df["session"] != held).to_numpy())[0]
                per_fold[held] = fit_eval(X, df, y, c, n_cls, tr, te,
                                          args.seeds, args.max_epochs)
            loso = {"per_fold": per_fold,
                    "mean": float(np.mean([per_fold[h]["median"] for h in sessions])),
                    "p90_mean": float(np.mean([per_fold[h]["p90"] for h in sessions]))}
        results[lab] = {"boards": list(subset), "n_features": len(names),
                        "in_domain": indom, "loso": loso}
        lm = f"{loso['mean']:6.1f}" if loso else "   n/a"
        print(f"  {lab:12s} {len(names):>4d} feat   in-domain {indom['median']:6.1f} cm "
              f"(p90 {indom['p90']:6.1f})   LOSO mean {lm} cm   ({time.time()-t0:.0f}s)",
              flush=True)

    # ── grouped bars: in-domain vs LOSO mean, single -> pair -> triple order ──
    labs = [subset_label(s) for s in subsets]
    ind = [results[l]["in_domain"]["median"] for l in labs]
    los = [results[l]["loso"]["mean"] if results[l]["loso"] else np.nan for l in labs]
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x - 0.2, ind, 0.4, label="in-domain (per-cell-temporal 0.8)", color="tab:blue")
    ax.bar(x + 0.2, los, 0.4, label="zero-shot LOSO (mean over folds)", color="tab:orange")
    for xi, v in zip(x - 0.2, ind):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    for xi, v in zip(x + 0.2, los):
        if np.isfinite(v):
            ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_xlabel("RX subset (singles → pairs → triple)")
    ax.set_ylabel("median error (cm)")
    ax.set_title("RX-count ablation — fixed rows + fixed backbone, only feature columns vary")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "ablation_bars.png"), dpi=120); plt.close()

    card = {
        "kind": "RX-count ablation",
        "sessions": [s.rstrip("/") for s in args.sessions],
        "boards": list(boards), "n_samples": int(len(df)), "n_cells": int(n_cls),
        "backbone": M.BACKBONE, "seeds": args.seeds, "max_epochs": args.max_epochs,
        "smoke": bool(args.smoke),
        "features": "masked-phase v2 (amp+phase+rssi, null subcarriers dropped) per board",
        "fairness_note": ("Frame built ONCE with all boards (phase_mode='masked'), so the row "
                          "set is fixed by all-boards freshness + dedup; per subset only the "
                          "feature COLUMNS change. In-domain uses one shared per-cell-temporal "
                          "0.8 split; LOSO refits scaler+model per fold on training sessions "
                          "only, with an inner 0.85 split for early stop."),
        "subsets": results,
        "runtime_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(out_dir, "ablation_card.json"), "w") as f:
        json.dump(card, f, indent=2, default=str)
    print(f"\nwrote {out_dir}/ablation_card.json + ablation_bars.png "
          f"({card['runtime_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
