#!/usr/bin/env python3
"""finalize_drift.py — turn the ml_drift study into deliverables:
  1. calibration_curve.png  (the same-day calibration lever, per held-out session)
  2. zero_shot_bars.png      (zero-shot LOSO per method, mean + seed band)
  3. model_drift.joblib + model_card_drift.json — the deployable model: an amp+PHASE
     seed-ensemble torch (phase robustly helps cross-session, the study's finding)
     trained on ALL sessions, with the honest LOSO/calibration context recorded.

Usage: python3 finalize_drift.py sessions/A sessions/B sessions/C --study sessions/drift_full
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, warnings
warnings.filterwarnings("ignore")
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dataset import (build_multiboard_dataset, grid_cell_id, load_session,
                     multiboard_feature_matrix, boards_in_multiboard_df)
from torch_net import TorchLocalizer, fit_seed_ensemble


def plot_calibration(card, out):
    cal = card["calibration_sweep_median_cm"]
    fracs = sorted({float(k) for d in cal.values() for k in d}, key=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for held, d in cal.items():
        ys = [d.get(str(f), np.nan) for f in fracs]
        ax.plot(fracs, ys, "o-", label=held.replace("20260", "…"), alpha=0.8)
    mean_ys = [np.mean([cal[h].get(str(f), np.nan) for h in cal]) for f in fracs]
    ax.plot(fracs, mean_ys, color="k", marker="s", ls="--", lw=2.5, label="mean", zorder=5)
    ax.axhline(card["zero_shot_loso"]["baseline (amp+rssi)"]["mean"], ls=":", color="grey",
               label=f"zero-shot baseline {card['zero_shot_loso']['baseline (amp+rssi)']['mean']:.0f} cm")
    ax.set_xlabel("fraction of each cell labelled today (calibration pass)")
    ax.set_ylabel("same-day test median error (cm)")
    ax.set_title("Calibration closes the drift gap (same-day, test-half held out)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=120); plt.close()


def plot_zero_shot(card, out):
    zs = card["zero_shot_loso"]
    names = list(zs.keys())
    means = [zs[n]["mean"] for n in names]
    bands = [zs[n]["seed_band"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#888" if n == "baseline (amp+rssi)" else
              ("#2a9d8f" if n in card["robust_wins_over_baseline"] else "#e76f51") for n in names]
    ax.bar(range(len(names)), means, yerr=bands, color=colors, capsize=4)
    ax.axhline(zs["baseline (amp+rssi)"]["mean"], ls=":", color="grey")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" (amp+rssi)", "") for n in names], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("zero-shot LOSO mean median (cm)")
    ax.set_title("Zero-shot cross-session error by method (green = robust win over baseline)")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--study", required=True, help="ml_drift out dir with drift_card.json")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    card = json.load(open(os.path.join(args.study, "drift_card.json")))

    plot_calibration(card, os.path.join(args.study, "calibration_curve.png"))
    plot_zero_shot(card, os.path.join(args.study, "zero_shot_bars.png"))
    print("wrote calibration_curve.png + zero_shot_bars.png")

    # ── deployable model: amp+PHASE seed-ensemble on ALL sessions ──
    dfs = []
    for s in args.sessions:
        csi, cam, clap = load_session(s.rstrip("/"))
        d = build_multiboard_dataset(csi, cam, clap)
        d["session"] = os.path.basename(s.rstrip("/"))
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).sort_values(["session", "wall_time_s"]).reset_index(drop=True)
    df["cell_id"] = grid_cell_id(df)
    enc = LabelEncoder().fit(df["cell_id"])
    c = enc.transform(df["cell_id"])
    n_cls = len(enc.classes_)
    X, names = multiboard_feature_matrix(df, include_phase=True, include_rssi=True)
    X = X.astype(np.float32)
    y = df[["x_cm", "y_cm"]].to_numpy(np.float64)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    # small held-out tail per cell just for early stopping (not an honest test — the
    # honest generalization number is the LOSO study, recorded in the card below)
    from train_final import per_cell_temporal_split
    tr, va = per_cell_temporal_split(df, 0.9)
    itr, iva = tr.index.to_numpy(), va.index.to_numpy()
    models = fit_seed_ensemble(
        lambda s: TorchLocalizer(n_cls=n_cls, seed=s, batch_size=2048, max_epochs=300,
                                 patience=30, depth=3, width=256, dropout=0.10, norm="layer",
                                 residual=False, lr=2.66e-3, weight_decay=1.1e-5, noise_std=0.037,
                                 feat_drop=0.114, label_smoothing=0.07, w_cls=0.29),
        Xs[itr], y[itr], c[itr], Xs[iva], y[iva], n_seeds=args.seeds)
    for m in models:
        m.to_cpu()
    joblib.dump({"models": models, "scaler": sc, "encoder": enc, "feature": "amp+phase+rssi",
                 "board_ids": list(boards_in_multiboard_df(df)), "names": names},
                os.path.join(args.study, "model_drift.joblib"))

    model_card = {
        "model": "amp+phase+rssi seed-ensemble torch (5 seeds), trained on ALL sessions",
        "why_phase": "phase robustly lowers zero-shot LOSO (-13 cm vs amp-only, sign-consistent on all 3 held-out sessions, beyond the seed band) — the study's key finding",
        "honest_generalization_LOSO_cm": {
            "amp+rssi baseline": card["zero_shot_loso"]["baseline (amp+rssi)"]["mean"],
            "amp+phase (this model's config)": card["zero_shot_loso"]["+phase"]["mean"],
            "CORAL+phase (transductive, best zero-shot)": card["zero_shot_loso"]["CORAL +phase"]["mean"],
        },
        "same_day_calibration_cm": {"zero_shot": float(np.mean([card["calibration_sweep_median_cm"][h]["0.0"] for h in card["calibration_sweep_median_cm"]])),
                                    "full_pass": float(np.mean([card["calibration_sweep_median_cm"][h]["1.0"] for h in card["calibration_sweep_median_cm"]]))},
        "deploy_recipe": "train on all days; at a new session, optionally run a quick labelled calibration pass over the room and fine-tune (torch warm-start) — calibration is the biggest lever.",
        "sessions": card["sessions"], "n_samples": int(len(df)), "n_cells": int(n_cls),
        "study_card": os.path.join(args.study, "drift_card.json"),
    }
    json.dump(model_card, open(os.path.join(args.study, "model_card_drift.json"), "w"), indent=2, default=str)
    print("wrote model_drift.joblib + model_card_drift.json")


if __name__ == "__main__":
    main()
