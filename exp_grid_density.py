#!/usr/bin/env python3
"""exp_grid_density.py — classification accuracy vs grid cell size (in-domain only).

The 50 cm grid is a design choice, not a law: re-snap the continuous (x_cm, y_cm)
labels to 100 / 50 / 25 cm grids (same rule the marker layout uses: gx =
round(x/s)*s) and re-train the fixed ml_drift BACKBONE per spacing on the same
masked-phase v2 features. Per spacing: per-cell-temporal 0.8 split (the honest
split), seed ensemble, known-cell + full-test accuracy (the train_v3 cls_report
numbers) raw AND smoothed (causal w=9), plus the regression median as a sanity
check — regression reads the same labels, so it should be ~spacing-independent.

Caveat plotted on the figure: 25 cm cells are smaller than an adult's standing
footprint (~30-40 cm), so adjacent-cell confusions there are physically expected.

Writes grid_card.json + grid_curve.png into --out.

Usage:
    source venv/bin/activate
    python3 -u exp_grid_density.py sessions/A sessions/B ... --out sessions/grid_<date>
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dataset import load_session, build_multiboard_dataset, multiboard_feature_matrix
from train_final import per_cell_temporal_split, smooth_classification
from torch_net import TorchLocalizer
import ml_drift as M

SPACINGS = (100, 50, 25)          # cm — coarse -> finer than the recorded grid
SMOOTH_W = 9                      # causal smoothing window (frames)


def snap_cells(df, spacing):
    """Re-snap continuous labels to a `spacing`-cm grid: gx = round(x/s)*s.

    Returns '<gx>_<gy>' string ids (same format as dataset.grid_cell_id), built
    from x_cm/y_cm — NOT from the recorded grid_x_cm/grid_y_cm, which are frozen
    at the session's own spacing."""
    gx = (np.round(df["x_cm"].to_numpy() / spacing) * spacing).astype(int)
    gy = (np.round(df["y_cm"].to_numpy() / spacing) * spacing).astype(int)
    return pd.Series([f"{a}_{b}" for a, b in zip(gx, gy)], index=df.index)


def cls_report(y_true_cells, pred_cells, train_classes):
    """Honest classification accuracy (train_v3 pattern): all three numbers."""
    known = np.isin(y_true_cells, train_classes)
    n = len(y_true_cells)
    n_correct_known = int((pred_cells[known] == y_true_cells[known]).sum())
    return {
        "unknown_cell_fraction": float((~known).mean()),
        "known_cell_accuracy": float(n_correct_known / max(1, known.sum())),
        "full_test_accuracy": float(n_correct_known / n),  # unknowns count as wrong
    }


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
    out_dir = args.out or ("sessions/grid_" + args.sessions[0].rstrip("/").split("/")[-1])
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    # ── load (masked-phase v2 build) + ONE feature matrix for all spacings ──
    dfs = []
    for s in args.sessions:
        csi, cam, clap = load_session(s.rstrip("/"))
        d = build_multiboard_dataset(csi, cam, clap, phase_mode="masked")
        d["session"] = os.path.basename(s.rstrip("/"))
        if args.smoke:
            d = d.iloc[::5].reset_index(drop=True)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).sort_values(["session", "wall_time_s"]).reset_index(drop=True)
    y = df[["x_cm", "y_cm"]].to_numpy(np.float64)
    X, names = multiboard_feature_matrix(df, include_phase=True, include_rssi=True,
                                         drop_null_subcarriers=True)
    X = X.astype(np.float64)
    times = df["wall_time_s"].to_numpy()
    print(f"Loaded {len(df):,} samples · {len(args.sessions)} sessions · "
          f"{X.shape[1]} features", flush=True)

    results = {}
    for spacing in SPACINGS:
        t0 = time.time()
        # cell ids, split and encoder are all per-spacing (n_cls varies)
        df["cell_id"] = snap_cells(df, spacing)
        enc = LabelEncoder().fit(df["cell_id"])
        c = enc.transform(df["cell_id"])
        n_cls = len(enc.classes_)
        tr_df, te_df = per_cell_temporal_split(df, 0.8)
        tr_idx, te_idx = tr_df.index.to_numpy(), te_df.index.to_numpy()
        itr_df, iva_df = per_cell_temporal_split(tr_df.reset_index(drop=True), 0.85)
        i_tr = tr_idx[itr_df.index.to_numpy()]
        i_va = tr_idx[iva_df.index.to_numpy()]
        sc = StandardScaler().fit(X[i_tr])
        Xtr, Xva, Xte = sc.transform(X[i_tr]), sc.transform(X[i_va]), sc.transform(X[te_idx])
        models = [TorchLocalizer(n_cls=n_cls, seed=sd, batch_size=2048,
                                 max_epochs=args.max_epochs, patience=30, **M.BACKBONE)
                  .fit(Xtr, y[i_tr], c[i_tr], X_val=Xva, y_val_xy=y[i_va])
                  for sd in range(args.seeds)]
        proba = np.mean([m.predict_proba(Xte) for m in models], axis=0)
        pred_xy = np.mean([m.predict(Xte) for m in models], axis=0)

        te_cells = df["cell_id"].to_numpy()[te_idx]
        train_classes = np.unique(df["cell_id"].to_numpy()[tr_idx])
        raw = cls_report(te_cells, enc.inverse_transform(proba.argmax(1)), train_classes)
        order = np.argsort(times[te_idx], kind="stable")     # time-order before smoothing
        sm_idx = smooth_classification(proba[order], SMOOTH_W)
        smoothed = cls_report(te_cells[order], enc.inverse_transform(sm_idx), train_classes)
        reg_med = M.reg_metrics(y[te_idx], pred_xy)["median"]

        results[str(spacing)] = {
            "n_cells": int(n_cls), "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "random_baseline_pct": 100.0 / n_cls,
            "raw": raw, "smoothed_w9": smoothed, "reg_median_cm": reg_med,
        }
        print(f"  {spacing:>3d} cm  {n_cls:>3d} cells   known-cell "
              f"{raw['known_cell_accuracy']*100:5.1f}% raw / "
              f"{smoothed['known_cell_accuracy']*100:5.1f}% w={SMOOTH_W}   full-test "
              f"{raw['full_test_accuracy']*100:5.1f}% / {smoothed['full_test_accuracy']*100:5.1f}%   "
              f"reg med {reg_med:5.1f} cm   ({time.time()-t0:.0f}s)", flush=True)

    # ── curve: accuracy vs spacing (reg median on a twin axis as the sanity) ──
    xs = list(SPACINGS)
    raw_acc = [results[str(s)]["raw"]["known_cell_accuracy"] * 100 for s in xs]
    sm_acc = [results[str(s)]["smoothed_w9"]["known_cell_accuracy"] * 100 for s in xs]
    reg_meds = [results[str(s)]["reg_median_cm"] for s in xs]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(xs, raw_acc, "o-", lw=2, color="tab:blue", label="known-cell accuracy (raw)")
    ax.plot(xs, sm_acc, "s--", lw=2, color="tab:cyan",
            label=f"known-cell accuracy (smoothed w={SMOOTH_W})")
    for s, a in zip(xs, raw_acc):
        ax.text(s, a + 1.2, f"{a:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xlabel("grid cell spacing (cm) — finer cells to the right? no: smaller = harder")
    ax.set_ylabel("classification accuracy (%) — per-cell-temporal 0.8 test")
    ax.invert_xaxis()                                # 100 -> 25: difficulty increases
    ax2 = ax.twinx()
    ax2.plot(xs, reg_meds, "^:", lw=1.5, color="grey",
             label="regression median (sanity)")
    ax2.set_ylabel("regression median error (cm) — ~spacing-independent", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Cell-classification accuracy vs grid density (in-domain, fixed backbone)")
    ax.annotate("25 cm < body footprint (~30-40 cm):\nadjacent-cell confusion is physical",
                xy=(25, raw_acc[-1]), xytext=(0.62, 0.12), textcoords="axes fraction",
                fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "grid_curve.png"), dpi=120)
    plt.close()

    card = {
        "kind": "grid-density sweep (in-domain classification)",
        "sessions": [s.rstrip("/") for s in args.sessions],
        "spacings_cm": list(SPACINGS), "n_samples": int(len(df)),
        "backbone": M.BACKBONE, "seeds": args.seeds, "max_epochs": args.max_epochs,
        "smoke": bool(args.smoke), "smooth_window": SMOOTH_W,
        "features": "masked-phase v2 (amp+phase+rssi, null subcarriers dropped), all boards",
        "note": ("Cell ids are re-snapped from continuous x_cm/y_cm (gx=round(x/s)*s) per "
                 "spacing; split/encoder/n_cls vary with spacing, features and rows do not. "
                 "Regression median is the sanity check (same labels at every spacing). "
                 "25 cm cells are smaller than a standing adult's footprint (~30-40 cm), so "
                 "label quantization + body size bound the achievable accuracy there."),
        "per_spacing": results,
        "runtime_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(out_dir, "grid_card.json"), "w") as f:
        json.dump(card, f, indent=2, default=str)
    print(f"\nwrote {out_dir}/grid_card.json + grid_curve.png "
          f"({card['runtime_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
