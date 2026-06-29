#!/usr/bin/env python3
"""exp_figures.py — thesis figures from a finished ml_drift --dump-preds run.

No training happens here: this consumes the per-fold LOSO prediction dumps
(preds__<method>__<fold>.npz with keys idx / y_xy / pred_xy / wall_time_s /
grid_x_cm / grid_y_cm, see ml_drift.dump_fold_preds) plus the same sessions list
— the sessions are only used to rebuild the walking SPEED from the camera
LABELS (np.gradient of (x, y) over wall_time_s, 9-sample median filter), never
the predictions. Produces into --drift-out (or --out):

  cdf_methods.png               error CDF per method (folds pooled)
  error_map_6session.png/.csv   per-cell median error, baseline method
  error_vs_speed.png            median error per walking-speed bin per method
  smoothing_compare.json/.png   raw vs causal rolling(w=9) vs Kalman, per method
  figures_card.json             all the numbers in one place

Usage:
    source venv/bin/activate
    python3 -u exp_figures.py sessions/A sessions/B ... --drift-out sessions/drift_<date>
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, glob, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import trim_to_session
from mlpipe import kalman_smooth
import ml_drift as M

SPEED_EDGES = (2.0, 10.0, 25.0, 50.0)                       # cm/s bin boundaries
SPEED_LABELS = ("0-2", "2-10", "10-25", "25-50", ">50")
ROLL_W = 9                                                  # causal smoothing window
NPZ_KEYS = ("idx", "y_xy", "pred_xy", "wall_time_s", "grid_x_cm", "grid_y_cm")


def rolling_mean_xy(pred_xy, window=ROLL_W):
    """Causal rolling mean over a time-ordered (n, 2) track (inline copy of
    train_final.smooth_regression so this script stays self-contained)."""
    if window <= 1:
        return pred_xy.copy()
    out = np.empty_like(pred_xy)
    for i in range(len(pred_xy)):
        lo = max(0, i - window + 1)
        out[i] = pred_xy[lo:i + 1].mean(axis=0)
    return out


def median_filter(v, w=9):
    """Centered w-sample median filter, edge-padded (denoises the label speed)."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 2 or w <= 1:
        return v.copy()
    half = w // 2
    pad = np.pad(v, (half, half), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, w)
    return np.median(win, axis=1)


def label_speed_table(sessions, smoke=False):
    """Pooled (times, speeds) lookup table from the camera LABELS of every session.

    Per session: detected camera frames (clap-trimmed when clap.csv parses),
    speed = |d(x,y)/dt| via np.gradient over wall_time_s, then a 9-sample median
    filter. Sessions' wall clocks are disjoint (different days), so one pooled
    time-sorted table serves nearest-time lookups for every fold."""
    ts, sp = [], []
    for sdir in sessions:
        sdir = sdir.rstrip("/")
        cam = pd.read_csv(os.path.join(sdir, "camera.csv"))
        if "timestamp_s" in cam.columns and "wall_time_s" not in cam.columns:
            cam = cam.rename(columns={"timestamp_s": "wall_time_s"})
        try:
            clap = pd.read_csv(os.path.join(sdir, "clap.csv"))
            cam = trim_to_session(cam, clap, "wall_time_s")
        except Exception:
            pass                                            # no clap -> use the full track
        cam = cam[cam["detected"] == 1].sort_values("wall_time_s").reset_index(drop=True)
        if smoke:
            cam = cam.iloc[::5].reset_index(drop=True)
        t = cam["wall_time_s"].to_numpy(np.float64)
        keep = np.concatenate([[True], np.diff(t) > 0])     # drop repeated timestamps
        t = t[keep]
        x = cam["x_cm"].to_numpy(np.float64)[keep]
        yy = cam["y_cm"].to_numpy(np.float64)[keep]
        if len(t) < 3:
            print(f"  skip speed for {sdir}: only {len(t)} usable camera frames", flush=True)
            continue
        speed = median_filter(np.hypot(np.gradient(x, t), np.gradient(yy, t)), 9)
        ts.append(t)
        sp.append(speed)
        print(f"  speed track {os.path.basename(sdir):34s} {len(t):>7,} frames  "
              f"median {np.median(speed):5.1f} cm/s", flush=True)
    if not ts:
        raise SystemExit("no usable camera label tracks — cannot rebuild speed")
    t_all = np.concatenate(ts)
    s_all = np.concatenate(sp)
    order = np.argsort(t_all, kind="stable")
    return t_all[order], s_all[order]


def nearest_speed(table_t, table_s, t):
    """Speed at the label sample nearest in time to each t (vectorized)."""
    if len(table_t) == 1:
        return np.full(len(t), table_s[0])
    j = np.clip(np.searchsorted(table_t, t), 1, len(table_t) - 1)
    pick_right = np.abs(table_t[j] - t) < np.abs(t - table_t[j - 1])
    return np.where(pick_right, table_s[j], table_s[j - 1])


def load_dumps(drift_out):
    """Read every preds__<method>__<fold>.npz; skip unreadable ones with a log
    line. Each fold is returned TIME-ORDERED (sorted by wall_time_s) with its
    raw per-frame error attached. Returns ({method: [fold dicts]}, skipped)."""
    files = sorted(glob.glob(os.path.join(drift_out, "preds__*.npz")))
    folds, skipped = {}, []
    for p in files:
        stem = os.path.basename(p)[len("preds__"):-len(".npz")]
        if "__" not in stem:
            print(f"  skip (unparseable name): {p}", flush=True)
            skipped.append(os.path.basename(p))
            continue
        meth, fold = stem.split("__", 1)
        try:
            with np.load(p) as z:
                d = {k: np.asarray(z[k]) for k in NPZ_KEYS}
        except Exception as e:
            print(f"  skip (unreadable: {e}): {p}", flush=True)
            skipped.append(os.path.basename(p))
            continue
        order = np.argsort(d["wall_time_s"], kind="stable")  # time-order within fold
        d = {k: v[order] for k, v in d.items()}
        d["fold"] = fold
        d["err"] = np.linalg.norm(d["y_xy"] - d["pred_xy"], axis=1)
        folds.setdefault(meth, []).append(d)
    return folds, skipped


def pooled(folds_m, key="err"):
    return np.concatenate([f[key] for f in folds_m])


def med_p90(err):
    return {"median": float(np.median(err)), "p90": float(np.percentile(err, 90)),
            "n": int(len(err))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="+",
                    help="the SAME sessions the drift run used (label speed rebuild)")
    ap.add_argument("--drift-out", required=True,
                    help="ml_drift output dir containing the preds__*.npz dumps")
    ap.add_argument("--out", "-o", default=None,
                    help="output dir (default: --drift-out itself)")
    ap.add_argument("--seeds", type=int, default=2,
                    help="unused (no training here; kept for a uniform exp_* CLI)")
    ap.add_argument("--max-epochs", type=int, default=250,
                    help="unused (no training here; kept for a uniform exp_* CLI)")
    ap.add_argument("--smoke", action="store_true",
                    help="wiring check: every 5th camera frame for the speed track")
    args = ap.parse_args()
    out_dir = args.out or args.drift_out
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    folds, skipped = load_dumps(args.drift_out)
    if not folds:
        raise SystemExit(f"no readable preds__*.npz in {args.drift_out} "
                         "(run ml_drift with --dump-preds first)")
    methods = sorted(folds)
    n_files = sum(len(v) for v in folds.values())
    print(f"Loaded {n_files} fold dumps · {len(methods)} methods: {methods}", flush=True)

    print("\nRebuilding label speed from camera tracks:", flush=True)
    table_t, table_s = label_speed_table(args.sessions, smoke=args.smoke)

    # ── (a) error CDF per method, folds pooled ──
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    pooled_raw = {}
    for meth in methods:
        err = pooled(folds[meth])
        pooled_raw[meth] = med_p90(err)
        es = np.sort(err)
        ax.plot(es, np.arange(1, len(es) + 1) / len(es), lw=1.6,
                label=f"{meth} (med {np.median(err):.0f} cm)")
    ax.axhline(0.5, ls="--", color="grey", lw=0.5)
    ax.axhline(0.9, ls="--", color="grey", lw=0.5)
    ax.set_xlabel("zero-shot LOSO error (cm)"); ax.set_ylabel("CDF")
    ax.set_title("Error CDF per method (all LOSO folds pooled)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "cdf_methods.png"), dpi=120); plt.close()
    print("wrote cdf_methods.png", flush=True)

    # ── (b) per-cell error map for the baseline method (exp_extra EXP2 style) ──
    base_meth = M.file_slug(M.BASELINE_NAME)
    if base_meth not in folds:
        base_meth = methods[0]
        print(f"  baseline dump missing — error map falls back to '{base_meth}'", flush=True)
    bm = folds[base_meth]
    cell_df = pd.DataFrame({
        "gx": np.concatenate([f["grid_x_cm"] for f in bm]),
        "gy": np.concatenate([f["grid_y_cm"] for f in bm]),
        "err": pooled(bm),
    })
    cell_df["cell_id"] = (cell_df["gx"].astype(int).astype(str) + "_"
                          + cell_df["gy"].astype(int).astype(str))
    g = (cell_df.groupby("cell_id")
         .agg(gx=("gx", "first"), gy=("gy", "first"),
              med=("err", "median"), n=("err", "size")).reset_index())
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    sct = ax.scatter(g.gx, g.gy, c=g.med, s=520, marker="s", cmap="RdYlGn_r",
                     vmin=float(np.percentile(g.med, 5)), vmax=float(np.percentile(g.med, 95)),
                     edgecolors="k", linewidths=0.5)
    for _, r in g.iterrows():
        ax.text(r.gx, r.gy, f"{r.med:.0f}", ha="center", va="center", fontsize=6.5)
    plt.colorbar(sct, label="zero-shot LOSO median error (cm)")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_title(f"Per-cell zero-shot error, all folds pooled ({base_meth})")
    ax.set_aspect("equal"); ax.grid(alpha=0.2); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_map_6session.png"), dpi=120); plt.close()
    g.to_csv(os.path.join(out_dir, "error_map_6session.csv"), index=False)
    print(f"wrote error_map_6session.png + .csv  ({len(g)} cells, min {g.med.min():.0f} / "
          f"median {g.med.median():.0f} / max {g.med.max():.0f} cm)", flush=True)

    # ── (c) median error vs label walking speed, per method ──
    speed_tab = {}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for meth in methods:
        sp = nearest_speed(table_t, table_s,
                           np.concatenate([f["wall_time_s"] for f in folds[meth]]))
        err = pooled(folds[meth])
        bins = np.digitize(sp, SPEED_EDGES)                 # 0..len(SPEED_LABELS)-1
        meds = [float(np.median(err[bins == b])) if (bins == b).any() else float("nan")
                for b in range(len(SPEED_LABELS))]
        ns = [int((bins == b).sum()) for b in range(len(SPEED_LABELS))]
        speed_tab[meth] = {lab: {"median": m, "n": n}
                           for lab, m, n in zip(SPEED_LABELS, meds, ns)}
        ax.plot(range(len(SPEED_LABELS)), meds, "o-", lw=1.6, label=meth)
    ax.set_xticks(range(len(SPEED_LABELS))); ax.set_xticklabels(SPEED_LABELS)
    ax.set_xlabel("label walking speed (cm/s)"); ax.set_ylabel("median error (cm)")
    ax.set_title("Zero-shot error vs walking speed (speed from camera labels)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "error_vs_speed.png"), dpi=120); plt.close()
    print("wrote error_vs_speed.png", flush=True)

    # ── (d) smoothing replay: raw vs causal rolling(w=9) vs Kalman, per method ──
    smooth_tab = {}
    for meth in methods:
        raw_e, roll_e, kal_e = [], [], []
        for f in folds[meth]:
            if len(f["err"]) < 2:
                print(f"  skip smoothing on tiny fold {meth}/{f['fold']}", flush=True)
                continue
            dt = float(np.median(np.diff(f["wall_time_s"])))
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0 / 30.0                              # degenerate timestamps
            raw_e.append(f["err"])
            roll_e.append(np.linalg.norm(
                f["y_xy"] - rolling_mean_xy(f["pred_xy"], ROLL_W), axis=1))
            kal_e.append(np.linalg.norm(
                f["y_xy"] - kalman_smooth(f["pred_xy"], dt), axis=1))
        if not raw_e:
            continue
        smooth_tab[meth] = {
            "raw": med_p90(np.concatenate(raw_e)),
            f"rolling_w{ROLL_W}": med_p90(np.concatenate(roll_e)),
            "kalman": med_p90(np.concatenate(kal_e)),
        }
    with open(os.path.join(out_dir, "smoothing_compare.json"), "w") as f:
        json.dump(smooth_tab, f, indent=2)
    ms = [m for m in methods if m in smooth_tab]
    x = np.arange(len(ms))
    fig, ax = plt.subplots(figsize=(max(8.5, 1.1 * len(ms) + 3), 5))
    for off, (key, lab, col) in zip((-0.25, 0.0, 0.25), [
            ("raw", "raw", "tab:blue"),
            (f"rolling_w{ROLL_W}", f"causal rolling w={ROLL_W}", "tab:orange"),
            ("kalman", "Kalman (CV)", "tab:green")]):
        vals = [smooth_tab[m][key]["median"] for m in ms]
        ax.bar(x + off, vals, 0.25, label=lab, color=col)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.5, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(ms, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("pooled median error (cm)")
    ax.set_title("Prediction smoothing replay on the LOSO dumps (causal, per fold)")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "smoothing_compare.png"), dpi=120); plt.close()
    print("wrote smoothing_compare.json + .png", flush=True)

    # ── (e) one card with every number ──
    card = {
        "kind": "drift-dump figures (no training)",
        "drift_out": args.drift_out.rstrip("/"),
        "sessions": [s.rstrip("/") for s in args.sessions],
        "methods": methods, "n_fold_dumps": n_files, "skipped_files": skipped,
        "smoke": bool(args.smoke),
        "speed_bins_cm_s": list(SPEED_LABELS), "rolling_window": ROLL_W,
        "pooled_raw_error": pooled_raw,
        "error_map_method": base_meth,
        "error_map_cells": {"n_cells": int(len(g)), "min_cm": float(g.med.min()),
                            "median_cm": float(g.med.median()), "max_cm": float(g.med.max())},
        "error_vs_speed": speed_tab,
        "smoothing_compare": smooth_tab,
        "runtime_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(out_dir, "figures_card.json"), "w") as f:
        json.dump(card, f, indent=2, default=str)
    print(f"\nwrote {out_dir}/figures_card.json ({card['runtime_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
