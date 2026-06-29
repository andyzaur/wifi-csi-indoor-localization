#!/usr/bin/env python3
"""repro_v3.py — reproducibility & rigor report for the train_v3 honest result.

train_v3 runs a stochastic, timeout-bounded search and writes model_card_v3.json.
That single number ("33.3 cm") is not yet defensible on its own: we don't know its
seed variance, we have no confidence interval, and we haven't shown a tuned net
beats trivial baselines on the SAME honest split. This script closes those gaps
WITHOUT re-running the search — it reads the frozen choices from the fresh card
and re-evaluates them, so it also serves as an artifact<->code consistency check.

It reproduces train_v3's nested per-cell-temporal protocol EXACTLY (same
deterministic splits, same feature config, same outer scaler) and reports:

  1. seed stability   — refit the chosen torch arch with N seeds on tr_outer,
                        score the untouched test fold each time -> mean +/- std
  2. bootstrap CI     — 95% CI on the ensemble's smoothed test median (resample
                        the test rows; refit from card params, blend card weights)
  3. baseline ladder  — constant / kNN / Ridge / lgbm-default on the same split,
                        so the tuned result is contextualised, not free-floating
  4. label-shuffle    — shuffle train labels, refit lgbm-default: must collapse to
                        ~the room half-spread (proves the result is signal, not leak)
  5. consistency      — recomputed ensemble median vs the card's stored median

Usage:
    python3 repro_v3.py sessions/20260603_01_real30hztest          # 5 seeds, B=2000
    python3 repro_v3.py sessions/<s> --seeds 5 --bootstrap 2000
    python3 repro_v3.py sessions/<s> --quick                       # 2 seeds, B=200
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# ── pure helpers (unit-tested in tests/test_repro_v3.py) ─────────────────────

def bootstrap_ci(errors, B=2000, ci=0.95, seed=0, stat=np.median):
    """Percentile bootstrap CI for a statistic of a per-sample error vector.

    errors: 1-D array of per-sample localization errors (cm). Resamples rows with
    replacement B times, recomputes `stat` each time, returns (point, lo, hi)
    where point = stat(errors) and (lo, hi) is the central `ci` interval.
    """
    errors = np.asarray(errors, dtype=float)
    n = len(errors)
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for b in range(B):
        boot[b] = stat(errors[rng.integers(0, n, n)])
    alpha = (1.0 - ci) / 2.0
    return float(stat(errors)), float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def per_sample_error(y, pred):
    """Euclidean per-row error (cm)."""
    return np.linalg.norm(np.asarray(y) - np.asarray(pred), axis=1)


def room_half_spread(y):
    """Expected error of a label-agnostic guess: distance from the centroid.

    A model that has learned nothing useful (e.g. trained on shuffled labels)
    can do no better in median than predicting a constant; its error floor is the
    median distance of true positions from their centroid. The label-shuffle
    control should land near this value.
    """
    y = np.asarray(y, dtype=float)
    centroid = y.mean(axis=0)
    return float(np.median(np.linalg.norm(y - centroid, axis=1)))


# ── report ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.bootstrap = 2, 200

    # import train_v3 FIRST: it imports lightgbm/xgboost before torch (OpenMP
    # load order) and gives us the exact feature/split pipeline the card was made with.
    import train_v3 as tv
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from dataset import (build_multiboard_dataset, grid_cell_id, load_session,
                         boards_in_multiboard_df)
    from train_final import per_cell_temporal_split, smooth_regression
    from torch_net import TorchLocalizer

    out_dir = args.sessions[0].rstrip("/")
    card_path = os.path.join(out_dir, "model_card_v3.json")
    with open(card_path) as f:
        card = json.load(f)
    print(f"Read {card_path}  (git_commit {card.get('git_commit','?')[:9]}, "
          f"families {list(card.get('per_family_inner_val_cm', {}).keys())})")
    if not card.get("model_params"):
        raise SystemExit("Card has empty model_params — rerun train_v3 first (stale card).")

    # ── rebuild df EXACTLY as train_v3 does ──
    dfs = []
    for s in args.sessions:
        csi, cam, clap = load_session(s.rstrip("/"))
        d = build_multiboard_dataset(csi, cam, clap)
        d["session"] = os.path.basename(s.rstrip("/"))
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).sort_values(["session", "wall_time_s"]).reset_index(drop=True)
    df["cell_id"] = grid_cell_id(df)
    enc = LabelEncoder().fit(df["cell_id"])
    df["cls"] = enc.transform(df["cell_id"])
    n_cls = len(enc.classes_)

    cfg = card["feature_params"]
    X = tv.build_features(df, cfg)
    y = df[["x_cm", "y_cm"]].to_numpy(np.float64)
    c = df["cls"].to_numpy()

    tr_outer, te_outer = per_cell_temporal_split(df, 0.8)
    inner_tr, inner_val = per_cell_temporal_split(tr_outer, 0.8)
    iTR, iVA = inner_tr.index.to_numpy(), inner_val.index.to_numpy()
    trO, teO = tr_outer.index.to_numpy(), te_outer.index.to_numpy()
    print(f"Protocol reproduced: outer train {len(trO):,} / TEST {len(teO):,} | "
          f"feature config '{card['feature_config']}' ({X.shape[1]} feats)")

    # outer scaler (fit on tr_outer ONLY), test stream time-ordered for causal smoothing
    sc = StandardScaler().fit(X[trO])
    Xtr, Xte, Xval = sc.transform(X[trO]), sc.transform(X[teO]), sc.transform(X[iVA])
    ytr, yte = y[trO], y[teO]
    ctr = c[trO]
    order = np.argsort(df.loc[te_outer.index, "wall_time_s"].to_numpy(), kind="stable")
    swin = int(card.get("smoothing_window_samples", 1))

    def test_median(pred, smoothed=True):
        if smoothed and swin > 1:
            p = smooth_regression(pred[order], swin)
            return float(np.median(per_sample_error(yte[order], p)))
        return float(np.median(per_sample_error(yte, pred)))

    report = {"session": out_dir, "card_git_commit": card.get("git_commit"),
              "smoothing_window_samples": swin, "n_test": int(len(teO))}

    # ── 1. SEED STABILITY (the chosen torch arch refit on tr_outer, N seeds) ──
    # Keep every seed's test prediction: their MEAN is the card's headline (TORCH-ENS).
    print(f"\n── 1. SEED STABILITY — torch refit x{args.seeds} on tr_outer (test median, cm) ──")
    tp = card["model_params"]["torch"]
    seed_raw, seed_sm, seed_preds = [], [], []
    for s in range(args.seeds):
        m = TorchLocalizer(n_cls=n_cls, seed=s, batch_size=2048, max_epochs=300,
                           patience=30, **tp).fit(Xtr, ytr, ctr, X_val=Xval, y_val_xy=y[iVA])
        pred = m.predict(Xte)
        seed_preds.append(pred)
        r, sm = test_median(pred, False), test_median(pred, True)
        seed_raw.append(r); seed_sm.append(sm)
        print(f"  seed {s}: raw {r:5.1f}  smoothed {sm:5.1f}", flush=True)
    report["seed_stability"] = {
        "seeds": args.seeds,
        "raw_median_cm": {"mean": float(np.mean(seed_raw)), "std": float(np.std(seed_raw)),
                          "values": [round(v, 2) for v in seed_raw]},
        "smoothed_median_cm": {"mean": float(np.mean(seed_sm)), "std": float(np.std(seed_sm)),
                               "values": [round(v, 2) for v in seed_sm]},
    }
    print(f"  -> single-seed smoothed {np.mean(seed_sm):.1f} +/- {np.std(seed_sm):.1f} cm  "
          f"(raw {np.mean(seed_raw):.1f} +/- {np.std(seed_raw):.1f})")

    # ── 2. HEADLINE = TORCH-ENS (mean of the seed refits) + bootstrap 95% CI ──
    print(f"\n── 2. HEADLINE TORCH-ENS (mean of {args.seeds} seeds) + bootstrap 95% CI ──")
    tens = np.mean(seed_preds, axis=0)
    tens_raw, tens_sm = test_median(tens, False), test_median(tens, True)
    err_sm = per_sample_error(yte[order], smooth_regression(tens[order], swin) if swin > 1 else tens[order])
    pt, lo, hi = bootstrap_ci(err_sm, B=args.bootstrap, ci=0.95, seed=0)
    report["headline_torch_ens"] = {
        "n_seeds": args.seeds, "raw_median_cm": round(tens_raw, 2),
        "smoothed_median_cm": round(tens_sm, 2),
        "smoothed_median_95ci_cm": [round(lo, 2), round(hi, 2)],
    }
    print(f"  TORCH-ENS raw {tens_raw:.1f}  smoothed {tens_sm:.1f}  95% CI [{lo:.1f}, {hi:.1f}] cm")

    # ── 2b. convex-blend ENSEMBLE (card weights) — completeness + consistency ──
    # Reuse seed-0 torch for the torch slot (matches train_v3's single final torch).
    fam_order = card.get("ensemble_weights", {})
    fams = list(fam_order.keys())
    w = np.array([fam_order[k] for k in fams], dtype=float)
    fam_pred = {}
    for kind in fams:
        fam_pred[kind] = (seed_preds[0] if kind == "torch"
                          else tv.make_regressor(kind, card["model_params"][kind]).fit(Xtr, ytr).predict(Xte))
    blend = np.tensordot(w, np.stack([fam_pred[k] for k in fams]), axes=(0, 0))
    blend_raw, blend_sm = test_median(blend, False), test_median(blend, True)
    report["convex_ensemble"] = {"weights": {k: float(fam_order[k]) for k in fams},
                                 "raw_median_cm": round(blend_raw, 2),
                                 "smoothed_median_cm": round(blend_sm, 2)}
    print(f"  convex blend({', '.join(f'{k}={fam_order[k]:.2f}' for k in fams)}) "
          f"raw {blend_raw:.1f} smoothed {blend_sm:.1f} cm "
          f"(inner-val weights under-transferred — headline stays TORCH-ENS)")

    # ── 3. BASELINE LADDER (same split, raw median) ──
    print(f"\n── 3. BASELINE LADDER (fit tr_outer, score test; raw median, cm) ──")
    baselines = {}
    const_pred = np.repeat(ytr.mean(0, keepdims=True), len(yte), axis=0)
    baselines["constant (train mean)"] = test_median(const_pred, False)
    baselines["kNN (k=10)"] = test_median(
        KNeighborsRegressor(n_neighbors=10).fit(Xtr, ytr).predict(Xte), False)
    baselines["Ridge"] = test_median(Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte), False)
    baselines["lgbm (default)"] = test_median(
        tv.make_regressor("lgbm", dict(n_estimators=400, learning_rate=0.05,
                                       num_leaves=63)).fit(Xtr, ytr).predict(Xte), False)
    baselines["TUNED torch-ens (raw)"] = tens_raw
    report["baseline_ladder_raw_median_cm"] = {k: round(v, 2) for k, v in baselines.items()}
    for k, v in baselines.items():
        print(f"  {k:24s} {v:6.1f}")

    # ── 4. LABEL-SHUFFLE CONTROL (must collapse to ~room half-spread) ──
    print(f"\n── 4. LABEL-SHUFFLE NEGATIVE CONTROL ──")
    floor = room_half_spread(yte)
    rng = np.random.default_rng(0)
    ytr_shuf = ytr[rng.permutation(len(ytr))]
    shuf_pred = tv.make_regressor("lgbm", dict(n_estimators=400, learning_rate=0.05,
                                               num_leaves=63)).fit(Xtr, ytr_shuf).predict(Xte)
    shuf_med = test_median(shuf_pred, False)
    report["label_shuffle_control"] = {
        "shuffled_test_median_cm": round(shuf_med, 2),
        "room_half_spread_cm": round(floor, 2),
        "passes": bool(shuf_med > 2.5 * tens_raw),
    }
    print(f"  shuffled-label lgbm test median {shuf_med:.1f} cm  "
          f"(room half-spread {floor:.1f}; real torch-ens {tens_raw:.1f})")
    print(f"  -> {'PASS' if shuf_med > 2.5 * tens_raw else 'CHECK'}: shuffling destroys the signal")

    # ── 5. ARTIFACT<->CODE CONSISTENCY (recomputed vs card-stored) ──
    fm = card.get("final_test_metrics_cm", {})
    cons = {}
    print(f"\n── 5. CONSISTENCY (recomputed vs card-stored smoothed median) ──")
    for label, recomputed in [("TORCH-ENS", tens_sm), ("ENSEMBLE", blend_sm)]:
        stored_sm = fm.get(label, {}).get("smoothed", {}).get("median")
        if stored_sm is not None:
            cons[label] = {"recomputed_cm": round(recomputed, 2),
                           "card_cm": round(stored_sm, 2), "delta_cm": round(recomputed - stored_sm, 2)}
            print(f"  {label:10s} recomputed {recomputed:5.1f}  card {stored_sm:5.1f}  "
                  f"(delta {recomputed - stored_sm:+.1f} cm)")
    report["consistency"] = cons

    out_path = os.path.join(out_dir, "repro_report_v3.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")
    print("\n" + "=" * 70)
    print("CITABLE (honest, with uncertainty):")
    print(f"  median localization error {tens_sm:.1f} cm "
          f"[95% CI {lo:.1f}-{hi:.1f}], single-seed spread +/-{np.std(seed_sm):.1f} cm")
    print(f"  vs best trivial baseline {min(v for k, v in baselines.items() if 'TUNED' not in k):.1f} cm; "
          f"label-shuffle floor {shuf_med:.1f} cm")
    print("=" * 70)


if __name__ == "__main__":
    main()
