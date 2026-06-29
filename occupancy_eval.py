#!/usr/bin/env python3
"""Cross-occupancy generalization: train on SOLO sessions only, test on each
occupancy session (a seated bystander present). The standard LOSO can't isolate
this — when an occupancy fold is held out it still trains on the OTHER occupancy
sessions. Here the training pool is strictly solo, so the number answers
"trained on empty-room walks, can it localize a person with a bystander present?"

Honest protocol mirrors ml_drift: scaler+model fit on the solo pool only,
occupancy session scored once, seed-averaged, per the chosen method spec.

    python3 occupancy_eval.py --solo S1 S2 ... --occ O1 O2 O3 --out DIR
"""
import os, sys, json, time, argparse, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from dataset import build_multiboard_dataset, grid_cell_id, load_session
from train_final import per_cell_temporal_split
import ml_drift as M

ap = argparse.ArgumentParser()
ap.add_argument("--solo", nargs="+", required=True)
ap.add_argument("--occ", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--max-epochs", type=int, default=250)
ap.add_argument("--methods", default="baseline,phasev2,persessstdphasev2")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
t_start = time.time()

sel = M.select_methods(M.ROSTER, args.methods)


def build(sessions, phase_mode):
    dfs = []
    for s in sessions:
        csi, cam, clap = load_session(s.rstrip("/"))
        d = build_multiboard_dataset(csi, cam, clap, phase_mode=phase_mode)
        d["session"] = os.path.basename(s.rstrip("/"))
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True).sort_values(["session", "wall_time_s"]).reset_index(drop=True)

all_sess = list(args.solo) + list(args.occ)
df = build(all_sess, "legacy")
df_v2 = build(all_sess, "masked")
assert len(df) == len(df_v2) and np.allclose(df["wall_time_s"], df_v2["wall_time_s"])
for d in (df, df_v2):
    d["cell_id"] = grid_cell_id(df)
enc = LabelEncoder().fit(df["cell_id"]); df["cls"] = enc.transform(df["cell_id"])
n_cls = len(enc.classes_)
y = df[["x_cm", "y_cm"]].to_numpy(np.float64); c = df["cls"].to_numpy()
solo_names = {os.path.basename(s.rstrip("/")) for s in args.solo}
occ_names = [os.path.basename(s.rstrip("/")) for s in args.occ]
is_solo = df["session"].isin(solo_names).to_numpy()
print(f"Loaded {len(df):,} samples · {n_cls} cells · {is_solo.sum():,} solo / {(~is_solo).sum():,} occupied", flush=True)

names_cache, baseline_vec = {}, {}
results = {}
for name, spec in sel:
    X = M.build_raw_features(df, spec, baseline_vec, names_cache, df_v2=df_v2)
    per_occ = {}
    for occ in occ_names:
        te = np.where((df["session"] == occ).to_numpy())[0]
        tr = np.where(is_solo)[0]                       # SOLO ONLY
        tr_df = df.iloc[tr].reset_index(drop=True)
        itr, iva = per_cell_temporal_split(tr_df, 0.85)
        i_tr, i_va = tr[itr.index.to_numpy()], tr[iva.index.to_numpy()]
        sc = StandardScaler().fit(X[i_tr])
        Xtr, Xva, Xte = sc.transform(X[i_tr]), sc.transform(X[i_va]), sc.transform(X[te])
        preds = M.seed_predictions(Xtr, y[i_tr], c[i_tr], Xva, y[i_va], Xte,
                                   n_cls, args.seeds, args.max_epochs)
        med = M.reg_metrics(y[te], np.mean(preds, axis=0))["median"]
        per_occ[occ] = round(med, 1)
        print(f"  {name:28s} test {occ:34s} {med:5.1f} cm", flush=True)
    per_occ["mean"] = round(float(np.mean([per_occ[o] for o in occ_names])), 1)
    results[name] = per_occ

json.dump({"kind": "cross-occupancy (train solo, test occupied)",
           "solo_sessions": list(solo_names), "occ_sessions": occ_names,
           "results_median_cm": results, "seeds": args.seeds,
           "runtime_s": round(time.time() - t_start, 1)},
          open(os.path.join(args.out, "occupancy_card.json"), "w"), indent=2)
print("\nwrote", os.path.join(args.out, "occupancy_card.json"))
