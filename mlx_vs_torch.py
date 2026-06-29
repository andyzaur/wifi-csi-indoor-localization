#!/usr/bin/env python3
"""Cross-framework reproducibility + throughput: TorchLocalizer (MPS) vs
MLXLocalizer (MLX) on identical data/splits/config. Confirms MLX reproduces the
torch result within seed noise, and benchmarks fit wall-time. NOT in the result
lineage — an engineering/reproducibility experiment.

    python3 mlx_vs_torch.py sessions/A sessions/B ... --out DIR --seeds 3
"""
import os, sys, json, time, argparse, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from dataset import build_multiboard_dataset, grid_cell_id, load_session, multiboard_feature_matrix
from train_final import per_cell_temporal_split, smooth_regression
from torch_net import TorchLocalizer
from mlx_net import MLXLocalizer
import ml_drift as M

ap = argparse.ArgumentParser()
ap.add_argument("sessions", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--max-epochs", type=int, default=200)
ap.add_argument("--phase-v2", action="store_true", help="use masked phase_v2 + drop nulls")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

dfs = []
for s in args.sessions:
    csi, cam, clap = load_session(s.rstrip("/"))
    d = build_multiboard_dataset(csi, cam, clap, phase_mode=("masked" if args.phase_v2 else "legacy"))
    d["session"] = os.path.basename(s.rstrip("/"))
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True).sort_values(["session", "wall_time_s"]).reset_index(drop=True)
df["cell_id"] = grid_cell_id(df)
enc = LabelEncoder().fit(df["cell_id"]); df["cls"] = enc.transform(df["cell_id"])
n_cls = len(enc.classes_)
X, _ = multiboard_feature_matrix(df, include_phase=args.phase_v2, include_rssi=True,
                                drop_null_subcarriers=args.phase_v2)
X = X.astype(np.float32)
y = df[["x_cm", "y_cm"]].to_numpy(np.float64); c = df["cls"].to_numpy()
tr, te = per_cell_temporal_split(df, 0.8); itr, iva = per_cell_temporal_split(tr, 0.8)
iTR, iVA, iTE = itr.index.to_numpy(), iva.index.to_numpy(), te.index.to_numpy()
sc = StandardScaler().fit(X[iTR])
Xtr, Xva, Xte = sc.transform(X[iTR]), sc.transform(X[iVA]), sc.transform(X[iTE])
order = np.argsort(df.loc[te.index, "wall_time_s"].to_numpy(), kind="stable")
print(f"Loaded {len(df):,} samples · {n_cls} cells · train {len(iTR):,} / test {len(iTE):,} · "
      f"{X.shape[1]} feats ({'phase_v2' if args.phase_v2 else 'amp+rssi'})", flush=True)

def med(pred, sm=False):
    p = smooth_regression(pred[order], 9) if sm else pred
    yy = y[iTE][order] if sm else y[iTE]
    return float(np.median(np.linalg.norm(yy - p, axis=1)))

BB = dict(M.BACKBONE)
res = {"torch": {"raw": [], "smoothed": [], "fit_s": []},
       "mlx": {"raw": [], "smoothed": [], "fit_s": []}}
for s in range(args.seeds):
    for name, Cls in (("torch", TorchLocalizer), ("mlx", MLXLocalizer)):
        t0 = time.time()
        m = Cls(n_cls=n_cls, seed=s, batch_size=2048, max_epochs=args.max_epochs, patience=30, **BB)
        m.fit(Xtr, y[iTR], c[iTR], X_val=Xva, y_val_xy=y[iVA])
        dt = time.time() - t0
        pred = m.predict(Xte)
        res[name]["raw"].append(med(pred)); res[name]["smoothed"].append(med(pred, True))
        res[name]["fit_s"].append(dt)
        print(f"  seed {s} {name:5s}: raw {res[name]['raw'][-1]:5.1f}  "
              f"smoothed {res[name]['smoothed'][-1]:5.1f}  fit {dt:5.1f}s", flush=True)

def stat(a): return {"mean": round(float(np.mean(a)), 2), "std": round(float(np.std(a)), 2)}
summary = {f: {k: stat(v) for k, v in res[f].items()} for f in res}
summary["throughput_torch_per_mlx"] = round(np.mean(res["torch"]["fit_s"]) / np.mean(res["mlx"]["fit_s"]), 2)
summary["repro_gap_cm"] = round(abs(np.mean(res["torch"]["smoothed"]) - np.mean(res["mlx"]["smoothed"])), 2)
summary["meta"] = {"sessions": [s.rstrip("/") for s in args.sessions], "n": len(df),
                   "feats": int(X.shape[1]), "seeds": args.seeds,
                   "torch_seed_std": stat(res["torch"]["smoothed"])["std"],
                   "mlx_seed_std": stat(res["mlx"]["smoothed"])["std"]}
json.dump(summary, open(os.path.join(args.out, "mlx_vs_torch.json"), "w"), indent=2)
print("\n=== SUMMARY ===")
print(f"  torch smoothed {summary['torch']['smoothed']['mean']} ±{summary['torch']['smoothed']['std']} cm  "
      f"(fit {summary['torch']['fit_s']['mean']}s)")
print(f"  mlx   smoothed {summary['mlx']['smoothed']['mean']} ±{summary['mlx']['smoothed']['std']} cm  "
      f"(fit {summary['mlx']['fit_s']['mean']}s)")
print(f"  reproduction gap {summary['repro_gap_cm']} cm (vs seed std ~{summary['meta']['torch_seed_std']}); "
      f"throughput torch/mlx ×{summary['throughput_torch_per_mlx']}")
print(f"\nwrote {args.out}/mlx_vs_torch.json")
