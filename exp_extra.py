#!/usr/bin/env python3
"""exp_extra.py — two follow-up experiments on the current 3 sessions, reusing the
ml_drift harness verbatim (same backbone, same honest protocol):

  EXP1  Does the calibration pass STACK with phase? The drift_full calibration
        curve used amp-only features. Re-run it with amp+phase and compare to the
        stored amp-only curve (same seeds/fracs → directly comparable).
  EXP2  Per-cell spatial error map: where in the room does the zero-shot cross-
        session (LOSO) baseline fail worst? -> error_map.png (a thesis figure +
        tells us where to place RX / collect more).

Writes calib_stack.json + calib_stack.png + error_map.png into sessions/drift_full/.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dataset import (load_session, build_multiboard_dataset, grid_cell_id,
                     multiboard_feature_matrix, multiboard_static_baseline)
from train_final import per_cell_temporal_split
import ml_drift as M

SESS = ['sessions/20260603_0027_real30hztest', 'sessions/20260603_2321_ActualSession1',
        'sessions/20260604_0025_RealSession2']
EMPTY = 'sessions/20260603_2047_empty_camera_1'
OUT = 'sessions/drift_full'
SEEDS, MAXEP = 5, 250
FRACS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)

# ── load exactly as ml_drift does ──
dfs = []
for s in SESS:
    csi, cam, clap = load_session(s)
    d = build_multiboard_dataset(csi, cam, clap)
    d['session'] = os.path.basename(s)
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True).sort_values(['session', 'wall_time_s']).reset_index(drop=True)
df['cell_id'] = grid_cell_id(df)
enc = LabelEncoder().fit(df['cell_id'])
df['cls'] = enc.transform(df['cell_id'])
n_cls = len(enc.classes_)
y = df[['x_cm', 'y_cm']].to_numpy(float)
c = df['cls'].to_numpy()
sessions = list(dict.fromkeys(df['session'].tolist()))
empty_csi = pd.read_csv(os.path.join(EMPTY, 'csi.csv'))
baseline_vec = {}
for key, phase in [('amp', False), ('phase', True)]:
    _, names = multiboard_feature_matrix(df.head(2), include_phase=phase, include_rssi=True)
    baseline_vec[key] = multiboard_static_baseline(empty_csi, names)
names_cache = {}
print(f"Loaded {len(df):,} samples, {n_cls} cells, {len(sessions)} sessions", flush=True)


def mean_curve(cal, f):
    v = [cal[h][f] for h in cal if f in cal[h]]
    return float(np.mean(v)) if v else float('nan')


# ── EXP1: calibration stacked with phase vs the stored amp-only curve ──
print("\n=== EXP1: does calibration STACK with phase? ===", flush=True)
card = json.load(open(os.path.join(OUT, 'drift_card.json')))
cal_amp = {h: {float(k): v for k, v in card['calibration_sweep_median_cm'][h].items()}
           for h in card['calibration_sweep_median_cm']}           # amp-only (from drift_full)
cal_phase, _ = M.calibration_sweep(df, baseline_vec, names_cache, y, c, n_cls,
                                   SEEDS, MAXEP, fracs=FRACS, spec=dict(phase=True))
print(f"{'frac':>6} {'amp-only':>9} {'amp+phase':>10} {'Δ':>7}")
rows = []
for f in FRACS:
    a, p = mean_curve(cal_amp, f), mean_curve(cal_phase, f)
    rows.append((f, a, p))
    print(f"{f:>6} {a:>9.1f} {p:>10.1f} {p-a:>+7.1f}", flush=True)
json.dump({"amp_only": {str(f): mean_curve(cal_amp, f) for f in FRACS},
           "amp_phase": {str(f): mean_curve(cal_phase, f) for f in FRACS},
           "per_session_amp_phase": {h: {str(k): v for k, v in cal_phase[h].items()} for h in cal_phase}},
          open(os.path.join(OUT, 'calib_stack.json'), 'w'), indent=2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(FRACS, [mean_curve(cal_amp, f) for f in FRACS], 'o-', label='calibration (amp+rssi)')
ax.plot(FRACS, [mean_curve(cal_phase, f) for f in FRACS], 's--', label='calibration (amp+phase)')
ax.set_xlabel('fraction of each cell labelled today'); ax.set_ylabel('same-day test median (cm)')
ax.set_title('Does the calibration pass stack with phase?'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'calib_stack.png'), dpi=120); plt.close()
print("wrote calib_stack.json + calib_stack.png", flush=True)


# ── EXP2: per-cell zero-shot LOSO error map (baseline amp+rssi) ──
print("\n=== EXP2: per-cell zero-shot error map (baseline) ===", flush=True)
X = M.build_raw_features(df, dict(), baseline_vec, names_cache)
err = np.full(len(df), np.nan)
for held in sessions:
    te = np.where((df['session'] == held).to_numpy())[0]
    tr_idx = np.where((df['session'] != held).to_numpy())[0]
    tr_df = df.iloc[tr_idx].reset_index(drop=True)
    itr, iva = per_cell_temporal_split(tr_df, 0.85)
    i_tr, i_va = tr_idx[itr.index.to_numpy()], tr_idx[iva.index.to_numpy()]
    sc = StandardScaler().fit(X[i_tr])
    Xtr, Xva, Xte = sc.transform(X[i_tr]), sc.transform(X[i_va]), sc.transform(X[te])
    preds = M.seed_predictions(Xtr, y[i_tr], c[i_tr], Xva, y[i_va], Xte, n_cls, 3, MAXEP)
    err[te] = np.linalg.norm(y[te] - np.mean(preds, 0), axis=1)
    print(f"  fold hold {held}: median {np.nanmedian(err[te]):.1f} cm", flush=True)
df['err'] = err
g = (df.dropna(subset=['err']).groupby('cell_id')
       .agg(gx=('grid_x_cm', 'first'), gy=('grid_y_cm', 'first'),
            med=('err', 'median'), n=('err', 'size')).reset_index())

fig, ax = plt.subplots(figsize=(7.5, 7.5))
s = ax.scatter(g.gx, g.gy, c=g.med, s=520, marker='s', cmap='RdYlGn_r',
               vmin=40, vmax=140, edgecolors='k', linewidths=0.5)
for _, r in g.iterrows():
    ax.text(r.gx, r.gy, f"{r.med:.0f}", ha='center', va='center', fontsize=6.5)
plt.colorbar(s, label='zero-shot LOSO median error (cm)')
ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
ax.set_title('Per-cell zero-shot cross-session error (baseline amp+rssi)')
ax.set_aspect('equal'); ax.grid(alpha=0.2); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'error_map.png'), dpi=120); plt.close()
g.to_csv(os.path.join(OUT, 'error_map.csv'), index=False)
print(f"wrote error_map.png + error_map.csv  ({len(g)} cells, "
      f"min {g.med.min():.0f} / median {g.med.median():.0f} / max {g.med.max():.0f} cm)", flush=True)
print("\nDONE", flush=True)
