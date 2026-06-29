#!/usr/bin/env python3
"""Deep analysis of an overnight empty-room CSI capture (memory-safe, chunked).
Drift curve (per-board shape cosine vs t0) + RSSI + agitation + packet-rate/gaps
over elapsed hours, periodicity (FFT) and the drift autocorrelation timescale."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SESS = sys.argv[1] if len(sys.argv) > 1 else "sessions/12h_drift_overnight"
BIN = 120.0  # seconds per bin
csi_path = os.path.join(SESS, "csi.csv")
clap = pd.read_csv(os.path.join(SESS, "clap.csv"))
t0 = clap["wall_time_s"].iloc[0]; t_end = clap["wall_time_s"].iloc[-1]
dur_h = (t_end - t0) / 3600.0
CSICOLS = [f"csi_{i}" for i in range(128)]
USE = ["wall_time_s", "board_id", "rssi", "rx_seq"] + CSICOLS

# accumulators keyed by (board, bin)
acc = {}   # (b,bin) -> [count, sum_amp(64), sum_rssi, sum_absdiff, ndiff]
boards = set()
last_wall, last_amp, last_seq = {}, {}, {}
seq_gaps, big_time_gaps, max_time_gap = {}, {}, {}
n_rows = 0

for chunk in pd.read_csv(csi_path, usecols=USE, chunksize=300_000):
    n_rows += len(chunk)
    bt = chunk["board_id"].to_numpy()
    wt = chunk["wall_time_s"].to_numpy(np.float64)
    rs = chunk["rssi"].to_numpy(np.float64)
    sq = chunk["rx_seq"].to_numpy(np.int64)
    iq = chunk[CSICOLS].to_numpy(np.float32).reshape(-1, 64, 2)
    amp = np.sqrt((iq ** 2).sum(-1))                 # (n,64)
    binix = ((wt - t0) // BIN).astype(int)
    for b in np.unique(bt):
        boards.add(int(b))
        m = bt == b
        wb, ab, rb, sb, bib = wt[m], amp[m], rs[m], sq[m], binix[m]
        order = np.argsort(wb, kind="stable")
        wb, ab, rb, sb, bib = wb[order], ab[order], rb[order], sb[order], bib[order]
        # bridge from previous chunk
        if b in last_wall:
            wb = np.concatenate([[last_wall[b]], wb])
            ab = np.vstack([last_amp[b], ab])
            sb = np.concatenate([[last_seq[b]], sb])
            bib = np.concatenate([[bib[0]], bib])
            bridged = True
        else:
            bridged = False
        dwt = np.diff(wb)
        mg = max_time_gap.get(b, 0.0); max_time_gap[b] = max(mg, float(dwt.max()) if len(dwt) else 0.0)
        big_time_gaps[b] = big_time_gaps.get(b, 0) + int((dwt > 2.0).sum())
        dseq = (np.diff(sb) - 1) % 65536
        seq_gaps[b] = seq_gaps.get(b, 0) + int(dseq[(dseq > 0) & (dseq < 1000)].sum())
        adiff = np.abs(np.diff(ab, axis=0)).mean(1)   # per-frame mean |Δamp|
        for j in range(1, len(wb)):
            key = (b, int(bib[j]))
            a = acc.get(key)
            if a is None:
                a = [0, np.zeros(64), 0.0, 0.0, 0]; acc[key] = a
            a[0] += 1; a[1] += ab[j]; a[3] += adiff[j-1]; a[4] += 1
        # rssi: include all rows (use the post-sort rb, minus the bridged dummy)
        rb_real = rb
        bib_real = bib[1:] if bridged else bib
        for j in range(len(rb_real)):
            key = (b, int(bib_real[j]))
            if key in acc: acc[key][2] += rb_real[j]
        last_wall[b], last_amp[b], last_seq[b] = wb[-1], ab[-1], sb[-1]

boards = sorted(boards)
nb = max((bn for (_, bn) in acc), default=0) + 1
hrs = (np.arange(nb) * BIN) / 3600.0

# per-board series
cos_s, rssi_s, agit_s, rate_s, level_s = {}, {}, {}, {}, {}
def cos(a, b):
    na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b)/(na*nb_)) if na>0 and nb_>0 else np.nan
for b in boards:
    means = {bn: acc[(b,bn)][1]/acc[(b,bn)][0] for bn in range(nb) if (b,bn) in acc and acc[(b,bn)][0]>0}
    ref = means[0] if 0 in means else means[min(means)]
    cos_s[b]  = np.array([cos(means[bn], ref) if bn in means else np.nan for bn in range(nb)])
    level_s[b]= np.array([float(np.linalg.norm(means[bn])) if bn in means else np.nan for bn in range(nb)])
    rssi_s[b] = np.array([acc[(b,bn)][2]/acc[(b,bn)][0] if (b,bn) in acc and acc[(b,bn)][0]>0 else np.nan for bn in range(nb)])
    agit_s[b] = np.array([100*acc[(b,bn)][3]/acc[(b,bn)][4]/level_s[b][bn]*np.sqrt(64) if (b,bn) in acc and acc[(b,bn)][4]>0 and level_s[b][bn]>0 else np.nan for bn in range(nb)])
    rate_s[b] = np.array([acc[(b,bn)][0]/BIN if (b,bn) in acc else 0.0 for bn in range(nb)])

# periodicity + autocorr on the level series (detrended), per board
def autocorr_tau(x):
    x = x[~np.isnan(x)];
    if len(x) < 10: return None
    x = x - x.mean();
    ac = np.correlate(x, x, "full")[len(x)-1:]; ac /= ac[0]
    below = np.where(ac < 1/np.e)[0]
    return float(below[0]*BIN/60.0) if len(below) else None   # minutes
def dom_period(x):
    x = x[~np.isnan(x)]
    if len(x) < 20: return None
    x = x - x.mean(); f = np.fft.rfftfreq(len(x), BIN); P = np.abs(np.fft.rfft(x))**2
    f, P = f[1:], P[1:]
    if not len(P): return None
    pk = int(np.argmax(P));
    return float(1/f[pk]/60.0) if f[pk]>0 else None            # minutes

summary = {"session": SESS, "duration_h": round(dur_h,2), "n_csi_rows": int(n_rows),
           "bin_s": BIN, "n_bins": int(nb), "boards": boards, "per_board": {}}
for b in boards:
    c = cos_s[b][~np.isnan(cos_s[b])]
    summary["per_board"][f"b{b}"] = {
        "cos_t0_end": round(float(c[-1]),5) if len(c) else None,
        "cos_min": round(float(np.nanmin(cos_s[b])),5),
        "level_drift_pct": round(float((np.nanmax(level_s[b])-np.nanmin(level_s[b]))/np.nanmedian(level_s[b])*100),2),
        "rssi_range_db": round(float(np.nanmax(rssi_s[b])-np.nanmin(rssi_s[b])),1),
        "rssi_med": round(float(np.nanmedian(rssi_s[b])),1),
        "agit_med_pct": round(float(np.nanmedian(agit_s[b])),2),
        "mean_rate_hz": round(float(np.nanmean(rate_s[b])),1),
        "max_time_gap_s": round(max_time_gap.get(b,0),1),
        "n_time_gaps_gt2s": big_time_gaps.get(b,0),
        "lost_pkts_seqgap": seq_gaps.get(b,0),
        "drift_tau_min": (round(autocorr_tau(level_s[b]),0) if autocorr_tau(level_s[b]) else None),
        "dom_period_min": (round(dom_period(level_s[b]),0) if dom_period(level_s[b]) else None),
    }

fig, ax = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
col = {boards[0]:"#1f77b4", boards[1]:"#d62728", boards[2]:"#2ca02c"} if len(boards)>=3 else {}
for b in boards:
    c = col.get(b);
    ax[0].plot(hrs, cos_s[b], label=f"b{b}", color=c)
    ax[1].plot(hrs, rssi_s[b], label=f"b{b}", color=c)
    ax[2].plot(hrs, agit_s[b], label=f"b{b}", color=c)
    ax[3].plot(hrs, rate_s[b], label=f"b{b}", color=c)
ax[0].set_ylabel("amp-shape cosine\nvs t=0"); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title(f"Overnight empty-room drift — {dur_h:.1f} h, {n_rows:,} packets")
ax[1].set_ylabel("RSSI (dBm)"); ax[1].grid(alpha=.3)
ax[2].set_ylabel("agitation\n(% frame-to-frame)"); ax[2].grid(alpha=.3)
ax[3].set_ylabel("packet rate (Hz)"); ax[3].set_xlabel("elapsed hours"); ax[3].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(SESS,"overnight_drift.png"), dpi=110); plt.close()
json.dump(summary, open(os.path.join(SESS,"overnight_drift.json"),"w"), indent=2)
print(json.dumps(summary, indent=2))
print("\nwrote overnight_drift.png + overnight_drift.json")
