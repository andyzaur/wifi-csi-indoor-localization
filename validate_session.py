#!/usr/bin/env python3
"""Validate a recording session before trusting it for training.

Run this immediately after a recording. It catches the failure modes that have
actually bitten us: silent CSI capture (empty csi.csv), a board that dropped
out, a missing/duplicated clapper event, poor camera detection, and broken
CSI<->camera time alignment.

Usage:
    python3 validate_session.py sessions/20260524_01_newsample

Exit code is non-zero if any HARD check fails, so it can gate a training script:
    python3 validate_session.py sessions/<name> && python3 train_final.py sessions/<name>
"""

import sys
import os
import json
import argparse

import numpy as np
import pandas as pd

# Reuse the canonical loaders so column-name handling stays in one place.
from dataset import (load_session, build_aligned_dataset, build_multiboard_dataset,
                     grid_cell_id, boards_in_multiboard_df)


OK, WARN, FAIL = "OK", "WARN", "FAIL"
GLYPH = {OK: "✓", WARN: "⚠", FAIL: "✗"}

EXPECTED_N_BOARDS = 3  # we run 3 RX boards; their actual IDs are data-driven
RATE_WARN = 20.0     # per-board CSI rate (Hz) below this warns (unicast should be ~33)
DETECT_WARN = 0.85   # camera detection fraction below this warns
DETECT_FAIL = 0.50   # below this fails
GAP_WARN_MS = 100.0  # median CSI<->camera gap above this warns
GAP_FAIL_MS = 300.0  # above this fails
WINDOW_WARN_S = 300.0  # session window (s) below this warns (real sessions are 15-20 min)
CAM_FPS_WARN = 20.0  # estimated camera fps below this warns (GUI preview stall)
CAM_GAP_WARN_S = 1.0  # max camera frame-to-frame gap (s) above this warns
CAM_GAP_FAIL_S = 5.0  # above this fails (capture stalled mid-session)


class Report:
    def __init__(self):
        self.rows = []  # (level, label, detail)

    def add(self, level, label, detail=""):
        self.rows.append((level, label, detail))

    def worst(self):
        if any(l == FAIL for l, _, _ in self.rows):
            return FAIL
        if any(l == WARN for l, _, _ in self.rows):
            return WARN
        return OK

    @property
    def exit_code(self):
        return 1 if self.worst() == FAIL else 0

    @property
    def ok(self):
        return self.worst() != FAIL

    def print(self):
        print()
        for level, label, detail in self.rows:
            line = f"  {GLYPH[level]} [{level:4}] {label}"
            if detail:
                line += f" — {detail}"
            print(line)
        print()
        verdict = self.worst()
        print(f"  ==> {GLYPH[verdict]} SESSION {verdict}")
        print()


def check_files(session_dir, rep):
    for name in ("csi.csv", "camera.csv", "clap.csv"):
        path = os.path.join(session_dir, name)
        if not os.path.exists(path):
            rep.add(FAIL, f"{name} exists", "file missing")
            continue
        # data rows = total lines - 1 header
        with open(path) as f:
            n = sum(1 for _ in f) - 1
        if n <= 0:
            rep.add(FAIL, f"{name} non-empty", "header only, no data rows")
        else:
            rep.add(OK, f"{name} non-empty", f"{n:,} data rows")


def check_clap(clap, rep):
    if clap is None or clap.empty:
        rep.add(FAIL, "clapper events", "no clap rows")
        return None, None
    starts = clap[clap["event_name"] == "start"]
    stops = clap[clap["event_name"] == "stop"]
    if len(starts) != 1 or len(stops) != 1:
        rep.add(FAIL, "clapper START/STOP count",
                f"got {len(starts)} START, {len(stops)} STOP (need exactly 1 each)")
        # still try to use first/last for downstream
    t_start = starts["wall_time_s"].iloc[0] if len(starts) else None
    t_stop = stops["wall_time_s"].iloc[-1] if len(stops) else None
    if t_start is not None and t_stop is not None:
        if t_stop <= t_start:
            rep.add(FAIL, "clapper ordering", "STOP not after START")
        else:
            dur = t_stop - t_start
            level = OK if dur > WINDOW_WARN_S else WARN
            rep.add(level, "session window", f"{dur:.0f} s between START and STOP")
        if len(starts) == 1 and len(stops) == 1 and t_stop > t_start:
            rep.add(OK, "clapper START/STOP count", "exactly 1 each")
    return t_start, t_stop


def check_csi(csi, t_start, t_stop, rep):
    if csi is None or csi.empty:
        rep.add(FAIL, "CSI present", "no CSI rows")
        return
    win = csi
    if t_start is not None and t_stop is not None:
        win = csi[(csi["wall_time_s"] >= t_start) & (csi["wall_time_s"] <= t_stop)]
        dur = max(t_stop - t_start, 1e-6)
    else:
        dur = max(csi["wall_time_s"].max() - csi["wall_time_s"].min(), 1e-6)

    present = sorted(int(b) for b in win["board_id"].unique())
    if len(present) < EXPECTED_N_BOARDS:
        rep.add(FAIL, f"{EXPECTED_N_BOARDS} boards present",
                f"only {len(present)} distinct board(s): {present}")
    elif len(present) > EXPECTED_N_BOARDS:
        rep.add(WARN, f"{EXPECTED_N_BOARDS} boards present",
                f"{len(present)} distinct boards (more than expected): {present}")
    else:
        rep.add(OK, f"{EXPECTED_N_BOARDS} boards present", f"boards {present} all sent CSI")

    rates = []
    for b in present:
        n = int((win["board_id"] == b).sum())
        rate = n / dur
        rates.append(f"b{b}={rate:.1f}/s ({n:,})")
        if rate < RATE_WARN:
            rep.add(WARN, f"board {b} packet rate",
                    f"{rate:.1f}/s (unicast expects ~33/s; ~10/s = still broadcast?)")
    rep.add(OK, "per-board CSI rate", ", ".join(rates))


def check_stream_purity(csi, rep):
    """One TX, one CSI format: frames from a stray MAC mean the RX filter let
    foreign traffic through, and mixed csi_len means mixed CSI formats — both
    silently corrupt the feature space."""
    if csi is None or csi.empty:
        return
    if "mac" in csi.columns:
        macs = csi["mac"].value_counts()
        if len(macs) != 1:
            mix = ", ".join(f"{m}={n:,}" for m, n in macs.items())
            rep.add(WARN, "TX MAC purity",
                    f"{len(macs)} source MACs ({mix}) — unexpected source frames")
        else:
            rep.add(OK, "TX MAC purity", f"single TX MAC {macs.index[0]}")
    if "csi_len" in csi.columns:
        lens = csi["csi_len"].value_counts()
        if len(lens) != 1:
            mix = ", ".join(f"len={int(v)}: {n:,}" for v, n in lens.items())
            rep.add(WARN, "csi_len purity", f"mixed CSI formats ({mix})")
        else:
            rep.add(OK, "csi_len purity", f"csi_len uniform {int(lens.index[0])}")


def check_camera(camera, rep):
    if camera is None or camera.empty:
        rep.add(FAIL, "camera frames present", "no camera rows")
        return
    det = (camera["detected"] == 1).mean()
    if det < DETECT_FAIL:
        rep.add(FAIL, "camera detection rate", f"{det*100:.1f}% frames had a marker")
    elif det < DETECT_WARN:
        rep.add(WARN, "camera detection rate", f"{det*100:.1f}% (low — occlusion or lighting?)")
    else:
        rep.add(OK, "camera detection rate", f"{det*100:.1f}% frames detected")


def check_camera_flow(camera, rep):
    """Frames must keep flowing: the known failure mode is a GUI preview stall
    starving the capture thread (fps sags) or dropping multi-second holes that
    leave CSI stretches unlabeled."""
    if camera is None or len(camera) < 10:
        return
    dt = camera["wall_time_s"].diff().dropna()
    med = float(dt.median())
    fps_est = (1.0 / med) if med > 0 else float("inf")
    max_gap = float(dt.max())
    detail = f"~{fps_est:.1f} fps (median frame interval), max gap {max_gap:.2f} s"
    if max_gap > CAM_GAP_FAIL_S:
        rep.add(FAIL, "camera frame flow",
                f"{detail} — capture stalled >{CAM_GAP_FAIL_S:.0f} s mid-session")
    elif fps_est < CAM_FPS_WARN:
        rep.add(WARN, "camera frame flow",
                f"{detail} — below {CAM_FPS_WARN:.0f} fps (GUI preview stall "
                f"starving capture?)")
    elif max_gap > CAM_GAP_WARN_S:
        rep.add(WARN, "camera frame flow",
                f"{detail} — frame gap >{CAM_GAP_WARN_S:.0f} s leaves CSI unlabeled")
    else:
        rep.add(OK, "camera frame flow", detail)


def check_alignment(csi, camera, clap, rep):
    try:
        aligned = build_aligned_dataset(csi, camera, clap)
    except Exception as e:
        rep.add(FAIL, "CSI<->camera alignment", f"could not build aligned dataset: {e}")
        return
    if aligned.empty:
        rep.add(FAIL, "CSI<->camera alignment", "0 aligned samples after trim + gap filter")
        return
    gap_ms = aligned["time_gap_s"].median() * 1000
    gap_max = aligned["time_gap_s"].max() * 1000
    detail = f"median {gap_ms:.0f} ms, max {gap_max:.0f} ms, {len(aligned):,} aligned samples"
    if gap_ms > GAP_FAIL_MS:
        rep.add(FAIL, "CSI<->camera time gap", detail)
    elif gap_ms > GAP_WARN_MS:
        rep.add(WARN, "CSI<->camera time gap", detail)
    else:
        rep.add(OK, "CSI<->camera time gap", detail)


def check_quality(csi, camera, clap, rep):
    """Two-foot label mix, duplicate-CSI %, per-board CSI age, grid coverage —
    the metrics tracked at validation time.
    Uses the same tested dataset helpers as training."""
    # two-foot label mix (camera 'method' column from two-leg tracking)
    if "method" in camera.columns and len(camera):
        vc = camera["method"].value_counts(dropna=False)
        tot = len(camera)
        both = vc.get("both", 0) / tot
        mix = ", ".join(f"{k}={100 * v / tot:.0f}%" for k, v in vc.items())
        if both < 0.50:
            rep.add(FAIL, "two-foot label mix",
                    f"both={both * 100:.0f}% ({mix}) — both<50%, poor labels "
                    f"inflate the LOSO fold (RealSession2 precedent)")
        else:
            rep.add(OK if both >= 0.75 else WARN, "two-foot label mix",
                    f"both={both * 100:.0f}% ({mix})"
                    + ("" if both >= 0.75 else " — both<75%, single-foot fallback heavy"))

    try:
        full = build_multiboard_dataset(csi, camera, clap, dedup=False)
        dd = build_multiboard_dataset(csi, camera, clap, dedup=True)
    except Exception as e:
        rep.add(WARN, "multiboard quality metrics", f"could not build: {e}")
        return
    if len(full) == 0:
        rep.add(WARN, "multiboard quality metrics", "0 multiboard samples")
        return

    dedup_pct = 100 * (1 - len(dd) / len(full))
    level = OK if dedup_pct < 5 else (WARN if dedup_pct < 20 else FAIL)
    rep.add(level, "duplicate-CSI fraction",
            f"{dedup_pct:.1f}% ({len(full):,}->{len(dd):,} unique); "
            f">20% = CSI delivery capped (broadcast/rate)")

    ages = []
    for b in boards_in_multiboard_df(dd):
        col = f"b{b}_age_s"
        if col in dd:
            a = dd[col].dropna()
            if len(a):
                ages.append(f"b{b}={a.median() * 1000:.0f}/{a.quantile(0.9) * 1000:.0f}ms")
    if ages:
        rep.add(OK, "per-board CSI age (median/p90)", ", ".join(ages))

    per = grid_cell_id(dd).value_counts()
    rep.add(OK, "grid coverage",
            f"{len(per)} cells; samples/cell min={per.min()} "
            f"median={int(per.median())} max={per.max()}")


def check_metadata(session_dir, rep):
    """A session without metadata can't be reasoned about for drift.
    Surface missing/incomplete metadata so it gets filled while still fresh."""
    path = os.path.join(session_dir, "metadata.json")
    if not os.path.exists(path):
        rep.add(WARN, "session metadata",
                "no metadata.json — run session_metadata.py (room, walk style, "
                "furniture, people, board placement) while you still remember")
        return
    try:
        meta = json.load(open(path))
    except Exception as e:
        rep.add(WARN, "session metadata", f"metadata.json unreadable: {e}")
        return
    key_human = ["room", "walk_style", "board_placement", "person", "furniture_notes"]
    missing = [k for k in key_human if not meta.get(k)]
    if missing:
        rep.add(WARN, "session metadata", f"present but unfilled: {', '.join(missing)}")
    else:
        rep.add(OK, "session metadata", "present")


def build_report(session_dir):
    """Build the validation Report for a session directory.

    Pure of process-control side effects: never prints the report body and
    never calls sys.exit. Returns a Report whose .rows hold every PASS/WARN/FAIL
    check, and whose .worst()/.exit_code/.ok summarise the verdict. The CLI in
    main() prints and exits; callers (e.g. a GUI) can consume the Report rows.
    """
    rep = Report()

    check_files(session_dir, rep)

    # Load what we can; loaders may fail on a totally broken session.
    try:
        csi, camera, clap = load_session(session_dir)
    except Exception as e:
        rep.add(FAIL, "load session CSVs", str(e))
        return rep

    t_start, t_stop = check_clap(clap, rep)
    check_csi(csi, t_start, t_stop, rep)
    check_stream_purity(csi, rep)
    check_camera(camera, rep)
    check_camera_flow(camera, rep)
    check_alignment(csi, camera, clap, rep)
    check_quality(csi, camera, clap, rep)
    check_metadata(session_dir, rep)

    return rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", help="path to sessions/<name>")
    args = parser.parse_args()
    session_dir = args.session.rstrip("/")

    print(f"Validating session: {session_dir}")
    rep = build_report(session_dir)
    rep.print()
    sys.exit(1 if rep.worst() == FAIL else 0)


if __name__ == "__main__":
    main()
