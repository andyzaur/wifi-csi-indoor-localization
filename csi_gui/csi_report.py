"""CSI-only session report for empty-room (camera-less) captures.

``validate_session.build_report`` assumes the full three-file session
(csi.csv + camera.csv + clap.csv) and FAILs an empty-room capture on the
missing camera artifacts. This builder reuses the SAME Report class and the
camera-independent checks (clap, CSI rates/purity, metadata) so an empty-room
session gets an honest verdict for what it actually is: a CSI + clapper
recording. The backend module is imported, never modified.
"""

from __future__ import annotations

import os

import pandas as pd

from validate_session import (
    FAIL,
    OK,
    Report,
    WARN,
    check_clap,
    check_csi,
    check_metadata,
    check_stream_purity,
)


def _read_csv(session_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(session_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(name)
    return pd.read_csv(path)


def build_csi_report(session_dir: str) -> Report:
    """Validate an empty-room session: csi.csv + clap.csv only.

    Returns a ``validate_session.Report`` (same rows/worst() contract the
    ValidatePanel renders), so the GUI needs no special rendering path.
    """
    rep = Report()
    rep.add(OK, "session mode", "empty room (CSI-only) — camera checks skipped")

    for name in ("csi.csv", "clap.csv"):
        path = os.path.join(session_dir, name)
        if not os.path.exists(path):
            rep.add(FAIL, f"{name} exists", "file missing")
            continue
        with open(path) as f:
            n = sum(1 for _ in f) - 1
        if n <= 0:
            rep.add(FAIL, f"{name} non-empty", "header only, no data rows")
        else:
            rep.add(OK, f"{name} non-empty", f"{n:,} data rows")

    try:
        csi = _read_csv(session_dir, "csi.csv")
    except Exception as exc:  # noqa: BLE001 — a broken file is a FAIL row, not a crash
        rep.add(FAIL, "load csi.csv", str(exc))
        return rep
    try:
        clap = _read_csv(session_dir, "clap.csv")
    except Exception:  # noqa: BLE001 — clapper is optional for a baseline capture
        clap = pd.DataFrame()
        rep.add(WARN, "load clap.csv", "missing/unreadable — no session window")

    t_start, t_stop = check_clap(clap, rep)
    check_csi(csi, t_start, t_stop, rep)
    check_stream_purity(csi, rep)
    check_metadata(session_dir, rep)
    return rep
