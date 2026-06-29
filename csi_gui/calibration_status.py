"""Read-only status of the three calibration JSON files.

This module NEVER writes and NEVER imports the calibration tools — it only reads
the on-disk JSON that those tools produce, so the Calibrate page can show whether
each calibration exists and summarise its key fields. A missing (or unparseable)
file yields ``present=False`` ("not calibrated").

Three calibrations, in the order the pipeline needs them:

  * lens_profile.json     — one-time intrinsic lens calibration (lens_calibrate.py)
  * marker_layout.json    — one-time floor-marker map (marker_layout.py)
  * floor_calibration.json — PRIMARY: camera extrinsics; re-run whenever the
                             camera moves (aruco_setup.py)

Field names mirror exactly what the tools write (verified against the on-disk
files): floor uses ``grid_spacing_cm`` / ``grid_bounds_cm`` / ``camera_resolution``
/ ``marker_positions_cm`` / ``aruco_dict``; lens uses ``resolution`` /
``reprojection_error``; marker uses ``positions_cm``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from csi_gui.app_context import ROOT

FLOOR_CALIBRATION = "floor_calibration.json"
LENS_PROFILE = "lens_profile.json"
MARKER_LAYOUT = "marker_layout.json"


@dataclass
class CalibrationStatus:
    """Status of a single calibration JSON file."""

    name: str
    path: str
    present: bool
    error: str | None = None          # set if the file exists but failed to parse
    mtime: float | None = None        # POSIX timestamp of the file, if present
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def mtime_str(self) -> str:
        """Human-readable modification time, or 'never' when absent."""
        if self.mtime is None:
            return "never"
        return datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def summary(self) -> str:
        """One-line headline: 'calibrated' / 'not calibrated' / 'unreadable'."""
        if not self.present:
            return "not calibrated"
        if self.error is not None:
            return f"unreadable ({self.error})"
        return "calibrated"


def _mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _load(name: str, path: str):
    """Read ``path`` and return ``(data, CalibrationStatus|None)``.

    Three outcomes for the on-disk file:
      * absent       -> ``(None, status(present=False))``       — "not calibrated"
      * present+bad  -> ``(None, status(present=True, error))`` — "unreadable"
      * present+ok   -> ``(data, None)``  (caller fills in the fields)
    """
    if not os.path.isfile(path):
        return None, CalibrationStatus(name, path, present=False)
    try:
        with open(path) as f:
            return json.load(f), None
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        return None, CalibrationStatus(name, path, present=True,
                                       error=type(exc).__name__, mtime=_mtime(path))


def floor_status(root: str | None = None) -> CalibrationStatus:
    """Status of floor_calibration.json (PRIMARY — re-run when camera moves)."""
    root = root or ROOT
    path = os.path.join(root, FLOOR_CALIBRATION)
    data, status = _load("floor", path)
    if status is not None:
        return status
    marker_positions = data.get("marker_positions_cm") or {}
    fields = {
        "grid_spacing_cm": data.get("grid_spacing_cm"),
        "grid_bounds_cm": data.get("grid_bounds_cm"),
        "camera_resolution": data.get("camera_resolution"),
        "n_marker_positions": len(marker_positions),
        "aruco_dict": data.get("aruco_dict"),
    }
    return CalibrationStatus("floor", path, present=True, mtime=_mtime(path),
                             fields=fields)


def lens_status(root: str | None = None) -> CalibrationStatus:
    """Status of lens_profile.json (one-time intrinsic calibration)."""
    root = root or ROOT
    path = os.path.join(root, LENS_PROFILE)
    data, status = _load("lens", path)
    if status is not None:
        return status
    fields = {
        "resolution": data.get("resolution"),
        "reprojection_error": data.get("reprojection_error"),
        "num_captures": data.get("num_captures"),
    }
    return CalibrationStatus("lens", path, present=True, mtime=_mtime(path),
                             fields=fields)


def marker_status(root: str | None = None) -> CalibrationStatus:
    """Status of marker_layout.json (one-time floor-marker map)."""
    root = root or ROOT
    path = os.path.join(root, MARKER_LAYOUT)
    data, status = _load("marker", path)
    if status is not None:
        return status
    positions = data.get("positions_cm") or {}
    fields = {
        "n_positions": len(positions),
    }
    return CalibrationStatus("marker", path, present=True, mtime=_mtime(path),
                             fields=fields)


def all_status(root: str | None = None) -> dict[str, CalibrationStatus]:
    """All three statuses keyed by name ('floor', 'lens', 'marker')."""
    return {
        "floor": floor_status(root),
        "lens": lens_status(root),
        "marker": marker_status(root),
    }
