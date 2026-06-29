#!/usr/bin/env python3
"""Write/update a metadata.json for a recording session.

Captures the context that the CSVs don't: room, board/antenna placement,
calibration provenance, walk style, environment, purpose. This is what makes a
session reproducible and lets us reason about drift across sessions later
("which sessions were the same room? same day? same clapper board?").

Auto-derives what it can (timestamps, git commit, calibration file hashes,
grid spacing/bounds, marker count). Human context comes from CLI flags or
--interactive prompts. Re-running merges into the existing file, so you can add
fields after the fact.

Usage:
    python3 session_metadata.py sessions/20260524_01 \
        --room "bedroom" --walk-style "random, faster" \
        --clapper "ESP32-S3 stand-in (C6 clapper on loan)" \
        --purpose "drift test, +5 days from session 02"

    python3 session_metadata.py sessions/20260524_01 --interactive
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

# Calibration files (in repo root) whose hashes pin the spatial setup.
CALIB_FILES = ["lens_profile.json", "marker_layout.json", "floor_calibration.json"]

# Human-context fields: (key, prompt text)
HUMAN_FIELDS = [
    ("room", "Room / location"),
    ("board_placement", "RX board placement (where are boards 1/2/3 + TX?)"),
    ("antenna_config", "Antenna config (which antenna tier on which board?)"),
    ("clapper", "Clapper board used (C6 / S3 stand-in / etc.)"),
    ("person", "Person (height, build)"),
    ("n_other_people", "Number of other people in the room during capture"),
    ("furniture_notes", "Furniture / environment notes (changes vs last session?)"),
    ("camera_mode", "Camera mode (resolution, fps, lens)"),
    ("walk_style", "Walk style (slow/structured, random/fast, standing, etc.)"),
    ("purpose", "Session purpose (what is this recording FOR?)"),
    ("notes", "Free-form notes"),
]


def file_hash(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def auto_fields(repo_root, marker_offset):
    out = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "calibration_hashes": {f: file_hash(os.path.join(repo_root, f)) for f in CALIB_FILES},
        "marker_offset_cm": {"side": marker_offset[0], "back": marker_offset[1]},
    }
    floor = os.path.join(repo_root, "floor_calibration.json")
    if os.path.exists(floor):
        try:
            fc = json.load(open(floor))
            out["grid_spacing_cm"] = fc.get("grid_spacing_cm")
            out["grid_bounds_cm"] = fc.get("grid_bounds_cm")
            out["aruco_dict"] = fc.get("aruco_dict")
            out["n_floor_markers"] = len(fc.get("marker_positions_cm", {}))
        except Exception:
            pass
    return out


def write_metadata(session_dir, repo_root, human, offsets,
                   interactive=False, prompt_fn=None):
    """Merge/auto-derive/write a session's metadata.json and return the dict.

    Behaviour mirrors the original main() exactly:
      * session_dir is created if absent; existing metadata.json is merged.
      * `human` maps each HUMAN_FIELDS key to a flag value (None = not given).
      * `offsets` is (side, back) for marker_offset_cm.
      * auto fields (timestamps, git, calibration hashes, grid) are re-derived.
      * When `interactive` and a field's human value is None, the field is
        resolved via `prompt_fn(prompt, existing)`; default prompt_fn replicates
        the original input()-based prompt (so on_log=None style: stdlib I/O).

    This is GUI-agnostic: callers pass plain Python values / a callable; nothing
    here imports any UI toolkit.
    """
    session_dir = session_dir.rstrip("/")
    os.makedirs(session_dir, exist_ok=True)
    meta_path = os.path.join(session_dir, "metadata.json")

    if prompt_fn is None:
        def prompt_fn(prompt, existing):
            shown = f" [{existing}]" if existing else ""
            entered = input(f"{prompt}{shown}: ").strip()
            return entered or existing or None

    # Merge into existing if present
    meta = {}
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            pass

    meta["session_name"] = os.path.basename(session_dir)
    meta.update(auto_fields(repo_root, offsets))

    for key, prompt in HUMAN_FIELDS:
        val = human.get(key)
        if val is None and interactive:
            existing = meta.get(key, "")
            val = prompt_fn(prompt, existing)
        if val is not None:
            meta[key] = val
        elif key not in meta:
            meta[key] = None  # keep the slot so it's visible as TODO

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {meta_path}")
    missing = [k for k, _ in HUMAN_FIELDS if not meta.get(k)]
    if missing:
        print(f"  (still unfilled: {', '.join(missing)} — rerun with flags or --interactive)")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", help="path to sessions/<name>")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="prompt for any human-context fields not given as flags")
    parser.add_argument("--offset-side", type=float, default=-20.0,
                        help="marker->body side offset cm (matches aruco_track default)")
    parser.add_argument("--offset-back", type=float, default=-15.0,
                        help="marker->body back offset cm (matches aruco_track default)")
    for key, help_text in HUMAN_FIELDS:
        parser.add_argument(f"--{key.replace('_', '-')}", default=None, help=help_text)
    args = parser.parse_args()

    session_dir = args.session.rstrip("/")
    repo_root = os.path.dirname(os.path.abspath(__file__))
    human = {key: getattr(args, key) for key, _ in HUMAN_FIELDS}
    write_metadata(session_dir, repo_root, human,
                   (args.offset_side, args.offset_back),
                   interactive=args.interactive)


if __name__ == "__main__":
    main()
