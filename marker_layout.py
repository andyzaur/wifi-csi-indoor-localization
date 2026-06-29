#!/usr/bin/env python3
"""Compute floor marker positions from arbitrary distance measurements.

Usage:
    python3 marker_layout.py

Workflow:
    1. Enter anchor markers with their known (x, y) coordinates in cm
       (typically the markers placed on the X-axis, Y-axis, or at the origin)
    2. Enter any distance measurements you made — free format:
       'ID1 ID2 distance_cm'   e.g. '1 5 200'  means marker 1 to marker 5 is 200cm
    3. Type 'compute' to run trilateration
    4. The algorithm iteratively places markers:
       - 3+ distances to known markers  => exact position, no question asked
       - 2 distances to known markers   => asks you which side
       - 1 or 0 distances               => skipped, needs more measurements
    5. Layout map opens for visual verification
    6. Confirm to save marker_layout.json

All distances in centimeters.
"""

import json
import math
import subprocess
import sys
from itertools import combinations

import numpy as np
import cv2


LAYOUT_FILE = "marker_layout.json"
LAYOUT_PNG = "marker_layout.png"
CANDIDATE_PNG = "marker_candidates.png"


def _render_markers(positions, candidates=None, candidate_mid=None,
                    used_anchors=None, anchor_id=None, x_axis_id=None,
                    out_path=None, title=""):
    """Render a top-down map of known markers; optionally overlay 2 candidates."""
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    if candidates:
        for c in candidates:
            all_x.append(c[0])
            all_y.append(c[1])

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    margin_cm = 40
    span_x = max(x_max - x_min, 50) + 2 * margin_cm
    span_y = max(y_max - y_min, 50) + 2 * margin_cm

    px_per_cm = 4
    W = int(span_x * px_per_cm)
    H = int(span_y * px_per_cm)
    img = np.full((H, W, 3), 250, dtype=np.uint8)

    def to_px(x, y):
        px = int(round((x - x_min + margin_cm) * px_per_cm))
        py = int(round((y_max - y + margin_cm) * px_per_cm))
        return px, py

    grid_step = 50
    x_lo = math.floor(x_min / grid_step) * grid_step
    x_hi = math.ceil(x_max / grid_step) * grid_step
    y_lo = math.floor(y_min / grid_step) * grid_step
    y_hi = math.ceil(y_max / grid_step) * grid_step

    for x in range(int(x_lo), int(x_hi) + 1, grid_step):
        cv2.line(img, to_px(x, y_lo - margin_cm), to_px(x, y_hi + margin_cm),
                 (220, 220, 220), 1)
    for y in range(int(y_lo), int(y_hi) + 1, grid_step):
        cv2.line(img, to_px(x_lo - margin_cm, y), to_px(x_hi + margin_cm, y),
                 (220, 220, 220), 1)

    # Axes
    cv2.line(img, to_px(x_lo - margin_cm, 0), to_px(x_hi + margin_cm, 0),
             (180, 180, 180), 2)
    cv2.line(img, to_px(0, y_lo - margin_cm), to_px(0, y_hi + margin_cm),
             (180, 180, 180), 2)

    # Distance circles around the used anchors (so candidates lie on these)
    if used_anchors:
        for aid, dist in used_anchors:
            ax, ay = positions[aid]
            cx, cy = to_px(ax, ay)
            r = int(round(dist * px_per_cm))
            cv2.circle(img, (cx, cy), r, (180, 200, 255), 1)

    # Existing placed markers
    for mid, (x, y) in sorted(positions.items()):
        cx, cy = to_px(x, y)
        if mid == anchor_id:
            color = (0, 0, 200)
        elif mid == x_axis_id:
            color = (200, 100, 0)
        else:
            color = (0, 130, 0)

        # Dim if not an anchor used for current trilateration
        if used_anchors:
            anchor_ids_used = {aid for aid, _ in used_anchors}
            if mid not in anchor_ids_used:
                color = tuple(int(c * 0.6 + 100) for c in color)

        cv2.circle(img, (cx, cy), 14, color, -1)
        cv2.circle(img, (cx, cy), 14, (50, 50, 50), 2)
        cv2.putText(img, str(mid), (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, f"({x:+.0f},{y:+.0f})", (cx + 20, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Candidates (if any)
    if candidates:
        labels = ["A", "B"]
        colors = [(0, 150, 255), (255, 0, 200)]  # orange, magenta
        for cand, lbl, col in zip(candidates, labels, colors):
            cx, cy = to_px(cand[0], cand[1])
            cv2.circle(img, (cx, cy), 18, col, 3)
            cv2.putText(img, lbl, (cx - 8, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3)
            cv2.putText(img, f"M{candidate_mid}-{lbl}: ({cand[0]:+.0f},{cand[1]:+.0f})",
                        (cx + 25, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    if title:
        cv2.putText(img, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.putText(img, "Top-down view (cm, +Y up, origin = red)",
                (10, H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(img, f"Bounds: x [{x_min:.0f}, {x_max:.0f}]  y [{y_min:.0f}, {y_max:.0f}] cm",
                (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    if out_path:
        cv2.imwrite(out_path, img)
        try:
            subprocess.run(["open", out_path], check=False)
        except Exception:
            pass


def trilaterate_2(p1, d1, p2, d2):
    """Two-circle intersection. Returns (cand_a, cand_b) or (None, None) if no solution."""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    delta = p2 - p1
    dist12 = np.linalg.norm(delta)
    if dist12 < 1e-6 or dist12 > d1 + d2 + 1e-3 or dist12 < abs(d1 - d2) - 1e-3:
        return None, None

    a = (d1 ** 2 - d2 ** 2 + dist12 ** 2) / (2 * dist12)
    h_sq = d1 ** 2 - a ** 2
    h = math.sqrt(max(0, h_sq))

    unit = delta / dist12
    perp = np.array([-unit[1], unit[0]])
    foot = p1 + a * unit
    return tuple(foot + h * perp), tuple(foot - h * perp)


def multilaterate(known_positions, measurements):
    """Solve for a position given 3+ distance measurements to known markers (least squares)."""
    pts = np.array([known_positions[mid] for mid, _ in measurements], dtype=float)
    ds = np.array([d for _, d in measurements], dtype=float)

    # Use marker 0 as reference, subtract its equation from others
    p0 = pts[0]
    d0 = ds[0]
    A = 2 * (pts[1:] - p0)
    b = (d0 ** 2 - ds[1:] ** 2) + np.sum(pts[1:] ** 2, axis=1) - np.sum(p0 ** 2)
    pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return tuple(pos)


def ask_yn(prompt):
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer y or n.")


def parse_3_ints_or_floats(line, types):
    parts = line.split()
    if len(parts) != 3:
        return None
    try:
        return tuple(t(p) for t, p in zip(types, parts))
    except ValueError:
        return None


def main():
    print("=== FLOOR MARKER LAYOUT ===")
    print()
    print("Step 1: Enter ANCHORS (markers with known coordinates)")
    print("  Format: 'ID X Y'  (e.g. '1 0 0' for origin)")
    print("  Type 'done' when finished. Need at least 1 anchor.")
    print()

    positions = {}
    while True:
        raw = input("anchor> ").strip()
        if raw.lower() == "done":
            break
        parsed = parse_3_ints_or_floats(raw, [int, float, float])
        if parsed is None:
            print("  Format: ID X Y  (e.g. '1 0 0')")
            continue
        mid, x, y = parsed
        positions[mid] = (x, y)
        print(f"  Marker {mid} -> ({x:+.1f}, {y:+.1f})")

    if len(positions) < 1:
        print("Need at least 1 anchor. Exiting.")
        sys.exit(1)

    print(f"\n{len(positions)} anchor(s): {sorted(positions.keys())}")
    print()
    print("Step 2: Enter DISTANCE MEASUREMENTS you took")
    print("  Format: 'ID1 ID2 distance_cm'  (e.g. '1 5 200')")
    print("  Type 'compute' when done, 'list' to see entered, 'undo' to remove last")
    print()

    distances = {}  # frozenset({a, b}) -> distance
    entry_order = []  # list of frozensets in entry order
    while True:
        raw = input("distance> ").strip().lower()
        if raw == "compute":
            break
        if raw == "list":
            for k in entry_order:
                a, b = sorted(k)
                print(f"  d({a}, {b}) = {distances[k]}")
            continue
        if raw == "undo":
            if entry_order:
                k = entry_order.pop()
                del distances[k]
                a, b = sorted(k)
                print(f"  Removed d({a}, {b})")
            continue
        parsed = parse_3_ints_or_floats(raw, [int, int, float])
        if parsed is None:
            print("  Format: ID1 ID2 distance_cm  (e.g. '1 5 200')")
            continue
        a, b, d = parsed
        if a == b:
            print("  ID1 and ID2 must be different.")
            continue
        if d <= 0:
            print("  Distance must be positive.")
            continue
        key = frozenset((a, b))
        if key not in distances:
            entry_order.append(key)
        distances[key] = d
        print(f"  d({a}, {b}) = {d}")

    if not distances:
        print("No distance measurements entered. Exiting.")
        sys.exit(1)

    # All marker IDs that appear anywhere
    all_ids = set(positions.keys())
    for key in distances:
        all_ids.update(key)
    unknowns = sorted(all_ids - set(positions.keys()))

    print(f"\nAnchors: {sorted(positions.keys())}")
    print(f"Unknowns: {unknowns}")
    print()

    # Iteratively place unknowns
    progress = True
    while progress and unknowns:
        progress = False
        still_unknown = []
        for mid in unknowns:
            # Collect distances to known markers
            known_distances = []
            for key, dist in distances.items():
                if mid not in key:
                    continue
                other = next(iter(key - {mid}))
                if other in positions:
                    known_distances.append((other, dist))

            if len(known_distances) == 0:
                print(f"Marker {mid}: no distances to known markers yet, skipping for now.")
                still_unknown.append(mid)
            elif len(known_distances) == 1:
                other, dist = known_distances[0]
                print(f"Marker {mid}: only 1 known distance (to {other} = {dist}cm). "
                      f"Need at least 2. Skipping.")
                still_unknown.append(mid)
            elif len(known_distances) == 2:
                # Two candidates — show map and ask user
                (a, d_a), (b, d_b) = known_distances
                cand_a, cand_b = trilaterate_2(positions[a], d_a, positions[b], d_b)
                if cand_a is None:
                    print(f"Marker {mid}: distances inconsistent "
                          f"(d({a},{mid})={d_a}, d({b},{mid})={d_b}). Skipping.")
                    still_unknown.append(mid)
                    continue

                # Auto-pick if user said all positive AND only one is in positive quadrant
                a_positive = cand_a[0] >= -1 and cand_a[1] >= -1
                b_positive = cand_b[0] >= -1 and cand_b[1] >= -1

                anchor_id_local = min(positions.keys())
                x_axis_id_local = _guess_x_axis(positions, anchor_id_local)

                _render_markers(
                    positions,
                    candidates=[cand_a, cand_b],
                    candidate_mid=mid,
                    used_anchors=[(a, d_a), (b, d_b)],
                    anchor_id=anchor_id_local,
                    x_axis_id=x_axis_id_local,
                    out_path=CANDIDATE_PNG,
                    title=f"Marker {mid}: pick A (orange) or B (magenta)",
                )

                print(f"\nMarker {mid}: 2 candidate positions from distances to {a} and {b}:")
                print(f"  A (orange): ({cand_a[0]:+.1f}, {cand_a[1]:+.1f})"
                      f"{'  [POSITIVE quadrant]' if a_positive else ''}")
                print(f"  B (magenta): ({cand_b[0]:+.1f}, {cand_b[1]:+.1f})"
                      f"{'  [POSITIVE quadrant]' if b_positive else ''}")
                print(f"  Map opened: {CANDIDATE_PNG}")

                while True:
                    pick = input(f"  Which one is marker {mid}? [a/b]: ").strip().lower()
                    if pick == "a":
                        positions[mid] = cand_a
                        break
                    if pick == "b":
                        positions[mid] = cand_b
                        break
                    print("    Pick 'a' or 'b'.")
                progress = True
            else:
                # 3+ known distances — exact least-squares solve
                pos = multilaterate(positions, known_distances)
                positions[mid] = pos
                anchors_used = [a for a, _ in known_distances]
                print(f"Marker {mid}: placed exactly via {len(known_distances)} distances "
                      f"to {anchors_used} -> ({pos[0]:+.1f}, {pos[1]:+.1f})")
                progress = True
        unknowns = still_unknown

    if unknowns:
        print()
        print(f"WARNING: Could not place markers {unknowns}.")
        print("  Each needs at least 2 distance measurements to already-placed markers.")
        print("  Add more measurements and re-run.")

    print()
    print("=== Final layout ===")
    for mid in sorted(positions.keys()):
        x, y = positions[mid]
        print(f"  Marker {mid:2d}: ({x:+8.1f}, {y:+8.1f}) cm")

    # Identify anchor + x-axis IDs for visualization coloring
    anchor_id = min(positions.keys())
    x_axis_id = None
    for mid, (x, y) in positions.items():
        if mid != anchor_id and abs(y) < 1e-3 and x > 0:
            x_axis_id = mid
            break
    if x_axis_id is None:
        x_axis_id = sorted(positions.keys())[1] if len(positions) > 1 else anchor_id

    draw_layout(positions, anchor_id, x_axis_id)

    if not ask_yn("\nDoes the map look right? Save to marker_layout.json? [y/n]: "):
        print("Not saved.")
        return

    layout = {
        "positions_cm": {str(k): list(v) for k, v in positions.items()},
        "anchors": {str(k): list(v) for k, v in positions.items() if k in [anchor_id]},
        "distance_measurements_cm": [
            {"a": min(k), "b": max(k), "d": distances[k]} for k in entry_order
        ],
    }
    with open(LAYOUT_FILE, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"Saved to {LAYOUT_FILE}")


def _guess_x_axis(positions, anchor_id):
    """Heuristic: x-axis marker is one on y=0 with x>0."""
    for mid, (x, y) in positions.items():
        if mid != anchor_id and abs(y) < 1e-3 and x > 0:
            return mid
    others = [mid for mid in positions if mid != anchor_id]
    return others[0] if others else anchor_id


def draw_layout(positions, anchor_id, x_axis_id):
    """Render the final top-down view of all markers."""
    _render_markers(positions, anchor_id=anchor_id, x_axis_id=x_axis_id,
                    out_path=LAYOUT_PNG, title="Final marker layout")
    print(f"\nLayout map saved to {LAYOUT_PNG}")


if __name__ == "__main__":
    main()
