#!/usr/bin/env python3
"""Plot a tracking session path on top of the floor marker layout.

Usage:
    python3 plot_session.py session_test.csv
    python3 plot_session.py session_test.csv --mode grid    # use snapped grid_x/y instead
    python3 plot_session.py session_test.csv --out path.png

Reads:
    - marker_layout.json   (floor anchors)
    - the CSV from aruco_track.py

Produces:
    - PNG with anchors + tracked path
    - Path color goes blue (start) -> green -> yellow -> red (end), so you can
      see direction over time
"""

import argparse
import csv
import json
import math
import subprocess
import sys

import numpy as np
import cv2


LAYOUT_FILE = "marker_layout.json"


def load_layout():
    with open(LAYOUT_FILE) as f:
        layout = json.load(f)
    return {int(k): tuple(v) for k, v in layout["positions_cm"].items()}


def load_session(path, mode):
    points = []  # list of (timestamp_s, x_cm, y_cm)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("detected") not in ("1", 1):
                continue
            try:
                t = float(row["timestamp_s"])
                if mode == "continuous":
                    x = float(row["x_cm"])
                    y = float(row["y_cm"])
                else:
                    x = float(row["grid_x_cm"])
                    y = float(row["grid_y_cm"])
                points.append((t, x, y))
            except (KeyError, ValueError):
                continue
    return points


def gradient_color(t01):
    """Map t in [0, 1] to a BGR color via OpenCV JET colormap."""
    val = int(round(max(0, min(1, t01)) * 255))
    arr = np.array([[val]], dtype=np.uint8)
    bgr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    return tuple(int(c) for c in bgr[0, 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="CSV log from aruco_track.py")
    parser.add_argument("--mode", choices=["continuous", "grid"], default="continuous",
                        help="continuous = raw x/y, grid = snapped grid_x/y")
    parser.add_argument("--out", "-o", default=None,
                        help="Output PNG (default: <csv_name>_<mode>.png)")
    args = parser.parse_args()

    out_path = args.out or args.csv_file.rsplit(".", 1)[0] + f"_{args.mode}.png"

    anchors = load_layout()
    points = load_session(args.csv_file, args.mode)

    if not points:
        print(f"No detected points in {args.csv_file}.")
        sys.exit(1)

    print(f"Loaded {len(points)} points from {args.csv_file} (mode={args.mode})")
    print(f"Duration: {points[-1][0] - points[0][0]:.1f} s")

    # Bounds — include both anchors and path
    all_x = [p[0] for p in anchors.values()] + [p[1] for p in points]
    all_y = [p[1] for p in anchors.values()] + [p[2] for p in points]
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

    # 50cm grid lines
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

    # Floor anchors (drawn underneath the path)
    for mid, (x, y) in sorted(anchors.items()):
        cx, cy = to_px(x, y)
        cv2.circle(img, (cx, cy), 14, (0, 130, 0), -1)
        cv2.circle(img, (cx, cy), 14, (50, 50, 50), 2)
        cv2.putText(img, str(mid), (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Path with time gradient
    t0 = points[0][0]
    tN = points[-1][0]
    duration = max(tN - t0, 1e-6)

    prev_pixel = None
    for (t, x, y) in points:
        t01 = (t - t0) / duration
        color = gradient_color(t01)
        cur_pixel = to_px(x, y)
        if prev_pixel is not None:
            cv2.line(img, prev_pixel, cur_pixel, color, 2, cv2.LINE_AA)
        cv2.circle(img, cur_pixel, 2, color, -1)
        prev_pixel = cur_pixel

    # Mark start and end clearly
    start_px = to_px(points[0][1], points[0][2])
    end_px = to_px(points[-1][1], points[-1][2])
    cv2.circle(img, start_px, 12, (255, 0, 0), 3)
    cv2.putText(img, "START", (start_px[0] + 15, start_px[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.circle(img, end_px, 12, (0, 0, 255), 3)
    cv2.putText(img, "END", (end_px[0] + 15, end_px[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Color legend (gradient bar bottom-right)
    legend_w, legend_h = 200, 14
    legend_x = W - legend_w - 20
    legend_y = H - 50
    for i in range(legend_w):
        c = gradient_color(i / legend_w)
        cv2.line(img, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_h), c, 1)
    cv2.rectangle(img, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                  (50, 50, 50), 1)
    cv2.putText(img, "t=0", (legend_x - 5, legend_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
    cv2.putText(img, f"t={duration:.1f}s", (legend_x + legend_w - 30, legend_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)

    # Title + summary
    cv2.putText(img, f"Session: {args.csv_file}  ({args.mode})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.putText(img, f"{len(points)} points, {duration:.1f}s",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")
    try:
        subprocess.run(["open", out_path], check=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
