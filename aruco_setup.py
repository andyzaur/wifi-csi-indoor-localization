#!/usr/bin/env python3
"""ArUco extrinsics calibration — compute camera-to-floor transformation from floor markers.

Usage:
    1. Place 4-8 ArUco markers at measured positions on the floor
    2. Run lens_calibrate.py first (need lens_profile.json)
    3. Run this script — it detects floor markers and computes the transformation
    4. Verify with grid overlay, press 's' to save

The extrinsics are saved to floor_calibration.json and used by aruco_track.py.
"""

import argparse
import json
import sys
import numpy as np
import cv2
from cv2 import aruco

LENS_PROFILE = "lens_profile.json"
FLOOR_CALIBRATION = "floor_calibration.json"
MARKER_LAYOUT = "marker_layout.json"

# Foot-tracking markers worn by the person (IDs match aruco_track.py's
# --marker-right 0 / --marker-left 9 defaults). They are NEVER floor references,
# so the calibrator ignores them even when they are in the camera's view.
FOOT_MARKER_IDS = {0, 9}
GRID_SPACING = 0.50  # meters

# ArUco dictionary — 4x4 with 50 markers is plenty
ARUCO_DICT = aruco.DICT_4X4_50


def load_marker_layout():
    """Load pre-computed marker positions from marker_layout.json if available."""
    try:
        with open(MARKER_LAYOUT) as f:
            data = json.load(f)
        positions = {int(k): tuple(v) for k, v in data["positions_cm"].items()}
        print(f"Loaded {len(positions)} marker positions from {MARKER_LAYOUT}")
        for mid in sorted(positions.keys()):
            x, y = positions[mid]
            print(f"  Marker {mid:2d}: ({x:+7.1f}, {y:+7.1f}) cm")
        return positions
    except FileNotFoundError:
        return None


def load_lens_profile():
    try:
        with open(LENS_PROFILE) as f:
            p = json.load(f)
        mtx = np.array(p["camera_matrix"], dtype=np.float64)
        dist = np.array(p["dist_coeffs"], dtype=np.float64)
        print(f"Lens profile loaded (error: {p['reprojection_error']:.4f} px)")
        return mtx, dist
    except FileNotFoundError:
        print(f"WARNING: {LENS_PROFILE} not found. Running without lens correction.")
        print("  Results will be less accurate. Run lens_calibrate.py first.")
        return None, None


def open_camera(cam_idx):
    if cam_idx is not None:
        # Accept either an integer index ("2") or a URL ("http://127.0.0.1:8080/video")
        source = int(cam_idx) if str(cam_idx).isdigit() else cam_idx
        cap = cv2.VideoCapture(source)
    else:
        cap = None
        for i in [3, 2, 1, 0]:
            c = cv2.VideoCapture(i)
            if c.isOpened():
                cap = c
                cam_idx = i
                break
    if cap is None or not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: index {cam_idx}, {w}x{h}")
    return cap, cam_idx, w, h


def get_floor_marker_positions(layout=None):
    """Get real-world positions for detected markers.

    If `layout` is provided (from marker_layout.json), use those positions for
    any detected marker whose ID is in the layout. Prompt for any missing IDs.
    """
    positions = {}
    ids_needed = sorted(detected_markers.keys())
    layout = layout or {}

    auto = [mid for mid in ids_needed if mid in layout]
    manual = [mid for mid in ids_needed if mid not in layout]

    if auto:
        print(f"\nUsing positions from {MARKER_LAYOUT} for: {auto}")
        for mid in auto:
            positions[mid] = layout[mid]

    if manual:
        print(f"\nNeed manual entry for: {manual}")
        print("Coordinates in CENTIMETERS. Use: x,y (e.g. 0,0)")
        for marker_id in manual:
            while True:
                try:
                    raw = input(f"  Marker ID {marker_id} — x,y in cm: ")
                    parts = raw.replace(" ", "").split(",")
                    x, y = float(parts[0]), float(parts[1])
                    positions[marker_id] = (x, y)
                    break
                except (ValueError, IndexError):
                    print("    Invalid format. Use: x,y (e.g. 0,0 or 300,150)")

    return positions


def compute_floor_homography(marker_corners_px, marker_positions_cm):
    """Compute homography from pixel coordinates to floor coordinates (cm)."""
    src_points = []  # pixel coords
    dst_points = []  # floor coords in cm

    for marker_id, corners in marker_corners_px.items():
        if marker_id not in marker_positions_cm:
            continue
        # Use the center of each marker
        center_px = corners.mean(axis=0)
        center_world = marker_positions_cm[marker_id]
        src_points.append(center_px)
        dst_points.append(center_world)

    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    if len(src) < 4:
        print(f"ERROR: Need at least 4 markers, found {len(src)}")
        return None, None

    H, status = cv2.findHomography(src, dst)
    H_inv, _ = cv2.findHomography(dst, src)
    return H, H_inv


def pixel_to_floor(px, py, H):
    pt = np.array([px, py, 1.0])
    wp = H @ pt
    if abs(wp[2]) < 1e-10:
        return None
    wp /= wp[2]
    return wp[0], wp[1]


def floor_to_pixel(fx, fy, H_inv):
    pt = np.array([fx, fy, 1.0])
    pp = H_inv @ pt
    if abs(pp[2]) < 1e-10:
        return None
    pp /= pp[2]
    px, py = int(round(pp[0])), int(round(pp[1]))
    if -5000 < px < 5000 and -5000 < py < 5000:
        return px, py
    return None


def draw_grid(frame, H_inv, grid_bounds, grid_spacing):
    x_min, x_max, y_min, y_max = grid_bounds
    xs = np.arange(x_min, x_max + grid_spacing / 2, grid_spacing)
    ys = np.arange(y_min, y_max + grid_spacing / 2, grid_spacing)
    h_frame, w_frame = frame.shape[:2]

    for x in xs:
        p1 = floor_to_pixel(x, y_min, H_inv)
        p2 = floor_to_pixel(x, y_max, H_inv)
        if p1 and p2:
            cv2.line(frame, p1, p2, (0, 255, 0), 1)
    for y in ys:
        p1 = floor_to_pixel(x_min, y, H_inv)
        p2 = floor_to_pixel(x_max, y, H_inv)
        if p1 and p2:
            cv2.line(frame, p1, p2, (0, 255, 0), 1)

    for x in xs:
        for y in ys:
            pt = floor_to_pixel(x, y, H_inv)
            if pt is None:
                continue
            px, py = pt
            if 0 <= px < w_frame and 0 <= py < h_frame:
                cv2.circle(frame, (px, py), 3, (0, 200, 0), -1)
                label = f"{x:.0f},{y:.0f}"
                cv2.putText(frame, label, (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)


detected_markers = {}  # id -> corners (4x2 array)


def main():
    global detected_markers

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", type=str, default=None,
                        help="OpenCV camera index OR URL (e.g. http://127.0.0.1:8080/video)")
    parser.add_argument("--marker-size", type=float, default=10.0,
                        help="Marker size in cm (default: 10)")
    args = parser.parse_args()

    lens_mtx, lens_dist = load_lens_profile()
    marker_layout = load_marker_layout()
    cap, cam_idx, w, h = open_camera(args.camera)

    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    print()
    print("=== FLOOR MARKER SETUP ===")
    print("Place ArUco markers (4x4_50 dictionary) on the floor.")
    print("  SPACE = capture current markers")
    print("  'q'   = quit")
    print()

    H = None
    H_inv = None
    grid_bounds = None
    marker_positions = None
    verified = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if lens_mtx is not None:
            frame = cv2.undistort(frame, lens_mtx, lens_dist, None, lens_mtx)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners_list, ids)
            seen = sorted(ids.flatten().tolist())
            floor = [m for m in seen if m not in FOOT_MARKER_IDS]
            feet = [m for m in seen if m in FOOT_MARKER_IDS]
            status = f"Floor markers ({len(floor)}): {floor}"
            if feet:
                status += f"  | ignoring feet {feet}"
            color = (0, 255, 0)
        else:
            status = "No markers detected"
            color = (0, 0, 255)

        if H is not None:
            draw_grid(frame, H_inv, grid_bounds, GRID_SPACING * 100)
            cv2.putText(frame, "Grid OK? Click window: 's'=save 'r'=redo 'q'=quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("ArUco Floor Setup", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' ') and ids is not None:
            detected_markers.clear()
            for i, marker_id in enumerate(ids.flatten()):
                mid = int(marker_id)
                if mid in FOOT_MARKER_IDS:
                    continue  # worn on the feet, not a floor reference — ignore
                detected_markers[mid] = corners_list[i][0]  # 4x2 array

            if len(detected_markers) < 4:
                print(f"Need at least 4 floor markers, detected {len(detected_markers)}. Add more.")
                continue

            print(f"\nCaptured {len(detected_markers)} markers: {sorted(detected_markers.keys())}")
            marker_positions = get_floor_marker_positions(marker_layout)

            H, H_inv = compute_floor_homography(detected_markers, marker_positions)
            if H is None:
                continue

            # Grid bounds in cm
            xs = [p[0] for p in marker_positions.values()]
            ys = [p[1] for p in marker_positions.values()]
            gs = GRID_SPACING * 100  # cm
            x_min = np.floor(min(xs) / gs) * gs
            x_max = np.ceil(max(xs) / gs) * gs
            y_min = np.floor(min(ys) / gs) * gs
            y_max = np.ceil(max(ys) / gs) * gs
            grid_bounds = (x_min, x_max, y_min, y_max)

            verified = True
            print("\nHomography computed. Check grid overlay. 's' to save, 'r' to redo.")

        elif key == ord('s') and verified:
            calib = {
                "homography": H.tolist(),
                "homography_inv": H_inv.tolist(),
                "marker_positions_cm": {str(k): list(v) for k, v in marker_positions.items()},
                "grid_spacing_cm": GRID_SPACING * 100,
                "grid_bounds_cm": list(grid_bounds),
                "camera_resolution": [w, h],
                "aruco_dict": "DICT_4X4_50",
            }
            with open(FLOOR_CALIBRATION, "w") as f:
                json.dump(calib, f, indent=2)
            print(f"\nFloor calibration saved to {FLOOR_CALIBRATION}")
            break

        elif key == ord('r'):
            detected_markers.clear()
            H = None
            H_inv = None
            grid_bounds = None
            verified = False
            print("\nReset. Capture markers again with SPACE.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
