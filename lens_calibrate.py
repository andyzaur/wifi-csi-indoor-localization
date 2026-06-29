#!/usr/bin/env python3
"""Lens calibration — auto-captures chessboard frames across the FOV.

Usage:
    1. Print a chessboard pattern (default: 9x6 inner corners)
    2. Run this script with the camera connected
    3. Move the chessboard through each of the 9 grid regions shown on screen
       — captures happen automatically when the board is detected and held steady
    4. Once all 9 regions have a capture (or enough overall), press 'c' to compute
    5. Review the undistorted preview, press 's' to save

Controls:
    SPACE  manual capture (if auto isn't triggering)
    c      compute calibration (need 8+ captures, ideally 15+)
    r      reset captures
    q      quit

The calibration is saved to lens_profile.json.
"""

import argparse
import json
import sys
import time
import numpy as np
import cv2

LENS_PROFILE = "lens_profile.json"

DEFAULT_COLS = 9
DEFAULT_ROWS = 6
SQUARE_SIZE_MM = 25  # affects scale but not distortion

# Detection downscale factor (faster) — corner refinement uses full res
DETECT_SCALE = 0.5

# Auto-capture: chessboard must be steady for STEADY_FRAMES with low motion
STEADY_FRAMES = 8
STEADY_MAX_MOTION_PX = 4.0
MIN_TIME_BETWEEN_CAPTURES_S = 0.5

# Grid regions for coverage guidance (3x3)
GRID_ROWS = 3
GRID_COLS = 3


def open_camera(cam_arg):
    if cam_arg is not None:
        source = int(cam_arg) if cam_arg.isdigit() else cam_arg
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera: {cam_arg}")
            sys.exit(1)
        return cap
    for i in [3, 2, 1, 0]:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    print("ERROR: No camera found.")
    sys.exit(1)


def find_chessboard_fast(frame_gray, board_size):
    """Try to find chessboard on a downscaled image for speed, refine on full."""
    h, w = frame_gray.shape
    small = cv2.resize(frame_gray, (int(w * DETECT_SCALE), int(h * DETECT_SCALE)))

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    found, corners_small = cv2.findChessboardCorners(small, board_size, flags=flags)
    if not found:
        return False, None

    corners_full = corners_small / DETECT_SCALE

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(frame_gray, corners_full.astype(np.float32),
                                        (11, 11), (-1, -1), criteria)
    return True, corners_refined


def region_of(center, w, h):
    """Return (row, col) of the 3x3 region containing center."""
    cx, cy = center
    col = min(GRID_COLS - 1, int(cx / w * GRID_COLS))
    row = min(GRID_ROWS - 1, int(cy / h * GRID_ROWS))
    return row, col


def draw_coverage_grid(frame, captured_regions, current_region=None):
    """Draw 3x3 grid showing which regions have captures."""
    h, w = frame.shape[:2]
    cell_w = w // GRID_COLS
    cell_h = h // GRID_ROWS

    overlay = frame.copy()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            captured = captured_regions.get((r, c), 0)
            is_current = current_region == (r, c)

            if captured > 0:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), -1)
            elif is_current:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), -1)

    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    for r in range(GRID_ROWS + 1):
        y = min(r * cell_h, h - 1)
        cv2.line(frame, (0, y), (w, y), (255, 255, 255), 1)
    for c in range(GRID_COLS + 1):
        x = min(c * cell_w, w - 1)
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 1)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            count = captured_regions.get((r, c), 0)
            tx = c * cell_w + 10
            ty = r * cell_h + 25
            cv2.putText(frame, str(count), (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", type=str, default=None,
                        help="Camera index or URL (e.g. http://127.0.0.1:8080/video)")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    args = parser.parse_args()

    board_size = (args.cols, args.rows)

    cap = open_camera(args.camera)
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame.")
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"Camera opened: {w}x{h}")
    print()
    print("=== LENS CALIBRATION ===")
    print(f"Chessboard: {args.cols}x{args.rows} inner corners")
    print(f"Auto-capture when chessboard is held steady ({STEADY_FRAMES} frames)")
    print("Move the board through all 9 grid regions for best coverage.")
    print()

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points = []
    img_points = []
    captured_regions = {}
    last_capture_time = 0

    steady_count = 0
    prev_corners = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = find_chessboard_fast(gray, board_size)

        current_region = None
        is_steady = False

        if found:
            center = corners.mean(axis=0).flatten()
            current_region = region_of(center, w, h)

            if prev_corners is not None and prev_corners.shape == corners.shape:
                motion = np.linalg.norm(corners - prev_corners, axis=2).mean()
                if motion < STEADY_MAX_MOTION_PX:
                    steady_count += 1
                else:
                    steady_count = 0
            prev_corners = corners.copy()
            is_steady = steady_count >= STEADY_FRAMES

            cv2.drawChessboardCorners(display, board_size, corners, found)
        else:
            steady_count = 0
            prev_corners = None

        draw_coverage_grid(display, captured_regions, current_region)

        now = time.time()
        can_capture = (found and is_steady and
                       (now - last_capture_time) >= MIN_TIME_BETWEEN_CAPTURES_S)
        if can_capture:
            obj_points.append(objp)
            img_points.append(corners)
            captured_regions[current_region] = captured_regions.get(current_region, 0) + 1
            last_capture_time = now
            steady_count = 0
            print(f"  Auto-captured #{len(obj_points)} (region {current_region})")
            display = cv2.bitwise_not(display)

        regions_covered = len(captured_regions)
        total_captures = len(obj_points)

        if found:
            if is_steady:
                msg = f"STEADY - auto-capturing... (region {current_region})"
                color = (0, 255, 0)
            else:
                bar = int(steady_count / STEADY_FRAMES * 30)
                msg = f"Hold steady: [{'#' * bar}{' ' * (30 - bar)}] (region {current_region})"
                color = (0, 255, 255)
        else:
            msg = "Show chessboard — fill empty grid cells"
            color = (200, 200, 200)

        cv2.putText(display, msg, (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display,
                    f"Captures: {total_captures} | Regions: {regions_covered}/9 | "
                    f"'c'=compute  'r'=reset  'q'=quit",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Lens Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and found:
            obj_points.append(objp)
            img_points.append(corners)
            captured_regions[current_region] = captured_regions.get(current_region, 0) + 1
            last_capture_time = time.time()
            print(f"  Manual capture #{len(obj_points)} (region {current_region})")

        elif key == ord('r'):
            obj_points.clear()
            img_points.clear()
            captured_regions.clear()
            print("Reset.")

        elif key == ord('c'):
            if len(obj_points) < 8:
                print(f"  Need at least 8 captures, have {len(obj_points)}")
                continue

            print(f"\nComputing calibration from {len(obj_points)} frames...")
            ret_calib, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, (w, h), None, None
            )

            total_error = 0
            for i in range(len(obj_points)):
                projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i],
                                                  camera_matrix, dist_coeffs)
                error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
                total_error += error
            mean_error = total_error / len(obj_points)
            print(f"Reprojection error: {mean_error:.4f} px (< 0.5 is good, < 1.0 is acceptable)")

            new_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )

            print("\nUndistorted preview. 's' to save, 'r' to redo, 'q' to quit")
            while True:
                ret2, frame2 = cap.read()
                if not ret2:
                    continue
                undistorted = cv2.undistort(frame2, camera_matrix, dist_coeffs, None, new_matrix)
                rx, ry, rw, rh = roi
                if rw > 0 and rh > 0:
                    cv2.rectangle(undistorted, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)
                cv2.putText(undistorted, "UNDISTORTED  's'=save  'r'=redo  'q'=quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(undistorted, f"Error: {mean_error:.4f} px",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Lens Calibration", undistorted)
                key2 = cv2.waitKey(30) & 0xFF
                if key2 == ord('s'):
                    profile = {
                        "camera_matrix": camera_matrix.tolist(),
                        "dist_coeffs": dist_coeffs.tolist(),
                        "new_camera_matrix": new_matrix.tolist(),
                        "roi": list(roi),
                        "resolution": [w, h],
                        "reprojection_error": mean_error,
                        "num_captures": len(obj_points),
                    }
                    with open(LENS_PROFILE, "w") as f:
                        json.dump(profile, f, indent=2)
                    print(f"Saved to {LENS_PROFILE}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key2 == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key2 == ord('r'):
                    break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
