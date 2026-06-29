"""FROZEN golden reference for ArUco-tracker parity tests.

This is the ORIGINAL aruco_track.main() per-frame loop, copied VERBATIM from the
pre-refactor on-disk version of aruco_track.py (the projection -> offsets ->
two-foot fusion -> grid snap -> camera.csv/corners.csv/keyframes path). It runs
on a frame iterator and writes the same stdout (collected to a list) + CSVs.

It exists so the parity test has an INDEPENDENT golden generator that does NOT
import ArucoTracker — otherwise the parity assertion would be circular. The pure
geometry helpers (pixel_to_floor, marker_axes, fuse_feet, ...) are unchanged by
the refactor, so the reference imports them from aruco_track; only the LOOP body
(which moved into the class) is frozen here.
"""

import csv
import os
import time

import numpy as np
import cv2
from cv2 import aruco

from aruco_track import (
    ARUCO_DICT,
    load_lens_profile, load_floor_calibration,
    pixel_to_floor, floor_to_pixel, snap_to_grid,
    marker_axes, body_center_from_axes, ema_unit,
    ramp_weight, fuse_feet, draw_grid,
    legacy_frame_source,
)


def run_reference(video, log, marker_right=0, marker_left=9,
                  offset_side=20.0, offset_back=-15.0, orient_smooth=0.3,
                  workers=6, no_display=True, display_scale=0.5,
                  keyframe_stride=30, foot_ramp=8, out=None):
    """Reproduce the original aruco_track.main() for a --video run.

    `out` is a list that collects every stdout line (so the test can compare
    without capturing the real stdout). Returns nothing; writes log/corners CSV.
    """
    if out is None:
        out = []

    def emit(msg):
        out.append(msg)

    lens_mtx, lens_dist = load_lens_profile()
    H, H_inv, grid_bounds, grid_spacing, floor_marker_ids = load_floor_calibration()

    wearable_right = marker_right
    wearable_left = marker_left
    emit(f"Tracking foot markers: right={wearable_right} left={wearable_left}")
    emit(f"Floor anchor markers (ignored): {sorted(floor_marker_ids)}")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    emit(f"Video: {video} ({total_frames} frames, {fps:.1f} fps)")
    is_live = False

    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(ARUCO_DICT),
                                   aruco.DetectorParameters())

    csv_file = None
    csv_writer = None
    if log:
        os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
        csv_file = open(log, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "timestamp_s", "x_cm", "y_cm",
                             "grid_x_cm", "grid_y_cm", "detected",
                             "n_markers", "method",
                             "right_x", "right_y", "left_x", "left_y"])

    corners_file = corners_writer = keyframe_dir = None
    if log:
        session_dir = os.path.dirname(os.path.abspath(log))
        corners_file = open(os.path.join(session_dir, "corners.csv"), "w", newline="")
        corners_writer = csv.writer(corners_file)
        corners_writer.writerow(["frame", "timestamp_s", "marker_id",
                                 "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"])
        if keyframe_stride > 0:
            keyframe_dir = os.path.join(session_dir, "keyframes")
            os.makedirs(keyframe_dir, exist_ok=True)

    emit("")
    emit("=== ARUCO TRACKER ===")
    emit("  'q' = quit")
    emit("")

    frame_num = 0
    last_print = 0
    detections = 0
    total_processed = 0
    fps_t0 = time.time()
    fps_n = 0
    show = not no_display

    frame_iter = legacy_frame_source(cap, lens_mtx, lens_dist, detector, is_live)

    smooth_fwd = {"right": None, "left": None}
    smooth_rt = {"right": None, "left": None}
    last_seen_frame = {"right": -10**9, "left": -10**9}
    foot_w = {"right": 0.0, "left": 0.0}
    last_est = {"right": None, "left": None}
    ORIENT_RESET_GAP = 20
    for frame, corners_list, ids, timestamp in frame_iter:
        frame_num += 1
        total_processed += 1
        h, w = frame.shape[:2]

        active_cell = None
        floor_pos = None
        method = "none"
        n_markers = 0
        mc_right = None
        mc_left = None
        est_right = None
        est_left = None

        if ids is not None:
            if show:
                aruco.drawDetectedMarkers(frame, corners_list, ids)

            ids_flat = ids.flatten()
            for foot, target_id in (("right", wearable_right), ("left", wearable_left)):
                if target_id < 0 or target_id not in ids_flat:
                    continue
                i = int(np.where(ids_flat == target_id)[0][0])
                corners_px = corners_list[i][0]

                floor_corners = []
                ok = True
                for cx_px, cy_px in corners_px:
                    fp = pixel_to_floor(cx_px, cy_px, H)
                    if fp is None:
                        ok = False
                        break
                    floor_corners.append(fp)
                if not ok:
                    continue
                floor_corners = np.array(floor_corners)

                axes = marker_axes(floor_corners)
                if axes is None:
                    continue
                center, fwd_raw, rt_raw = axes
                if frame_num - last_seen_frame[foot] > ORIENT_RESET_GAP:
                    sf, sr = fwd_raw, rt_raw
                else:
                    sf = ema_unit(smooth_fwd[foot], fwd_raw, orient_smooth)
                    sr = ema_unit(smooth_rt[foot], rt_raw, orient_smooth)
                smooth_fwd[foot], smooth_rt[foot] = sf, sr
                last_seen_frame[foot] = frame_num

                est = body_center_from_axes(center, sf, sr, foot,
                                            offset_side, offset_back)
                if foot == "right":
                    est_right = est
                    mc_right = center
                else:
                    est_left = est
                    mc_left = center

        for foot, est in (("right", est_right), ("left", est_left)):
            foot_w[foot] = ramp_weight(foot_w[foot], est is not None,
                                       foot_ramp, foot_ramp)
            if est is not None:
                last_est[foot] = est
        position, n_markers, method = fuse_feet(last_est["right"], foot_w["right"],
                                                last_est["left"], foot_w["left"])
        if position is not None:
            fx, fy = float(position[0]), float(position[1])
            gx, gy = snap_to_grid(fx, fy, grid_spacing, grid_bounds)
            active_cell = (gx, gy)
            detections += 1
            floor_pos = (fx, fy)

            if show:
                for mc, color, label in ((mc_right, (0, 0, 255), "R"),
                                         (mc_left, (255, 0, 255), "L")):
                    if mc is not None:
                        p = floor_to_pixel(mc[0], mc[1], H_inv)
                        if p:
                            cv2.circle(frame, p, 6, color, -1)
                            cv2.putText(frame, label, (p[0] + 8, p[1] - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                bc = floor_to_pixel(fx, fy, H_inv)
                if bc:
                    cv2.circle(frame, bc, 14, (0, 200, 0), 3)
                    cv2.putText(frame, f"YOU ({fx:.0f},{fy:.0f}) {method}",
                                (bc[0] + 16, bc[1] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

            now = time.time()
            if now - last_print >= 0.5:
                emit(f"You: ({fx:+7.1f}, {fy:+7.1f}) cm -> grid ({gx:.0f}, {gy:.0f}) "
                     f"| {method} ({n_markers} marker(s))")
                last_print = now

        if csv_writer:
            rx = f"{mc_right[0]:.1f}" if mc_right is not None else ""
            ry = f"{mc_right[1]:.1f}" if mc_right is not None else ""
            lx = f"{mc_left[0]:.1f}" if mc_left is not None else ""
            ly = f"{mc_left[1]:.1f}" if mc_left is not None else ""
            if floor_pos:
                fx, fy = floor_pos
                gx, gy = active_cell
                csv_writer.writerow([frame_num, f"{timestamp:.4f}",
                                     f"{fx:.1f}", f"{fy:.1f}",
                                     f"{gx:.0f}", f"{gy:.0f}", 1,
                                     n_markers, method, rx, ry, lx, ly])
            else:
                csv_writer.writerow([frame_num, f"{timestamp:.4f}",
                                     "", "", "", "", 0,
                                     n_markers, method, rx, ry, lx, ly])
            if frame_num % 30 == 0:
                csv_file.flush()

        if corners_writer is not None and ids is not None:
            for mi, mid in enumerate(ids.flatten()):
                c = corners_list[mi][0]
                corners_writer.writerow([frame_num, f"{timestamp:.4f}", int(mid)]
                                        + [f"{v:.1f}" for xy in c for v in xy])
            if frame_num % 30 == 0:
                corners_file.flush()
        if keyframe_dir is not None and frame_num % keyframe_stride == 0:
            cv2.imwrite(os.path.join(keyframe_dir, f"f{frame_num:06d}.jpg"),
                        frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        if show:
            draw_grid(frame, H_inv, grid_bounds, grid_spacing, active_cell)

            if floor_pos:
                fx, fy = floor_pos
                gx, gy = active_cell
                status = (f"{method} ({n_markers}): ({fx:.0f},{fy:.0f})cm | "
                          f"Grid: ({gx:.0f},{gy:.0f})")
                color = (0, 255, 0)
            else:
                status = "No foot marker visible"
                color = (0, 0, 255)

            cv2.putText(frame, status, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not is_live:
                progress = f"Frame {frame_num}/{total_frames}"
                cv2.putText(frame, progress, (w - 250, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            disp = frame if display_scale >= 0.999 else cv2.resize(
                frame, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
            cv2.imshow("ArUco Tracker", disp)

        fps_n += 1
        now_fps = time.time()
        if now_fps - fps_t0 >= 2.0:
            emit(f"[camera] {fps_n / (now_fps - fps_t0):.1f} fps")
            fps_t0 = now_fps
            fps_n = 0

        if show:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and not is_live:
                cv2.waitKey(0)

    if not is_live:
        emit(f"\nVideo complete. {detections}/{total_processed} frames with detection.")

    if csv_file:
        csv_file.close()
        emit(f"Position log saved to {log}")
    if corners_file:
        corners_file.close()
        kf = len(os.listdir(keyframe_dir)) if keyframe_dir else 0
        emit(f"Audit trail saved: corners.csv + {kf} keyframe(s)")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return out
