#!/usr/bin/env python3
"""ArUco position tracker — tracks a wearable marker and maps to floor grid.

Usage:
    1. Run lens_calibrate.py (if not done) → lens_profile.json
    2. Run aruco_setup.py → floor_calibration.json
    3. Run this script
    4. Walk around with the wearable ArUco marker visible to camera

Outputs live position to screen and optionally logs to CSV.

Supports two input modes:
    --camera N    : live camera feed (default)
    --video FILE  : process a recorded video file (for offline labeling)
"""

import argparse
import collections
import csv
import itertools
import json
import os
import socket as _socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
import numpy as np
import cv2
from cv2 import aruco

LENS_PROFILE = "lens_profile.json"
FLOOR_CALIBRATION = "floor_calibration.json"
ARUCO_DICT = aruco.DICT_4X4_50

# Wearable marker IDs — configurable via --marker-right / --marker-left.
# Right foot keeps the historical default (0). Floor anchors use 1-8, so the
# left foot uses 9 (DICT_4X4_50 supports IDs 0-49). Set an ID to -1 to disable
# that foot (single-marker mode).
DEFAULT_RIGHT_ID = 0
DEFAULT_LEFT_ID = 9

# Floor anchor marker IDs that should be ignored for tracking
# (loaded from floor_calibration.json)
floor_marker_ids = set()


# ─── Live MJPEG pipeline (threaded, to hit the full camera framerate) ────────
# cv2.VideoCapture's FFmpeg MJPEG-over-HTTP reader caps ~15 fps because it
# couples network read + decode on one thread. We bypass it: read raw bytes
# from the socket, split JPEG frames, and decode+undistort+detect in a thread
# pool (OpenCV releases the GIL in native calls → real parallelism).

def mjpeg_blobs(url, timeout=5.0, reconnect=True, max_failures=120):
    """Yield (jpeg_bytes, arrival_wall_time) from an MJPEG-over-HTTP stream.

    The timestamp is captured the moment the frame's bytes arrive, NOT when it
    is later decoded — this keeps CSI<->camera alignment accurate despite
    pipeline latency.

    Resilient to transient stream gaps: a socket timeout / closed connection /
    network error reconnects (with a short backoff and a visible warning)
    instead of ending the capture. This is what protects a long recording from
    a momentary iproxy/phone hiccup. Gives up only after `max_failures`
    consecutive reconnect attempts (~tens of seconds), so a truly dead camera
    eventually ends the loop and lets the CSV close cleanly.
    """
    u = urlparse(url)
    host, port, path = u.hostname, (u.port or 80), (u.path or "/")

    def connect():
        s = _socket.create_connection((host, port), timeout=timeout)
        s.sendall(f"GET {path} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
        s.settimeout(timeout)
        hdr = b""
        while b"\r\n\r\n" not in hdr:
            chunk = s.recv(4096)
            if not chunk:
                raise ConnectionError("stream closed during HTTP headers")
            hdr += chunk
        return s, hdr.split(b"\r\n\r\n", 1)[1]

    failures = 0
    while True:
        try:
            sock, buf = connect()
            failures = 0  # a fresh good connection resets the counter
        except OSError as e:
            failures += 1
            if not reconnect or failures > max_failures:
                raise
            print(f"[camera] connect failed ({type(e).__name__}); retry {failures}/{max_failures}...")
            time.sleep(0.5)
            continue

        try:
            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
                if start != -1 and end != -1:
                    yield buf[start:end + 2], time.time()
                    buf = buf[end + 2:]
                else:
                    chunk = sock.recv(131072)
                    if not chunk:
                        raise ConnectionError("stream closed")
                    buf += chunk
        except OSError as e:
            if not reconnect:
                return
            failures += 1
            print(f"[camera] stream gap ({type(e).__name__}); reconnecting {failures}/{max_failures}...")
            time.sleep(0.3)
        finally:
            try:
                sock.close()
            except Exception:
                pass

        if failures > max_failures:
            print(f"[camera] gave up after {max_failures} reconnect attempts — camera gone?")
            return


_tls = threading.local()


def _thread_detector():
    """One ArucoDetector per worker thread (detectMarkers is not guaranteed
    thread-safe on a shared detector)."""
    d = getattr(_tls, "detector", None)
    if d is None:
        d = aruco.ArucoDetector(aruco.getPredefinedDictionary(ARUCO_DICT),
                                aruco.DetectorParameters())
        _tls.detector = d
    return d


def _decode_detect(item, maps):
    """Pool worker: (jpeg_bytes, ts) -> (frame, corners, ids, ts)."""
    jpeg, ts = item
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None
    if maps[0] is not None:
        frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _thread_detector().detectMarkers(gray)
    return frame, corners, ids, ts


def _ordered_pipeline(stream, fn, workers):
    """Run fn over an (infinite) stream in a thread pool, yielding results in
    INPUT ORDER with bounded concurrency. Unlike Executor.map, this does not
    eagerly drain the stream, so it works with a live/endless source."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inflight = collections.deque()
        depth = workers + 2

        def fill():
            while len(inflight) < depth:
                try:
                    inflight.append(pool.submit(fn, next(stream)))
                except StopIteration:
                    break

        fill()
        while inflight:
            result = inflight.popleft().result()
            fill()
            if result is not None:
                yield result


def threaded_frame_source(url, lens_mtx, lens_dist, workers):
    """Yield (frame, corners, ids, timestamp) from the threaded MJPEG pipeline."""
    blobs = mjpeg_blobs(url)
    first = next(blobs)  # peek one frame to size the undistortion maps once
    frame0 = cv2.imdecode(np.frombuffer(first[0], np.uint8), cv2.IMREAD_COLOR)
    if lens_mtx is not None and frame0 is not None:
        fh, fw = frame0.shape[:2]
        m1, m2 = cv2.initUndistortRectifyMap(
            lens_mtx, lens_dist, None, lens_mtx, (fw, fh), cv2.CV_16SC2)
        maps = (m1, m2)
    else:
        maps = (None, None)
    stream = itertools.chain([first], blobs)
    yield from _ordered_pipeline(stream, lambda it: _decode_detect(it, maps), workers)


def legacy_frame_source(cap, lens_mtx, lens_dist, detector, is_live):
    """Yield (frame, corners, ids, timestamp) via cv2.VideoCapture — used for
    --video file mode and the --legacy live fallback. Single-threaded, mirrors
    the original path."""
    maps = [None, None]
    frame_num = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while True:
        ret, frame = cap.read()
        if not ret:
            return
        ts = time.time() if is_live else frame_num / fps
        if lens_mtx is not None:
            if maps[0] is None:
                fh, fw = frame.shape[:2]
                maps[0], maps[1] = cv2.initUndistortRectifyMap(
                    lens_mtx, lens_dist, None, lens_mtx, (fw, fh), cv2.CV_16SC2)
            frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        frame_num += 1
        yield frame, corners, ids, ts


def load_lens_profile():
    try:
        with open(LENS_PROFILE) as f:
            p = json.load(f)
        mtx = np.array(p["camera_matrix"], dtype=np.float64)
        dist = np.array(p["dist_coeffs"], dtype=np.float64)
        return mtx, dist
    except FileNotFoundError:
        return None, None


def load_floor_calibration():
    try:
        with open(FLOOR_CALIBRATION) as f:
            c = json.load(f)
        H = np.array(c["homography"], dtype=np.float64)
        H_inv = np.array(c["homography_inv"], dtype=np.float64)
        grid_bounds = tuple(c["grid_bounds_cm"])
        grid_spacing = c["grid_spacing_cm"]
        floor_ids = set(int(k) for k in c["marker_positions_cm"].keys())
        return H, H_inv, grid_bounds, grid_spacing, floor_ids
    except FileNotFoundError:
        print(f"ERROR: {FLOOR_CALIBRATION} not found. Run aruco_setup.py first.")
        sys.exit(1)


def _load_floor_calibration_raising():
    """Same as load_floor_calibration() but RAISES FileNotFoundError instead of
    printing + sys.exit. Used by ArucoTracker so importers get an exception they
    can handle; the CLI shim maps it back to the original message + exit code."""
    with open(FLOOR_CALIBRATION) as f:
        c = json.load(f)
    H = np.array(c["homography"], dtype=np.float64)
    H_inv = np.array(c["homography_inv"], dtype=np.float64)
    grid_bounds = tuple(c["grid_bounds_cm"])
    grid_spacing = c["grid_spacing_cm"]
    floor_ids = set(int(k) for k in c["marker_positions_cm"].keys())
    return H, H_inv, grid_bounds, grid_spacing, floor_ids


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


def snap_to_grid(fx, fy, grid_spacing, grid_bounds):
    x_min, x_max, y_min, y_max = grid_bounds
    gx = round((fx - x_min) / grid_spacing) * grid_spacing + x_min
    gy = round((fy - y_min) / grid_spacing) * grid_spacing + y_min
    gx = max(x_min, min(x_max, gx))
    gy = max(y_min, min(y_max, gy))
    return gx, gy


def marker_axes(floor_corners):
    """Return (center, forward_unit, right_unit) for a marker's 4 floor corners,
    or None if degenerate. Corner order TL, TR, BR, BL; forward = toes direction."""
    floor_corners = np.asarray(floor_corners, dtype=float)
    center = floor_corners.mean(axis=0)
    top_mid = (floor_corners[0] + floor_corners[1]) / 2
    bot_mid = (floor_corners[2] + floor_corners[3]) / 2
    left_mid = (floor_corners[0] + floor_corners[3]) / 2
    right_mid = (floor_corners[1] + floor_corners[2]) / 2
    forward = top_mid - bot_mid
    right = right_mid - left_mid
    fwd_len = np.linalg.norm(forward)
    rt_len = np.linalg.norm(right)
    if fwd_len < 1e-3 or rt_len < 1e-3:  # cm; sub-mm edge = degenerate marker
        return None
    return center, forward / fwd_len, right / rt_len


def body_center_from_axes(center, forward_unit, right_unit, side, offset_side, offset_back):
    """Apply the per-foot body-center offset to precomputed marker axes.

    side "right" → body center to the marker's LEFT; "left" → to its RIGHT.
    offset_side: magnitude (cm) toward body center; offset_back: signed (cm)
    along forward (negative = toward heel).
    """
    side_sign = -1.0 if side == "right" else 1.0
    return center + side_sign * offset_side * right_unit + offset_back * forward_unit


def estimate_body_center(floor_corners, side, offset_side, offset_back):
    """Estimate body center (floor cm) from one foot marker's 4 floor-projected
    corners. Thin wrapper over marker_axes + body_center_from_axes. Returns
    np.ndarray (x, y) or None if the marker is degenerate."""
    axes = marker_axes(floor_corners)
    if axes is None:
        return None
    return body_center_from_axes(*axes, side, offset_side, offset_back)


def ema_unit(prev, new, alpha):
    """Exponential moving average of a unit vector, renormalized. `alpha` in
    (0,1] is the weight on the new sample (1 = no smoothing). prev=None returns
    new unchanged. Foot orientation changes slowly, so this de-noises the
    marker's facing direction that drives the body-center offset.

    Blending then renormalizing is a small-angle approximation of true angular
    (slerp) smoothing — accurate enough for the tiny per-frame orientation
    deltas of a slowly-rotating foot."""
    if prev is None:
        return new
    blended = alpha * new + (1.0 - alpha) * prev
    n = np.linalg.norm(blended)
    if n < 1e-9:
        return new
    return blended / n


def combine_foot_estimates(est_right, est_left):
    """Combine per-foot body-center estimates into a final position.

    Each estimate is an (x, y) np.ndarray or None. Returns
    (position, n_markers, method) where position is an np.ndarray or None and
    method is one of "both", "right_only", "left_only", "none".
    """
    ests = [e for e in (est_right, est_left) if e is not None]
    if not ests:
        return None, 0, "none"
    position = np.mean(ests, axis=0)
    if est_right is not None and est_left is not None:
        method = "both"
    elif est_right is not None:
        method = "right_only"
    else:
        method = "left_only"
    return position, len(ests), method


def ramp_weight(w, present, ramp_up=8, ramp_down=8):
    """Per-frame confidence weight in [0,1] for one foot's marker.

    Ramps toward 1 while the marker is present, decays toward 0 while absent,
    over `ramp_up`/`ramp_down` frames (~8 ≈ 0.25 s at 30 fps). This one
    mechanism gives BOTH the hysteresis (a one-frame dropout barely moves the
    weight, so we don't enter/leave "both" on a flicker) AND the smooth blend on
    (re)acquire that a hard mode-switch lacks.
    """
    step = 1.0 / max(1, ramp_up if present else ramp_down)
    return float(min(1.0, max(0.0, (w + step) if present else (w - step))))


def fuse_feet(est_right, w_right, est_left, w_left, on_threshold=0.5):
    """Confidence-weighted body center from per-foot estimates + ramp weights.

    Each foot contributes its (possibly last-known) estimate scaled by its
    weight, so the fused (x, y) ramps smoothly as a foot's weight rises/falls —
    no jump when switching between one- and two-foot tracking. Returns
    (position or None, n_markers, method); method/n_markers reflect which feet
    currently carry real weight (>= on_threshold), so they stay an honest record
    for label-quality stratification. Position label is left RAW (not EMA'd).
    """
    parts = [(np.asarray(e, dtype=float), w)
             for e, w in ((est_right, w_right), (est_left, w_left))
             if e is not None and w > 1e-3]
    if not parts:
        return None, 0, "none"
    wsum = sum(w for _, w in parts)
    position = sum(w * e for e, w in parts) / wsum
    r_on = est_right is not None and w_right >= on_threshold
    l_on = est_left is not None and w_left >= on_threshold
    if r_on and l_on:
        return position, 2, "both"
    if r_on:
        return position, 1, "right_only"
    if l_on:
        return position, 1, "left_only"
    # Transitional: weights present but below threshold (just (re)acquired or
    # decaying after a dropout). Still emit a position; label by dominant foot.
    both_ramp = w_right > 1e-3 and w_left > 1e-3
    method = "both" if both_ramp else ("right_only" if w_right >= w_left else "left_only")
    return position, len(parts), method


def draw_grid(frame, H_inv, grid_bounds, grid_spacing, active_cell=None):
    x_min, x_max, y_min, y_max = grid_bounds
    # Grid lines are drawn at cell BOUNDARIES (between cell centers), so they
    # align with the highlighted active cell which is centered on snapped points.
    hs = grid_spacing / 2
    xs = np.arange(x_min - hs, x_max + hs + grid_spacing / 4, grid_spacing)
    ys = np.arange(y_min - hs, y_max + hs + grid_spacing / 4, grid_spacing)
    h_frame, w_frame = frame.shape[:2]

    for x in xs:
        p1 = floor_to_pixel(x, y_min - hs, H_inv)
        p2 = floor_to_pixel(x, y_max + hs, H_inv)
        if p1 and p2:
            cv2.line(frame, p1, p2, (0, 255, 0), 1)
    for y in ys:
        p1 = floor_to_pixel(x_min - hs, y, H_inv)
        p2 = floor_to_pixel(x_max + hs, y, H_inv)
        if p1 and p2:
            cv2.line(frame, p1, p2, (0, 255, 0), 1)

    # Highlight active cell
    if active_cell is not None:
        ax, ay = active_cell
        hs = grid_spacing / 2
        corners = [
            floor_to_pixel(ax - hs, ay - hs, H_inv),
            floor_to_pixel(ax + hs, ay - hs, H_inv),
            floor_to_pixel(ax + hs, ay + hs, H_inv),
            floor_to_pixel(ax - hs, ay + hs, H_inv),
        ]
        if all(c is not None for c in corners):
            pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            # Blend only the cell's bounding-box ROI (clamped to the frame) rather
            # than copying + blending the whole frame. Pixel-identical (outside the
            # poly the blend is 0.3*p+0.7*p = p), but avoids an 8MP copy+addWeighted
            # every frame during tracking — the dominant overlay cost at full res.
            x0 = max(0, int(pts[:, 0, 0].min()))
            y0 = max(0, int(pts[:, 0, 1].min()))
            x1 = min(w_frame, int(pts[:, 0, 0].max()) + 1)
            y1 = min(h_frame, int(pts[:, 0, 1].max()) + 1)
            if x1 > x0 and y1 > y0:
                roi = frame[y0:y1, x0:x1]
                overlay = roi.copy()
                cv2.fillPoly(overlay, [pts - np.array([[[x0, y0]]], dtype=np.int32)],
                             (0, 255, 255))
                cv2.addWeighted(overlay, 0.3, roi, 0.7, 0, roi)

        cp = floor_to_pixel(ax, ay, H_inv)
        if cp:
            label = f"({ax:.0f}, {ay:.0f})cm"
            cv2.putText(frame, label, (cp[0] - 40, cp[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)


# ─── Importable, controllable tracker API ────────────────────────────────────
# The per-frame projection/offset/fusion/grid/CSV/imshow loop lives in
# ArucoTracker so a GUI (or a test) can drive it without re-running the CLI.
# The class stays GUI-agnostic: callbacks are plain Python callables (default
# None), and with the default knobs its behavior — stdout, CSV bytes, the
# imshow window, the q-to-quit — is byte-identical to running the script.


@dataclass(frozen=True)
class PositionState:
    detected: bool
    x_cm: Optional[float]
    y_cm: Optional[float]
    grid_x_cm: Optional[float]
    grid_y_cm: Optional[float]
    n_markers: int
    method: str
    right_center: Optional[tuple]
    left_center: Optional[tuple]
    fps: float


@dataclass
class FrameResult:
    frame_num: int
    timestamp_s: float
    preview_bgr: Optional[np.ndarray]   # full-res frame NOT emitted
    position: PositionState


class ArucoTracker:
    """Drive the ArUco position-tracking loop from code.

    With the default knobs (owns_window=True, display=True, undistort="full",
    emit_preview=False) this reproduces the CLI byte-for-byte: same stdout, same
    camera.csv / corners.csv / keyframes, the same imshow window and q-to-quit.
    With owns_window=False it never touches cv2.imshow/waitKey/destroyAllWindows
    so a host (GUI) owns the window. on_frame / on_position fire per processed
    frame; on_log (when given) receives every stdout line instead of print().
    """

    def __init__(self, camera=None, video=None,
                 marker_right=DEFAULT_RIGHT_ID, marker_left=DEFAULT_LEFT_ID,
                 log=None, offset_side=20.0, offset_back=-15.0,
                 orient_smooth=0.3, legacy=False, workers=6,
                 keyframe_stride=30, foot_ramp=8, display=True,
                 owns_window=True, display_scale=0.5, undistort="full",
                 emit_preview=False, on_frame=None, on_position=None,
                 on_log=None):
        global floor_marker_ids

        if undistort == "corners":
            raise NotImplementedError(
                'undistort="corners" is deferred to a later phase')
        if undistort != "full":
            raise ValueError(f"unknown undistort mode: {undistort!r}")

        self.camera = camera
        self.video = video
        self.marker_right = marker_right
        self.marker_left = marker_left
        self.log = log
        self.offset_side = offset_side
        self.offset_back = offset_back
        self.orient_smooth = orient_smooth
        self.legacy = legacy
        self.workers = workers
        self.keyframe_stride = keyframe_stride
        self.foot_ramp = foot_ramp
        self.display = display
        self.owns_window = owns_window
        self.display_scale = display_scale
        self.undistort = undistort
        self.emit_preview = emit_preview
        self.on_frame = on_frame
        self.on_position = on_position
        self.on_log = on_log

        # Drives the imshow window + q-to-quit. With owns_window=False the host
        # owns the window, so the tracker never calls cv2.imshow/waitKey.
        self._show = self.display
        self._owns_window = self.owns_window

        self.lens_mtx, self.lens_dist = load_lens_profile()
        (self.H, self.H_inv, self.grid_bounds,
         self.grid_spacing, floor_marker_ids) = _load_floor_calibration_raising()
        self.floor_marker_ids = floor_marker_ids

        self.wearable_right = self.marker_right
        self.wearable_left = self.marker_left
        for label, mid in (("right", self.wearable_right), ("left", self.wearable_left)):
            if mid >= 0 and mid in self.floor_marker_ids:
                raise ValueError(
                    f"--marker-{label} ID {mid} collides with a floor anchor "
                    f"({sorted(self.floor_marker_ids)}). Pick a free ID.")
        if self.wearable_right < 0 and self.wearable_left < 0:
            raise ValueError(
                "At least one of --marker-right / --marker-left must be enabled.")
        if self.wearable_right >= 0 and self.wearable_right == self.wearable_left:
            raise ValueError("--marker-right and --marker-left must be different IDs.")

        # Per-frame source / IO / loop state (populated in start()).
        self._cap = None
        self._total_frames = 0
        self._use_threaded = False
        self._stream_url = None
        self._is_live = True
        self._detector = None
        self._csv_file = None
        self._csv_writer = None
        self._corners_file = None
        self._corners_writer = None
        self._keyframe_dir = None
        self._frame_iter = None
        self._stop = False
        self._started = False

    def _emit_log(self, msg):
        """Route a stdout line through on_log (when set) else print() verbatim."""
        if self.on_log is None:
            print(msg)
        else:
            self.on_log(msg)

    def start(self):
        """Open the frame source, CSV/audit files and the detector, and emit the
        banner — everything the loop in run_forever() needs. Mirrors the CLI
        setup (and its stdout) exactly. Construct → start() → run_forever()."""
        self._emit_log(f"Tracking foot markers: right={self.wearable_right} left={self.wearable_left}")
        self._emit_log(f"Floor anchor markers (ignored): {sorted(self.floor_marker_ids)}")

        # ─── Decide frame source ────────────────────────────────────────────
        cap = None
        total_frames = 0
        use_threaded = False
        stream_url = None
        if self.video:
            cap = cv2.VideoCapture(self.video)
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {self.video}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._emit_log(f"Video: {self.video} ({total_frames} frames, {fps:.1f} fps)")
            is_live = False
        else:
            is_live = True
            url_like = (self.camera is not None and not str(self.camera).isdigit()
                        and str(self.camera).startswith("http"))
            if url_like and not self.legacy:
                # Fast path: raw-socket MJPEG + threaded decode/undistort/detect.
                use_threaded = True
                stream_url = self.camera
                self._emit_log(f"Live stream (threaded MJPEG, {self.workers} workers): {stream_url}")
            else:
                # Fallback: integer webcam index, or --legacy on a URL → VideoCapture.
                if self.camera is not None:
                    source = int(self.camera) if str(self.camera).isdigit() else self.camera
                    cap = cv2.VideoCapture(source)
                else:
                    cap = None
                    for i in [3, 2, 1, 0]:
                        c = cv2.VideoCapture(i)
                        if c.isOpened():
                            cap = c
                            break
                if cap is None or not cap.isOpened():
                    raise FileNotFoundError("Cannot open camera.")
                if self.legacy:
                    self._emit_log("Live stream (legacy single-threaded VideoCapture path)")

        detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(ARUCO_DICT),
                                       aruco.DetectorParameters())

        # CSV logger
        csv_file = None
        csv_writer = None
        if self.log:
            os.makedirs(os.path.dirname(self.log) or ".", exist_ok=True)
            csv_file = open(self.log, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["frame", "timestamp_s", "x_cm", "y_cm",
                                 "grid_x_cm", "grid_y_cm", "detected",
                                 "n_markers", "method",
                                 "right_x", "right_y", "left_x", "left_y"])

        # Audit trail: raw ArUco corner pixels for EVERY detected marker
        # (tiny — lets us re-derive any position offline and re-check the homography
        # without video) + sparse keyframe JPEGs for visual spot-checks. Both live in
        # the session dir so they travel with the data.
        corners_file = corners_writer = keyframe_dir = None
        if self.log:
            session_dir = os.path.dirname(os.path.abspath(self.log))
            corners_file = open(os.path.join(session_dir, "corners.csv"), "w", newline="")
            corners_writer = csv.writer(corners_file)
            corners_writer.writerow(["frame", "timestamp_s", "marker_id",
                                     "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"])
            if self.keyframe_stride > 0:
                keyframe_dir = os.path.join(session_dir, "keyframes")
                os.makedirs(keyframe_dir, exist_ok=True)

        self._emit_log("")
        self._emit_log("=== ARUCO TRACKER ===")
        self._emit_log("  'q' = quit")
        self._emit_log("")

        # The frame source yields (frame, corners, ids, timestamp). Decode,
        # undistort and ArUco detection all happen inside it (threaded for the live
        # HTTP stream); the timestamp is captured at frame ARRIVAL so CSI<->camera
        # alignment stays accurate despite pipeline latency. Everything below the
        # loop header is the original, validated projection/offset/grid/CSV logic.
        if use_threaded:
            frame_iter = threaded_frame_source(stream_url, self.lens_mtx, self.lens_dist, self.workers)
        else:
            frame_iter = legacy_frame_source(cap, self.lens_mtx, self.lens_dist, detector, is_live)

        self._cap = cap
        self._total_frames = total_frames
        self._use_threaded = use_threaded
        self._stream_url = stream_url
        self._is_live = is_live
        self._detector = detector
        self._csv_file = csv_file
        self._csv_writer = csv_writer
        self._corners_file = corners_file
        self._corners_writer = corners_writer
        self._keyframe_dir = keyframe_dir
        self._frame_iter = frame_iter
        self._stop = False
        self._started = True

    def stop(self, join_timeout=2.0):
        """Request the loop to stop after the current frame."""
        self._stop = True

    def _build_preview(self, frame):
        """Downscale a BGR copy for emit_preview=True — built on the worker (the
        run loop) so the host gets a small frame without the full-res render."""
        h, w = frame.shape[:2]
        scale = self.display_scale if self.display_scale < 0.999 else (640.0 / w)
        if scale >= 0.999:
            return frame.copy()
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def run_forever(self):
        """Run the per-frame tracking loop until the source ends, q is pressed
        (owns_window only), or stop() is called. Construct → start() → here."""
        if not self._started:
            self.start()

        cap = self._cap
        total_frames = self._total_frames
        is_live = self._is_live
        csv_file = self._csv_file
        csv_writer = self._csv_writer
        corners_file = self._corners_file
        corners_writer = self._corners_writer
        keyframe_dir = self._keyframe_dir
        frame_iter = self._frame_iter

        H = self.H
        H_inv = self.H_inv
        grid_bounds = self.grid_bounds
        grid_spacing = self.grid_spacing
        wearable_right = self.wearable_right
        wearable_left = self.wearable_left

        frame_num = 0
        last_print = 0
        detections = 0
        total_processed = 0
        fps_t0 = time.time()
        fps_n = 0
        show = self._show
        # Draw the marker/grid/YOU overlays whenever the frame will be *looked at*
        # — the CLI imshow window (display) OR a host GUI consuming preview frames
        # (emit_preview). Previously gated on `display` only, so a GUI running
        # display=False got a raw preview with no detection feedback drawn on it.
        draw_overlays = self.display or self.emit_preview
        display_scale = self.display_scale

        smooth_fwd = {"right": None, "left": None}
        smooth_rt = {"right": None, "left": None}
        last_seen_frame = {"right": -10**9, "left": -10**9}
        foot_w = {"right": 0.0, "left": 0.0}        # per-foot confidence weight (ramped)
        last_est = {"right": None, "left": None}    # last valid body-center per foot
        # 20 frames ≈ 0.7s at 30 fps. 5 was too aggressive — it reset the orientation
        # EMA on every brief flicker of the more-often-missed left marker.
        ORIENT_RESET_GAP = 20
        for frame, corners_list, ids, timestamp in frame_iter:
            if self._stop:
                break
            frame_num += 1
            total_processed += 1
            h, w = frame.shape[:2]
            cur_fps = 0.0

            active_cell = None
            floor_pos = None
            method = "none"
            n_markers = 0
            mc_right = None
            mc_left = None
            est_right = None
            est_left = None

            if ids is not None:
                if draw_overlays:
                    aruco.drawDetectedMarkers(frame, corners_list, ids)

                ids_flat = ids.flatten()
                for foot, target_id in (("right", wearable_right), ("left", wearable_left)):
                    if target_id < 0 or target_id not in ids_flat:
                        continue
                    i = int(np.where(ids_flat == target_id)[0][0])
                    corners_px = corners_list[i][0]   # (4, 2): TL, TR, BR, BL

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
                        sf, sr = fwd_raw, rt_raw            # re-acquired after a gap → reset
                    else:
                        sf = ema_unit(smooth_fwd[foot], fwd_raw, self.orient_smooth)
                        sr = ema_unit(smooth_rt[foot], rt_raw, self.orient_smooth)
                    smooth_fwd[foot], smooth_rt[foot] = sf, sr
                    last_seen_frame[foot] = frame_num

                    est = body_center_from_axes(center, sf, sr, foot,
                                                self.offset_side, self.offset_back)
                    if foot == "right":
                        est_right = est
                        mc_right = center
                    else:
                        est_left = est
                        mc_left = center

            # Confidence-weighted foot fusion — runs EVERY frame so a foot's weight
            # decays on dropout and ramps on (re)acquire (hysteresis + blend, no hard
            # one-/two-foot mode-switch jump). The position LABEL stays RAW — we do
            # NOT EMA-smooth ground truth, which would lag a walking target.
            for foot, est in (("right", est_right), ("left", est_left)):
                foot_w[foot] = ramp_weight(foot_w[foot], est is not None,
                                           self.foot_ramp, self.foot_ramp)
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

                if draw_overlays:
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
                    self._emit_log(
                        f"You: ({fx:+7.1f}, {fy:+7.1f}) cm -> grid ({gx:.0f}, {gy:.0f}) "
                        f"| {method} ({n_markers} marker(s))")
                    last_print = now

            # Log to CSV
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
                # Periodic flush so a Ctrl-C (e.g. headless --no-display recording)
                # loses at most ~1s of rows instead of the whole buffer.
                if frame_num % 30 == 0:
                    csv_file.flush()

            # Audit trail: raw corners for every detected marker + sparse keyframe.
            if corners_writer is not None and ids is not None:
                for mi, mid in enumerate(ids.flatten()):
                    c = corners_list[mi][0]   # (4,2) TL,TR,BR,BL pixel corners
                    corners_writer.writerow([frame_num, f"{timestamp:.4f}", int(mid)]
                                            + [f"{v:.1f}" for xy in c for v in xy])
                if frame_num % 30 == 0:
                    corners_file.flush()
            if keyframe_dir is not None and frame_num % self.keyframe_stride == 0:
                cv2.imwrite(os.path.join(keyframe_dir, f"f{frame_num:06d}.jpg"),
                            frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            if draw_overlays:
                draw_grid(frame, H_inv, grid_bounds, grid_spacing, active_cell)

                # Status bar
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

                # The full-res display resize is only for the CLI imshow window;
                # a GUI host (owns_window=False) downscales via _build_preview instead.
                if self._owns_window:
                    disp = frame if display_scale >= 0.999 else cv2.resize(
                        frame, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow("ArUco Tracker", disp)

            # Achieved-framerate counter (prints every 2s, with or without a marker)
            fps_n += 1
            now_fps = time.time()
            if now_fps - fps_t0 >= 2.0:
                cur_fps = fps_n / (now_fps - fps_t0)
                self._emit_log(f"[camera] {cur_fps:.1f} fps")
                fps_t0 = now_fps
                fps_n = 0

            # Build the per-frame result + fire callbacks BEFORE the GUI key pump,
            # so a host sees every processed frame even if the next waitKey quits.
            if self.on_frame is not None or self.on_position is not None:
                if floor_pos is not None:
                    px_state, py_state = floor_pos
                    gx_state, gy_state = active_cell
                    state = PositionState(
                        detected=True, x_cm=px_state, y_cm=py_state,
                        grid_x_cm=gx_state, grid_y_cm=gy_state,
                        n_markers=n_markers, method=method,
                        right_center=(tuple(mc_right) if mc_right is not None else None),
                        left_center=(tuple(mc_left) if mc_left is not None else None),
                        fps=cur_fps)
                else:
                    state = PositionState(
                        detected=False, x_cm=None, y_cm=None,
                        grid_x_cm=None, grid_y_cm=None,
                        n_markers=n_markers, method=method,
                        right_center=(tuple(mc_right) if mc_right is not None else None),
                        left_center=(tuple(mc_left) if mc_left is not None else None),
                        fps=cur_fps)
                if self.on_position is not None:
                    self.on_position(state)
                if self.on_frame is not None:
                    preview = self._build_preview(frame) if self.emit_preview else None
                    self.on_frame(FrameResult(frame_num=frame_num,
                                              timestamp_s=timestamp,
                                              preview_bgr=preview,
                                              position=state))

            # waitKey takes MILLISECONDS, not fps. Live used to pass 30 → a forced
            # 30 ms/frame stall. 1 ms is enough to pump the GUI event loop. Skipped
            # entirely when headless (no window to receive keys; stop with Ctrl-C).
            if show and self._owns_window:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' ') and not is_live:
                    cv2.waitKey(0)  # pause on video

        if not is_live:
            self._emit_log(f"\nVideo complete. {detections}/{total_processed} frames with detection.")

        if csv_file:
            csv_file.close()
            self._emit_log(f"Position log saved to {self.log}")
        if corners_file:
            corners_file.close()
            kf = len(os.listdir(keyframe_dir)) if keyframe_dir else 0
            self._emit_log(f"Audit trail saved: corners.csv + {kf} keyframe(s)")

        if cap is not None:
            cap.release()
        if self._owns_window:
            cv2.destroyAllWindows()


def main():
    global floor_marker_ids

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", type=str, default=None,
                        help="OpenCV camera index OR URL (e.g. http://127.0.0.1:8080/video)")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Process a recorded video instead of live camera")
    parser.add_argument("--marker-right", type=int, default=DEFAULT_RIGHT_ID,
                        help=f"Right-foot marker ID (default: {DEFAULT_RIGHT_ID}; -1 to disable)")
    parser.add_argument("--marker-left", type=int, default=DEFAULT_LEFT_ID,
                        help=f"Left-foot marker ID (default: {DEFAULT_LEFT_ID}; -1 to disable)")
    parser.add_argument("--log", "-l", type=str, default=None,
                        help="Save position log to CSV file")
    # Body-center offsets, applied per foot (auto-mirrored side-to-side):
    #   --offset-side = MAGNITUDE (cm) toward body center, perpendicular to forward
    #   --offset-back = signed (cm) along marker forward; negative = toward heel
    # Markers are mounted on the FRONT of each foot with the top edge toward the
    # toes, so body center is offset_side to the inside and offset_back to the heel.
    parser.add_argument("--offset-side", type=float, default=20.0,
                        help="Lateral offset magnitude toward body center (cm, default 20)")
    parser.add_argument("--offset-back", type=float, default=-15.0,
                        help="Offset along marker forward (cm); negative = toward heel (default -15)")
    parser.add_argument("--orient-smooth", type=float, default=0.3,
                        help="EMA weight on each new frame's marker orientation "
                             "(0<a<=1; 1=no smoothing, lower=smoother; default 0.3 ~5-6 frames). "
                             "Reduces position jitter from the offset lever-arm at oblique angles.")
    parser.add_argument("--legacy", action="store_true",
                        help="Force the single-threaded cv2.VideoCapture path for a "
                             "live HTTP stream (fallback if the threaded pipeline misbehaves)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Thread-pool size for the live MJPEG decode/detect pipeline (default 6)")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip all drawing + the imshow window (headless). Detection/CSV still run. "
                             "Removes the main-thread 8MP render cost — use for max framerate / recording.")
    parser.add_argument("--display-scale", type=float, default=0.5,
                        help="Downscale factor for the preview window only (detection stays full-res). "
                             "Default 0.5; rendering a full 8MP window every frame is the main-thread bottleneck.")
    parser.add_argument("--keyframe-stride", type=int, default=30,
                        help="Audit trail: save 1 JPEG every N frames to <session>/keyframes/ "
                             "(0 = off). Default 30 (~1/s); raw ArUco corners are always logged to "
                             "corners.csv regardless. ~70-150 MB/session at the default.")
    parser.add_argument("--foot-ramp", type=int, default=8,
                        help="Frames over which a foot's confidence weight ramps up on acquire / "
                             "decays on dropout (~8 = 0.25s at 30fps). Gives hysteresis + smooth "
                             "blend instead of a hard one-/two-foot mode switch.")
    args = parser.parse_args()

    # Construct the tracker; map its constructor-time errors back to the original
    # CLI messages + exit codes (the class RAISES instead of sys.exit so importers
    # can handle the failure, but the CLI must stay byte-identical).
    try:
        tracker = ArucoTracker(
            camera=args.camera, video=args.video,
            marker_right=args.marker_right, marker_left=args.marker_left,
            log=args.log, offset_side=args.offset_side, offset_back=args.offset_back,
            orient_smooth=args.orient_smooth, legacy=args.legacy, workers=args.workers,
            keyframe_stride=args.keyframe_stride, foot_ramp=args.foot_ramp,
            display=not args.no_display, owns_window=True,
            display_scale=args.display_scale, undistort="full",
            emit_preview=False, on_frame=None, on_position=None, on_log=None)
    except FileNotFoundError as e:
        msg = str(e)
        if msg == f"Cannot open video: {args.video}":
            print(f"ERROR: Cannot open video: {args.video}")
        else:
            print(f"ERROR: {FLOOR_CALIBRATION} not found. Run aruco_setup.py first.")
        sys.exit(1)
    except ValueError as e:
        raise SystemExit(str(e))

    floor_marker_ids = tracker.floor_marker_ids

    # start() can still hit a frame-source failure (camera/video open); map it.
    try:
        tracker.start()
    except FileNotFoundError as e:
        msg = str(e)
        if msg == f"Cannot open video: {args.video}":
            print(f"ERROR: Cannot open video: {args.video}")
        elif msg == "Cannot open camera.":
            print("ERROR: Cannot open camera.")
        else:
            print(f"ERROR: {msg}")
        sys.exit(1)

    tracker.run_forever()


if __name__ == "__main__":
    main()
