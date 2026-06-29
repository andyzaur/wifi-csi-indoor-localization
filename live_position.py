#!/usr/bin/env python3
"""Real-time position inference using the multi-board MLP model.

Listens for CSI on UDP :5500, maintains the most recent packet per board, and
when all three boards have recent data (< max-age seconds) builds the same
195-feature vector used during training and predicts position.

Usage:
    # 1. Train a model first
    python3 train_final.py sessions/<name>

    # 2. Run live demo
    python3 live_position.py --model sessions/<name>/model_final.joblib

A window shows the floor marker layout + a red dot for the smoothed predicted
position. Press 'q' to quit.
"""

import argparse
import json
import math
import socket
import struct
import threading
import time
from collections import deque

import cv2
import joblib
import numpy as np


PORT = 5500
CSI_HDR_FMT = "<B6BbBIHH"
CSI_HDR_SIZE = struct.calcsize(CSI_HDR_FMT)
CSI_DATA_LEN = 128
CLAP_MAGIC = 0xCA


def load_layout(path="marker_layout.json"):
    with open(path) as f:
        layout = json.load(f)
    return {int(k): tuple(v) for k, v in layout["positions_cm"].items()}


# ─── Live socket listener ───────────────────────────────────────────────────

class LiveCSI:
    """Tracks the most recent CSI packet per board (1, 2, 3)."""

    def __init__(self):
        self.latest = {}  # board_id -> (wall_t, rssi, amps_64)
        self.lock = threading.Lock()
        self.running = True

    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.bind(("", PORT))
        s.settimeout(0.5)
        t = threading.Thread(target=self._loop, args=(s,), daemon=True)
        t.start()
        return self

    def _loop(self, sock):
        while self.running:
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            if not data or data[0] == CLAP_MAGIC:
                continue
            if data[0] not in (1, 2, 3) or len(data) < CSI_HDR_SIZE:
                continue
            fields = struct.unpack_from(CSI_HDR_FMT, data, 0)
            board_id = fields[0]
            rssi = fields[7]
            csi_len = fields[11]
            csi = data[CSI_HDR_SIZE:CSI_HDR_SIZE + csi_len]
            csi = csi + b"\x00" * (CSI_DATA_LEN - len(csi))
            csi = csi[:CSI_DATA_LEN]
            iq = np.frombuffer(csi, dtype=np.int8).reshape(-1, 2).astype(np.float32)
            amps = np.sqrt(iq[:, 0] ** 2 + iq[:, 1] ** 2)
            with self.lock:
                self.latest[board_id] = (time.time(), float(rssi), amps)

    def stop(self):
        self.running = False

    def multiboard_snapshot(self, max_age_s=0.5):
        """Return (feature_vector_195, ages_dict) if all 3 boards have fresh data."""
        with self.lock:
            if not all(b in self.latest for b in (1, 2, 3)):
                return None, None
            now = time.time()
            ages = {}
            feats = []
            for b in (1, 2, 3):
                wall_t, rssi, amps = self.latest[b]
                age = now - wall_t
                ages[b] = age
                if age > max_age_s:
                    return None, ages
                feats.append(amps)
                feats.append(np.array([rssi], dtype=np.float32))
            return np.concatenate(feats).reshape(1, -1), ages


# ─── GUI ────────────────────────────────────────────────────────────────────

def make_layout_image(anchors, pred_history, current_pred, status_lines):
    xs_all = [p[0] for p in anchors.values()]
    ys_all = [p[1] for p in anchors.values()]
    if current_pred is not None:
        xs_all.append(current_pred[0])
        ys_all.append(current_pred[1])
    margin = 60
    x_min = min(xs_all) - margin
    x_max = max(xs_all) + margin
    y_min = min(ys_all) - margin
    y_max = max(ys_all) + margin
    px_per_cm = 3
    W = int((x_max - x_min) * px_per_cm)
    H = int((y_max - y_min) * px_per_cm)
    img = np.full((H, W, 3), 245, dtype=np.uint8)

    def to_px(x, y):
        return int((x - x_min) * px_per_cm), int((y_max - y) * px_per_cm)

    for x in range(int(math.floor(x_min / 50)) * 50,
                   int(math.ceil(x_max / 50)) * 50 + 1, 50):
        cv2.line(img, to_px(x, y_min), to_px(x, y_max), (220, 220, 220), 1)
    for y in range(int(math.floor(y_min / 50)) * 50,
                   int(math.ceil(y_max / 50)) * 50 + 1, 50):
        cv2.line(img, to_px(x_min, y), to_px(x_max, y), (220, 220, 220), 1)

    for mid, (x, y) in anchors.items():
        cx, cy = to_px(x, y)
        cv2.circle(img, (cx, cy), 12, (0, 130, 0), -1)
        cv2.circle(img, (cx, cy), 12, (50, 50, 50), 2)
        cv2.putText(img, str(mid), (cx - 5, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    n = len(pred_history)
    for i, (x, y) in enumerate(pred_history):
        if x is None:
            continue
        a = (i + 1) / n
        col = (int(255 * (1 - a)), int(180 * (1 - a)), int(50 + 200 * a))
        cx, cy = to_px(x, y)
        cv2.circle(img, (cx, cy), 3, col, -1)

    if current_pred is not None:
        x, y = current_pred
        cx, cy = to_px(x, y)
        cv2.circle(img, (cx, cy), 16, (0, 0, 220), -1)
        cv2.circle(img, (cx, cy), 16, (255, 255, 255), 2)
        cv2.putText(img, f"({x:.0f}, {y:.0f})", (cx + 22, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

    for i, line in enumerate(status_lines):
        cv2.putText(img, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layout", default="marker_layout.json")
    parser.add_argument("--task", choices=["regression", "classification"],
                        default="regression")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Rolling-mean smoothing window")
    parser.add_argument("--trail", type=int, default=80)
    parser.add_argument("--max-age", type=float, default=0.5,
                        help="Drop prediction if any board's CSI is older than this (s)")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    if "scaler" not in bundle:
        raise SystemExit("Model bundle has no 'scaler' — was this trained with train_final.py?")
    scaler = bundle["scaler"]
    model = bundle["regressor"] if args.task == "regression" else bundle["classifier"]
    encoder = bundle.get("encoder")
    anchors = load_layout(args.layout)
    print(f"Loaded model from {args.model} (task={args.task})")
    print(f"Loaded {len(anchors)} floor anchors")

    listener = LiveCSI().start()
    print(f"Listening for multi-board CSI on UDP :{PORT}.  Press 'q' to quit.")

    raw_history = deque(maxlen=args.smooth)
    pred_history = deque(maxlen=args.trail)
    smoothed = None
    last_pred_t = 0
    last_render = 0

    cv2.namedWindow("Live position", cv2.WINDOW_NORMAL)
    try:
        while True:
            now = time.time()
            X, ages = listener.multiboard_snapshot(max_age_s=args.max_age)

            if X is not None and now - last_pred_t >= 0.05:
                X_s = scaler.transform(X)
                if args.task == "regression":
                    p = model.predict(X_s)[0]
                    raw_xy = (float(p[0]), float(p[1]))
                else:
                    cls = model.predict(X_s)[0]
                    cell_label = encoder.inverse_transform([cls])[0] if encoder is not None else cls
                    gx_s, gy_s = str(cell_label).split("_")
                    raw_xy = (float(gx_s), float(gy_s))

                raw_history.append(raw_xy)
                xs = np.mean([h[0] for h in raw_history])
                ys = np.mean([h[1] for h in raw_history])
                smoothed = (xs, ys)
                pred_history.append(smoothed)
                last_pred_t = now

            if now - last_render >= 1 / 30:
                if ages is None:
                    status = ["Waiting for all 3 boards to send CSI..."]
                else:
                    age_strs = "  ".join([f"b{b}: {ages[b]*1000:.0f}ms" for b in (1, 2, 3)])
                    status = [f"CSI ages: {age_strs}",
                              f"smooth window {args.smooth}   task={args.task}"]
                img = make_layout_image(anchors, list(pred_history), smoothed, status)
                cv2.imshow("Live position", img)
                last_render = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        listener.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
