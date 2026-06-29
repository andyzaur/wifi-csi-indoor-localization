"""Live position inference core — Qt-free, importable, unit-tested.

Real-time localization shared by the GUI's Live-validate page and the
``live_position.py`` CLI:

  * a UDP listener keeps the most recent CSI packet per RX board,
  * a feature vector identical to the training layout is assembled on demand
    (per board, in ascending id order: 64 subcarrier amplitudes then RSSI),
  * a multiboard amp+rssi model bundle (saved by ``train_final.py``) turns that
    vector into an ``(x, y)`` estimate in floor centimetres, and
  * a floor-marker map image is rendered for display.

A :class:`ReplaySource` can feed a recorded session's ``csi.csv`` through the
exact same path, so the demo shows real model output without live hardware.

The wire format matches ``csi_collector.py`` (``<B6BbBIHH`` header, 128 bytes of
int8 I/Q). Board ids are auto-detected, so this works for boards (1, 2, 3),
(1, 4, 5), or any other labelling, as long as the count matches the model.
"""
from __future__ import annotations

import glob
import json
import math
import os
import socket
import struct
import threading
import time
from collections import deque

import cv2
import numpy as np

PORT = 5500
CSI_HDR_FMT = "<B6BbBIHH"
CSI_HDR_SIZE = struct.calcsize(CSI_HDR_FMT)   # 17 bytes
CSI_DATA_LEN = 128
CLAP_MAGIC = 0xCA
AMPS_PER_BOARD = 64
FEATS_PER_BOARD = AMPS_PER_BOARD + 1          # + rssi


# ── CSI parsing ──────────────────────────────────────────────────────────────

def load_layout(path: str) -> dict[int, tuple[float, float]]:
    """Read ``marker_layout.json`` -> ``{marker_id: (x_cm, y_cm)}``."""
    with open(path) as f:
        layout = json.load(f)
    return {int(k): (float(v[0]), float(v[1])) for k, v in layout["positions_cm"].items()}


def amps_from_iq_bytes(csi: bytes) -> np.ndarray:
    """Raw int8 I/Q bytes -> 64 subcarrier amplitudes ``sqrt(I^2 + Q^2)``."""
    csi = (bytes(csi) + b"\x00" * CSI_DATA_LEN)[:CSI_DATA_LEN]
    iq = np.frombuffer(csi, dtype=np.int8).reshape(-1, 2).astype(np.float32)
    return np.sqrt(iq[:, 0] ** 2 + iq[:, 1] ** 2)


def parse_csi_packet(data: bytes):
    """Parse one UDP datagram -> ``(board_id, rssi, amps[64])`` or ``None``."""
    if not data or data[0] == CLAP_MAGIC or len(data) < CSI_HDR_SIZE:
        return None
    fields = struct.unpack_from(CSI_HDR_FMT, data, 0)
    board_id = fields[0]
    rssi = fields[7]
    csi_len = fields[11]
    if board_id < 1 or board_id > 200:
        return None
    amps = amps_from_iq_bytes(data[CSI_HDR_SIZE:CSI_HDR_SIZE + csi_len])
    return board_id, int(rssi), amps


# ── Per-board state + feature assembly ───────────────────────────────────────

class BoardState:
    """Thread-safe store of the most recent ``(wall_t, rssi, amps)`` per board."""

    def __init__(self) -> None:
        self._latest: dict[int, tuple[float, float, np.ndarray]] = {}
        self._lock = threading.Lock()

    def update(self, board_id: int, wall_t: float, rssi: float, amps: np.ndarray) -> None:
        with self._lock:
            self._latest[int(board_id)] = (wall_t, float(rssi), np.asarray(amps, dtype=np.float32))

    def snapshot(self, n_boards: int, max_age_s: float, now: float | None = None):
        """Feature vector ``[1, n_boards*65]`` from the freshest boards.

        Returns ``(features | None, ages_by_board)``. Column order matches
        training: for each board (ascending id) 64 amplitudes then RSSI.
        ``features`` is ``None`` until at least ``n_boards`` boards are fresher
        than ``max_age_s``.
        """
        now = time.time() if now is None else now
        with self._lock:
            items = dict(self._latest)
        ages = {b: now - t for b, (t, _, _) in items.items()}
        fresh = sorted(b for b in items if ages[b] <= max_age_s)
        if len(fresh) < n_boards:
            return None, ages
        boards = fresh[:n_boards]
        feats: list[np.ndarray] = []
        for b in boards:
            _, rssi, amps = items[b]
            feats.append(amps.astype(np.float32))
            feats.append(np.array([rssi], dtype=np.float32))
        return np.concatenate(feats).reshape(1, -1), ages


# ── Model bundle ─────────────────────────────────────────────────────────────

class Predictor:
    """Wraps a ``train_final.py`` model bundle for single-vector inference."""

    def __init__(self, bundle: dict) -> None:
        for key in ("scaler", "regressor"):
            if key not in bundle:
                raise ValueError(
                    f"Model bundle is missing '{key}' — expected a train_final.py "
                    "bundle (scaler + regressor + n_features)."
                )
        self.scaler = bundle["scaler"]
        self.regressor = bundle["regressor"]
        self.encoder = bundle.get("encoder")
        n = bundle.get("n_features") or getattr(self.scaler, "n_features_in_", None)
        if not n:
            raise ValueError("Model bundle has no feature count.")
        self.n_features = int(n)
        self.n_boards = max(1, self.n_features // FEATS_PER_BOARD)

    @classmethod
    def from_path(cls, path: str) -> "Predictor":
        import joblib
        return cls(joblib.load(path))

    def predict_xy(self, features: np.ndarray) -> tuple[float, float]:
        xs = self.scaler.transform(features)
        p = self.regressor.predict(xs)[0]
        return float(p[0]), float(p[1])


# ── Live UDP source ──────────────────────────────────────────────────────────

class UdpCsiListener:
    """Background UDP :5500 reader that updates a :class:`BoardState`."""

    def __init__(self, state: BoardState, port: int = PORT) -> None:
        self._state = state
        self._port = port
        self._running = False
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> "UdpCsiListener":
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except OSError:
            pass
        s.bind(("", self._port))
        s.settimeout(0.5)
        self._sock = s
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            parsed = parse_csi_packet(data)
            if parsed is None:
                continue
            board_id, rssi, amps = parsed
            self._state.update(board_id, time.time(), rssi, amps)

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# ── Replay source (demo / offline) ───────────────────────────────────────────

class ReplaySource:
    """Replays a recorded session's ``csi.csv`` into a :class:`BoardState`.

    Lets the Live-validate page show real model output without hardware. Each
    row is stamped with the current wall clock as it is fed, so age checks pass.
    Reads only the columns it needs and caps the row count to stay responsive.
    """

    def __init__(self, state: BoardState, csi_csv_path: str, *,
                 speed: float = 4.0, loop: bool = True, max_rows: int = 120_000) -> None:
        import pandas as pd
        cols = ["board_id", "rssi", "wall_time_s"] + [f"csi_{i}" for i in range(CSI_DATA_LEN)]
        df = pd.read_csv(csi_csv_path, usecols=lambda c: c in cols, nrows=max_rows)
        df = df.sort_values("wall_time_s").reset_index(drop=True)
        csi_cols = [f"csi_{i}" for i in range(CSI_DATA_LEN)]
        iq = df[csi_cols].to_numpy(dtype=np.float32).reshape(len(df), AMPS_PER_BOARD, 2)
        self._amps = np.sqrt(iq[:, :, 0] ** 2 + iq[:, :, 1] ** 2)
        self._board = df["board_id"].to_numpy()
        self._rssi = df["rssi"].to_numpy()
        t = df["wall_time_s"].to_numpy(dtype=np.float64)
        self._dt = np.diff(t, prepend=t[0]) if len(t) else np.array([])
        self._state = state
        self._speed = max(0.1, speed)
        self._loop = loop
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def n_rows(self) -> int:
        return len(self._board)

    def feed_row(self, i: int) -> None:
        """Push row ``i`` into the shared state (used by the run loop and tests)."""
        self._state.update(int(self._board[i]), time.time(),
                           float(self._rssi[i]), self._amps[i])

    def start(self) -> "ReplaySource":
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while self._running:
            for i in range(self.n_rows):
                if not self._running:
                    return
                self.feed_row(i)
                dt = min(float(self._dt[i]) / self._speed, 0.05) if i < len(self._dt) else 0.0
                if dt > 0:
                    time.sleep(dt)
            if not self._loop:
                return

    def stop(self) -> None:
        self._running = False


# ── Floor-map renderer (dark theme, matches the GUI) ─────────────────────────

_BG = (18, 14, 12)          # near-black, BGR
_GRID = (40, 34, 28)
_ANCHOR = (146, 212, 58)    # accent green, BGR of #3ad492
_ANCHOR_RING = (90, 140, 40)
_TEXT = (241, 235, 232)
_PRED = (120, 230, 90)      # bright accent dot
_PRED_RING = (255, 255, 255)


def render_floor_map(anchors, pred_history, current_pred, status_lines,
                     *, px_per_cm: int = 3, margin: int = 60) -> np.ndarray:
    """Render the floor markers, prediction trail, and current estimate.

    Returns an ``H x W x 3`` uint8 BGR image (use ``QImage.Format_BGR888``).
    """
    if not anchors:
        img = np.full((360, 640, 3), _BG, dtype=np.uint8)
        cv2.putText(img, "marker_layout.json not found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _TEXT, 1, cv2.LINE_AA)
        return img

    xs_all = [p[0] for p in anchors.values()]
    ys_all = [p[1] for p in anchors.values()]
    if current_pred is not None:
        xs_all.append(current_pred[0])
        ys_all.append(current_pred[1])
    x_min, x_max = min(xs_all) - margin, max(xs_all) + margin
    y_min, y_max = min(ys_all) - margin, max(ys_all) + margin
    w = max(320, int((x_max - x_min) * px_per_cm))
    h = max(240, int((y_max - y_min) * px_per_cm))
    img = np.full((h, w, 3), _BG, dtype=np.uint8)

    def to_px(x, y):
        return int((x - x_min) * px_per_cm), int((y_max - y) * px_per_cm)

    for x in range(int(math.floor(x_min / 50)) * 50, int(math.ceil(x_max / 50)) * 50 + 1, 50):
        cv2.line(img, to_px(x, y_min), to_px(x, y_max), _GRID, 1, cv2.LINE_AA)
    for y in range(int(math.floor(y_min / 50)) * 50, int(math.ceil(y_max / 50)) * 50 + 1, 50):
        cv2.line(img, to_px(x_min, y), to_px(x_max, y), _GRID, 1, cv2.LINE_AA)

    for mid, (x, y) in anchors.items():
        cx, cy = to_px(x, y)
        cv2.circle(img, (cx, cy), 11, _ANCHOR, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 11, _ANCHOR_RING, 2, cv2.LINE_AA)
        cv2.putText(img, str(mid), (cx - 5, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 25, 15), 1, cv2.LINE_AA)

    n = max(1, len(pred_history))
    for i, pt in enumerate(pred_history):
        if pt is None or pt[0] is None:
            continue
        a = (i + 1) / n
        col = (int(60 + 120 * a), int(120 + 110 * a), int(50 + 60 * a))
        cx, cy = to_px(pt[0], pt[1])
        cv2.circle(img, (cx, cy), 3, col, -1, cv2.LINE_AA)

    if current_pred is not None:
        cx, cy = to_px(current_pred[0], current_pred[1])
        cv2.circle(img, (cx, cy), 15, _PRED, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 15, _PRED_RING, 2, cv2.LINE_AA)
        cv2.putText(img, f"({current_pred[0]:.0f}, {current_pred[1]:.0f})", (cx + 20, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _TEXT, 1, cv2.LINE_AA)

    for i, line in enumerate(status_lines):
        cv2.putText(img, line, (12, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT, 1, cv2.LINE_AA)
    return img


# ── Discovery helpers ────────────────────────────────────────────────────────

def find_model_bundles(sessions_dir: str) -> list[tuple[str, str]]:
    """``[(label, path)]`` of model bundles one level under ``sessions_dir``."""
    out: list[tuple[str, str]] = []
    if not os.path.isdir(sessions_dir):
        return out
    for path in sorted(glob.glob(os.path.join(sessions_dir, "*", "*.joblib"))):
        name = os.path.basename(path)
        if "model" in name or "master" in name:
            sess = os.path.basename(os.path.dirname(path))
            out.append((f"{sess}/{name}", path))
    return out


def find_replay_sessions(sessions_dir: str) -> list[tuple[str, str]]:
    """``[(label, csi.csv path)]`` for recorded sessions, smallest file first."""
    out: list[tuple[int, str, str]] = []
    if not os.path.isdir(sessions_dir):
        return []
    for path in glob.glob(os.path.join(sessions_dir, "*", "csi.csv")):
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        sess = os.path.basename(os.path.dirname(path))
        out.append((size, f"{sess}  ({size / 1e6:.0f} MB)", path))
    out.sort(key=lambda t: t[0])
    return [(label, path) for _, label, path in out]
