"""Session VISUALIZATIONS — Qt-free compute + matplotlib Agg rendering.

Each public ``render_*`` function takes a *loaded* session (the
``(csi, camera, clap)`` triple from :func:`dataset.load_session`, or a
:class:`LoadedSession` bundle) and returns a :class:`RenderedImage` — a flat
RGBA byte buffer plus its width/height/stride — that the GUI wraps in a
``QImage`` / ``QPixmap`` with **zero** matplotlib-in-Qt coupling.

Design rules (per the Sessions-explorer brief):

  * **Agg only.** We never touch ``pyplot`` and never open a window. Figures are
    built with the object API (``Figure`` + ``FigureCanvasAgg``) so this is
    headless and thread-safe — these run on a WORKER thread, on demand, never on
    import or the GUI hot path.
  * **Robust to schema drift.** Camera CSVs come in a 7-col and a 13-col flavour
    (the latter adds ``method``/``right_*``/``left_*``); we only ever rely on the
    common columns (``x_cm``, ``y_cm``, ``grid_x_cm``, ``grid_y_cm``,
    ``detected``, ``wall_time_s``).
  * **Robust to size.** Big sessions (tens of thousands of rows) are sampled /
    limited so a render stays fast and bounded in memory.

The plots reuse the ideas in ``explore.py`` (timeline, walked path, CSI
amplitude, distributions) but recompute GUI-side from raw CSI via
:func:`dataset.amplitudes_from_csi` so we don't depend on ``explore.main``'s
side-effecting file output.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless: no pyplot windows, ever.

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dataset import (
    CSI_FEATURE_COLS,
    amplitudes_from_csi,
    detect_board_ids,
    grid_cell_id,
    trim_to_session,
)

# Caps so a huge session renders fast and bounded. These only sub-SAMPLE for the
# plot; they never change the underlying data on disk.
MAX_HEATMAP_PACKETS = 1500   # per-board CSI rows fed into the amplitude heatmap
MAX_PATH_POINTS = 8000       # scatter points in the walked-path / coverage plots
MAX_RATE_PACKETS = 200_000   # cap on rows scanned for the rate timeline

# Palette matching the GUI's dark QSS so embedded plots look native.
_BG = "#14141a"
_FG = "#e0e0e0"
_GRID = "#2a2a34"
_MUTED = "#9a9aa6"
_ACCENT = "#3ddc84"


@dataclass
class RenderedImage:
    """A rendered RGBA image: a flat byte buffer + geometry.

    ``buffer`` is tightly packed RGBA8888 (``stride == width * 4``), row-major,
    top-to-bottom — exactly what ``QImage(buffer, w, h, QImage.Format_RGBA8888)``
    wants. ``empty`` is True for a placeholder rendered when there's nothing to
    plot (still a valid, non-empty image so the GUI shows *something*).
    """

    buffer: bytes
    width: int
    height: int
    stride: int
    empty: bool = False

    def __len__(self) -> int:  # convenience for "non-empty buffer" assertions
        return len(self.buffer)


@dataclass
class LoadedSession:
    """A loaded session bundle the render functions can share.

    Holds the three raw frames plus the clap-trimmed CSI/camera (so every plot
    sees the same in-window data) and the detected board IDs. Build it once on
    the worker thread via :func:`load_for_viz`, then hand it to each plot.
    """

    csi: pd.DataFrame
    camera: pd.DataFrame
    clap: pd.DataFrame
    board_ids: tuple[int, ...]


def load_for_viz(csi: pd.DataFrame, camera: pd.DataFrame,
                 clap: Optional[pd.DataFrame] = None) -> LoadedSession:
    """Bundle + clap-trim raw frames into a :class:`LoadedSession` for plotting.

    Trims CSI and camera to the START/STOP clap window when a usable clap frame
    is present (so plots show the real session, not warm-up noise). Tolerates a
    missing/empty clap by leaving the frames untrimmed.
    """
    csi_t, cam_t = csi, camera
    if clap is not None and not clap.empty:
        try:
            csi_t = trim_to_session(csi, clap, "wall_time_s")
            cam_t = trim_to_session(camera, clap, "wall_time_s")
        except Exception:
            csi_t, cam_t = csi, camera
        # If trimming nuked everything (bad clap), fall back to untrimmed.
        if len(csi_t) == 0:
            csi_t = csi
        if len(cam_t) == 0:
            cam_t = camera
    try:
        board_ids = detect_board_ids(csi_t) if len(csi_t) else ()
    except Exception:
        board_ids = ()
    return LoadedSession(csi=csi_t, camera=cam_t,
                         clap=clap if clap is not None else pd.DataFrame(),
                         board_ids=board_ids)


# ---------------------------------------------------------------------------
# Figure <-> RGBA buffer plumbing
# ---------------------------------------------------------------------------

def _new_figure(width_px: int = 760, height_px: int = 460, dpi: int = 100) -> Figure:
    """A dark-themed Agg Figure sized in pixels (no pyplot, no window)."""
    fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(_BG)
    FigureCanvasAgg(fig)  # attach an Agg canvas (sets fig.canvas)
    return fig


def _style_axes(ax) -> None:
    """Apply the dark palette to a single Axes (face, ticks, spines, grid)."""
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    ax.grid(alpha=0.25, color=_GRID)


def _figure_to_rendered(fig: Figure, empty: bool = False) -> RenderedImage:
    """Render an Agg Figure to a packed RGBA :class:`RenderedImage`, then free it.

    Uses ``buffer_rgba`` (always available on the Agg canvas). We copy the bytes
    out so the figure (and its big numpy buffer) can be garbage-collected
    immediately — important when these run repeatedly on a worker thread.
    """
    canvas = fig.canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    rgba = np.asarray(canvas.buffer_rgba())  # (h, w, 4) uint8, top-to-bottom
    buf = bytes(np.ascontiguousarray(rgba).tobytes())
    fig.clf()
    return RenderedImage(buffer=buf, width=int(w), height=int(h),
                         stride=int(w) * 4, empty=empty)


def _placeholder(message: str, width_px: int = 760, height_px: int = 260) -> RenderedImage:
    """A dark card with a centered message — used when a plot has no data."""
    fig = _new_figure(width_px, height_px)
    ax = fig.add_subplot(111)
    ax.set_facecolor(_BG)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center",
            color=_MUTED, fontsize=12, transform=ax.transAxes, wrap=True)
    return _figure_to_rendered(fig, empty=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _amps_for_board(csi: pd.DataFrame, board_id: int,
                    max_packets: int = MAX_HEATMAP_PACKETS) -> Optional[np.ndarray]:
    """Amplitudes (N, 64) for one board's CSI rows, time-ordered + sub-sampled.

    Returns None when the board has no rows. Evenly sub-samples to at most
    ``max_packets`` rows (preserving temporal spread) so a long session's heatmap
    stays cheap to compute and draw.
    """
    sub = csi[csi["board_id"] == board_id]
    if len(sub) == 0:
        return None
    if "wall_time_s" in sub.columns:
        sub = sub.sort_values("wall_time_s")
    if len(sub) > max_packets:
        idx = np.linspace(0, len(sub) - 1, max_packets).astype(int)
        sub = sub.iloc[idx]
    raw = sub[CSI_FEATURE_COLS].to_numpy()
    out = np.empty((raw.shape[0], 64), dtype=np.float32)
    for i, row in enumerate(raw):
        out[i] = amplitudes_from_csi(row)
    return out


def _detected_camera(camera: pd.DataFrame) -> pd.DataFrame:
    """Camera rows with a marker detection, sorted by time.

    Tolerates a missing ``detected`` column (treats all rows as detected) and an
    empty frame (returns it unchanged).
    """
    cam = camera
    if "detected" in cam.columns:
        cam = cam[cam["detected"] == 1]
    if "wall_time_s" in cam.columns:
        cam = cam.sort_values("wall_time_s")
    return cam


# ---------------------------------------------------------------------------
# (a) CSI amplitude heatmap (subcarrier 0-63 x time, per RX board)
# ---------------------------------------------------------------------------

def render_csi_heatmap(session: LoadedSession) -> RenderedImage:
    """CSI amplitude heatmap: subcarrier (y) vs time/packet-index (x), per board.

    One column per RX board (sorted ascending). Each cell's color is the CSI
    amplitude (from :func:`dataset.amplitudes_from_csi`) for that subcarrier at
    that packet. Long sessions are sub-sampled to :data:`MAX_HEATMAP_PACKETS`
    packets per board.
    """
    boards = session.board_ids
    if not boards:
        return _placeholder("No CSI rows to build an amplitude heatmap.")

    n = len(boards)
    fig = _new_figure(width_px=max(360, 300 * n), height_px=420)
    axes = fig.subplots(1, n, squeeze=False)[0]
    last_im = None
    for ax, b in zip(axes, boards):
        amps = _amps_for_board(session.csi, b)
        _style_axes(ax)
        if amps is None or amps.size == 0:
            ax.text(0.5, 0.5, f"board {b}\nno CSI", ha="center", va="center",
                    color=_MUTED, transform=ax.transAxes)
            ax.set_title(f"Board {b}")
            continue
        # amps: (packets, 64) -> show as (subcarrier, packet)
        last_im = ax.imshow(amps.T, aspect="auto", origin="lower",
                            cmap="viridis", interpolation="nearest")
        ax.set_title(f"Board {b}")
        ax.set_xlabel("packet (time →)")
    axes[0].set_ylabel("subcarrier 0–63")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=list(axes), fraction=0.025, pad=0.02)
        cbar.set_label("CSI amplitude", color=_FG, fontsize=8)
        cbar.ax.tick_params(colors=_MUTED, labelsize=7)
    fig.suptitle("CSI amplitude per subcarrier over time", color=_FG, fontsize=12)
    return _figure_to_rendered(fig)


# ---------------------------------------------------------------------------
# (b) spatial coverage map (samples per 50 cm grid cell)
# ---------------------------------------------------------------------------

def render_coverage_map(session: LoadedSession) -> RenderedImage:
    """Spatial coverage: detected-camera sample count per 50 cm grid cell.

    Bins detected camera frames by their ``(grid_x_cm, grid_y_cm)`` cell (via
    :func:`dataset.grid_cell_id` parsing) and draws a 2-D histogram of the room,
    so under-covered corners are obvious. Robust to the 7-col / 13-col camera
    schema (uses only the grid columns).
    """
    cam = _detected_camera(session.camera)
    needed = {"grid_x_cm", "grid_y_cm"}
    if len(cam) == 0 or not needed.issubset(cam.columns):
        return _placeholder("No detected camera frames with grid cells "
                            "to map coverage.")

    gx = pd.to_numeric(cam["grid_x_cm"], errors="coerce")
    gy = pd.to_numeric(cam["grid_y_cm"], errors="coerce")
    ok = gx.notna() & gy.notna()
    gx, gy = gx[ok].to_numpy(), gy[ok].to_numpy()
    if gx.size == 0:
        return _placeholder("No valid grid-cell coordinates to map coverage.")

    cell = 50.0  # the floor grid spacing (cm)
    xs = np.unique(gx)
    ys = np.unique(gy)
    x_edges = np.append(xs, xs.max() + cell) - cell / 2
    y_edges = np.append(ys, ys.max() + cell) - cell / 2
    counts, _, _ = np.histogram2d(gx, gy, bins=[x_edges, y_edges])

    fig = _new_figure(width_px=560, height_px=480)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    im = ax.imshow(counts.T, origin="lower", aspect="equal", cmap="magma",
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    ax.set_xlabel("grid X (cm)")
    ax.set_ylabel("grid Y (cm)")
    ax.set_title(f"Coverage: samples per 50 cm cell ({len(xs)}×{len(ys)} grid)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("samples in cell", color=_FG, fontsize=8)
    cbar.ax.tick_params(colors=_MUTED, labelsize=7)
    return _figure_to_rendered(fig)


# ---------------------------------------------------------------------------
# (c) walked path (x_cm, y_cm over time, time gradient)
# ---------------------------------------------------------------------------

def render_walked_path(session: LoadedSession) -> RenderedImage:
    """Walked path: ``(x_cm, y_cm)`` scattered, colored by normalized time.

    Blue → yellow over the session, so you can see the walk's shape and where it
    spent time. Sub-samples to :data:`MAX_PATH_POINTS` points for big sessions.
    Needs only ``x_cm``/``y_cm`` (+ ``wall_time_s`` for the gradient).
    """
    cam = _detected_camera(session.camera)
    if len(cam) == 0 or not {"x_cm", "y_cm"}.issubset(cam.columns):
        return _placeholder("No detected camera positions to draw a path.")

    x = pd.to_numeric(cam["x_cm"], errors="coerce")
    y = pd.to_numeric(cam["y_cm"], errors="coerce")
    ok = x.notna() & y.notna()
    cam = cam[ok]
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    if x.size == 0:
        return _placeholder("No valid camera positions to draw a path.")

    if "wall_time_s" in cam.columns:
        t = pd.to_numeric(cam["wall_time_s"], errors="coerce").to_numpy()
        span = np.nanmax(t) - np.nanmin(t)
        t_norm = (t - np.nanmin(t)) / span if span > 1e-9 else np.zeros_like(t)
    else:
        t_norm = np.linspace(0.0, 1.0, x.size)

    if x.size > MAX_PATH_POINTS:
        idx = np.linspace(0, x.size - 1, MAX_PATH_POINTS).astype(int)
        x, y, t_norm = x[idx], y[idx], t_norm[idx]

    fig = _new_figure(width_px=560, height_px=480)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    sc = ax.scatter(x, y, c=t_norm, cmap="viridis", s=8, alpha=0.75)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title("Walked path (color = time, blue → yellow)")
    ax.set_aspect("equal", adjustable="datalim")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("normalized time", color=_FG, fontsize=8)
    cbar.ax.tick_params(colors=_MUTED, labelsize=7)
    return _figure_to_rendered(fig)


# ---------------------------------------------------------------------------
# (d) per-board CSI rate over time / detection timeline
# ---------------------------------------------------------------------------

def render_rate_timeline(session: LoadedSession) -> RenderedImage:
    """Per-board CSI packet rate (Hz) over the session, in 1 s bins.

    Shows whether every RX board stayed alive at the expected ~33 Hz and where
    drops happened — the most common silent failure. One line per board, plus a
    camera-detection rate line if the camera carries timestamps. Robust to a
    very long session (caps rows scanned at :data:`MAX_RATE_PACKETS`).
    """
    csi = session.csi
    if len(csi) == 0 or "wall_time_s" not in csi.columns:
        return _placeholder("No timestamped CSI to plot a rate timeline.")

    if len(csi) > MAX_RATE_PACKETS:
        csi = csi.iloc[:: max(1, len(csi) // MAX_RATE_PACKETS)]

    t = pd.to_numeric(csi["wall_time_s"], errors="coerce")
    ok = t.notna()
    csi = csi[ok]
    t = t[ok].to_numpy()
    if t.size == 0:
        return _placeholder("No valid CSI timestamps to plot a rate timeline.")

    t0 = t.min()
    rel = t - t0
    duration = max(rel.max(), 1e-6)
    n_bins = max(2, int(np.ceil(duration)) + 1)  # ~1 s bins
    edges = np.linspace(0, duration, n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_w = max(edges[1] - edges[0], 1e-6)

    fig = _new_figure(width_px=760, height_px=420)
    ax = fig.add_subplot(111)
    _style_axes(ax)

    boards = session.board_ids or detect_board_ids(csi)
    for b in boards:
        bmask = (csi["board_id"].to_numpy() == b)
        if not bmask.any():
            continue
        hist, _ = np.histogram(rel[bmask], bins=edges)
        ax.plot(centers, hist / bin_w, lw=1.0, marker="", label=f"board {b}")

    # Optional camera-detection rate overlay (relative to the SAME t0).
    cam = session.camera
    if len(cam) and "wall_time_s" in cam.columns:
        ct = pd.to_numeric(cam["wall_time_s"], errors="coerce")
        det = cam["detected"] == 1 if "detected" in cam.columns else pd.Series(
            True, index=cam.index)
        cm_ok = ct.notna() & det
        crel = ct[cm_ok].to_numpy() - t0
        crel = crel[(crel >= 0) & (crel <= duration)]
        if crel.size:
            chist, _ = np.histogram(crel, bins=edges)
            ax.plot(centers, chist / bin_w, lw=1.0, color=_MUTED, alpha=0.8,
                    linestyle="--", label="camera (detected)")

    ax.set_xlabel("time since start (s)")
    ax.set_ylabel("rate (Hz, 1 s bins)")
    ax.set_title("Per-board CSI rate over time")
    leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.2)
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_color(_FG)
    return _figure_to_rendered(fig)


# Registry the GUI iterates to build one tab per plot (label, fn).
PLOTS = (
    ("CSI heatmap", render_csi_heatmap),
    ("Coverage", render_coverage_map),
    ("Walked path", render_walked_path),
    ("Rate timeline", render_rate_timeline),
)
