"""Subprocess WORKERS for the Sessions explorer — top-level, picklable, GIL-free.

The heavy Sessions-page work — :func:`dataset.load_session` reading the (large)
``csi.csv``, the per-packet ``amplitudes_from_csi`` loop in :mod:`csi_gui.viz`,
the matplotlib Agg render, and ``validate_session.build_report`` (which builds the
multiboard dataset TWICE) — is all pure-Python/pandas/numpy and therefore HOLDS
THE GIL. Running it on a ``QThreadPool`` thread starves the GUI thread, so every
tab switch / session click beachballs on macOS.

The fix is to run that work in a SEPARATE PROCESS (a
``concurrent.futures.ProcessPoolExecutor``) so it has its own GIL and the GUI
event loop stays responsive. The two entry points the subprocess runs live here:

  * :func:`render_session_plot` — load only what a single plot needs (camera-only
    for the coverage / walked-path plots, CSI for the heatmap / rate timeline),
    render it via :mod:`csi_gui.viz`, and return the RGBA bytes as a plain dict.
  * :func:`compute_report` — run ``validate_session.build_report`` and return its
    rows as a plain, PICKLABLE dict (never the live ``Report`` object).

Both are MODULE-LEVEL functions (so ``ProcessPoolExecutor`` under the macOS
``spawn`` start method can pickle them by qualified name) and return only
picklable, primitive-typed payloads (``bytes`` / ``str`` / ``int`` / lists /
dicts) — never a pandas frame, a numpy array, or a Qt object.

macOS spawn safety: importing this module must do NO work and must NOT construct
a process pool — the GUI side owns the pool's lifecycle. Everything imported here
(``dataset``, ``csi_gui.viz``, ``validate_session``) is itself import-safe.
"""

from __future__ import annotations

from typing import Any, Dict


# Picklable string keys for the report verdict (mirrors validate_session levels
# without importing it at module scope — keeps this import cheap).
_VERDICT_OK = "OK"
_VERDICT_WARN = "WARN"
_VERDICT_FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Plot rendering (runs in the subprocess)
# ---------------------------------------------------------------------------

# Which plots need the CSI frame vs only the camera frame. Loading only what a
# plot needs keeps a single render cheap (the coverage / path plots don't touch
# the big csi.csv at all). Indices match csi_gui.viz.PLOTS order:
#   0 CSI heatmap (CSI), 1 Coverage (camera), 2 Walked path (camera),
#   3 Rate timeline (CSI + camera overlay).
_PLOT_NEEDS_CSI = (True, False, False, True)


def render_session_plot(session_path: str, plot_idx: int) -> Dict[str, Any]:
    """Load + render a SINGLE session plot in this (sub)process; return RGBA bytes.

    Loads the session at ``session_path`` (only the frames the chosen plot needs:
    camera-only for the coverage / walked-path plots, CSI for the heatmap / rate
    timeline), builds the :class:`csi_gui.viz.LoadedSession` bundle, renders the
    plot at ``plot_idx`` via :data:`csi_gui.viz.PLOTS`, and returns a PICKLABLE
    dict the GUI wraps in a ``QImage(Format_RGBA8888)`` -> ``QPixmap``::

        {"buffer": bytes,  # tightly-packed RGBA8888, row-major, top-to-bottom
         "width": int, "height": int, "stride": int, "empty": bool}

    All heavy imports (pandas, matplotlib via viz, dataset) happen HERE, in the
    worker process, never on the GUI side. Raises on a bad index so the caller's
    failure path can surface it; otherwise viz itself renders a placeholder image
    for empty/degenerate data rather than raising.
    """
    from csi_gui import viz

    if plot_idx < 0 or plot_idx >= len(viz.PLOTS):
        raise IndexError(f"plot_idx {plot_idx} out of range (have {len(viz.PLOTS)})")

    bundle = _load_bundle_for_plot(session_path, plot_idx)
    _label, fn = viz.PLOTS[plot_idx]
    rendered = fn(bundle)
    # Hand back only picklable primitives — bytes(...) ensures a real bytes copy
    # (not a numpy view) crosses the process boundary cleanly.
    return {
        "buffer": bytes(rendered.buffer),
        "width": int(rendered.width),
        "height": int(rendered.height),
        "stride": int(rendered.stride),
        "empty": bool(rendered.empty),
    }


def _load_bundle_for_plot(session_path: str, plot_idx: int):
    """Load just enough of the session to render plot ``plot_idx``.

    The coverage / walked-path plots need only the camera frame, so we skip
    parsing the big ``csi.csv`` for them and feed viz an empty CSI frame. The
    clap is always loaded (cheap) so CSI/camera get trimmed to the session
    window consistently with the full pipeline.
    """
    from csi_gui import viz

    csi, camera, clap = _load_session_tolerant(
        session_path, need_csi=_PLOT_NEEDS_CSI[plot_idx])
    return viz.load_for_viz(csi, camera, clap)


# Columns viz actually reads off the camera frame (7-col flavour). An empty
# camera frame with these columns lets the coverage / walked-path renderers fall
# through to their own "no detected camera frames" placeholders instead of the
# pipeline blowing up on a missing file.
_CAMERA_COLS = ("wall_time_s", "x_cm", "y_cm", "grid_x_cm", "grid_y_cm", "detected")


def _read_csv_or_empty(path: str, columns=()):
    """``pd.read_csv(path)`` but tolerant of a missing or 0-byte file.

    Returns an empty frame (with ``columns`` if given) when the file is absent or
    has no parseable header, so a partial session never raises mid-render.
    """
    import os

    import pandas as pd

    if not os.path.isfile(path):
        return pd.DataFrame(columns=list(columns))
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(columns))


def _load_session_tolerant(session_path: str, need_csi: bool = True):
    """Like ``dataset.load_session`` but tolerant + lazy about the big CSI read.

    Two robustness fixes over ``dataset.load_session``:

    * **Tolerant** — a capture with ``csi.csv`` but no ``camera.csv`` (the
      ``diag_*`` / no-camera runs) still appears in the explorer, so selecting it
      must not raise. Missing / empty camera / clap files become empty frames;
      viz then renders graceful "no camera data" placeholders rather than the
      worker dying with ``FileNotFoundError`` (which broke the Sessions page).
    * **Lazy** — the coverage / walked-path plots need ONLY the camera frame, so
      with ``need_csi=False`` we never parse ``csi.csv`` at all. That matters: a
      no-camera capture can carry a 130 MB ``csi.csv``, and reading it just to
      throw it away made those camera-only renders pathologically slow.

    Column-name normalization mirrors ``dataset.load_session``.
    """
    import os

    import pandas as pd

    if need_csi:
        csi = _read_csv_or_empty(os.path.join(session_path, "csi.csv"))
    else:
        csi = pd.DataFrame()  # camera-only plot: never touch the big csi.csv
    camera = _read_csv_or_empty(os.path.join(session_path, "camera.csv"),
                                columns=_CAMERA_COLS)
    if "timestamp_s" in camera.columns and "wall_time_s" not in camera.columns:
        camera = camera.rename(columns={"timestamp_s": "wall_time_s"})
    clap = _read_csv_or_empty(os.path.join(session_path, "clap.csv"))
    return csi, camera, clap


# ---------------------------------------------------------------------------
# Quality report (runs in the subprocess)
# ---------------------------------------------------------------------------

def compute_report(session_path: str) -> Dict[str, Any]:
    """Run ``validate_session.build_report`` here; return PICKLABLE rows + verdict.

    ``build_report`` builds the multiboard dataset TWICE (genuinely expensive and
    GIL-bound), so it runs in this worker process. The live ``Report`` object is
    NOT picklable-friendly to depend on across versions, so we flatten it to::

        {"rows": [{"status": "OK"|"WARN"|"FAIL", "label": str, "message": str},
                  ...],
         "verdict": "OK"|"WARN"|"FAIL"}

    which the GUI renders directly (no Report object ever crosses the boundary).
    """
    from validate_session import build_report

    report = build_report(session_path)
    rows = [
        {"status": str(level), "label": str(label), "message": str(detail or "")}
        for (level, label, detail) in report.rows
    ]
    return {"rows": rows, "verdict": str(report.worst())}
