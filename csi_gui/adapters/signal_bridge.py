"""CameraBridge: the tracker-worker -> GUI-thread frame/position relay.

``on_frame`` runs on the ArUco tracker's worker thread. It must stay cheap and
must never touch a QML/QQuickItem directly. Its job:

  1. MEASURE the true capture/tracking fps. ``on_frame`` is invoked for EVERY
     tracker frame (before any throttling), so the bridge is the right place to
     measure the real rate. We keep a rolling window of recent frame arrival
     times (``time.monotonic()``) and compute fps = (n-1)/(t_last - t_first).
     This replaces the broken ``PositionState.fps`` (aruco_track resets its
     internal cur_fps to 0.0 every frame and only recomputes it every ~2s, so
     the old readout showed ~0 almost always).
  2. STEADY time-based throttle of the preview. The tracker may run at 25-60 fps;
     rendering every frame is wasted CPU. We forward a frame only when at least
     ``1.0 / target_fps`` seconds have elapsed since the last forwarded frame
     (wall-clock, not a stride derived from the broken in_fps). This decouples
     delivery from the tracker and gives steady, even frames.
  3. For forwarded frames: convert the small preview BGR->RGB (a single
     vectorised channel reverse + contiguity copy — no Python pixel loop) and
     hand it to the LiveFrameProvider (latest-frame-only).
  4. Emit ``frameReady(frame_id)``, ``positionUpdated(position)`` and
     ``fpsMeasured(fps)``. These are auto-connected as *queued* signals across
     the thread boundary, so their slots run on the GUI thread; the worker just
     enqueues and returns.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from PySide6.QtCore import QObject, Signal

# Preview delivery cap. The throttle quantizes to input-frame boundaries, so a
# value just above the camera's rate forwards every frame (smoothest) while still
# capping a high-fps source. At ~25-30 fps cameras this delivers the full rate;
# preflight is paused during recording + the preview is ~800px, so it's cheap.
TARGET_FPS = 30.0
_FPS_WINDOW = 30  # rolling-window size for the fps estimate

# The overlay text (position numbers) + the fps readout don't need a per-frame
# update — refreshing them ~5x/s reads as "live" while keeping the GUI event
# rate low. frameReady (which drives the actual image) stays at the preview
# cadence; only positionUpdated + fpsMeasured are clamped to this rate.
_OVERLAY_FPS = 5.0


class CameraBridge(QObject):
    """Relay worker-thread frames to the GUI thread, throttled to ~target_fps."""

    # frameReady carries a monotonically increasing frame id (cache-buster for
    # the QML Image source). positionUpdated carries the PositionState object.
    # fpsMeasured carries the bridge-measured true capture/tracking fps.
    frameReady = Signal(int)
    positionUpdated = Signal(object)
    fpsMeasured = Signal(float)

    def __init__(self, provider, target_fps: float = TARGET_FPS,
                 fps_window: int = _FPS_WINDOW, parent=None,
                 clock=time.monotonic, overlay_fps: float = _OVERLAY_FPS):
        super().__init__(parent)
        self._provider = provider
        self._target_fps = float(target_fps)
        # Minimum wall-clock gap between forwarded frames (the steady throttle).
        self._min_interval = (1.0 / self._target_fps) if self._target_fps > 0 else 0.0
        self._frame_id = 0          # monotonic id of frames PUSHED to the GUI
        self._clock = clock         # injectable monotonic clock (tests)
        # Rolling window of recent frame arrival times for the fps estimate.
        self._arrivals: deque[float] = deque(maxlen=int(fps_window))
        self._last_emit = None      # monotonic time of the last FORWARDED frame
        # Separate, slower throttle for positionUpdated + fpsMeasured: these only
        # drive overlay TEXT, not the image, so a few Hz is plenty and keeps the
        # GUI event rate down. None until the first overlay emit.
        self._overlay_fps = float(overlay_fps)
        self._overlay_interval = (1.0 / self._overlay_fps) if self._overlay_fps > 0 else 0.0
        self._last_overlay_emit = None

    def set_target_fps(self, fps: float) -> None:
        """Live-adjust the preview delivery cap (recomputes the throttle gap).

        The user can trade smoothness for CPU mid-recording: a higher target
        forwards more frames (smoother, more CPU), a lower one fewer. Detection
        is unaffected — it always runs full-res on the tracker; only the GUI
        preview cadence changes. ``fps <= 0`` forwards every frame (no throttle).
        """
        self._target_fps = float(fps)
        self._min_interval = (1.0 / self._target_fps) if self._target_fps > 0 else 0.0

    @property
    def target_fps(self) -> float:
        return self._target_fps

    def measured_fps(self) -> float:
        """True capture/tracking fps over the rolling arrival window (0 if <2)."""
        arrivals = self._arrivals
        n = len(arrivals)
        if n < 2:
            return 0.0
        span = arrivals[-1] - arrivals[0]
        if span <= 0.0:
            return 0.0
        return (n - 1) / span

    def on_frame(self, fr) -> None:
        """Tracker WORKER-THREAD callback. Measure, throttle, convert, emit.

        Stays cheap: records every arrival for the fps estimate, then forwards a
        frame only on the steady wall-clock cadence.
        """
        now = self._clock()
        # Record EVERY arrival so the fps estimate reflects the true rate, even
        # for frames we throttle away below.
        self._arrivals.append(now)

        # Steady time-based throttle: forward only if enough wall-clock time has
        # elapsed since the last forwarded frame. First frame always forwards.
        if self._last_emit is not None and (now - self._last_emit) < self._min_interval:
            return
        self._last_emit = now

        preview = fr.preview_bgr
        if preview is not None:
            # BGR -> RGB: reverse the last axis, then make contiguous so the
            # provider can wrap it without a stride surprise. Vectorised; no loop.
            rgb = np.ascontiguousarray(preview[..., ::-1])
            self._provider.set_frame(rgb)

        self._frame_id += 1
        # Queued across the thread boundary: slots fire on the GUI thread.
        # frameReady drives the actual image -> every forwarded (preview-cadence)
        # frame.
        self.frameReady.emit(self._frame_id)

        # positionUpdated + fpsMeasured only refresh overlay TEXT, so clamp them
        # to the slower overlay cadence (~5 Hz). This is the dominant saving in
        # the GUI event rate during recording.
        if (self._last_overlay_emit is None
                or (now - self._last_overlay_emit) >= self._overlay_interval):
            self._last_overlay_emit = now
            self.positionUpdated.emit(fr.position)
            self.fpsMeasured.emit(self.measured_fps())
