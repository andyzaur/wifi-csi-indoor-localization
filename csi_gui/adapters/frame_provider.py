"""LiveFrameProvider: a one-slot QImage provider for the QML live view.

The ArUco tracker's worker thread produces a small RGB ndarray; the QML GUI
thread asks for a QImage via ``image://live/...``. Those two threads never share
the numpy buffer:

  * ``set_frame(rgb)`` (worker thread) wraps the ndarray in a QImage and
    immediately ``.copy()``-es it, so the QImage owns its own pixels and the
    numpy array can be freed/reused the instant set_frame returns. No dangling
    buffer, ever.
  * ``requestImage(...)`` (GUI thread) hands back the stored QImage under the
    same lock.

Only the *latest* frame is kept — a newer set_frame overwrites the previous one
outright. Nothing is queued, so a slow GUI thread can never make the worker
thread back up: it just drops intermediate frames.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider


def _placeholder(width: int = 16, height: int = 9) -> QImage:
    """A tiny opaque gray QImage shown before the first real frame arrives."""
    img = QImage(width, height, QImage.Format.Format_RGB888)
    img.fill(0xFF404040)  # ARGB; alpha ignored for RGB888 but keeps it opaque
    return img


class LiveFrameProvider(QQuickImageProvider):
    """Holds exactly one current QImage, swapped atomically under a lock."""

    def __init__(self) -> None:
        super().__init__(QQuickImageProvider.ImageType.Image)
        # A plain threading.Lock is enough: the critical section is a single
        # reference swap / read, and both ends are ordinary Python threads.
        self._lock = threading.Lock()
        self._image: QImage = _placeholder()

    def set_frame(self, rgb: np.ndarray) -> None:
        """Store ``rgb`` (H x W x 3, uint8, RGB) as the new current frame.

        Callable from ANY thread (typically the tracker worker). The ndarray is
        made contiguous, wrapped, and *copied* so the stored QImage no longer
        references the caller's buffer.
        """
        if rgb is None:
            return
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"set_frame expects an H x W x 3 RGB array, got shape {rgb.shape!r}")

        # QImage(memoryview) does not own the bytes; .copy() detaches it onto
        # Qt-owned storage so the numpy buffer's lifetime cannot dangle.
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line,
                      QImage.Format.Format_RGB888).copy()

        with self._lock:
            self._image = qimg

    def requestImage(self, _id: str, size, _requested):  # noqa: N802 (Qt name)
        """Return the current QImage (GUI thread). Never blocks on the worker.

        ``size`` is an out-param Qt reads to learn the native frame size; we set
        it so QML's PreserveAspectFit has the real aspect ratio even before the
        Image element measures it.
        """
        with self._lock:
            img = self._image
        if size is not None:
            size.setWidth(img.width())
            size.setHeight(img.height())
        return img
