"""AppContext: a tiny shared-state object passed to every shell page.

The shell builds one AppContext and hands it to each page. It holds the camera
URL so the Calibrate page and the Record page agree on which source to use (edit
it on one, both subprocesses / the tracker pick it up), plus an OPTIONAL
diagnostic ``logger`` callable (see :mod:`csi_gui.applog`) that the Record page
and pre-flight panel route their events into when ``--log-file`` was passed.

Kept deliberately framework-light (plain object, no Qt) so it stays trivially
importable and testable. ``ROOT`` is the repository root — the directory that
holds ``floor_calibration.json`` etc. — used both as the subprocess ``cwd`` and
as the base for reading the read-only calibration JSONs.
"""

from __future__ import annotations

import os
from typing import Callable

# csi_gui/ -> repo root (one level up from this file's package dir).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CAMERA_URL = "http://127.0.0.1:8080/video"


class AppContext:
    """Mutable shared state for the shell pages.

    ``logger`` is an optional 1-arg callable ``log(message: str) -> None`` (the
    :class:`csi_gui.applog.AppLogger` instance is itself such a callable). It is
    ``None`` unless ``--log-file`` was given, in which case Record/pre-flight
    events are appended to that file. Use :meth:`log` to write through it safely
    (a no-op when no logger is configured).
    """

    def __init__(self, camera_url: str | None = None, root: str | None = None,
                 logger: Callable[[str], None] | None = None) -> None:
        self.camera_url: str = camera_url or DEFAULT_CAMERA_URL
        self.root: str = root or ROOT
        self.logger: Callable[[str], None] | None = logger

    def log(self, message: str) -> None:
        """Route ``message`` to the configured logger; a no-op if none is set."""
        if self.logger is not None:
            self.logger(message)
