"""Lightweight diagnostic logger for the CSI GUI.

A tiny, dependency-free file logger that appends TIMESTAMPED lines to a path so a
recording session can be inspected mid-run. It is intentionally NOT the stdlib
``logging`` module: the GUI needs a single callable it can hand to the
ArucoTracker's ``on_log`` and call from the preflight panel / Start/Stop events,
and it must stay readable while the file is open (each line is flushed).

Usage::

    log = AppLogger("/tmp/csi.log")     # opens + appends a session banner
    log("Start tracker: http://...")    # the logger *is* the callable
    log.close()                         # flush + close (idempotent)

When no ``--log-file`` is given the GUI keeps the previous behaviour (the tracker
prints to stdout, preflight transitions are not persisted): callers simply leave
the logger as ``None`` and fall back to ``print`` / nothing.

The logger is thread-safe: the ArucoTracker runs on a daemon worker thread while
the preflight scheduler emits on the GUI thread, so both can call it
concurrently. A small lock serialises the ``write``+``flush`` so lines never
interleave.
"""

from __future__ import annotations

import threading
import time
from typing import TextIO


def _timestamp() -> str:
    """A millisecond wall-clock stamp, e.g. ``2026-06-06 14:03:21.482``."""
    now = time.time()
    local = time.localtime(now)
    ms = int((now - int(now)) * 1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", local) + f".{ms:03d}"


class AppLogger:
    """Append timestamped lines to a file; the instance is itself callable.

    Opening uses line buffering (``buffering=1``) and every write is followed by
    an explicit ``flush`` so the file is tailable while the GUI runs. All public
    methods are safe to call after :meth:`close` (they become no-ops), so the
    crash/shutdown paths never have to guard the logger.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._fh: TextIO | None = open(path, "a", buffering=1, encoding="utf-8")
        self.log(f"=== csi_gui log opened ({_timestamp()}) ===")

    def log(self, message: str) -> None:
        """Write one timestamped line (``message`` may be any printable object)."""
        with self._lock:
            fh = self._fh
            if fh is None:
                return
            try:
                fh.write(f"{_timestamp()}  {message}\n")
                fh.flush()
            except (ValueError, OSError):
                # File closed underneath us, or disk error — never propagate a
                # logging failure into the GUI / tracker thread.
                pass

    # The logger is its own callable so it can be passed straight as the
    # ArucoTracker's ``on_log`` and as the panel's log target.
    def __call__(self, message: str) -> None:
        self.log(message)

    def close(self) -> None:
        """Flush + close the file. Idempotent and safe to call from any thread."""
        with self._lock:
            fh, self._fh = self._fh, None
            if fh is None:
                return
            try:
                fh.flush()
                fh.close()
            except (ValueError, OSError):
                pass
