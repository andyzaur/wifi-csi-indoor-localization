"""Entry point: ``python -m csi_gui.app``.

Launches the 5-section sidebar SHELL for the WiFi-CSI data-collection GUI:

    Calibrate -> Record -> Sessions -> Train -> Live-validate

No camera is required at launch. ``--camera`` / ``--video`` are OPTIONAL and, if
given, just prefill the camera field shared by the Calibrate and Record pages.
The ArucoTracker is no longer created here — the Record page constructs + starts
it when the user presses **Start**, and stops/joins it on **Stop** and on app
shutdown (``QApplication.aboutToQuit`` is wired to the shell's ``stop_tracker``).

The live-preview path itself (LiveFrameProvider + CameraBridge + the QQuickWidget
hosting live_view.qml, with queued frameReady / positionUpdated signals) is reused
verbatim on the Record page.
"""

from __future__ import annotations

import argparse
import signal
import sys
import traceback

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from csi_gui.app_context import DEFAULT_CAMERA_URL, AppContext
from csi_gui.applog import AppLogger
from csi_gui.preflight import netconfig
from csi_gui.ui import theme
from csi_gui.ui.main_window import MainWindow


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="csi_gui.app",
        description="Sidebar shell for the WiFi-CSI data-collection GUI "
                    "(Calibrate / Record / Sessions / Train / Live-validate).")
    parser.add_argument("--camera", "-c", type=str, default=None,
                        help="Optional camera URL or OpenCV index to prefill the "
                             "shared camera field (e.g. http://127.0.0.1:8080/video).")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Optional recorded video path to prefill the shared "
                             "camera field instead of a live camera.")
    parser.add_argument("--log-file", type=str, default=None, metavar="PATH",
                        help="Optional file to append TIMESTAMPED diagnostic lines "
                             "to (tracker stdout, pre-flight check transitions, "
                             "Start/Stop events). Default: log to stdout only.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    app = QApplication(sys.argv[:1])
    # Global "lab instrument" theme (base chrome); pages add their specifics.
    theme.apply(app)

    # Optional diagnostic file logger. Without --log-file behaviour is unchanged
    # (tracker -> stdout, pre-flight transitions not persisted).
    logger = AppLogger(args.log_file) if args.log_file else None
    if logger is not None:
        app.aboutToQuit.connect(logger.close)

    # Defense-in-depth: a stray exception escaping a Qt slot (e.g. a worker-pool
    # failure on the Sessions page) terminates the app under PySide6. Installing
    # our own sys.excepthook makes such an exception LOG instead of abort, so one
    # bad session can never take the whole app down mid-recording. The individual
    # slots still handle their own errors; this is only the last-resort net.
    def _excepthook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(text)
        if logger is not None:
            logger("UNCAUGHT EXCEPTION (kept alive by excepthook):\n" + text)

    sys.excepthook = _excepthook

    camera_url = args.camera or args.video or DEFAULT_CAMERA_URL
    context = AppContext(camera_url=camera_url, logger=logger)
    # Building the shell (QML compile + session scan) takes a few seconds; say
    # so, or a quiet terminal reads as a hang and invites a Ctrl+C mid-build.
    print("CSI Collector: starting… (the window appears in a few seconds)",
          flush=True)
    window = MainWindow(context=context)

    # Stop any tracker the Record page started before the event loop tears down.
    app.aboutToQuit.connect(window.stop_tracker)

    # Shut the Sessions page's process pool down cleanly on exit so no worker
    # subprocess is orphaned (zombie) after the GUI closes.
    app.aboutToQuit.connect(window.sessions_page.shutdown)

    # In-process belt for the static-IP revert: wires atexit + aboutToQuit +
    # SIGINT/SIGTERM to revert_dhcp() so the laptop returns to DHCP on any clean
    # exit (the out-of-process root watchdog covers SIGKILL / power loss).
    netconfig.register_crash_safe_revert(app)

    # Make Ctrl+C in the launching terminal actually stop the app. Qt's C++ event
    # loop normally never hands control back to the Python interpreter, so the
    # SIGINT handler would never run (the classic "Ctrl+C does nothing" on a Qt
    # app). A periodic no-op timer wakes the interpreter ~5x/s so the handler
    # fires; the handler asks Qt to quit cleanly, which runs aboutToQuit (DHCP
    # revert + tracker stop + pool shutdown) on the way out.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    _sigint_timer = QTimer()
    _sigint_timer.start(200)
    _sigint_timer.timeout.connect(lambda: None)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
