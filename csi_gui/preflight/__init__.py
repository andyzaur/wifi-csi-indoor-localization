"""Stage-1 guided pre-flight for the Record section.

This package maps section A of ``SESSION_CHECKLIST.md`` ("Pre-flight — do once at
start of the day") into the native PySide6 GUI. It is deliberately split so the
*logic* stays Qt-free and unit-testable, while only the driver + panel touch Qt:

  * ``probes``    — PURE check functions. Each runs a small read-only system
                    command (argv lists, never ``shell=True``, hard timeout),
                    parses stdout, and returns a :class:`CheckResult`
                    (GREEN / YELLOW / RED + detail + one-line fix hint).
  * ``netconfig`` — static-IP set/revert with a CRASH-SAFE out-of-process
                    watchdog plus an in-process belt (atexit / aboutToQuit /
                    signals) so the laptop reverts to DHCP even on SIGKILL.
  * ``actions``   — remediation side effects (connect Wi-Fi, start/stop iproxy).
  * ``engine``    — the Qt-free PreflightEngine: the ordered list of Checks and a
                    persistent CsiCollector board-rate listener.
  * ``scheduler`` — the Qt driver (QTimer + QThreadPool) that runs the checks off
                    the GUI thread on a cadence and emits ``checkUpdated``.

The panel widget lives in ``csi_gui.ui.preflight_panel``.
"""

from csi_gui.preflight.probes import GREEN, RED, YELLOW, CheckResult

__all__ = ["CheckResult", "GREEN", "YELLOW", "RED"]
