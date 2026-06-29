"""Remediation actions for the pre-flight panel (subprocess, argv lists).

These are the one-click "Fix" buttons next to RED checks. They wrap the exact
commands from ``SESSION_CHECKLIST.md`` section A. All use argv lists (never
``shell=True``); the long-running ``iproxy`` is launched fully DETACHED so it
outlives the GUI process exactly like ``iproxy 8080 8080 &`` in a terminal.

Static-IP setting is NOT here — it needs admin + a crash-safe watchdog, so it
lives in :mod:`csi_gui.preflight.netconfig`.
"""

from __future__ import annotations

import subprocess

from csi_gui.preflight.probes import TARGET_SSID, WIFI_DEV

# Checklist step 4: join CSI_TX (the SoftAP password from the checklist).
_WIFI_PASSWORD = "23456789"

# Checklist step 7: iproxy 8080 8080 (iPhone USB camera tunnel).
_IPROXY_PORT = "8080"


def connect_wifi() -> tuple[bool, str]:
    """Join the CSI_TX network (``networksetup -setairportnetwork en0 CSI_TX ...``).

    Returns ``(ok, message)``. Note ``-setairportnetwork`` takes the DEVICE
    (en0), unlike ``-setmanual``/``-setdhcp`` which take the service name.
    """
    argv = ["networksetup", "-setairportnetwork", WIFI_DEV,
            TARGET_SSID, _WIFI_PASSWORD]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"connect failed: {exc}"
    out = (proc.stdout or "") + (proc.stderr or "")
    # networksetup returns 0 and prints nothing on success; some errors still
    # exit 0 but print text — surface any output.
    if proc.returncode != 0:
        return False, out.strip() or f"exit {proc.returncode}"
    return True, out.strip() or f"joining {TARGET_SSID}…"


def start_iproxy() -> tuple[bool, str]:
    """Launch ``iproxy 8080 8080`` DETACHED (survives the GUI). Returns quickly.

    Uses ``start_new_session=True`` so the child is in its own session/process
    group and is not killed when the GUI exits — matching ``iproxy ... &`` in the
    checklist. stdin/out/err are sent to /dev/null so it never blocks.
    """
    argv = ["iproxy", _IPROXY_PORT, _IPROXY_PORT]
    try:
        with open("/dev/null", "wb") as devnull:
            subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=devnull,
                stderr=devnull,
                start_new_session=True,
            )
    except OSError as exc:
        return False, f"could not start iproxy: {exc}"
    return True, "iproxy started on :8080"


def stop_iproxy() -> tuple[bool, str]:
    """Stop iproxy (``pkill -x iproxy``). Returns ``(ok, message)``.

    ``pkill`` exits 1 when nothing matched — treated as a benign no-op here.
    """
    argv = ["pkill", "-x", "iproxy"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not stop iproxy: {exc}"
    if proc.returncode == 0:
        return True, "iproxy stopped"
    if proc.returncode == 1:
        return True, "iproxy was not running"
    return False, (proc.stderr or "").strip() or f"exit {proc.returncode}"
