"""Static-IP set/revert for the CSI_TX Wi-Fi, with a CRASH-SAFE revert.

The laptop must hold the static IP ``192.168.4.200`` on the ``Wi-Fi`` service
while recording (RX boards unicast CSI there). That setting *sticks across
networks*, so if the GUI dies without reverting, the user's normal Wi-Fi has no
internet afterwards. This module makes the revert robust on three fronts:

  1. An **out-of-process root watchdog**, spawned (detached) inside the same
     elevated ``osascript`` that sets the IP. It reverts to DHCP as soon as the
     GUI pid dies OR the sentinel file is removed — so the IP reverts even on
     SIGKILL / power loss, with no further admin prompt.
  2. An **in-process belt** (:func:`register_crash_safe_revert`) wiring
     ``atexit`` + ``QApplication.aboutToQuit`` + SIGINT/SIGTERM to
     :func:`revert_dhcp` for the normal clean-exit path.
  3. A **sentinel file** (``~/.csi_static_ip_active``) that both sides watch, so
     either side reverting (and removing it) tells the other it is done.

Everything runs via argv lists / quoted osascript — never ``shell=True`` on
attacker-controllable input.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading

# ── constants ───────────────────────────────────────────────────────────────
IP = "192.168.4.200"
MASK = "255.255.255.0"
ROUTER = "192.168.4.1"
SERVICE = "Wi-Fi"
WIFI_DEVICE = "en0"
SENTINEL = os.path.expanduser("~/.csi_static_ip_active")
# Optional home-network config (LOCAL ONLY, never committed): lets "Revert now"
# rejoin the user's normal Wi-Fi and restore internet. JSON:
# {"home_ssid": "...", "home_password": "..."} (password optional — macOS keychain
# can reconnect by SSID). Absent -> revert just goes to DHCP, no reconnect.
HOME_CONFIG = os.path.expanduser("~/.csi_gui_local.json")

# Guard so revert_dhcp() only does real work once, and so we don't double-wire
# the in-process handlers. Protected by a lock for thread-safety.
_lock = threading.Lock()
_reverted = False
_handlers_registered = False


# ── osascript builders (pure — the tests assert these argv shapes) ───────────
def build_set_static_osascript(gui_pid: int) -> list[str]:
    """The exact ``osascript`` argv that sets the IP + spawns the watchdog.

    One elevated ``do shell script`` that (a) sets the manual IP, (b) touches the
    sentinel, and (c) backgrounds a DETACHED root watchdog reverting to DHCP when
    ``gui_pid`` dies OR the sentinel is removed. Backgrounding inside the single
    elevated script means the watchdog inherits root with no second prompt.
    """
    # The service name must be quoted to the SHELL ("Wi-Fi" has no space, but
    # other service names can), and those quotes live INSIDE the AppleScript
    # ``do shell script "..."`` string — so they must be escaped as \" or the
    # AppleScript string would terminate early. q = the escaped quote.
    q = '\\"'
    inner = (
        f'networksetup -setmanual {q}{SERVICE}{q} {IP} {MASK} {ROUTER}; '
        f'touch {SENTINEL}; '
        f'(while kill -0 {int(gui_pid)} 2>/dev/null && [ -f {SENTINEL} ]; '
        f'do sleep 5; done; '
        f'networksetup -setdhcp {q}{SERVICE}{q}; '
        f'rm -f {SENTINEL}) >/dev/null 2>&1 &'
    )
    return [
        "osascript",
        "-e",
        f'do shell script "{inner}" with administrator privileges',
    ]


def build_revert_osascript() -> list[str]:
    """``osascript`` argv that reverts to DHCP (admin) — the elevated fallback."""
    q = '\\"'
    inner = f'networksetup -setdhcp {q}{SERVICE}{q}'
    return [
        "osascript",
        "-e",
        f'do shell script "{inner}" with administrator privileges',
    ]


# ── sentinel ─────────────────────────────────────────────────────────────────
def is_static_active() -> bool:
    """True while the sentinel exists (static IP believed active)."""
    return os.path.exists(SENTINEL)


def _remove_sentinel() -> None:
    try:
        os.remove(SENTINEL)
    except OSError:
        pass


# ── set / revert ─────────────────────────────────────────────────────────────
def set_static_ip(gui_pid: int | None = None) -> bool:
    """Set the static IP and arm the crash-safe watchdog. Returns True on success.

    Runs ONE elevated ``osascript`` (single admin prompt) built by
    :func:`build_set_static_osascript`. On success the in-process revert guard is
    re-armed so a later clean exit still reverts.
    """
    global _reverted
    if gui_pid is None:
        gui_pid = os.getpid()
    argv = build_set_static_osascript(int(gui_pid))
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    with _lock:
        _reverted = False
    return True


def revert_dhcp(use_admin: bool = False) -> bool:
    """Revert the Wi-Fi service to DHCP and drop the sentinel. Idempotent.

    Guarded by a module flag so it only acts ONCE per static-IP session (repeated
    calls — atexit + aboutToQuit + a signal — collapse to a single real revert).
    ``networksetup -setdhcp`` usually needs no admin; pass ``use_admin=True`` to
    wrap it in an elevated ``osascript`` if it does.
    """
    global _reverted
    with _lock:
        if _reverted:
            return True
        _reverted = True

    ok = False
    try:
        if use_admin:
            argv = build_revert_osascript()
        else:
            argv = ["networksetup", "-setdhcp", SERVICE]
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=60, check=False)
        ok = proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    finally:
        # Always drop the sentinel: it both reflects "no longer static" and
        # signals the out-of-process watchdog to exit (it ANDs on the sentinel).
        _remove_sentinel()
    return ok


def load_home_network() -> tuple[str, str | None] | None:
    """Read (ssid, password) from ``~/.csi_gui_local.json``, or None if unset.

    LOCAL-ONLY file (outside the repo) so the Wi-Fi password is never in tracked
    source. ``home_password`` may be omitted (keychain reconnects by SSID alone).
    """
    try:
        import json
        with open(HOME_CONFIG) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    ssid = (data.get("home_ssid") or "").strip()
    if not ssid:
        return None
    return ssid, (data.get("home_password") or None)


def reconnect_home() -> bool:
    """Best-effort: rejoin the configured home Wi-Fi to restore internet.

    No-op (returns False) when no home network is configured. Joins via
    ``networksetup -setairportnetwork en0 <ssid> [password]``.
    """
    home = load_home_network()
    if home is None:
        return False
    ssid, password = home
    argv = ["networksetup", "-setairportnetwork", WIFI_DEVICE, ssid]
    if password:
        argv.append(password)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=30, check=False)
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def revert_and_reconnect(use_admin: bool = False) -> bool:
    """The "Revert now" action: revert the static IP to DHCP, then rejoin the home
    Wi-Fi. Reconnect is best-effort and never blocks the (critical) DHCP revert."""
    ok = revert_dhcp(use_admin=use_admin)
    reconnect_home()
    return ok


def _arm_for_new_session() -> None:
    """Re-arm the revert guard (used by tests / a fresh set_static_ip)."""
    global _reverted
    with _lock:
        _reverted = False


# ── in-process crash-safe wiring ─────────────────────────────────────────────
def register_crash_safe_revert(qapp=None) -> None:
    """Wire atexit + QApplication.aboutToQuit + SIGINT/SIGTERM to revert_dhcp().

    The in-process belt complementing the out-of-process watchdog: on any normal
    teardown path the laptop returns to DHCP. Idempotent — only wires once. The
    signal handlers chain to any previously installed handler (so Qt / the
    interpreter still see the signal) and only revert when the static IP is
    actually active, leaving unrelated Ctrl-C untouched.
    """
    global _handlers_registered
    with _lock:
        if _handlers_registered:
            already = True
        else:
            _handlers_registered = True
            already = False
    if already:
        if qapp is not None:
            _connect_about_to_quit(qapp)
        return

    atexit.register(_atexit_revert)

    if qapp is not None:
        _connect_about_to_quit(qapp)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev = signal.getsignal(sig)
            signal.signal(sig, _make_signal_handler(sig, prev))
        except (ValueError, OSError, RuntimeError):
            # e.g. called off the main thread — atexit/aboutToQuit still cover us.
            pass


def _connect_about_to_quit(qapp) -> None:
    try:
        qapp.aboutToQuit.connect(lambda: revert_dhcp())
    except (RuntimeError, AttributeError):
        pass


def _atexit_revert() -> None:
    if is_static_active():
        revert_dhcp()


def _make_signal_handler(sig, prev_handler):
    def _handler(signum, frame):
        if is_static_active():
            revert_dhcp()
        # Chain to whatever was installed before us so default behavior holds.
        if callable(prev_handler):
            prev_handler(signum, frame)
        elif prev_handler == signal.SIG_DFL:
            signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)
    return _handler
