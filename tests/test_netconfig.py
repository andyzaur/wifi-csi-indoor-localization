"""Tests for csi_gui.preflight.netconfig — static-IP set/revert builders + guards.

These NEVER touch the real network: subprocess is monkeypatched so we only assert
the osascript / networksetup argv that *would* run, plus sentinel + idempotency
behavior. The sentinel path is redirected to a temp file.
"""

import os

import pytest

netconfig = pytest.importorskip("csi_gui.preflight.netconfig")


@pytest.fixture(autouse=True)
def _temp_sentinel(monkeypatch, tmp_path):
    """Redirect the sentinel to a temp file and reset the module guards."""
    sentinel = str(tmp_path / ".csi_static_ip_active")
    monkeypatch.setattr(netconfig, "SENTINEL", sentinel)
    # Reset module-level guards between tests.
    netconfig._reverted = False
    yield sentinel
    netconfig._reverted = False


# ---------------------------------------------------------------------------
# build_set_static_osascript — the exact crash-safe form.
# ---------------------------------------------------------------------------
def test_build_set_static_osascript_shape():
    argv = netconfig.build_set_static_osascript(4242)
    assert argv[0] == "osascript"
    assert argv[1] == "-e"
    script = argv[2]

    # One elevated do-shell-script with admin privileges.
    assert script.startswith("do shell script ")
    assert "with administrator privileges" in script

    # (a) sets the manual IP on the Wi-Fi SERVICE (not en0); the service-name
    #     quotes are escaped (\") so the AppleScript string survives intact.
    assert 'networksetup -setmanual \\"Wi-Fi\\" 192.168.4.200 ' \
           '255.255.255.0 192.168.4.1' in script
    # The escaped-quote form is exactly what the task spec mandates.
    assert '\\"Wi-Fi\\"' in script
    # (b) touches the sentinel
    assert f"touch {netconfig.SENTINEL}" in script
    # (c) spawns a DETACHED watchdog keyed on the real pid AND the sentinel,
    #     reverting to DHCP + removing the sentinel, backgrounded with '&'.
    assert "kill -0 4242" in script
    assert f"[ -f {netconfig.SENTINEL} ]" in script
    assert 'networksetup -setdhcp \\"Wi-Fi\\"' in script
    assert f"rm -f {netconfig.SENTINEL}" in script
    # The watchdog subshell is backgrounded (&) right before the AppleScript
    # string closes and the privileges clause begins.
    assert '&" with administrator privileges' in script


def test_build_revert_osascript_shape():
    argv = netconfig.build_revert_osascript()
    assert argv[0] == "osascript"
    assert 'networksetup -setdhcp \\"Wi-Fi\\"' in argv[2]
    assert "with administrator privileges" in argv[2]


# ---------------------------------------------------------------------------
# set_static_ip — builds + runs the osascript with the REAL pid; no real net.
# ---------------------------------------------------------------------------
def test_set_static_ip_uses_real_pid_and_runs_osascript(monkeypatch):
    captured = {}

    class _Proc:
        returncode = 0

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _Proc()
    monkeypatch.setattr(netconfig.subprocess, "run", fake_run)

    ok = netconfig.set_static_ip()  # default pid = os.getpid()
    assert ok is True
    argv = captured["argv"]
    assert argv[0] == "osascript"
    assert f"kill -0 {os.getpid()}" in argv[2]


def test_set_static_ip_explicit_pid(monkeypatch):
    captured = {}

    class _Proc:
        returncode = 0

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _Proc()
    monkeypatch.setattr(netconfig.subprocess, "run", fake_run)

    assert netconfig.set_static_ip(98765) is True
    assert "kill -0 98765" in captured["argv"][2]


def test_set_static_ip_returns_false_on_nonzero(monkeypatch):
    class _Proc:
        returncode = 1

    monkeypatch.setattr(netconfig.subprocess, "run", lambda argv, **k: _Proc())
    assert netconfig.set_static_ip() is False


def test_set_static_ip_returns_false_on_oserror(monkeypatch):
    def boom(argv, **kwargs):
        raise OSError("nope")
    monkeypatch.setattr(netconfig.subprocess, "run", boom)
    assert netconfig.set_static_ip() is False


# ---------------------------------------------------------------------------
# revert_dhcp — idempotent, drops the sentinel, runs setdhcp once.
# ---------------------------------------------------------------------------
def test_revert_dhcp_runs_setdhcp_and_removes_sentinel(monkeypatch, _temp_sentinel):
    calls = []

    class _Proc:
        returncode = 0

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _Proc()
    monkeypatch.setattr(netconfig.subprocess, "run", fake_run)

    # Pretend the static IP is active.
    open(netconfig.SENTINEL, "w").close()
    assert netconfig.is_static_active() is True

    assert netconfig.revert_dhcp() is True
    assert calls == [["networksetup", "-setdhcp", "Wi-Fi"]]
    assert netconfig.is_static_active() is False


def test_revert_dhcp_is_idempotent(monkeypatch):
    calls = []

    class _Proc:
        returncode = 0

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _Proc()
    monkeypatch.setattr(netconfig.subprocess, "run", fake_run)

    assert netconfig.revert_dhcp() is True
    # Second call must NOT run setdhcp again (guarded by the module flag).
    assert netconfig.revert_dhcp() is True
    assert len(calls) == 1


def test_revert_dhcp_admin_path_uses_osascript(monkeypatch):
    captured = {}

    class _Proc:
        returncode = 0

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _Proc()
    monkeypatch.setattr(netconfig.subprocess, "run", fake_run)

    assert netconfig.revert_dhcp(use_admin=True) is True
    assert captured["argv"][0] == "osascript"


# ---------------------------------------------------------------------------
# is_static_active via a temp sentinel.
# ---------------------------------------------------------------------------
def test_is_static_active_tracks_sentinel():
    assert netconfig.is_static_active() is False
    open(netconfig.SENTINEL, "w").close()
    assert netconfig.is_static_active() is True
    os.remove(netconfig.SENTINEL)
    assert netconfig.is_static_active() is False


# ---------------------------------------------------------------------------
# register_crash_safe_revert — wires atexit/signals once, reverts only when
# the sentinel is active (does not hijack unrelated Ctrl-C).
# ---------------------------------------------------------------------------
def test_register_crash_safe_revert_is_idempotent(monkeypatch):
    registered = []
    monkeypatch.setattr(netconfig.atexit, "register",
                        lambda fn: registered.append(fn))
    # Avoid touching real signal handlers in the test process.
    monkeypatch.setattr(netconfig.signal, "signal", lambda *a, **k: None)
    netconfig._handlers_registered = False

    netconfig.register_crash_safe_revert(None)
    netconfig.register_crash_safe_revert(None)
    # atexit.register called exactly once despite two register calls.
    assert len(registered) == 1
    netconfig._handlers_registered = False  # reset for other tests


def test_atexit_revert_noop_when_not_static(monkeypatch):
    called = {"reverted": False}
    monkeypatch.setattr(netconfig, "revert_dhcp",
                        lambda *a, **k: called.__setitem__("reverted", True))
    # Sentinel absent -> no revert attempted.
    netconfig._atexit_revert()
    assert called["reverted"] is False
