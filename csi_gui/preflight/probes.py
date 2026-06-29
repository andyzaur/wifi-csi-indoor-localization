"""Pure pre-flight check functions (Qt-free, unit-testable).

Each public ``check_*`` runs a small READ-ONLY system command and returns a
:class:`CheckResult` whose ``status`` is one of :data:`GREEN`, :data:`YELLOW`,
:data:`RED`. RED/YELLOW results carry a one-line ``hint`` — the matching fix from
``SESSION_CHECKLIST.md`` ("Common failure modes").

Design rules (so this stays safe + testable):

  * Subprocess is invoked with an **argv list** (never ``shell=True``) and a hard
    ``timeout`` — a hung command degrades to YELLOW, never blocks the caller.
  * The *parsing* is split out into pure ``parse_*`` helpers that take captured
    stdout text. The unit tests feed those parsers recorded fixtures, so the
    suite never touches the real network / system.
  * A command that is missing or errors out is reported, not raised.

These functions are called off the GUI thread by ``scheduler``.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

# ── status constants ────────────────────────────────────────────────────────
GREEN = "GREEN"
YELLOW = "YELLOW"
RED = "RED"

# Default hard timeout for every probe subprocess (seconds). ping/curl override.
_DEFAULT_TIMEOUT = 6.0

# The static IP / network constants this whole stage is built around.
STATIC_IP = "192.168.4.200"
ROUTER_IP = "192.168.4.1"
TARGET_SSID = "CSI_TX"
WIFI_DEV = "en0"

# CSI_TX SoftAP hands out 192.168.4.x; holding an en0 IPv4 in this /24 is a
# location-permission-free signal that we are joined to CSI_TX. On macOS Tahoe
# the SSID itself is masked as "<redacted>" without Location Services.
CSI_TX_SUBNET_PREFIX = "192.168.4."
# Values macOS reports when the SSID is unreadable (Location Services off).
_REDACTED_SSID = "<redacted>"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one pre-flight probe.

    ``status`` is GREEN/YELLOW/RED. ``detail`` is a short human-readable line for
    the row. ``hint`` is the one-line fix (empty for GREEN, set for RED/YELLOW).
    ``raw`` keeps the captured stdout for debugging (never shown in the UI).
    """

    status: str
    detail: str
    hint: str = ""
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.status == GREEN


# ── subprocess helper ───────────────────────────────────────────────────────
def _run(argv: list[str], timeout: float = _DEFAULT_TIMEOUT) -> tuple[int, str]:
    """Run ``argv`` (argv list, no shell), return ``(returncode, stdout+stderr)``.

    Never raises for an ordinary failure: a missing binary, a non-zero exit, or a
    timeout all return a sentinel ``returncode`` (-1 timeout, -2 OSError) plus
    whatever text was captured, so callers can map them to YELLOW/RED.
    """
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        return -1, f"timeout after {timeout}s: {exc}"
    except OSError as exc:
        return -2, f"could not run {argv[0]!r}: {exc}"


# ── pure parsers (fed recorded stdout fixtures by the tests) ─────────────────
def parse_ethernet_devices(listallhardwareports_stdout: str) -> list[str]:
    """Devices whose Hardware Port mentions 'Ethernet' (e.g. en4, en5, en6).

    ``networksetup -listallhardwareports`` prints blocks of::

        Hardware Port: Ethernet Adapter (en4)
        Device: en4
        Ethernet Address: ...

    We return the Device for any block whose Hardware Port contains 'Ethernet'
    (case-insensitive) but NOT 'Wi-Fi' (so the Wi-Fi port — which also has an
    "Ethernet Address" line — is never treated as wired).
    """
    devices: list[str] = []
    port = None
    for line in listallhardwareports_stdout.splitlines():
        line = line.strip()
        if line.startswith("Hardware Port:"):
            port = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and port is not None:
            dev = line.split(":", 1)[1].strip()
            low = port.lower()
            if "ethernet" in low and "wi-fi" not in low and "wifi" not in low:
                if dev:
                    devices.append(dev)
            port = None
    return devices


def parse_ifconfig_active(ifconfig_stdout: str) -> bool:
    """True if an ``ifconfig <dev>`` dump shows an ACTIVE link with an IPv4.

    A wired port is "active" only when BOTH ``status: active`` and a real
    ``inet`` (IPv4) address are present. An unplugged adapter reports
    ``status: inactive`` / ``media: none`` and no inet line.
    """
    has_status_active = bool(
        re.search(r"^\s*status:\s*active\b", ifconfig_stdout, re.MULTILINE))
    has_inet = bool(
        re.search(r"^\s*inet\s+\d+\.\d+\.\d+\.\d+", ifconfig_stdout, re.MULTILINE))
    return has_status_active and has_inet


def parse_ifconfig_inet(ifconfig_stdout: str) -> str | None:
    """First IPv4 address from an ``ifconfig`` dump, or None."""
    m = re.search(r"^\s*inet\s+(\d+\.\d+\.\d+\.\d+)",
                  ifconfig_stdout, re.MULTILINE)
    return m.group(1) if m else None


def normalize_ssid(ssid: str | None) -> str | None:
    """Collapse an UNREADABLE SSID to ``None``.

    macOS Tahoe masks the SSID as ``<redacted>`` when Location Services is off;
    treat that (and empty/whitespace) the same as "not reported" so it is never
    mistaken for the joined network name. A real name passes through unchanged.
    """
    if ssid is None:
        return None
    ssid = ssid.strip()
    if not ssid or ssid == _REDACTED_SSID:
        return None
    return ssid


def parse_summary_ssid(getsummary_stdout: str) -> str | None:
    """SSID from ``ipconfig getsummary en0`` ('  SSID : CSI_TX'), or None.

    Returns None when the SSID line is absent OR present-but-empty (not joined).
    """
    # Horizontal-whitespace only around the colon: a bare "SSID :\n" (empty
    # value) must NOT vacuum up the next line's text via \s matching the newline.
    m = re.search(r"^[^\S\n]*SSID[^\S\n]*:[^\S\n]*(.*)$",
                  getsummary_stdout, re.MULTILINE)
    if not m:
        return None
    ssid = m.group(1).strip()
    return ssid or None


def parse_airport_ssid(getairportnetwork_stdout: str) -> str | None:
    """SSID from ``networksetup -getairportnetwork en0``, or None.

    Joined  -> 'Current Wi-Fi Network: CSI_TX'
    Not     -> 'You are not associated with an AirPort network.'
    """
    m = re.search(r"Current Wi-?Fi Network:\s*(.*)$",
                  getairportnetwork_stdout, re.MULTILINE)
    if not m:
        return None
    ssid = m.group(1).strip()
    return ssid or None


def parse_ping_loss(ping_stdout: str) -> float | None:
    """Packet-loss percentage from ``ping`` summary, or None if unparseable.

    Looks for 'X.X% packet loss'.
    """
    m = re.search(r"([\d.]+)%\s*packet loss", ping_stdout)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_http_code(curl_stdout: str) -> int | None:
    """HTTP status code from ``curl -w "%{http_code}"`` output, or None.

    curl prints '000' when it could not connect; that maps to None here so the
    caller reports a RED 'unreachable' rather than a literal status 0.
    """
    s = curl_stdout.strip()
    m = re.search(r"\b(\d{3})\b", s)
    if not m:
        return None
    code = int(m.group(1))
    return code if code != 0 else None


# ── checks (compose _run + parser + status mapping) ──────────────────────────
def check_ethernet_active() -> CheckResult:
    """RED if any wired Ethernet port has an active IPv4 link (must be UNPLUGGED).

    Enumerates ethernet devices, then inspects each with ``ifconfig``. GREEN when
    none is active. (macOS deprioritizes the no-internet CSI_TX Wi-Fi while a
    wired link with internet is up, silently dropping the association.)
    """
    rc, out = _run(["networksetup", "-listallhardwareports"])
    if rc < 0:
        return CheckResult(YELLOW, "could not list hardware ports", out.strip(),
                           raw=out)
    devices = parse_ethernet_devices(out)
    active: list[str] = []
    for dev in devices:
        drc, dout = _run(["ifconfig", dev])
        if drc < 0:
            continue
        if parse_ifconfig_active(dout):
            active.append(dev)
    if active:
        return CheckResult(
            RED,
            f"wired link active on {', '.join(active)}",
            "Unplug the ethernet cable — macOS drops CSI_TX while wired internet is up.",
            raw=out)
    return CheckResult(GREEN, "no active wired link", raw=out)


def check_wifi_ssid() -> CheckResult:
    """GREEN if joined to CSI_TX — location-permission-free.

    On macOS Tahoe the SSID is masked as ``<redacted>`` without Location
    Services (and ``-getairportnetwork`` is unreliable), so a readable SSID is
    not guaranteed. We fall back to a location-free signal: holding an en0 IPv4
    in the CSI_TX SoftAP subnet ``192.168.4.0/24`` (you only get a 192.168.4.x
    address while joined to CSI_TX, whose AP is 192.168.4.1).

      1. Read SSID (getsummary, fallback getairportnetwork); ``<redacted>`` /
         empty -> UNREADABLE (None).
      2. Read en0 inet; ``on_subnet`` = it starts with ``192.168.4.``.
      3. SSID == CSI_TX            -> GREEN.
      4. SSID is a real OTHER name -> RED (wrong network).
      5. SSID unreadable           -> GREEN if on_subnet else RED.
    """
    # 1) SSID, normalising "<redacted>"/empty to None (unreadable).
    rc, out = _run(["ipconfig", "getsummary", WIFI_DEV])
    ssid = normalize_ssid(parse_summary_ssid(out)) if rc >= 0 else None
    used = out
    if ssid is None:
        frc, fout = _run(["networksetup", "-getairportnetwork", WIFI_DEV])
        ssid = normalize_ssid(parse_airport_ssid(fout)) if frc >= 0 else None
        used = (out + "\n" + fout) if out else fout

    # 2) en0 IPv4 + subnet membership (the location-free signal).
    irc, iout = _run(["ifconfig", WIFI_DEV])
    ip = parse_ifconfig_inet(iout) if irc >= 0 else None
    on_subnet = bool(ip) and ip.startswith(CSI_TX_SUBNET_PREFIX)

    # 3) Readable + correct.
    if ssid == TARGET_SSID:
        return CheckResult(GREEN, f"joined {TARGET_SSID}", raw=used)
    # 4) Readable + wrong network.
    if ssid is not None:
        return CheckResult(
            RED, f"joined wrong network: {ssid}",
            f"Click 'Connect Wi-Fi' to switch to {TARGET_SSID}.", raw=used)
    # 5) SSID unreadable: fall back to the subnet signal.
    if on_subnet:
        return CheckResult(
            GREEN,
            f"on {TARGET_SSID} subnet (en0 {ip}; SSID hidden by macOS)",
            raw=used)
    return CheckResult(
        RED, f"not on {TARGET_SSID} (en0 {ip or 'none'})",
        f"Click 'Connect Wi-Fi' (joins {TARGET_SSID}).", raw=used)


def check_static_ip() -> CheckResult:
    """GREEN if ``ifconfig en0`` inet == 192.168.4.200, else RED.

    RX boards unicast CSI to this address; a wrong/missing IP silently drops CSI.
    """
    rc, out = _run(["ifconfig", WIFI_DEV])
    if rc < 0:
        return CheckResult(YELLOW, f"could not read {WIFI_DEV}", out.strip(),
                           raw=out)
    ip = parse_ifconfig_inet(out)
    if ip == STATIC_IP:
        return CheckResult(GREEN, f"static IP {STATIC_IP}", raw=out)
    detail = f"IP is {ip}" if ip else "no IPv4 on Wi-Fi"
    return CheckResult(
        RED, detail,
        f"Click 'Set static IP' — RX boards unicast CSI to {STATIC_IP}.", raw=out)


def check_tx_reachable() -> CheckResult:
    """GREEN on 0% loss pinging the TX board (the CSI anchor at 192.168.4.1)."""
    rc, out = _run(["ping", "-c", "3", "-t", "3", ROUTER_IP], timeout=8.0)
    loss = parse_ping_loss(out)
    if loss is None:
        return CheckResult(
            RED, f"{ROUTER_IP} unreachable",
            "Power-cycle the TX board, then re-join CSI_TX.", raw=out)
    if loss == 0.0:
        return CheckResult(GREEN, f"{ROUTER_IP} reachable (0% loss)", raw=out)
    if loss >= 100.0:
        return CheckResult(
            RED, f"{ROUTER_IP} unreachable (100% loss)",
            "Power-cycle the TX board, then re-join CSI_TX.", raw=out)
    return CheckResult(
        YELLOW, f"{ROUTER_IP} flaky ({loss:.0f}% loss)",
        "Check the TX board / Wi-Fi link; power-cycle if it persists.", raw=out)


def check_camera_http(url: str) -> CheckResult:
    """GREEN when the camera stream URL returns HTTP 200 (via curl)."""
    if not url:
        return CheckResult(
            YELLOW, "no camera URL set",
            "Set the camera URL on the Record page.", raw="")
    rc, out = _run(
        ["curl", "-s", "--max-time", "4", "-o", "/dev/null",
         "-w", "%{http_code}", url],
        timeout=6.0)
    code = parse_http_code(out)
    if code == 200:
        return CheckResult(GREEN, "camera stream HTTP 200", raw=out)
    if code is None:
        return CheckResult(
            RED, "camera unreachable",
            "Start iproxy and tap Start Streaming in AnyCamStream.", raw=out)
    return CheckResult(
        RED, f"camera HTTP {code}",
        "Restart iproxy + AnyCamStream (tap Start Streaming).", raw=out)


def check_iproxy() -> CheckResult:
    """GREEN if an ``iproxy`` process exists (the iPhone USB tunnel)."""
    rc, out = _run(["pgrep", "-x", "iproxy"])
    pids = [p for p in out.split() if p.strip().isdigit()]
    if rc == 0 and pids:
        return CheckResult(GREEN, f"iproxy running (pid {pids[0]})", raw=out)
    return CheckResult(
        RED, "iproxy not running",
        "Click 'Start iproxy' (tunnels iPhone camera on :8080).", raw=out)


def check_floor_calibration(root: str | None = None) -> CheckResult:
    """GREEN if floor calibration is present (reuses calibration_status read-only).

    Imported lazily so this module stays importable without the rest of the GUI.
    """
    from csi_gui import calibration_status as cs

    st = cs.floor_status(root)
    if st.present and st.error is None:
        return CheckResult(
            GREEN, f"floor calibrated ({st.mtime_str})", raw="")
    if st.present:
        return CheckResult(
            YELLOW, f"floor cal unreadable ({st.error})",
            "Re-run floor calibration on the Calibrate page.", raw="")
    return CheckResult(
        RED, "floor not calibrated",
        "Run floor calibration on the Calibrate page (aruco_setup.py).", raw="")
