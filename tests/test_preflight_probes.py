"""Tests for csi_gui.preflight.probes — pure parsers + status mapping.

These feed CAPTURED stdout fixtures (real shapes of ifconfig / ipconfig
getsummary / ping / curl / pgrep / networksetup -listallhardwareports) into the
parsers and assert GREEN/RED + the right hints. The ``check_*`` wrappers are
exercised with subprocess fully monkeypatched so the suite NEVER touches the real
network or runs ping/curl/pgrep against the system.
"""

import pytest

probes = pytest.importorskip("csi_gui.preflight.probes")
GREEN, YELLOW, RED = probes.GREEN, probes.YELLOW, probes.RED


# ---------------------------------------------------------------------------
# Captured stdout fixtures (verbatim shapes from a real macOS host).
# ---------------------------------------------------------------------------
LISTALLHWPORTS = """\

Hardware Port: Ethernet Adapter (en4)
Device: en4
Ethernet Address: ca:7e:f2:2e:b9:f4

Hardware Port: Ethernet Adapter (en5)
Device: en5
Ethernet Address: ca:7e:f2:2e:b9:f5

Hardware Port: Wi-Fi
Device: en0
Ethernet Address: c8:89:f3:dd:57:80

Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: 36:f5:a2:47:b2:00
"""

IFCONFIG_INACTIVE = """\
en4: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\toptions=400<CHANNEL_IO>
\tether ca:7e:f2:2e:b9:f4
\tnd6 options=201<PERFORMNUD,DAD>
\tmedia: none
\tstatus: inactive
"""

IFCONFIG_ACTIVE = """\
en4: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\toptions=400<CHANNEL_IO>
\tether ca:7e:f2:2e:b9:f4
\tinet6 fe80::1%en4 prefixlen 64 secured scopeid 0x10
\tinet 192.168.1.50 netmask 0xffffff00 broadcast 192.168.1.255
\tnd6 options=201<PERFORMNUD,DAD>
\tmedia: 1000baseT <full-duplex>
\tstatus: active
"""

IFCONFIG_EN0_STATIC = """\
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether c8:89:f3:dd:57:80
\tinet6 fe80::1ccc%en0 prefixlen 64 secured scopeid 0x10
\tinet 192.168.4.200 netmask 0xffffff00 broadcast 192.168.4.255
\tmedia: autoselect
\tstatus: active
"""

IFCONFIG_EN0_DHCP = IFCONFIG_EN0_STATIC.replace("192.168.4.200", "192.168.0.233")

IFCONFIG_EN0_NO_INET = """\
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether c8:89:f3:dd:57:80
\tnd6 options=201<PERFORMNUD,DAD>
\tmedia: autoselect
\tstatus: active
"""

GETSUMMARY_JOINED = """\
SSID : CSI_TX
  BSSID : a1:b2:c3:d4:e5:f6
  SSID : CSI_TX
  Security : WPA2 Personal
"""

GETSUMMARY_OTHER = GETSUMMARY_JOINED.replace("CSI_TX", "HomeNet")

GETSUMMARY_NO_SSID = """\
  BSSID : a1:b2:c3:d4:e5:f6
  Security : None
"""

GETSUMMARY_EMPTY_SSID = """\
  SSID :
  Security : None
"""

# macOS Tahoe masks the SSID without Location Services.
GETSUMMARY_REDACTED = """\
  SSID : <redacted>
  BSSID : <redacted>
  Security : WPA2 Personal
"""

# en0 ifconfig dumps for the location-free subnet check.
IFCONFIG_EN0_ON_SUBNET = IFCONFIG_EN0_STATIC  # inet 192.168.4.200 (in 192.168.4.0/24)
IFCONFIG_EN0_OFF_SUBNET = IFCONFIG_EN0_STATIC.replace("192.168.4.200", "192.168.1.42")

AIRPORT_JOINED = "Current Wi-Fi Network: CSI_TX\n"
AIRPORT_NONE = "You are not associated with an AirPort network.\n"

PING_0_LOSS = """\
PING 192.168.4.1 (192.168.4.1): 56 data bytes
64 bytes from 192.168.4.1: icmp_seq=0 ttl=64 time=2.1 ms

--- 192.168.4.1 ping statistics ---
3 packets transmitted, 3 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 2.0/2.1/2.2/0.1 ms
"""

PING_100_LOSS = """\
PING 192.168.4.1 (192.168.4.1): 56 data bytes
Request timeout for icmp_seq 0

--- 192.168.4.1 ping statistics ---
3 packets transmitted, 0 packets received, 100.0% packet loss
"""

PING_33_LOSS = """\
--- 192.168.4.1 ping statistics ---
3 packets transmitted, 2 packets received, 33.3% packet loss
round-trip min/avg/max/stddev = 2.0/2.1/2.2/0.1 ms
"""

PGREP_RUNNING = "48213\n"
PGREP_NONE = ""


# ---------------------------------------------------------------------------
# parse_ethernet_devices — only wired ports, never Wi-Fi.
# ---------------------------------------------------------------------------
def test_parse_ethernet_devices_excludes_wifi_and_bridge():
    devs = probes.parse_ethernet_devices(LISTALLHWPORTS)
    assert devs == ["en4", "en5"]
    assert "en0" not in devs       # Wi-Fi port, even though it has an Ethernet Address line
    assert "bridge0" not in devs   # Thunderbolt Bridge


# ---------------------------------------------------------------------------
# parse_ifconfig_active — needs BOTH status:active AND an inet line.
# ---------------------------------------------------------------------------
def test_parse_ifconfig_active_true_only_when_active_with_inet():
    assert probes.parse_ifconfig_active(IFCONFIG_ACTIVE) is True
    assert probes.parse_ifconfig_active(IFCONFIG_INACTIVE) is False
    # active but no IPv4 -> not a real wired link
    assert probes.parse_ifconfig_active(IFCONFIG_EN0_NO_INET) is False


def test_parse_ifconfig_inet():
    assert probes.parse_ifconfig_inet(IFCONFIG_EN0_STATIC) == "192.168.4.200"
    assert probes.parse_ifconfig_inet(IFCONFIG_EN0_DHCP) == "192.168.0.233"
    assert probes.parse_ifconfig_inet(IFCONFIG_EN0_NO_INET) is None


# ---------------------------------------------------------------------------
# SSID parsers.
# ---------------------------------------------------------------------------
def test_parse_summary_ssid():
    assert probes.parse_summary_ssid(GETSUMMARY_JOINED) == "CSI_TX"
    assert probes.parse_summary_ssid(GETSUMMARY_OTHER) == "HomeNet"
    assert probes.parse_summary_ssid(GETSUMMARY_NO_SSID) is None
    # empty SSID line -> not joined
    assert probes.parse_summary_ssid(GETSUMMARY_EMPTY_SSID) is None


def test_parse_airport_ssid():
    assert probes.parse_airport_ssid(AIRPORT_JOINED) == "CSI_TX"
    assert probes.parse_airport_ssid(AIRPORT_NONE) is None


def test_normalize_ssid_treats_redacted_and_empty_as_unreadable():
    assert probes.normalize_ssid("CSI_TX") == "CSI_TX"
    assert probes.normalize_ssid("  HomeNet  ") == "HomeNet"
    assert probes.normalize_ssid("<redacted>") is None
    assert probes.normalize_ssid("") is None
    assert probes.normalize_ssid("   ") is None
    assert probes.normalize_ssid(None) is None


# ---------------------------------------------------------------------------
# ping / curl parsers.
# ---------------------------------------------------------------------------
def test_parse_ping_loss():
    assert probes.parse_ping_loss(PING_0_LOSS) == 0.0
    assert probes.parse_ping_loss(PING_100_LOSS) == 100.0
    assert probes.parse_ping_loss(PING_33_LOSS) == pytest.approx(33.3)
    assert probes.parse_ping_loss("garbage") is None


def test_parse_http_code():
    assert probes.parse_http_code("200") == 200
    assert probes.parse_http_code("404") == 404
    assert probes.parse_http_code("000") is None   # curl could-not-connect
    assert probes.parse_http_code("") is None


# ---------------------------------------------------------------------------
# check_* wrappers — subprocess monkeypatched, NEVER hits the real system.
# ---------------------------------------------------------------------------
class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_run(monkeypatch, table):
    """Patch probes._run with a router keyed on the FIRST argv token + extras.

    ``table`` maps a key string -> (returncode, stdout). The key is matched as a
    substring of ' '.join(argv) so callers can target e.g. 'ifconfig en0' vs
    'ifconfig en4'.
    """
    def fake_run(argv, timeout=None):
        joined = " ".join(argv)
        for key, (rc, out) in table.items():
            if key in joined:
                return rc, out
        return -2, f"unexpected argv: {joined}"
    monkeypatch.setattr(probes, "_run", fake_run)


def test_check_ethernet_active_green_when_all_inactive(monkeypatch):
    _patch_run(monkeypatch, {
        "listallhardwareports": (0, LISTALLHWPORTS),
        "ifconfig en4": (0, IFCONFIG_INACTIVE),
        "ifconfig en5": (0, IFCONFIG_INACTIVE),
    })
    res = probes.check_ethernet_active()
    assert res.status == GREEN


def test_check_ethernet_active_red_when_a_link_is_up(monkeypatch):
    _patch_run(monkeypatch, {
        "listallhardwareports": (0, LISTALLHWPORTS),
        "ifconfig en4": (0, IFCONFIG_ACTIVE),
        "ifconfig en5": (0, IFCONFIG_INACTIVE),
    })
    res = probes.check_ethernet_active()
    assert res.status == RED
    assert "en4" in res.detail
    assert "Unplug" in res.hint


def test_check_wifi_ssid_green_on_csi_tx(monkeypatch):
    # Readable CSI_TX SSID -> GREEN regardless of the subnet probe.
    _patch_run(monkeypatch, {
        "getsummary": (0, GETSUMMARY_JOINED),
        "ifconfig en0": (0, IFCONFIG_EN0_OFF_SUBNET),
    })
    res = probes.check_wifi_ssid()
    assert res.status == GREEN
    assert "CSI_TX" in res.detail


def test_check_wifi_ssid_wrong_network_is_red(monkeypatch):
    # Readable OTHER name -> RED even if (oddly) on the subnet.
    _patch_run(monkeypatch, {
        "getsummary": (0, GETSUMMARY_OTHER),
        "ifconfig en0": (0, IFCONFIG_EN0_ON_SUBNET),
    })
    res = probes.check_wifi_ssid()
    assert res.status == RED
    assert "HomeNet" in res.detail


def test_check_wifi_ssid_redacted_on_subnet_is_green(monkeypatch):
    # macOS Tahoe redacts the SSID -> use the location-free subnet signal.
    # en0 holds 192.168.4.x -> joined CSI_TX -> GREEN.
    _patch_run(monkeypatch, {
        "getsummary": (0, GETSUMMARY_REDACTED),
        "getairportnetwork": (0, AIRPORT_NONE),
        "ifconfig en0": (0, IFCONFIG_EN0_ON_SUBNET),
    })
    res = probes.check_wifi_ssid()
    assert res.status == GREEN
    assert "subnet" in res.detail
    assert "192.168.4.200" in res.detail


def test_check_wifi_ssid_redacted_off_subnet_is_red(monkeypatch):
    # Redacted SSID + en0 on a foreign 192.168.1.x subnet -> NOT on CSI_TX -> RED.
    _patch_run(monkeypatch, {
        "getsummary": (0, GETSUMMARY_REDACTED),
        "getairportnetwork": (0, AIRPORT_NONE),
        "ifconfig en0": (0, IFCONFIG_EN0_OFF_SUBNET),
    })
    res = probes.check_wifi_ssid()
    assert res.status == RED
    assert "192.168.1.42" in res.detail
    assert res.hint  # carries a Connect-Wi-Fi fix


def test_check_wifi_ssid_empty_ssid_off_subnet_is_red(monkeypatch):
    # No SSID anywhere + no 192.168.4.x address -> RED.
    _patch_run(monkeypatch, {
        "getsummary": (0, GETSUMMARY_NO_SSID),
        "getairportnetwork": (0, AIRPORT_NONE),
        "ifconfig en0": (0, IFCONFIG_EN0_NO_INET),
    })
    res = probes.check_wifi_ssid()
    assert res.status == RED
    assert "none" in res.detail
    assert res.hint  # carries a fix


def test_check_wifi_ssid_falls_back_to_airport(monkeypatch):
    # getsummary unavailable (rc<0) -> fallback path returns the airport SSID.
    _patch_run(monkeypatch, {
        "getsummary": (-2, "no such command"),
        "getairportnetwork": (0, AIRPORT_JOINED),
        "ifconfig en0": (0, IFCONFIG_EN0_OFF_SUBNET),
    })
    res = probes.check_wifi_ssid()
    assert res.status == GREEN


def test_check_static_ip_green_on_match(monkeypatch):
    _patch_run(monkeypatch, {"ifconfig en0": (0, IFCONFIG_EN0_STATIC)})
    res = probes.check_static_ip()
    assert res.status == GREEN
    assert probes.STATIC_IP in res.detail


def test_check_static_ip_red_on_wrong_ip(monkeypatch):
    _patch_run(monkeypatch, {"ifconfig en0": (0, IFCONFIG_EN0_DHCP)})
    res = probes.check_static_ip()
    assert res.status == RED
    assert "192.168.0.233" in res.detail
    assert probes.STATIC_IP in res.hint


def test_check_tx_reachable_green_on_zero_loss(monkeypatch):
    _patch_run(monkeypatch, {"ping": (0, PING_0_LOSS)})
    res = probes.check_tx_reachable()
    assert res.status == GREEN


def test_check_tx_reachable_red_on_total_loss(monkeypatch):
    _patch_run(monkeypatch, {"ping": (2, PING_100_LOSS)})
    res = probes.check_tx_reachable()
    assert res.status == RED
    assert "Power-cycle" in res.hint


def test_check_tx_reachable_yellow_on_partial_loss(monkeypatch):
    _patch_run(monkeypatch, {"ping": (0, PING_33_LOSS)})
    res = probes.check_tx_reachable()
    assert res.status == YELLOW


def test_check_camera_http_green_on_200(monkeypatch):
    _patch_run(monkeypatch, {"curl": (0, "200")})
    res = probes.check_camera_http("http://127.0.0.1:8080/video")
    assert res.status == GREEN


def test_check_camera_http_red_on_no_connect(monkeypatch):
    _patch_run(monkeypatch, {"curl": (0, "000")})
    res = probes.check_camera_http("http://127.0.0.1:8080/video")
    assert res.status == RED
    assert "iproxy" in res.hint.lower()


def test_check_camera_http_yellow_without_url(monkeypatch):
    # No subprocess should even run when the URL is empty.
    called = {"ran": False}

    def fake_run(argv, timeout=None):
        called["ran"] = True
        return 0, "200"
    monkeypatch.setattr(probes, "_run", fake_run)
    res = probes.check_camera_http("")
    assert res.status == YELLOW
    assert called["ran"] is False


def test_check_iproxy_green_when_running(monkeypatch):
    _patch_run(monkeypatch, {"pgrep": (0, PGREP_RUNNING)})
    res = probes.check_iproxy()
    assert res.status == GREEN
    assert "48213" in res.detail


def test_check_iproxy_red_when_absent(monkeypatch):
    _patch_run(monkeypatch, {"pgrep": (1, PGREP_NONE)})
    res = probes.check_iproxy()
    assert res.status == RED
    assert "iproxy" in res.hint.lower()


# ---------------------------------------------------------------------------
# floor calibration check reuses calibration_status (read-only).
# ---------------------------------------------------------------------------
def test_check_floor_calibration_green_when_present(tmp_path):
    import json
    from csi_gui import calibration_status as cs
    (tmp_path / cs.FLOOR_CALIBRATION).write_text(json.dumps({
        "grid_spacing_cm": 50.0, "marker_positions_cm": {"2": [0, 0]},
    }))
    res = probes.check_floor_calibration(str(tmp_path))
    assert res.status == GREEN


def test_check_floor_calibration_red_when_missing(tmp_path):
    res = probes.check_floor_calibration(str(tmp_path))
    assert res.status == RED
    assert "Calibrate" in res.hint


# ---------------------------------------------------------------------------
# _run never raises for a missing binary / timeout (degrades gracefully).
# ---------------------------------------------------------------------------
def test_run_returns_sentinel_for_missing_binary():
    rc, out = probes._run(["this_binary_does_not_exist_csi_xyz"], timeout=2.0)
    assert rc < 0
    assert isinstance(out, str)
