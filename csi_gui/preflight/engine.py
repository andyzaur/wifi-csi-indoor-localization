"""PreflightEngine: the Qt-free core of Stage 1.

Owns two things:

  * The ordered list of :class:`Check` rows (id, label, whether it is CRITICAL
    for "READY TO RECORD", and the callable that produces a CheckResult).
  * A persistent **board-rate listener**: a ``CsiCollector(write_csv=False)``
    bound to UDP :5500 so the "boards 1/4/5 are sending CSI" check can read live
    per-board rates (``rate_hz``). This is the one stateful check — it needs a
    running socket, unlike the one-shot probes.

CRITICAL ORDERING NOTE: the board listener binds :5500. A real recording
collector in Stage 2 binds the same port, so :meth:`stop_board_listener` MUST be
called before that. EADDRINUSE (another collector already bound) is handled
gracefully — reported as "port :5500 busy", not raised.

No Qt imports here; the scheduler drives this off the GUI thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from csi_collector import CsiCollector
from csi_gui.preflight import probes
from csi_gui.preflight.probes import GREEN, RED, YELLOW, CheckResult

# Board IDs that must be streaming CSI (TX=3, clapper=2 are excluded).
RX_BOARD_IDS = (1, 4, 5)
# A board counts as "sending" above this rate (Hz). At ~33 Hz nominal, 5 Hz is a
# generous floor that still flags a stalled/dropping board.
MIN_BOARD_HZ = 5.0

# Check ids (stable identifiers used by the scheduler + panel).
ETHERNET = "ethernet"
WIFI = "wifi"
STATIC_IP = "static_ip"
TX = "tx"
BOARDS = "boards"
IPROXY = "iproxy"
CAMERA = "camera"
FLOOR = "floor"


@dataclass(frozen=True)
class Check:
    """One pre-flight row: how to run it and whether it gates READY-TO-RECORD."""

    id: str
    label: str
    critical: bool
    run: Callable[[], CheckResult]


class PreflightEngine:
    """Ordered checks + the persistent board-rate listener.

    ``camera_url_getter`` is a 0-arg callable returning the current shared camera
    URL (so the camera check always uses the live AppContext value). ``root`` is
    the repo root for the floor-calibration read.
    """

    def __init__(self, camera_url_getter: Callable[[], str] | None = None,
                 root: str | None = None) -> None:
        self._camera_url_getter = camera_url_getter or (lambda: "")
        self._root = root
        self._collector: CsiCollector | None = None
        self._listener_error: str | None = None
        # Empty-room (CSI-only) sessions don't use the camera: the CAMERA check
        # then stops gating READY-TO-RECORD (it stays visible as informational).
        self.camera_required = True

        self._checks: list[Check] = [
            Check(ETHERNET, "Ethernet unplugged", True,
                  probes.check_ethernet_active),
            Check(WIFI, f"Wi-Fi on {probes.TARGET_SSID}", True,
                  probes.check_wifi_ssid),
            Check(STATIC_IP, f"Static IP {probes.STATIC_IP}", True,
                  probes.check_static_ip),
            Check(TX, "TX board reachable", True,
                  probes.check_tx_reachable),
            Check(BOARDS, "RX boards 1/4/5 streaming CSI", True,
                  self.board_rate_result),
            Check(IPROXY, "iproxy tunnel running", False,
                  probes.check_iproxy),
            Check(CAMERA, "Camera stream HTTP 200", True,
                  self._camera_check),
            Check(FLOOR, "Floor calibrated", False,
                  self._floor_check),
        ]

    # -- accessors -------------------------------------------------------------
    @property
    def checks(self) -> list[Check]:
        return list(self._checks)

    def check_by_id(self, check_id: str) -> Check | None:
        for c in self._checks:
            if c.id == check_id:
                return c
        return None

    # -- bound checks (need engine state) --------------------------------------
    def _camera_check(self) -> CheckResult:
        return probes.check_camera_http(self._camera_url_getter())

    def _floor_check(self) -> CheckResult:
        return probes.check_floor_calibration(self._root)

    # -- board-rate listener ---------------------------------------------------
    def start_board_listener(self) -> bool:
        """Start the CsiCollector(:5500, no CSV). Returns True if listening.

        Idempotent. EADDRINUSE (a real collector already bound the port) is
        caught and recorded so :meth:`board_rate_result` can report it instead of
        crashing.
        """
        if self._collector is not None:
            return True
        collector = CsiCollector(write_csv=False, quiet=True, on_log=lambda _m: None)
        try:
            collector.start()
        except OSError as exc:
            self._listener_error = f"port :5500 busy ({exc.errno or exc})"
            # Best-effort cleanup of a half-opened collector.
            try:
                collector.stop()
            except Exception:  # noqa: BLE001
                pass
            return False
        self._collector = collector
        self._listener_error = None
        return True

    def stop_board_listener(self) -> None:
        """Stop + release :5500. MUST run before a Stage-2 collector binds it."""
        collector, self._collector = self._collector, None
        if collector is not None:
            try:
                collector.stop()
            except Exception:  # noqa: BLE001
                pass

    @property
    def board_listener_running(self) -> bool:
        return self._collector is not None

    def board_rates(self) -> dict[int, float]:
        """Latest per-RX-board rate (Hz). Empty if the listener is not running."""
        if self._collector is None:
            return {bid: 0.0 for bid in RX_BOARD_IDS}
        return {bid: self._collector.rate_hz(bid) for bid in RX_BOARD_IDS}

    def board_rate_result(self) -> CheckResult:
        """GREEN if all three RX boards exceed ~5 Hz; YELLOW at 0; RED otherwise."""
        if self._collector is None:
            if self._listener_error:
                return CheckResult(
                    YELLOW, self._listener_error,
                    "Stop the other collector — only one can bind UDP :5500.")
            return CheckResult(
                YELLOW, "listener not started",
                "Open the Record page to start the board listener.")

        rates = self.board_rates()
        missing = [bid for bid in RX_BOARD_IDS if rates.get(bid, 0.0) <= 0.0]
        slow = [bid for bid in RX_BOARD_IDS
                if 0.0 < rates.get(bid, 0.0) < MIN_BOARD_HZ]
        summary = ", ".join(f"{bid}:{rates.get(bid, 0.0):.0f}Hz" for bid in RX_BOARD_IDS)

        if not missing and not slow:
            return CheckResult(GREEN, f"all RX streaming ({summary})")
        if len(missing) == len(RX_BOARD_IDS):
            return CheckResult(
                YELLOW, f"no CSI yet ({summary})",
                "Set the static IP + power the RX boards; give it a few seconds.")
        if missing:
            return CheckResult(
                RED, f"boards {missing} silent ({summary})",
                "Power-cycle the silent RX board(s); LED red = link dropped.")
        return CheckResult(
            YELLOW, f"boards {slow} slow ({summary})",
            "Low rate — check TX is up and the RX link is stable.")

    # -- ready gate ------------------------------------------------------------
    def _is_critical(self, check: Check) -> bool:
        """A check's EFFECTIVE criticality (camera is waived in empty-room mode)."""
        if check.id == CAMERA and not self.camera_required:
            return False
        return check.critical

    def critical_ids(self) -> list[str]:
        return [c.id for c in self._checks if self._is_critical(c)]

    def all_critical_green(self, statuses: dict[str, str]) -> bool:
        """True iff every CRITICAL check's status in ``statuses`` is GREEN.

        ``statuses`` maps check id -> last known status string. A missing or
        non-GREEN critical check means NOT ready. With ``camera_required``
        False (empty-room / CSI-only session) the camera check does not gate.
        """
        for c in self._checks:
            if self._is_critical(c) and statuses.get(c.id) != GREEN:
                return False
        return True
