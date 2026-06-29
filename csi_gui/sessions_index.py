"""Session DISCOVERY for the Sessions explorer (Qt-free, import-light).

Scans ``sessions/`` for recorded sessions (directories that contain a
``csi.csv``) and returns a list of :class:`SessionInfo` — one per session, with
the cheap-to-compute facts the Sessions page needs to render the LEFT list
without loading or parsing any data:

  * ``name`` / ``path``                       — directory name + absolute path
  * ``date`` / ``nn`` / ``purpose``           — parsed from the ``YYYYMMDD_NN_slug``
                                                naming convention (all optional;
                                                non-conforming names are tolerated)
  * ``csi_rows`` / ``camera_rows`` / ``clap_rows`` — wc-style line counts
                                                (total lines minus the header),
                                                NOT a pandas parse
  * ``has_camera`` / ``has_clap`` / ``has_metadata``
  * ``label``                                 — the GUI-owned rating sidecar
                                                (:mod:`csi_gui.session_labels`)

Everything here is plain Python (no Qt, no pandas, no matplotlib) so it stays
trivially importable, fast, and unit-testable on a temp directory. Heavy work
(the actual CSI/camera load + plotting) happens later, on a worker thread, only
for the *selected* session.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from csi_gui.session_labels import Label, load_label

# sessions/<name> directory name convention: YYYYMMDD_NN_<slug>.
# NN is the per-day sequence; the slug (purpose) is optional.
_NAME_RE = re.compile(r"^(?P<date>\d{8})_(?P<nn>\d+)(?:_(?P<purpose>.*))?$")


@dataclass
class SessionInfo:
    """Cheap, render-ready facts about one recorded session directory.

    None of these fields require parsing the CSVs — row counts are wc-style line
    counts. ``label`` is the GUI-owned rating sidecar (defaults to a "none"
    Label when no ``labels.json`` exists).
    """

    name: str
    path: str
    date: Optional[date] = None
    nn: Optional[int] = None
    purpose: str = ""
    csi_rows: int = 0
    camera_rows: int = 0
    clap_rows: int = 0
    has_camera: bool = False
    has_clap: bool = False
    has_metadata: bool = False
    label: Label = field(default_factory=Label)

    @property
    def date_str(self) -> str:
        """``YYYY-MM-DD`` if the name parsed a date, else ``""``."""
        return self.date.isoformat() if self.date is not None else ""

    @property
    def rating(self) -> str:
        """Convenience pass-through to ``label.rating`` (none/useful/best)."""
        return self.label.rating

    @property
    def row_summary(self) -> str:
        """One-line ``csi / cam / clap`` row-count summary for the list cell."""
        cam = f"{self.camera_rows:,}" if self.has_camera else "—"
        clap = f"{self.clap_rows:,}" if self.has_clap else "—"
        return f"{self.csi_rows:,} csi · {cam} cam · {clap} clap"


def _count_data_rows(path: str) -> int:
    """wc-style data-row count: total lines minus one header line.

    Cheap (a buffered line scan, no pandas). Returns 0 for a missing file or a
    header-only / empty file. Robust to a trailing newline-less last line.
    """
    if not os.path.isfile(path):
        return 0
    n = 0
    try:
        with open(path, "rb") as f:
            for n, _ in enumerate(f, start=1):
                pass
    except OSError:
        return 0
    # n counts the header too; clamp so a header-only file reports 0 data rows.
    return max(n - 1, 0)


def parse_session_name(name: str):
    """Parse ``YYYYMMDD_NN_slug`` -> ``(date|None, nn|None, purpose)``.

    Tolerant: a non-conforming name yields ``(None, None, "")`` rather than
    raising. A valid date prefix with an unparsable day still degrades to None
    for the date but keeps NN/purpose when those parsed.
    """
    m = _NAME_RE.match(name)
    if m is None:
        return None, None, ""
    d: Optional[date] = None
    try:
        d = date(int(m["date"][0:4]), int(m["date"][4:6]), int(m["date"][6:8]))
    except ValueError:
        d = None
    try:
        nn: Optional[int] = int(m["nn"])
    except (TypeError, ValueError):
        nn = None
    purpose = (m["purpose"] or "").strip()
    return d, nn, purpose


def build_session_info(session_dir: str) -> SessionInfo:
    """Build a :class:`SessionInfo` for one session directory.

    ``session_dir`` must already be known to contain ``csi.csv`` (use
    :func:`list_sessions` for discovery). Counts rows cheaply and reads the
    GUI-owned label sidecar; never parses the CSVs.
    """
    name = os.path.basename(os.path.normpath(session_dir))
    d, nn, purpose = parse_session_name(name)

    camera_path = os.path.join(session_dir, "camera.csv")
    clap_path = os.path.join(session_dir, "clap.csv")
    meta_path = os.path.join(session_dir, "metadata.json")

    return SessionInfo(
        name=name,
        path=os.path.abspath(session_dir),
        date=d,
        nn=nn,
        purpose=purpose,
        csi_rows=_count_data_rows(os.path.join(session_dir, "csi.csv")),
        camera_rows=_count_data_rows(camera_path),
        clap_rows=_count_data_rows(clap_path),
        has_camera=os.path.isfile(camera_path),
        has_clap=os.path.isfile(clap_path),
        has_metadata=os.path.isfile(meta_path),
        label=load_label(session_dir),
    )


def _sort_key(info: SessionInfo):
    """Newest-first sort key.

    Sort by (date, nn) descending where parsed; sessions without a parseable
    date sink to the bottom (date == None -> earliest), then break ties by name
    so the order is stable and deterministic.
    """
    has_date = info.date is not None
    d_ord = info.date.toordinal() if info.date is not None else 0
    nn = info.nn if info.nn is not None else -1
    # Primary: has-a-date (so undated sink last). Then date, then nn, then name.
    return (has_date, d_ord, nn, info.name)


def list_sessions(sessions_dir: str) -> list[SessionInfo]:
    """Discover recorded sessions under ``sessions_dir``, newest-first.

    A directory is a "session" iff it directly contains a ``csi.csv``. Scans one
    level deep, builds a cheap :class:`SessionInfo` for each, and sorts so the
    newest (by parsed date + NN) is first. A missing/empty ``sessions_dir``
    yields ``[]``.
    """
    if not os.path.isdir(sessions_dir):
        return []
    infos: list[SessionInfo] = []
    for entry in sorted(os.listdir(sessions_dir)):
        sub = os.path.join(sessions_dir, entry)
        if not os.path.isdir(sub):
            continue
        if not os.path.isfile(os.path.join(sub, "csi.csv")):
            continue
        try:
            infos.append(build_session_info(sub))
        except OSError:
            # An unreadable directory shouldn't kill the whole listing.
            continue
    infos.sort(key=_sort_key, reverse=True)
    return infos
