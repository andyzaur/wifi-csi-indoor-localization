"""Tests for csi_gui.session.next_session_name.

Naming convention: ``YYYYMMDD_HHMM_<slug>`` derived from the local clock at call
time (minute resolution — captures are identified by time of day, not a per-day
NN counter). ``now`` (a datetime) is injectable so the test never depends on the
wall clock.
"""

import datetime as dt
import re

from csi_gui.session import next_session_name, _slugify


def test_name_is_time_based(tmp_path):
    when = dt.datetime(2026, 6, 7, 22, 10)
    name = next_session_name("empty baseline", sessions_dir=str(tmp_path), now=when)
    assert name == "20260607_2210_empty_baseline"


def test_minute_resolution_zero_padded():
    when = dt.datetime(2026, 1, 2, 3, 4)  # single-digit month/day/hour/minute
    assert next_session_name("walk grid", now=when) == "20260102_0304_walk_grid"


def test_blank_purpose_defaults_slug():
    when = dt.datetime(2026, 6, 7, 9, 30)
    assert next_session_name("", now=when) == "20260607_0930_session"


def test_real_clock_used_when_now_none(tmp_path):
    name = next_session_name("p", sessions_dir=str(tmp_path))
    assert re.match(r"^\d{8}_\d{4}_p$", name), name


def test_sessions_dir_no_longer_affects_name(tmp_path):
    # Pre-existing dirs must not change the time-based name (no counter anymore).
    (tmp_path / "20260607_2210_old").mkdir()
    when = dt.datetime(2026, 6, 7, 22, 10)
    assert next_session_name("x", sessions_dir=str(tmp_path), now=when) == \
        "20260607_2210_x"


def test_slugify_collapses_and_lowercases():
    assert _slugify("Walk Grid (slow!)") == "walk_grid_slow"
    assert _slugify("  multiple   spaces  ") == "multiple_spaces"
    assert _slugify("---trim---") == "trim"
    assert _slugify("") == "session"
