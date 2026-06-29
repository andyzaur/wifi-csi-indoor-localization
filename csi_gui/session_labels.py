"""GUI-OWNED label sidecar for sessions (Qt-free).

This is how the user marks a recorded session in the Sessions explorer: a
rating (none / useful / best), free-form tags, and notes — persisted to a
``labels.json`` *inside* the session directory:

    sessions/<name>/labels.json
        {"rating": "best", "tags": ["clean", "33hz"], "notes": "the good one"}

It lives entirely on the GUI side and writes a NEW file the backend never reads
or touches (the rule is: don't edit backend files; this only *adds* a sidecar).
Reading a session with no sidecar returns a default "none" :class:`Label`.

Plain Python (json + dataclasses), no Qt, so it round-trips in unit tests and is
safe to import from the Qt-free :mod:`csi_gui.sessions_index`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List

# The rating vocabulary the segmented control in the Sessions page offers.
# "none" is the default (unrated); the rest are user-applied quality tiers.
RATINGS = ("none", "best", "useful", "test", "useless", "ignore")
DEFAULT_RATING = "none"

LABELS_FILENAME = "labels.json"


@dataclass
class Label:
    """A session's user label: rating + tags + notes.

    ``rating`` is constrained to :data:`RATINGS` (an out-of-vocabulary value is
    coerced to ``"none"`` on load). ``tags`` is a list of short strings; ``notes``
    is free text.
    """

    rating: str = DEFAULT_RATING
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {"rating": self.rating, "tags": list(self.tags), "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict) -> "Label":
        """Build a Label from a parsed labels.json dict, tolerating junk.

        Coerces an unknown/missing rating to ``"none"``, a non-list ``tags`` to
        ``[]`` (or splits a comma string), and a non-string ``notes`` to ``""``.
        """
        if not isinstance(data, dict):
            return cls()
        rating = data.get("rating", DEFAULT_RATING)
        if rating not in RATINGS:
            rating = DEFAULT_RATING
        tags = normalize_tags(data.get("tags", []))
        notes = data.get("notes", "")
        if not isinstance(notes, str):
            notes = ""
        return cls(rating=rating, tags=tags, notes=notes)


def normalize_tags(raw) -> List[str]:
    """Coerce ``raw`` into a clean list of non-empty, stripped tag strings.

    Accepts a list (its string items are stripped) or a single comma-separated
    string (``"a, b ,c"`` -> ``["a", "b", "c"]``). Anything else -> ``[]``.
    Empty entries are dropped; order is preserved.
    """
    if isinstance(raw, str):
        parts = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        parts = raw
    else:
        return []
    out: List[str] = []
    for p in parts:
        if not isinstance(p, str):
            p = str(p)
        t = p.strip()
        if t:
            out.append(t)
    return out


def labels_path(session_dir: str) -> str:
    """Path to a session's ``labels.json`` (does not check existence)."""
    return os.path.join(session_dir, LABELS_FILENAME)


def load_label(session_dir: str) -> Label:
    """Read ``sessions/<name>/labels.json`` -> :class:`Label`.

    Returns a default ``"none"`` Label when the sidecar is missing, the directory
    doesn't exist, or the file is unreadable/corrupt — never raises, so the
    Sessions list stays robust to a half-written or hand-edited file.
    """
    path = labels_path(session_dir)
    if not os.path.isfile(path):
        return Label()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return Label()
    return Label.from_dict(data)


def save_label(session_dir: str, label: Label) -> str:
    """Write ``label`` to ``sessions/<name>/labels.json``; returns the path.

    Creates ``session_dir`` if needed (so a brand-new session can be labelled).
    The ``rating`` is validated against :data:`RATINGS` and tags are normalised
    before writing, so the on-disk file is always well-formed.
    """
    os.makedirs(session_dir, exist_ok=True)
    rating = label.rating if label.rating in RATINGS else DEFAULT_RATING
    payload = {
        "rating": rating,
        "tags": normalize_tags(label.tags),
        "notes": label.notes if isinstance(label.notes, str) else "",
    }
    path = labels_path(session_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path
