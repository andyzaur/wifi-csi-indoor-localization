"""Tests for session_metadata.write_metadata (library entry point).

write_metadata() carries the merge / auto-derive / write logic that used to live
in main(). Contract:
  * merges into an existing metadata.json (preserving unrelated keys),
  * fills the auto-derived fields,
  * records the (side, back) offsets passed in,
  * applies the human dict (None -> keep slot visible as TODO),
  * returns the written dict, and the file on disk matches it.
"""

import json
import os
from datetime import datetime, timezone

import pytest

import session_metadata as sm
from session_metadata import write_metadata, HUMAN_FIELDS


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def frozen_clock(monkeypatch):
    # Make created_at deterministic so the written dict is reproducible.
    fixed = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed

    monkeypatch.setattr(sm, "datetime", FakeDT)
    return fixed


def _empty_human():
    return {key: None for key, _ in HUMAN_FIELDS}


def test_write_metadata_merges_into_existing(frozen_clock, tmp_path):
    sd = tmp_path / "sess"
    sd.mkdir()
    meta_path = sd / "metadata.json"
    # Pre-existing file with an unrelated key + a stale human value.
    meta_path.write_text(json.dumps({
        "room": "old_room",
        "notes": "keep_me",
        "custom_extra": "preserved",
    }))

    human = _empty_human()
    human["room"] = "newroom"      # flag overrides existing
    human["walk_style"] = "fast"   # flag fills a fresh field

    out = write_metadata(str(sd), REPO_ROOT, human, (-7.0, -3.0))

    # Unrelated key survives the merge.
    assert out["custom_extra"] == "preserved"
    # A field NOT supplied by a flag keeps its existing value.
    assert out["notes"] == "keep_me"
    # Flag overrides existing.
    assert out["room"] == "newroom"
    assert out["walk_style"] == "fast"
    # session_name derived from dir basename.
    assert out["session_name"] == "sess"

    # File on disk equals the returned dict.
    on_disk = json.loads(meta_path.read_text())
    assert on_disk == out


def test_write_metadata_fills_auto_fields(frozen_clock, tmp_path):
    sd = tmp_path / "sess"
    sd.mkdir()
    out = write_metadata(str(sd), REPO_ROOT, _empty_human(), (-20.0, -15.0))

    # created_at comes from the frozen clock, isoformat with seconds precision.
    assert out["created_at"] == frozen_clock.astimezone().isoformat(timespec="seconds")
    # auto fields present
    assert "git_commit" in out
    assert "calibration_hashes" in out
    assert set(out["calibration_hashes"].keys()) == {
        "lens_profile.json", "marker_layout.json", "floor_calibration.json"}
    # human fields with no flag value keep a visible None slot
    assert out["room"] is None
    assert out["purpose"] is None


def test_write_metadata_records_offsets(frozen_clock, tmp_path):
    sd = tmp_path / "sess"
    sd.mkdir()
    out = write_metadata(str(sd), REPO_ROOT, _empty_human(), (-12.5, 4.0))
    assert out["marker_offset_cm"] == {"side": -12.5, "back": 4.0}


def test_write_metadata_creates_session_dir(frozen_clock, tmp_path):
    # Dir does not exist yet; write_metadata should create it.
    sd = tmp_path / "newly" / "created"
    out = write_metadata(str(sd), REPO_ROOT, _empty_human(), (-20.0, -15.0))
    assert os.path.isdir(str(sd))
    assert os.path.exists(os.path.join(str(sd), "metadata.json"))
    assert out["session_name"] == "created"


def test_write_metadata_interactive_uses_prompt_fn(frozen_clock, tmp_path):
    # With interactive=True and a custom prompt_fn (None human values get
    # resolved through it); flag-provided values are NOT prompted.
    sd = tmp_path / "sess"
    sd.mkdir()
    (sd / "metadata.json").write_text(json.dumps({"person": "Bob"}))

    seen = []

    def prompt_fn(prompt, existing):
        seen.append((prompt, existing))
        # Return a value only for the room field; everything else stays None.
        if prompt.startswith("Room"):
            return "answered_room"
        return existing or None

    human = _empty_human()
    human["walk_style"] = "fast"  # provided by flag -> must not be prompted

    out = write_metadata(str(sd), REPO_ROOT, human, (-20.0, -15.0),
                         interactive=True, prompt_fn=prompt_fn)

    assert out["room"] == "answered_room"
    assert out["walk_style"] == "fast"
    # existing 'person' shown as default to the prompt, preserved when no input.
    assert out["person"] == "Bob"
    prompted_labels = [p for p, _ in seen]
    assert any(p.startswith("Room") for p in prompted_labels)
    # walk_style had a flag value -> never prompted
    assert not any(p.startswith("Walk style") for p in prompted_labels)
