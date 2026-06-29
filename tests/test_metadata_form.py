"""Tests for csi_gui.ui.metadata_form.

The field->dict mapping (collect_fields) and the write seam (save_metadata) are
Qt-free and tested directly: blank fields -> None, the offsets default to the
ArUco (20.0, -15.0), and write_metadata is called with the right args. An
offscreen smoke test drives the QWidget Save path with write_metadata
monkeypatched.
"""

import os

import pytest

from session_metadata import HUMAN_FIELDS
from csi_gui.ui.metadata_form import collect_fields, save_metadata, OFFSETS
from csi_gui.app_context import ROOT


def test_offsets_match_aruco_defaults():
    assert OFFSETS == (20.0, -15.0)


def test_collect_fields_maps_all_keys_and_blanks_to_none():
    values = {"room": "bedroom", "walk_style": "  slow  ", "notes": ""}
    out = collect_fields(values)
    # Every human key is present.
    assert set(out) == {k for k, _ in HUMAN_FIELDS}
    assert out["room"] == "bedroom"
    assert out["walk_style"] == "slow"          # stripped
    assert out["notes"] is None                 # blank -> None
    assert out["person"] is None                # missing -> None


def test_save_metadata_calls_write_with_right_dict_and_offsets():
    captured = {}

    def fake_write(session_dir, root, human, offsets):
        captured["session_dir"] = session_dir
        captured["root"] = root
        captured["human"] = human
        captured["offsets"] = offsets
        return {"ok": True}

    values = {"room": "lab", "person": "AG", "n_other_people": "2"}
    out = save_metadata("sessions/x", values, write_fn=fake_write)

    assert out == {"ok": True}
    assert captured["session_dir"] == "sessions/x"
    assert captured["root"] == ROOT
    assert captured["offsets"] == (20.0, -15.0)
    # Human dict: provided keys set, blanks/missing -> None.
    assert captured["human"]["room"] == "lab"
    assert captured["human"]["person"] == "AG"
    assert captured["human"]["n_other_people"] == "2"
    assert captured["human"]["furniture_notes"] is None
    assert set(captured["human"]) == {k for k, _ in HUMAN_FIELDS}


def test_save_metadata_default_offsets_and_root():
    seen = {}

    def fake_write(session_dir, root, human, offsets):
        seen["root"] = root
        seen["offsets"] = offsets
        return {}

    save_metadata("sessions/y", {}, write_fn=fake_write)
    assert seen["offsets"] == OFFSETS
    assert seen["root"] == ROOT


# ---------------------------------------------------------------------------
# Offscreen QWidget Save path.
# ---------------------------------------------------------------------------
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_metadata_form_save_invokes_write_metadata(monkeypatch, tmp_path):
    import csi_gui.ui.metadata_form as mf

    captured = {}

    def fake_write(session_dir, root, human, offsets):
        captured["session_dir"] = session_dir
        captured["human"] = human
        captured["offsets"] = offsets
        return {"session_name": os.path.basename(session_dir)}

    monkeypatch.setattr(mf, "write_metadata", fake_write)

    form = mf.MetadataForm()
    sd = str(tmp_path / "20260606_01_walk")
    os.makedirs(sd)
    form.set_session(sd, purpose="walk grid")

    # Fill a field, then Save.
    form._set_value("room", "bedroom")
    saved = []
    form.saved.connect(lambda m: saved.append(m))
    form._on_save()

    assert captured["session_dir"] == sd
    assert captured["offsets"] == (20.0, -15.0)
    assert captured["human"]["room"] == "bedroom"
    # purpose was prefilled by set_session.
    assert captured["human"]["purpose"] == "walk grid"
    assert saved and saved[0]["session_name"] == "20260606_01_walk"
    assert "Saved" in form._status.text()
    form.deleteLater()


def test_metadata_form_save_without_session_shows_error(monkeypatch):
    import csi_gui.ui.metadata_form as mf

    form = mf.MetadataForm()
    # No set_session() -> Save should not call write_metadata; shows an error.
    called = {"n": 0}
    monkeypatch.setattr(mf, "write_metadata",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    form._on_save()
    assert called["n"] == 0
    assert "no session" in form._status.text().lower()
    form.deleteLater()
