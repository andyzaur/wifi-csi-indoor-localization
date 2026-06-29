"""Tests for csi_gui.session_labels — the GUI-owned labels.json sidecar.

Qt-free round-trip: default "none" when missing, write+read, tolerance of junk,
tag normalisation, rating coercion, and creating the dir on save.
"""

import json
import os

import pytest

from csi_gui.session_labels import (
    DEFAULT_RATING,
    RATINGS,
    Label,
    labels_path,
    load_label,
    normalize_tags,
    save_label,
)


def test_ratings_vocabulary_extended():
    # The vocabulary now offers the extra quality tiers; default stays "none".
    assert DEFAULT_RATING == "none"
    assert set(RATINGS) == {"none", "best", "useful", "test", "useless", "ignore"}


@pytest.mark.parametrize("rating", ["best", "useful", "test", "useless", "ignore"])
def test_round_trip_each_new_rating(tmp_path, rating):
    d = str(tmp_path / f"sess_{rating}")
    os.makedirs(d)
    save_label(d, Label(rating=rating, tags=["t"], notes="n"))
    back = load_label(d)
    assert back.rating == rating
    assert back.tags == ["t"]
    assert back.notes == "n"


def test_junk_rating_coerces_to_none(tmp_path):
    d = tmp_path / "sess"
    d.mkdir()
    (d / "labels.json").write_text('{"rating": "spectacular", "tags": [], "notes": ""}')
    assert load_label(str(d)).rating == "none"


def test_load_missing_returns_default_none(tmp_path):
    label = load_label(str(tmp_path / "no_such_session"))
    assert label.rating == DEFAULT_RATING == "none"
    assert label.tags == []
    assert label.notes == ""


def test_load_existing_dir_without_sidecar_is_none(tmp_path):
    d = tmp_path / "session"
    d.mkdir()
    label = load_label(str(d))
    assert label.rating == "none"


def test_round_trip_write_read(tmp_path):
    d = str(tmp_path / "session")
    os.makedirs(d)
    save_label(d, Label(rating="best", tags=["clean", "33hz"], notes="great run"))
    back = load_label(d)
    assert back.rating == "best"
    assert back.tags == ["clean", "33hz"]
    assert back.notes == "great run"


def test_save_creates_missing_dir(tmp_path):
    d = str(tmp_path / "brand_new")
    assert not os.path.isdir(d)
    path = save_label(d, Label(rating="useful"))
    assert os.path.isfile(path)
    assert path == labels_path(d)
    assert load_label(d).rating == "useful"


def test_written_file_is_valid_json(tmp_path):
    d = str(tmp_path / "session")
    os.makedirs(d)
    path = save_label(d, Label(rating="useful", tags=["a"], notes="n"))
    with open(path) as f:
        data = json.load(f)
    assert data == {"rating": "useful", "tags": ["a"], "notes": "n"}


def test_invalid_rating_coerced_to_none_on_load(tmp_path):
    d = tmp_path / "session"
    d.mkdir()
    (d / "labels.json").write_text('{"rating": "amazing", "tags": [], "notes": ""}')
    assert load_label(str(d)).rating == "none"


def test_invalid_rating_coerced_on_save(tmp_path):
    d = str(tmp_path / "session")
    os.makedirs(d)
    save_label(d, Label(rating="garbage"))
    assert load_label(d).rating == "none"


def test_corrupt_json_returns_default(tmp_path):
    d = tmp_path / "session"
    d.mkdir()
    (d / "labels.json").write_text("{ this is not json ")
    assert load_label(str(d)).rating == "none"


def test_normalize_tags_from_comma_string():
    assert normalize_tags("a, b ,c,") == ["a", "b", "c"]


def test_normalize_tags_from_list_strips_empties():
    assert normalize_tags(["  x ", "", "y"]) == ["x", "y"]


def test_normalize_tags_rejects_junk():
    assert normalize_tags(123) == []
    assert normalize_tags(None) == []


def test_label_from_dict_tolerates_bad_types():
    label = Label.from_dict({"rating": 5, "tags": "p, q", "notes": 99})
    assert label.rating == "none"
    assert label.tags == ["p", "q"]
    assert label.notes == ""


def test_label_from_dict_non_dict():
    assert Label.from_dict("nope").rating == "none"


def test_save_then_load_default_when_empty(tmp_path):
    d = str(tmp_path / "session")
    os.makedirs(d)
    save_label(d, Label())  # all defaults
    back = load_label(d)
    assert back.rating == "none"
    assert back.tags == []
    assert back.notes == ""
