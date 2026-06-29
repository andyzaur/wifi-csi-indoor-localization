"""Offscreen tests for csi_gui.ui.validate_panel.ValidatePanel.

The panel renders a validate_session.Report (the real Report class) with one
colored row per check + an overall verdict, reusing the shared status palette.
We feed a synthetic Report and assert the rows + verdict reflect it.
"""

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from validate_session import Report, OK, WARN, FAIL  # noqa: E402
from csi_gui.ui.validate_panel import ValidatePanel  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _report(rows):
    rep = Report()
    for level, label, detail in rows:
        rep.add(level, label, detail)
    return rep


def test_renders_one_row_per_check_with_verdict():
    panel = ValidatePanel()
    rep = _report([
        (OK, "csi.csv non-empty", "24,418 data rows"),
        (WARN, "two-foot label mix", "both=62%"),
        (OK, "camera detection rate", "99.7% detected"),
    ])
    panel.render_report(rep)
    # One widget-tuple per row.
    assert len(panel._row_widgets) == 3
    # WARN row colored amber (state "warn"); OK rows green.
    glyph_states = [w[0].property("lvl") for w in panel._row_widgets]
    assert glyph_states == ["ok", "warn", "ok"]
    # Worst is WARN -> verdict amber.
    assert "WARN" in panel._verdict.text()
    assert panel._verdict.property("lvl") == "warn"
    panel.deleteLater()


def test_fail_row_drives_fail_verdict():
    panel = ValidatePanel()
    rep = _report([
        (OK, "a", ""),
        (FAIL, "CSI present", "no CSI rows"),
    ])
    panel.render_report(rep)
    assert "FAIL" in panel._verdict.text()
    assert panel._verdict.property("lvl") == "bad"
    panel.deleteLater()


def test_rerender_replaces_previous_rows():
    panel = ValidatePanel()
    panel.render_report(_report([(OK, "x", ""), (OK, "y", "")]))
    assert len(panel._row_widgets) == 2
    panel.render_report(_report([(FAIL, "z", "")]))
    assert len(panel._row_widgets) == 1
    assert "FAIL" in panel._verdict.text()
    panel.deleteLater()


def test_validate_button_emits_request():
    panel = ValidatePanel()
    fired = []
    panel.validateRequested.connect(lambda: fired.append(True))
    panel._run_btn.click()
    QApplication.processEvents()
    assert fired == [True]
    panel.deleteLater()
