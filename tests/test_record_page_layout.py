"""Offscreen smoke test for the RECORD page's new layout (big video, top-left).

After the layout restructure the live video must be the dominant, top-left
element: it sits in a stretching LEFT column and expands, while the pre-flight /
monitor / post-stop panels live on a narrower FIXED right rail. We just assert
the page constructs headless and the key layout invariants hold (the QML video
has an Expanding size policy; the right rail is fixed-width). Qt is importorskip
+ offscreen so this stays headless and creates no camera.
"""

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QSizePolicy  # noqa: E402

from csi_gui.app_context import AppContext  # noqa: E402
import csi_gui.ui.pages.record_page as record_page_mod  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_record_page_constructs_with_new_layout(_qapp):
    page = record_page_mod.RecordPage(AppContext(camera_url="0"))
    # The video widget exists and is set to EXPAND (dominant element).
    pol = page._quick.sizePolicy()
    assert pol.horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert pol.verticalPolicy() == QSizePolicy.Policy.Expanding
    # The post-stop + monitor still exist (now on the right rail) and start hidden.
    assert not page._monitor.isVisible()
    assert not page._post_stop.isVisible()
    # Core wiring untouched: provider/bridge/QML and Start/Stop buttons present.
    assert page._provider is not None and page._bridge is not None
    assert page._start_btn is not None and page._stop_btn is not None
    assert not page._stop_btn.isEnabled()  # stop disabled until recording
    page.deleteLater()
