"""MainWindow: the 4-section sidebar shell for the CSI data-collection GUI.

Layout (the redesigned "lab instrument" shell):

    +----------------+--------------------------------------------+
    | brand          | topbar (section title + subtitle)          |
    | WORKFLOW       +--------------------------------------------+
    |  1 Calibrate   |                                            |
    |  2 Record      |   QStackedWidget of pages                  |
    |  3 Sessions    |                                            |
    |  4 Live-validate                                            |
    |  (spacer)      |                                            |
    | global status  |                                            |
    +----------------+--------------------------------------------+

The LEFT column is a brand header + a numbered nav (a ``QListWidget`` kept as
``_sidebar`` so selection / tests are unchanged; the 1–5 badges are painted by a
delegate, so each item's ``text()`` stays the clean section name) + an
always-visible GLOBAL STATUS footer (camera + static-IP), so the static-IP-active
warning is never buried. Selecting a row shows that page in the stack and updates
the topbar.

The shell owns one AppContext (shared camera URL etc.) and hands it to the pages
that need it. The Record page owns the ArucoTracker lifecycle; the shell exposes
``stop_tracker()`` so app.py can wire ``QApplication.aboutToQuit`` to it.

Palette + base chrome come from :mod:`csi_gui.ui.theme`; this module adds only the
shell-specific rules on top.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from csi_gui.app_context import AppContext
from csi_gui.preflight import netconfig
from csi_gui.ui import theme
from csi_gui.ui.theme import C
from csi_gui.ui.pages.calibrate_page import CalibratePage
from csi_gui.ui.pages.live_validate_page import LiveValidatePage
from csi_gui.ui.pages.record_page import RecordPage
from csi_gui.ui.pages.sessions_page import SessionsPage

# (label, attribute-name) for each sidebar section, in workflow order.
_SECTIONS = ("Calibrate", "Record", "Sessions", "Live-validate")

# One-line subtitle per section, shown in the topbar.
_SUBTITLES = {
    "Calibrate": "Floor grid · lens · marker layout — check before you record.",
    "Record": "Guided pre-flight → capture → live monitor → validate.",
    "Sessions": "Browse recordings, check quality, visualize, label the good ones.",
    "Live-validate": "Live position estimate from the trained model, on the floor map.",
}

_SHELL_QSS = f"""
#sidebarCol {{ background: {C.SURFACE_1}; border-right: 1px solid {C.BORDER}; }}

#brandName {{ font-size: 14px; font-weight: 600; color: {C.TEXT}; }}
#brandSub {{ font-size: 10.5px; color: {C.TEXT_FAINT}; font-family: {theme.MONO}; }}
#brandGlyph {{
    background: {C.ACCENT}; border-radius: 8px; color: {C.TEXT_ON_ACCENT};
    font-size: 16px; font-weight: 700;
}}
#navLabel {{
    font-size: 10px; font-weight: 600; color: {C.TEXT_FAINT};
    letter-spacing: 1px; padding: 4px 6px;
}}

#sidebar {{ background: transparent; border: none; outline: 0; font-size: 13px; }}
#sidebar::item {{
    padding: 10px 38px 10px 12px;
    margin: 2px 0;
    border-radius: 9px;
    border: 1px solid transparent;
    color: {C.TEXT_DIM};
}}
#sidebar::item:selected {{
    background: {C.ACCENT_SOFT};
    color: {C.TEXT};
    border: 1px solid {C.ACCENT_LINE};
}}
#sidebar::item:hover:!selected {{ background: {C.SURFACE_2}; color: {C.TEXT}; }}

#sbStatus {{
    background: {C.SURFACE_2}; border: 1px solid {C.BORDER}; border-radius: 10px;
}}
#sbStatusRow {{ color: {C.TEXT_DIM}; font-size: 11.5px; }}
#sbStatusVal {{ color: {C.TEXT}; font-family: {theme.MONO}; font-size: 10.5px; }}
#sbStatusVal[tone="warn"] {{ color: {C.WARN}; }}
#sbStatusVal[tone="ok"] {{ color: {C.OK}; }}

#topbar {{ background: {C.SURFACE_1}; border-bottom: 1px solid {C.BORDER}; }}
#topbarTitle {{ font-size: 17px; font-weight: 600; color: {C.TEXT}; }}
#topbarSub {{ font-size: 11.5px; color: {C.TEXT_DIM}; }}

#calibCard, #calibCardPrimary {{
    background: {C.SURFACE_1}; border: 1px solid {C.BORDER};
    border-radius: {theme.RADIUS_CARD}px;
}}
#calibCardPrimary {{ border: 1px solid {C.ACCENT_LINE}; }}
#calibTitle {{ font-size: 14.5px; font-weight: 600; color: {C.TEXT}; }}
#calibTitlePrimary {{ font-size: 16px; font-weight: 600; color: {C.ACCENT_TEXT}; }}
#calibSubtitle {{ color: {C.TEXT_DIM}; font-size: 12px; }}
#calibFields {{
    color: {C.TEXT_DIM}; font-family: {theme.MONO}; font-size: 11px;
    background: {C.INSET}; border: 1px solid {C.HAIR};
    border-radius: 9px; padding: 9px 11px;
}}
#calibError {{ color: {C.BAD}; font-size: 12px; }}
#calibBadge {{
    padding: 3px 10px; border-radius: 8px; font-size: 11px; font-weight: bold;
    background: {C.SURFACE_3}; color: {C.TEXT_DIM};
}}
#calibBadge[state="ok"] {{ background: {C.OK_SOFT}; color: {C.OK}; }}
#calibBadge[state="warn"] {{ background: {C.WARN_SOFT}; color: {C.WARN}; }}
#calibBadge[state="missing"] {{ background: {C.BAD_SOFT}; color: {C.BAD}; }}
#calibButtonPrimary {{
    background: {C.ACCENT}; border: 1px solid {C.ACCENT_STRONG};
    color: {C.TEXT_ON_ACCENT}; font-weight: 600;
}}
#calibButtonPrimary:hover {{ background: {C.ACCENT_STRONG}; }}

#recordStatus {{ color: {C.TEXT_DIM}; font-family: {theme.MONO}; font-size: 12px; }}
"""


class _NavDelegate(QStyledItemDelegate):
    """Paint a 1-based step badge on the right of each sidebar row.

    Keeps the item's ``text()`` the clean section name (tests rely on that) while
    still showing the workflow numbering from the redesign.
    """

    def paint(self, painter, option, index):  # noqa: N802 - Qt override
        super().paint(painter, option, index)
        num = str(index.row() + 1)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        size = 17
        r = option.rect
        badge = QRect(r.right() - size - 11, r.center().y() - size // 2, size, size)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(C.ACCENT) if selected else QColor(C.SURFACE_3))
        painter.drawRoundedRect(badge, 5, 5)
        painter.setPen(QColor(C.TEXT_ON_ACCENT) if selected
                       else QColor(C.TEXT_FAINT))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, num)
        painter.restore()


class MainWindow(QMainWindow):
    """Sidebar shell hosting the four workflow pages."""

    def __init__(self, context: AppContext | None = None, parent=None) -> None:
        super().__init__(parent)
        self._context = context or AppContext()

        self.setWindowTitle("CSI Collector")
        self.resize(1280, 820)
        # Base chrome (so the shell is themed even when built without app.main,
        # e.g. in tests) + the shell-specific rules on top.
        self.setStyleSheet(theme.BASE_QSS + _SHELL_QSS)

        # --- sidebar column ---------------------------------------------------
        self._sidebar = QListWidget()
        self._sidebar.setObjectName("sidebar")
        self._sidebar.setItemDelegate(_NavDelegate(self._sidebar))
        for label in _SECTIONS:
            self._sidebar.addItem(QListWidgetItem(label))

        sidebar_col = self._build_sidebar_col()

        # --- pages ------------------------------------------------------------
        self._stack = QStackedWidget()
        self.calibrate_page = CalibratePage(self._context)
        self.record_page = RecordPage(self._context)
        self.sessions_page = SessionsPage()
        self.live_validate_page = LiveValidatePage(self._context)

        # Same order as _SECTIONS.
        self._pages = [
            self.calibrate_page,
            self.record_page,
            self.sessions_page,
            self.live_validate_page,
        ]
        for page in self._pages:
            self._stack.addWidget(page)

        main_col = self._build_main_col()

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(sidebar_col)
        root.addWidget(main_col, 1)
        self.setCentralWidget(central)

        self._sidebar.currentRowChanged.connect(self._on_section_changed)
        self._sidebar.setCurrentRow(0)

    # -- shell construction ----------------------------------------------------
    def _build_sidebar_col(self) -> QWidget:
        col = QWidget()
        col.setObjectName("sidebarCol")
        col.setFixedWidth(224)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(12, 14, 12, 12)
        lay.setSpacing(4)

        # brand header
        brand = QWidget()
        brow = QHBoxLayout(brand)
        brow.setContentsMargins(4, 2, 4, 10)
        brow.setSpacing(10)
        glyph = QLabel("◆")
        glyph.setObjectName("brandGlyph")
        glyph.setFixedSize(30, 30)
        glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        name = QLabel("CSI Collector")
        name.setObjectName("brandName")
        sub = QLabel("indoor localization")
        sub.setObjectName("brandSub")
        title_box.addWidget(name)
        title_box.addWidget(sub)
        brow.addWidget(glyph)
        brow.addLayout(title_box)
        brow.addStretch(1)

        nav_label = QLabel("WORKFLOW")
        nav_label.setObjectName("navLabel")

        lay.addWidget(brand)
        lay.addWidget(nav_label)
        lay.addWidget(self._sidebar, 1)
        lay.addWidget(self._build_status_footer())
        return col

    def _build_status_footer(self) -> QWidget:
        card = QWidget()
        card.setObjectName("sbStatus")
        grid = QVBoxLayout(card)
        grid.setContentsMargins(11, 9, 11, 9)
        grid.setSpacing(7)

        self._st_camera = self._status_row("Camera", grid)
        self._st_static = self._status_row("Static IP", grid)
        self._refresh_status_footer()
        return card

    def _status_row(self, label: str, parent_layout) -> QLabel:
        row = QWidget()
        row.setObjectName("sbStatusRow")
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        lab = QLabel(label)
        lab.setObjectName("sbStatusRow")
        val = QLabel("—")
        val.setObjectName("sbStatusVal")
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(lab)
        h.addStretch(1)
        h.addWidget(val)
        parent_layout.addWidget(row)
        return val

    def _refresh_status_footer(self) -> None:
        """Update the always-visible footer (cheap: a URL field + a stat call)."""
        url = self._context.camera_url or "not set"
        host = url.split("//", 1)[-1].split("/", 1)[0] if "//" in url else url
        self._st_camera.setText(host[:22])
        self._set_tone(self._st_camera, "")

        static_on = False
        try:
            static_on = os.path.exists(netconfig.SENTINEL)
        except Exception:  # noqa: BLE001 — footer must never raise
            static_on = False
        if static_on:
            self._st_static.setText("ACTIVE")
            self._set_tone(self._st_static, "warn")
        else:
            self._st_static.setText("off (DHCP)")
            self._set_tone(self._st_static, "ok")

    @staticmethod
    def _set_tone(widget: QLabel, tone: str) -> None:
        widget.setProperty("tone", tone)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _build_main_col(self) -> QWidget:
        col = QWidget()
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        topbar = QWidget()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(62)
        tb = QVBoxLayout(topbar)
        tb.setContentsMargins(22, 0, 22, 0)
        tb.setSpacing(1)
        self._topbar_title = QLabel("Calibrate")
        self._topbar_title.setObjectName("topbarTitle")
        self._topbar_sub = QLabel(_SUBTITLES["Calibrate"])
        self._topbar_sub.setObjectName("topbarSub")
        tb.addWidget(self._topbar_title)
        tb.addWidget(self._topbar_sub)

        lay.addWidget(topbar)
        lay.addWidget(self._stack, 1)
        return col

    def _on_section_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._pages):
            return
        page = self._pages[row]
        self._stack.setCurrentIndex(row)
        section = _SECTIONS[row]
        self._topbar_title.setText(section)
        self._topbar_sub.setText(_SUBTITLES.get(section, ""))
        # The static-IP / camera footer can change while away (Record toggles the
        # static IP), so refresh it whenever the visible page changes.
        self._refresh_status_footer()
        # Keep the shared camera URL field in sync when a page becomes visible.
        sync = getattr(page, "sync_from_context", None)
        if callable(sync):
            sync()

    def stop_tracker(self) -> None:
        """Stop any running tracker (delegated to the Record page). Safe at shutdown."""
        self.record_page.stop_tracker()
