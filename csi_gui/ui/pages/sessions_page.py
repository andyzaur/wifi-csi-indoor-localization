"""SessionsPage: the Sessions explorer / viewer.

LEFT: a list of recorded sessions (newest first), each cell showing the name,
parsed date, a row-count summary and a rating badge; plus a filter-by-rating and
sort control. RIGHT: a detail pane for the selected session with four parts, all
populated OFF the GUI thread — in a SEPARATE PROCESS:

  1. a QUALITY SCORECARD — ``validate_session.build_report`` (via
     :func:`csi_gui.session_worker.compute_report`), run ON DEMAND via an
     "Analyze / run quality report" button (build_report builds the multiboard
     dataset twice, so it never runs on selection), rendered with the shared
     :class:`~csi_gui.ui.validate_panel.ValidatePanel`;
  2. the session METADATA (``metadata.json``) if present;
  3. the VISUALIZATIONS (one tab per :data:`csi_gui.viz.PLOTS` entry), each
     rendered LAZILY: only the active tab renders on selection, and a tab renders
     the first time it becomes visible (``currentChanged``);
  4. LABEL controls — a rating segmented control + tags + notes + Save, writing
     the GUI-owned ``labels.json`` via :func:`session_labels.save_label`.

WHY A SEPARATE PROCESS (not a thread): the heavy work (the big ``csi.csv`` parse,
the per-packet amplitude loop, matplotlib Agg, the double multiboard build) is
pure-Python/pandas and so HOLDS THE GIL — on a ``QThreadPool`` thread it starves
the GUI thread and the app beachballs on every tab switch. Running it in a
:class:`concurrent.futures.ProcessPoolExecutor` gives it its OWN GIL, so the GUI
event loop stays responsive. The pool is created LAZILY (never at import — macOS
spawn safety) and shut down cleanly on app close (no zombie processes).

Workers return only PICKLABLE payloads (RGBA bytes for a plot, a rows dict for
the report). A ``future.add_done_callback`` (which runs on a pool thread) emits a
QUEUED Qt signal carrying the result; the GUI thread then wraps the bytes in a
``QImage(Format_RGBA8888)`` / renders the rows. The GUI thread NEVER calls
``load_session`` / ``build_report`` / matplotlib directly. Because a running
subprocess task can't be cheaply cancelled, STALE results (whose session no
longer matches the current selection) are simply ignored.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Optional

from PySide6.QtCore import (
    QObject,
    QRect,
    QSize,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from csi_gui import session_worker
from csi_gui.app_context import ROOT
from csi_gui.session_labels import RATINGS, Label, save_label
from csi_gui.sessions_index import SessionInfo, list_sessions
from csi_gui.ui import theme
from csi_gui.ui.theme import C
from csi_gui.ui.validate_panel import ValidatePanel

SESSIONS_QSS = f"""
#sessList {{ background: {C.SURFACE_1}; border: 1px solid {C.BORDER}; border-radius: {theme.RADIUS_CTRL}px; outline: 0; }}
#sessList::item {{ padding: 10px 11px; margin: 2px 4px; border-radius: 9px; color: {C.TEXT_DIM}; }}
#sessList::item:selected {{ background: {C.ACCENT_SOFT}; color: {C.TEXT}; border-left: 3px solid {C.ACCENT}; }}
#sessList::item:hover:!selected {{ background: {C.SURFACE_2}; color: {C.TEXT}; }}

#sessFilterRow QComboBox {{
    background: {C.SURFACE_2}; border: 1px solid {C.BORDER}; border-radius: {theme.RADIUS_CTRL}px;
    padding: 5px 9px; color: {C.TEXT};
}}
#sessDetailHeading {{ font-size: 18px; font-weight: 600; color: {C.TEXT}; font-family: {theme.MONO}; }}
#sessDetailSub {{ color: {C.TEXT_DIM}; font-size: 12px; }}
#sessSectionHeading {{ font-size: 14px; font-weight: 600; color: {C.TEXT}; }}
#sessMeta {{ color: {C.TEXT_DIM}; font-family: {theme.MONO}; font-size: 11px; }}
#sessMetaMissing {{ color: {C.WARN}; font-size: 12px; }}
#sessPlaceholder {{ color: {C.TEXT_FAINT}; font-size: 14px; }}
#sessVizLabel {{ background: {C.INSET}; border: 1px solid {C.BORDER}; border-radius: {theme.RADIUS_CTRL}px; }}
#sessVizStatus {{ color: {C.TEXT_DIM}; font-size: 12px; }}

#sessLabelCard {{ background: {C.SURFACE_1}; border: 1px solid {C.BORDER}; border-radius: {theme.RADIUS_CARD}px; }}
#sessLabelHeading {{ font-size: 14px; font-weight: 600; color: {C.TEXT}; }}
#sessRatingBtn {{
    background: {C.SURFACE_2}; border: 1px solid {C.BORDER}; border-radius: {theme.RADIUS_CTRL}px;
    padding: 6px 12px; color: {C.TEXT_DIM};
}}
#sessRatingBtn:hover {{ color: {C.TEXT}; border-color: {C.BORDER_STRONG}; }}
#sessRatingBtn:checked {{ background: {C.ACCENT_SOFT}; border: 1px solid {C.ACCENT_LINE}; color: {C.ACCENT_TEXT}; font-weight: 600; }}
#sessSaveBtn {{
    background: {C.ACCENT}; border: 1px solid {C.ACCENT_STRONG}; color: {C.TEXT_ON_ACCENT}; font-weight: 600;
    border-radius: {theme.RADIUS_CTRL}px; padding: 7px 16px;
}}
#sessSaveBtn:hover {{ background: {C.ACCENT_STRONG}; }}
#sessSaveStatus {{ color: {C.ACCENT}; font-size: 12px; }}

#sessBadge {{ padding: 2px 8px; border-radius: 9px; font-size: 10px; font-weight: bold; }}
"""

# Filter menu options + the rating they keep (None = "All ratings", no filter).
_FILTER_OPTIONS = (
    ("All ratings", None),
    ("Best", "best"),
    ("Useful", "useful"),
    ("Test", "test"),
    ("Useless", "useless"),
    ("Ignore", "ignore"),
    ("Unrated", "none"),
)
_SORT_OPTIONS = ("Newest first", "Oldest first", "Name A→Z", "Rating")

# Human-readable label per rating (for the segmented control buttons).
_RATING_LABEL = {
    "none": "Unrated",
    "best": "Best",
    "useful": "Useful",
    "test": "Test",
    "useless": "Useless",
    "ignore": "Ignore",
}
# Short uppercase badge shown in the list cell (none -> no badge).
_RATING_BADGE = {
    "best": "BEST",
    "useful": "USEFUL",
    "test": "TEST",
    "useless": "USELESS",
    "ignore": "IGNORE",
}
# Sort priority when "Rating" is selected (best first, unrated last).
_RATING_ORDER = {"best": 0, "useful": 1, "test": 2, "useless": 3, "ignore": 4, "none": 5}

# Process-pool worker count: enough to overlap a plot render with a report run.
_POOL_WORKERS = 2


# ---------------------------------------------------------------------------
# Worker result plumbing (ProcessPool + queued Qt signals)
# ---------------------------------------------------------------------------

class _WorkerSignals(QObject):
    """Queued signals the pool-thread done-callbacks emit back to the GUI thread.

    Emitting a queued signal from the executor's done-callback thread is
    thread-safe: Qt marshals the call onto the GUI thread's event loop. Each
    payload is PICKLABLE (came back from a subprocess): a plain dict for a viz
    render / report, or strings for a failure.
    """

    reportReady = Signal(str, object)            # (session_path, rows-dict)
    vizReady = Signal(str, int, object)          # (session_path, plot_index, rgba-dict)
    failed = Signal(str, str, str)               # (session_path, stage, message)


# ---------------------------------------------------------------------------
# Small reusable widgets
# ---------------------------------------------------------------------------

def _rated_badge_text(rating: str) -> str:
    return _RATING_BADGE.get(rating, "")


class _AspectImage(QWidget):
    """Paints a pixmap scaled-to-fit (KeepAspectRatio), centered, in paintEvent.

    CRITICAL — why this is a custom paint widget and NOT a ``QLabel.setPixmap``:
    a QLabel showing a pixmap reports the pixmap's size as its sizeHint, so
    setting a scaled pixmap *inside* ``resizeEvent`` changed the label geometry,
    which fired another ``resizeEvent`` → ``setPixmap`` → … an INFINITE resize
    loop that pegged the GUI thread (the Sessions-page "freeze"/beachball,
    confirmed via faulthandler on the real cocoa platform — offscreen never
    exercises the layout/resize cycle, which is why tests missed it).

    Here the size hint is CONSTANT (independent of the pixmap) and the image is
    drawn in ``paintEvent``, which is read-only w.r.t. layout. Setting an image
    just calls ``update()`` (a repaint, never a relayout), so the loop is
    structurally impossible.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sessVizLabel")
        self._pix: Optional[QPixmap] = None
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:  # constant — does NOT depend on the pixmap
        return QSize(480, 320)

    def minimumSizeHint(self) -> QSize:
        return QSize(1, 1)

    def set_pixmap(self, pix: Optional[QPixmap]) -> None:
        self._pix = pix
        self.update()  # repaint only — never relayout

    def clear(self) -> None:
        self._pix = None
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        area = self.rect()
        painter.fillRect(area, QColor(C.INSET))
        if self._pix is None or self._pix.isNull():
            return
        scaled = self._pix.size()
        scaled.scale(area.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = area.x() + (area.width() - scaled.width()) // 2
        y = area.y() + (area.height() - scaled.height()) // 2
        painter.drawPixmap(QRect(x, y, scaled.width(), scaled.height()), self._pix)


class _VizTab(QWidget):
    """One visualization tab: a status line + an image view (lazy-rendered)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._status = QLabel("loading…")
        self._status.setObjectName("sessVizStatus")
        self._image = _AspectImage()
        self._image.setMinimumHeight(300)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._status)
        layout.addWidget(self._image, 1)
        self._pixmap: Optional[QPixmap] = None

    def set_loading(self) -> None:
        self._status.setText("rendering… (in a worker process)")
        self._image.clear()
        self._pixmap = None

    def set_loading_mode(self, mode: str) -> None:
        if mode == "thread":
            self._status.setText("rendering… (fallback worker thread)")
        else:
            self.set_loading()

    def set_pending(self) -> None:
        """Placeholder shown for tabs not yet visited (rendered on demand)."""
        self._status.setText("open this tab to render the plot")
        self._image.clear()
        self._pixmap = None

    def set_image(self, rendered: dict) -> None:
        """Wrap a render result's RGBA buffer (dict) in a QPixmap and show it.

        ``rendered`` is the picklable dict from
        :func:`session_worker.render_session_plot`:
        ``{"buffer","width","height","stride","empty"}``. The pixmap is handed to
        the paint-based :class:`_AspectImage` (no setPixmap-in-resizeEvent loop).
        """
        img = QImage(rendered["buffer"], rendered["width"], rendered["height"],
                     rendered["stride"], QImage.Format.Format_RGBA8888)
        # Copy so the pixmap owns its bytes independently of the Python buffer.
        self._pixmap = QPixmap.fromImage(img.copy())
        self._status.setText("no plottable data" if rendered.get("empty") else "")
        self._image.set_pixmap(self._pixmap)

    def set_error(self, message: str) -> None:
        self._status.setText(f"plot failed: {message}")
        self._image.clear()
        self._pixmap = None


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

class SessionsPage(QWidget):
    """Explorer for recorded sessions: list + scorecard + viz + labels."""

    # Emitted (mainly for tests) once a selected session's active plot finished
    # rendering. Carries the session path.
    detailLoaded = Signal(str)

    def __init__(self, sessions_dir: Optional[str] = None, parent=None,
                 executor=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(SESSIONS_QSS)
        self._sessions_dir = sessions_dir or os.path.join(ROOT, "sessions")
        # The process pool is created LAZILY (first heavy task) so importing /
        # constructing the page does NO subprocess spawn — macOS spawn safety.
        # Tests may inject a synchronous fake executor to stay in-process.
        self._executor = executor
        self._owns_executor = executor is None
        self._executor_mode = "injected" if executor is not None else None
        self._worker_error = ""
        self._shutdown = False
        self._signals = _WorkerSignals()
        self._signals.reportReady.connect(self._on_report_ready)
        self._signals.vizReady.connect(self._on_viz_ready)
        self._signals.failed.connect(self._on_worker_failed)

        self._infos: list[SessionInfo] = []
        self._selected_path: Optional[str] = None       # the active token
        # Per-session render cache: path -> {"report":, "viz":{idx: rgba-dict}}
        self._cache: dict[str, dict] = {}
        # path -> set of plot indices whose render is currently in flight (so a
        # rapid tab-flip doesn't dispatch a duplicate render of the same plot).
        self._viz_inflight: dict[str, set] = {}
        self._report_inflight: set[str] = set()

        self._build_ui()
        self.refresh()

    # -- process pool ----------------------------------------------------------
    def _ensure_executor(self):
        """Return a worker executor, creating it LAZILY on first heavy task.

        Never created at import / construction (macOS spawn safety). Re-create is
        guarded: once the page is shut down we return None so no new task is
        submitted into a dying pool.

        Process pools are preferred because the worker code is CPU/Python-heavy,
        but pool construction itself can fail on restricted runtimes or exhausted
        semaphore limits. That happens inside GUI slots, so the failure must not
        escape. A single thread fallback keeps the page usable; it may stutter on
        very large sessions, but it is better than terminating the app.
        """
        if self._shutdown:
            return None
        if self._executor is None:
            try:
                self._executor = ProcessPoolExecutor(max_workers=_POOL_WORKERS)
                self._executor_mode = "process"
                self._worker_error = ""
            except Exception as exc:  # noqa: BLE001 - never escape a Qt slot
                self._worker_error = f"{type(exc).__name__}: {exc}"
                try:
                    self._executor = ThreadPoolExecutor(max_workers=1)
                    self._executor_mode = "thread"
                except Exception as thread_exc:  # noqa: BLE001
                    self._worker_error = (
                        f"{self._worker_error}; thread fallback failed: "
                        f"{type(thread_exc).__name__}: {thread_exc}"
                    )
                    self._executor = None
                    self._executor_mode = None
        return self._executor

    def _submit(self, fn, *args):
        """Submit a worker task, surviving a pool a prior worker may have broken.

        CRITICAL: every submit runs on the GUI thread inside a Qt slot. If
        ``executor.submit`` raised here (e.g. a worker died abnormally — OOM /
        segfault on a huge ``csi.csv`` — poisoning the pool with
        :class:`BrokenProcessPool`), the exception would propagate OUT of the slot
        and PySide6 would TERMINATE the app. That was the Sessions-page "crash."
        So we catch it, rebuild the pool ONCE (only a pool we own — never an
        injected test executor) and retry; any failure returns ``None`` and the
        caller shows an inline error instead of taking the app down.
        """
        try:
            executor = self._ensure_executor()
        except Exception as exc:  # noqa: BLE001 — must never escape the slot
            self._worker_error = f"{type(exc).__name__}: {exc}"
            return None
        if executor is None:
            return None
        try:
            return executor.submit(fn, *args)
        except BrokenProcessPool:
            self._worker_error = (
                "worker pool unavailable: process pool broke while submitting work"
            )
            if not self._owns_executor:
                return None
            self._rebuild_executor()
            executor = self._ensure_executor()
            if executor is None:
                return None
            try:
                return executor.submit(fn, *args)
            except Exception as exc:  # noqa: BLE001 — must never escape the slot
                self._worker_error = f"{type(exc).__name__}: {exc}"
                return None
        except Exception as exc:  # noqa: BLE001 — must never escape the slot
            self._worker_error = f"{type(exc).__name__}: {exc}"
            return None

    def _rebuild_executor(self) -> None:
        """Tear down a broken pool and clear in-flight bookkeeping for a retry."""
        old = self._executor
        self._executor = None
        self._executor_mode = None
        if old is not None and self._owns_executor:
            old.shutdown(wait=False, cancel_futures=True)
        # The dead pool's futures will never call back, so forget them; a later
        # re-selection / tab visit re-dispatches against the fresh pool.
        self._viz_inflight.clear()
        self._report_inflight.clear()

    def shutdown(self) -> None:
        """Shut the process pool down cleanly (no zombie processes).

        Idempotent and safe to call from ``aboutToQuit`` / the close event. Only
        shuts down a pool WE created (an injected test executor is left alone).
        """
        self._shutdown = True
        exec_ = self._executor
        self._executor = None
        self._executor_mode = None
        if exec_ is not None and self._owns_executor:
            exec_.shutdown(wait=False, cancel_futures=True)

    # -- UI construction -------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 760])

        # No in-page heading: the shell's topbar already shows "Sessions" + the
        # one-line description (the duplicate header read as clutter).
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(8)
        root.addWidget(splitter, 1)

    def _build_left(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        controls = QWidget()
        controls.setObjectName("sessFilterRow")
        crow = QHBoxLayout(controls)
        crow.setContentsMargins(0, 0, 0, 0)
        crow.setSpacing(6)
        self._filter = QComboBox()
        for text, _rating in _FILTER_OPTIONS:
            self._filter.addItem(text)
        self._filter.currentIndexChanged.connect(self._repopulate_list)
        self._sort = QComboBox()
        self._sort.addItems(_SORT_OPTIONS)
        self._sort.currentIndexChanged.connect(self._repopulate_list)
        crow.addWidget(self._filter, 1)
        crow.addWidget(self._sort, 1)

        self._refresh_btn = QPushButton("Rescan")
        self._refresh_btn.clicked.connect(self.refresh)

        self._list = QListWidget()
        self._list.setObjectName("sessList")
        self._list.currentItemChanged.connect(self._on_selection_changed)

        self._count_label = QLabel("")
        self._count_label.setObjectName("sessDetailSub")

        layout.addWidget(controls)
        layout.addWidget(self._refresh_btn)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._count_label)
        panel.setFixedWidth(330)
        return panel

    def _build_right(self) -> QWidget:
        self._detail_heading = QLabel("Select a session")
        self._detail_heading.setObjectName("sessDetailHeading")
        self._detail_sub = QLabel("")
        self._detail_sub.setObjectName("sessDetailSub")
        self._detail_sub.setWordWrap(True)

        # Scorecard (reuses ValidatePanel). build_report is expensive (double
        # multiboard build), so it runs ON DEMAND: repurpose the panel's own run
        # button as the "Analyze / run quality report" trigger.
        self._scorecard = ValidatePanel()
        self._scorecard._run_btn.setText("Analyze / run quality report")
        self._scorecard.validateRequested.connect(self._on_analyze)

        # Metadata block.
        meta_heading = QLabel("Metadata")
        meta_heading.setObjectName("sessSectionHeading")
        self._meta_heading = meta_heading
        self._meta_label = QLabel("")
        self._meta_label.setObjectName("sessMeta")
        self._meta_label.setWordWrap(True)
        self._meta_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)

        # Visualization tabs (one per viz.PLOTS entry).
        viz_heading = QLabel("Visualizations")
        viz_heading.setObjectName("sessSectionHeading")
        self._viz_heading = viz_heading
        self._tabs = QTabWidget()
        self._viz_tabs: list[_VizTab] = []
        from csi_gui import viz as _viz
        for label, _fn in _viz.PLOTS:
            tab = _VizTab()
            self._viz_tabs.append(tab)
            self._tabs.addTab(tab, label)
        # Lazily render a plot the first time its tab becomes visible.
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Label controls.
        label_card = self._build_label_card()

        inner = QWidget()
        ilayout = QVBoxLayout(inner)
        ilayout.setContentsMargins(2, 2, 12, 2)
        ilayout.setSpacing(10)
        ilayout.addWidget(self._detail_heading)
        ilayout.addWidget(self._detail_sub)
        ilayout.addWidget(self._scorecard)
        ilayout.addWidget(meta_heading)
        ilayout.addWidget(self._meta_label)
        ilayout.addWidget(viz_heading)
        ilayout.addWidget(self._tabs)
        ilayout.addWidget(label_card)
        ilayout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Stable viewport width (reserve the vertical scrollbar, no horizontal one)
        # so a height-for-width change can't toggle a scrollbar and oscillate the
        # resize. (Single, non-nested scroll — hardened to be safe.)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setWidget(inner)
        self._detail_scroll = scroll
        self._detail_inner = inner
        self._set_detail_enabled(False)
        return scroll

    def _build_label_card(self) -> QWidget:
        card = QWidget()
        card.setObjectName("sessLabelCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        heading = QLabel("Label this session")
        heading.setObjectName("sessLabelHeading")

        rating_row = QHBoxLayout()
        rating_row.setSpacing(6)
        self._rating_group = QButtonGroup(self)
        self._rating_buttons: dict[str, QPushButton] = {}
        # Show the tiers in a sensible order (best..ignore) then Unrated last,
        # matching the filter ordering.
        order = ("best", "useful", "test", "useless", "ignore", "none")
        for rating in order:
            if rating not in RATINGS:
                continue
            btn = QPushButton(_RATING_LABEL[rating])
            btn.setObjectName("sessRatingBtn")
            btn.setCheckable(True)
            self._rating_group.addButton(btn)
            self._rating_buttons[rating] = btn
            rating_row.addWidget(btn)
        rating_row.addStretch(1)
        self._rating_buttons["none"].setChecked(True)

        self._tags_input = QLineEdit()
        self._tags_input.setPlaceholderText("tags (comma-separated, e.g. clean, 33hz)")
        self._notes_input = QPlainTextEdit()
        self._notes_input.setPlaceholderText("notes about this session…")
        self._notes_input.setFixedHeight(64)

        self._save_btn = QPushButton("Save label")
        self._save_btn.setObjectName("sessSaveBtn")
        self._save_btn.clicked.connect(self._on_save_label)
        self._save_status = QLabel("")
        self._save_status.setObjectName("sessSaveStatus")

        save_row = QHBoxLayout()
        save_row.addWidget(self._save_btn)
        save_row.addWidget(self._save_status, 1)

        layout.addWidget(heading)
        layout.addLayout(rating_row)
        layout.addWidget(QLabel("Tags"))
        layout.addWidget(self._tags_input)
        layout.addWidget(QLabel("Notes"))
        layout.addWidget(self._notes_input)
        layout.addLayout(save_row)
        self._label_card = card
        return card

    # -- discovery + list ------------------------------------------------------
    def refresh(self) -> None:
        """Rescan ``sessions/`` and repopulate the list (cheap; no data parse)."""
        self._infos = list_sessions(self._sessions_dir)
        self._repopulate_list()

    def _filtered_sorted(self) -> list[SessionInfo]:
        infos = list(self._infos)
        idx = self._filter.currentIndex()
        keep = _FILTER_OPTIONS[idx][1] if 0 <= idx < len(_FILTER_OPTIONS) else None
        if keep is not None:
            infos = [i for i in infos if i.rating == keep]

        s = self._sort.currentText()
        if s == "Oldest first":
            infos = list(reversed(infos))  # list_sessions is newest-first
        elif s == "Name A→Z":
            infos = sorted(infos, key=lambda i: i.name.lower())
        elif s == "Rating":
            infos = sorted(infos,
                           key=lambda i: (_RATING_ORDER.get(i.rating, 6), i.name))
        # "Newest first" is the default order from list_sessions.
        return infos

    def _repopulate_list(self) -> None:
        prev_path = self._selected_path
        self._list.blockSignals(True)
        self._list.clear()
        infos = self._filtered_sorted()
        for info in infos:
            item = QListWidgetItem(self._item_text(info))
            item.setData(Qt.ItemDataRole.UserRole, info.path)
            badge = _rated_badge_text(info.rating)
            if badge:
                item.setToolTip(f"{info.name}\nrating: {info.rating}")
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._count_label.setText(
            f"{len(infos)} of {len(self._infos)} sessions")
        # Restore selection if the previously-selected session is still shown.
        if prev_path is not None:
            for row in range(self._list.count()):
                if self._list.item(row).data(Qt.ItemDataRole.UserRole) == prev_path:
                    self._list.setCurrentRow(row)
                    break

    def _item_text(self, info: SessionInfo) -> str:
        badge = _rated_badge_text(info.rating)
        prefix = f"[{badge}] " if badge else ""
        date = info.date_str or "(undated)"
        return f"{prefix}{info.name}\n{date} · {info.row_summary}"

    def _info_for_path(self, path: str) -> Optional[SessionInfo]:
        for info in self._infos:
            if info.path == path:
                return info
        return None

    # -- selection -> off-process load ----------------------------------------
    @Slot(object, object)
    def _on_selection_changed(self, current, _previous) -> None:
        if current is None:
            self._selected_path = None
            self._set_detail_enabled(False)
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        self.select_session(path)

    def select_session(self, path: str) -> None:
        """Show the session at ``path`` — CHEAP info immediately, no heavy work.

        Sets the active token and fills the header + metadata + label controls
        synchronously (all cheap). It then dispatches a render of ONLY the active
        visualization tab to the process pool; the GUI thread does NO load /
        render itself. It does NOT run the expensive build_report (that's the
        Analyze button) and does NOT render the non-visible tabs. Stale results
        from a previously-selected session are dropped on arrival (we compare the
        result's path to the current selection). Safe to call directly (tests do).
        """
        self._selected_path = path
        info = self._info_for_path(path)
        self._set_detail_enabled(True)
        self._populate_header(info, path)
        self._populate_metadata(path)
        self._populate_label_controls(info)
        self._save_status.setText("")

        cached = self._cache.get(path, {})

        # Scorecard: replay a cached report; otherwise show the idle "press
        # Analyze" prompt — never auto-run build_report.
        report = cached.get("report")
        if report is not None:
            self._scorecard.render_report_rows(report)
        else:
            self._scorecard.set_idle()

        # Visualization tabs: clear to a "select to render" placeholder, replay
        # any cached images, leave the rest unrendered until their tab is shown.
        viz_cache = cached.get("viz", {})
        for idx, tab in enumerate(self._viz_tabs):
            if idx in viz_cache:
                tab.set_image(viz_cache[idx])
            else:
                tab.set_pending()

        # Render ONLY the active tab now (others render lazily on tab change).
        self._render_active_tab(path)

    # -- lazy plot dispatch ----------------------------------------------------
    def _render_active_tab(self, path: str) -> None:
        """Render the currently-active viz tab for ``path`` if not yet cached."""
        idx = self._tabs.currentIndex()
        self._render_tab(path, idx)

    def _render_tab(self, path: str, idx: int) -> None:
        """Dispatch a single-plot subprocess render for tab ``idx``.

        No-op if the plot is already cached or a render for it is already in
        flight. Each task loads the session + renders one plot in a worker
        PROCESS; the GUI thread only shows a "rendering…" placeholder meanwhile.
        """
        if idx < 0 or idx >= len(self._viz_tabs):
            return
        cached = self._cache.get(path, {})
        viz_cache = cached.get("viz", {})
        if idx in viz_cache:
            self._viz_tabs[idx].set_image(viz_cache[idx])
            return
        inflight = self._viz_inflight.setdefault(path, set())
        if idx in inflight:
            return
        if self._shutdown:
            return
        self._viz_tabs[idx].set_loading_mode(self._executor_mode or "process")
        future = self._submit(session_worker.render_session_plot, path, idx)
        if future is None:
            # Pool unavailable / unrecoverable — show an inline error, never crash.
            msg = self._worker_error or "worker pool unavailable"
            self._viz_tabs[idx].set_error(f"{msg}; reselect the session to retry")
            return
        # Re-fetch the set: a pool rebuild inside _submit may have cleared it.
        self._viz_inflight.setdefault(path, set()).add(idx)
        future.add_done_callback(self._make_viz_callback(path, idx))

    def _make_viz_callback(self, path: str, idx: int):
        """A future done-callback (runs on a pool thread) -> queued Qt signal."""
        def _cb(fut):
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                self._safe_emit(self._signals.failed, path, f"plot:{idx}", str(exc))
                return
            self._safe_emit(self._signals.vizReady, path, idx, result)
        return _cb

    def _safe_emit(self, signal, *args) -> None:
        """Emit a queued signal unless the page is torn down (swallow the race)."""
        if self._shutdown:
            return
        try:
            signal.emit(*args)
        except RuntimeError:
            # The page (and its _WorkerSignals) was destroyed mid-flight.
            pass

    @Slot(int)
    def _on_tab_changed(self, idx: int) -> None:
        """A viz tab became visible -> render it lazily (first visit only)."""
        if self._selected_path is not None:
            self._render_tab(self._selected_path, idx)

    @Slot()
    def _on_analyze(self) -> None:
        """Run the expensive build_report for the selected session, on demand."""
        path = self._selected_path
        if path is None:
            return
        cached = self._cache.get(path, {})
        if cached.get("report") is not None:
            self._scorecard.render_report_rows(cached["report"])
            return
        if path in self._report_inflight:
            return
        if self._shutdown:
            return
        self._scorecard.set_running()
        future = self._submit(session_worker.compute_report, path)
        if future is None:
            self._scorecard.set_idle()
            msg = self._worker_error or "worker pool error"
            self._detail_sub.setText(f"{self._detail_sub.text()}  "
                                     f"(analysis unavailable: {msg})")
            return
        self._report_inflight.add(path)
        future.add_done_callback(self._make_report_callback(path))

    def _make_report_callback(self, path: str):
        """A report future done-callback (pool thread) -> queued Qt signal."""
        def _cb(fut):
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                self._safe_emit(self._signals.failed, path, "validate", str(exc))
                return
            self._safe_emit(self._signals.reportReady, path, result)
        return _cb

    def _populate_header(self, info: Optional[SessionInfo], path: str) -> None:
        name = info.name if info else os.path.basename(path)
        self._detail_heading.setText(name)
        bits = []
        if info:
            if info.date_str:
                bits.append(info.date_str)
            if info.purpose:
                bits.append(info.purpose)
            bits.append(info.row_summary)
        self._detail_sub.setText(" · ".join(bits) if bits else path)

    def _populate_metadata(self, path: str) -> None:
        meta_path = os.path.join(path, "metadata.json")
        if not os.path.isfile(meta_path):
            self._meta_label.setObjectName("sessMetaMissing")
            self._meta_label.setText("No metadata.json (run session_metadata.py "
                                     "to record room / walk style / boards).")
            self._restyle(self._meta_label)
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError) as exc:
            self._meta_label.setObjectName("sessMetaMissing")
            self._meta_label.setText(f"metadata.json unreadable: {exc}")
            self._restyle(self._meta_label)
            return
        self._meta_label.setObjectName("sessMeta")
        self._meta_label.setText(self._format_metadata(meta))
        self._restyle(self._meta_label)

    @staticmethod
    def _format_metadata(meta: dict) -> str:
        # Show the human-readable keys first, then anything else, one per line.
        priority = ("room", "person", "n_other_people", "walk_style",
                    "board_placement", "antenna_config", "camera_mode",
                    "furniture_notes", "purpose", "notes", "created_at",
                    "git_commit")
        lines = []
        seen = set()
        for key in priority:
            if key in meta and meta[key] not in (None, ""):
                lines.append(f"{key}: {meta[key]}")
                seen.add(key)
        for key, val in meta.items():
            if key in seen or isinstance(val, (dict, list)):
                continue
            if val in (None, ""):
                continue
            lines.append(f"{key}: {val}")
        return "\n".join(lines) if lines else "(metadata.json is empty)"

    def _populate_label_controls(self, info: Optional[SessionInfo]) -> None:
        label = info.label if info else Label()
        self._rating_buttons.get(label.rating,
                                 self._rating_buttons["none"]).setChecked(True)
        self._tags_input.setText(", ".join(label.tags))
        self._notes_input.setPlainText(label.notes)

    # -- worker result handlers (GUI thread) ----------------------------------
    @Slot(str, object)
    def _on_report_ready(self, path: str, result) -> None:
        self._report_inflight.discard(path)
        self._cache.setdefault(path, {})["report"] = result
        if path == self._selected_path:
            self._scorecard.render_report_rows(result)
        # else: STALE — the user moved on; keep it cached, don't touch the UI.

    @Slot(str, int, object)
    def _on_viz_ready(self, path: str, idx: int, rendered) -> None:
        self._cache.setdefault(path, {}).setdefault("viz", {})[idx] = rendered
        inflight = self._viz_inflight.get(path)
        if inflight is not None:
            inflight.discard(idx)
        # Drop STALE results: only paint if this session is still selected.
        if path == self._selected_path and 0 <= idx < len(self._viz_tabs):
            self._viz_tabs[idx].set_image(rendered)
            if idx == self._tabs.currentIndex():
                self.detailLoaded.emit(path)

    @Slot(str, str, str)
    def _on_worker_failed(self, path: str, stage: str, message: str) -> None:
        # A plot failure clears its in-flight flag regardless of selection, so a
        # later re-visit can retry the render.
        if stage.startswith("plot:"):
            idx = int(stage.split(":", 1)[1])
            inflight = self._viz_inflight.get(path)
            if inflight is not None:
                inflight.discard(idx)
        elif stage == "validate":
            self._report_inflight.discard(path)
        if path != self._selected_path:
            return
        if stage == "validate":
            # Leave the scorecard in its running state but note the failure.
            self._detail_sub.setText(f"{self._detail_sub.text()}  (validate "
                                     f"failed: {message})")
        elif stage.startswith("plot:"):
            idx = int(stage.split(":", 1)[1])
            if 0 <= idx < len(self._viz_tabs):
                self._viz_tabs[idx].set_error(message)

    # -- saving labels ---------------------------------------------------------
    @Slot()
    def _on_save_label(self) -> None:
        if self._selected_path is None:
            return
        rating = "none"
        for r, btn in self._rating_buttons.items():
            if btn.isChecked():
                rating = r
                break
        label = Label(
            rating=rating,
            tags=[t.strip() for t in self._tags_input.text().split(",") if t.strip()],
            notes=self._notes_input.toPlainText(),
        )
        try:
            save_label(self._selected_path, label)
        except OSError as exc:
            self._save_status.setText(f"save failed: {exc}")
            return
        # Reflect the new label in the in-memory info + the list cell badge.
        info = self._info_for_path(self._selected_path)
        if info is not None:
            info.label = label
        self._save_status.setText("Saved labels.json ✓")
        self._repopulate_list()

    # -- lifecycle -------------------------------------------------------------
    def closeEvent(self, event):  # noqa: N802 - Qt override
        # Shut the process pool down so no zombie subprocess survives the page.
        self.shutdown()
        super().closeEvent(event)

    # -- helpers ---------------------------------------------------------------
    def _set_detail_enabled(self, enabled: bool) -> None:
        self._detail_inner.setVisible(True)
        self._scorecard.setVisible(enabled)
        self._tabs.setVisible(enabled)
        self._label_card.setVisible(enabled)
        self._meta_label.setVisible(enabled)
        # The bare "Metadata" / "Visualizations" headings floated over an empty
        # pane when nothing was selected — hide them with their content.
        self._meta_heading.setVisible(enabled)
        self._viz_heading.setVisible(enabled)
        if not enabled:
            self._detail_heading.setText("Select a session")
            self._detail_sub.setText("Pick a session on the left to inspect "
                                     "its quality, plots, and metadata.")

    @staticmethod
    def _restyle(widget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
