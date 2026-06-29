"""Central design system for the CSI Collector GUI — the "lab instrument" theme.

Single source of truth for the GUI's visual design. Qt QSS has no CSS variables /
``color-mix`` / ``grid``, so the design tokens are resolved here to concrete hex
and exposed two ways:

  * :class:`C` — named colour constants, so any widget building its own QSS
    string references the SAME palette instead of re-hardcoding hex.
  * :data:`BASE_QSS` — a global stylesheet for the generic chrome (window
    background, buttons, inputs, combos, scrollbars, tabs, list, splitter). Apply
    it once at the application level with :func:`apply`; per-page modules then add
    only their object-name-specific rules on top.

Why a base sheet + per-file specifics (rather than one giant global sheet): each
page already encodes correct per-widget padding / sizing in its own QSS; keeping
that and only unifying the *palette* avoids regressing those carefully-tuned
metrics while still giving one consistent look.
"""

from __future__ import annotations


class C:
    """Resolved dark-theme colour tokens (mirror of the design's CSS variables)."""

    # surfaces / structure
    BG = "#0b0d11"           # window / page background
    BG_DEEP = "#08090c"      # deepest wells
    SURFACE_1 = "#13161c"    # cards, sidebar, panels
    SURFACE_2 = "#181c23"    # inputs, resting buttons, sub-rows
    SURFACE_3 = "#20242d"    # hover / pressed, segmented "on"
    INSET = "#0c0e13"        # video well, mono read-out blocks
    BORDER = "#232831"
    BORDER_STRONG = "#313845"
    HAIR = "#1a1e25"         # subtle in-card dividers

    # text
    TEXT = "#e8ebf1"
    TEXT_DIM = "#99a2b2"
    TEXT_FAINT = "#626b79"
    TEXT_ON_ACCENT = "#04130d"

    # accent (emerald) + soft fills (resolved ~16% over BG)
    ACCENT = "#3ad492"
    ACCENT_STRONG = "#2bbf80"
    ACCENT_TEXT = "#74e7b6"
    ACCENT_SOFT = "#12281f"   # accent ~16% on BG
    ACCENT_LINE = "#2a6b51"   # accent ~48% on BG (borders)

    # semantic status (fixed, independent of accent)
    OK = "#3ad492"
    OK_SOFT = "#12281f"
    WARN = "#f2c14e"
    WARN_TEXT = "#ffe0a3"
    WARN_SOFT = "#2a2413"
    WARN_LINE = "#5a4a1f"
    BAD = "#ff6b6b"
    BAD_TEXT = "#ff9a9a"
    BAD_SOFT = "#2a1518"
    BAD_LINE = "#5a2a2c"
    INFO = "#58b0ff"
    INFO_TEXT = "#9ccbff"
    INFO_SOFT = "#12202e"
    INFO_LINE = "#2c4a66"


# Font stacks. The design called for IBM Plex, but it is NOT installed on the
# target Mac — Qt then burned ~60 ms scanning aliases at startup and fell back
# to an arbitrary family with off metrics (part of the "everything feels weird"
# report). The UI font is now simply NOT set in QSS: Qt's application default
# IS the platform system font (SF on macOS), which both looks native and skips
# the alias scan entirely. Mono leads with Menlo (always present on macOS; SF
# Mono only ships with Xcode/Terminal).
MONO = '"Menlo", monospace'

RADIUS_CARD = 12
RADIUS_CTRL = 8


# Generic chrome: window, labels, buttons, inputs, combos, scrollbars, tabs,
# lists, splitters. Object-name specifics live in each page module.
BASE_QSS = f"""
QMainWindow, QWidget {{
    background: {C.BG};
    color: {C.TEXT};
    font-size: 13px;
}}
QToolTip {{
    background: {C.SURFACE_2}; color: {C.TEXT};
    border: 1px solid {C.BORDER_STRONG}; padding: 4px 8px;
}}

QLabel {{ color: {C.TEXT}; background: transparent; }}
#pageHeading {{ font-size: 21px; font-weight: 600; color: {C.TEXT}; }}
#pageIntro {{ color: {C.TEXT_DIM}; font-size: 12.5px; }}
#placeholderLabel {{ color: {C.TEXT_FAINT}; font-size: 16px; }}

/* default (secondary) button */
QPushButton {{
    background: {C.SURFACE_2};
    border: 1px solid {C.BORDER_STRONG};
    border-radius: {RADIUS_CTRL}px;
    padding: 7px 14px;
    color: {C.TEXT};
}}
QPushButton:hover {{ background: {C.SURFACE_3}; }}
QPushButton:pressed {{ background: {C.SURFACE_1}; }}
QPushButton:disabled {{
    color: {C.TEXT_FAINT}; background: {C.SURFACE_1}; border-color: {C.BORDER};
}}
QPushButton:focus {{ outline: none; }}

/* inputs */
QLineEdit, QPlainTextEdit, QTextEdit {{
    background: {C.SURFACE_2};
    border: 1px solid {C.BORDER};
    border-radius: {RADIUS_CTRL}px;
    padding: 6px 9px;
    color: {C.TEXT};
    selection-background-color: {C.ACCENT};
    selection-color: {C.TEXT_ON_ACCENT};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {C.ACCENT_LINE};
}}
QLineEdit:disabled {{ color: {C.TEXT_FAINT}; background: {C.SURFACE_1}; }}

/* combo box */
QComboBox {{
    background: {C.SURFACE_2};
    border: 1px solid {C.BORDER};
    border-radius: {RADIUS_CTRL}px;
    padding: 5px 9px;
    color: {C.TEXT};
}}
QComboBox:hover {{ border-color: {C.BORDER_STRONG}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {C.SURFACE_2};
    border: 1px solid {C.BORDER_STRONG};
    selection-background-color: {C.ACCENT_SOFT};
    selection-color: {C.TEXT};
    outline: 0;
}}

/* scrollbars — slim, no arrows */
QScrollBar:vertical {{
    background: transparent; width: 11px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C.SURFACE_3}; border-radius: 5px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {C.BORDER_STRONG}; }}
QScrollBar:horizontal {{
    background: transparent; height: 11px; margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {C.SURFACE_3}; border-radius: 5px; min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{ background: {C.BORDER_STRONG}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* tabs */
QTabWidget::pane {{ border: 1px solid {C.BORDER}; border-radius: {RADIUS_CTRL}px; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {C.TEXT_DIM};
    padding: 8px 14px; border: none; margin-right: 2px;
}}
QTabBar::tab:hover {{ color: {C.TEXT}; }}
QTabBar::tab:selected {{
    color: {C.TEXT};
    border-bottom: 2px solid {C.ACCENT};
}}

/* lists / splitter */
QListWidget {{
    background: {C.SURFACE_1}; border: 1px solid {C.BORDER};
    border-radius: {RADIUS_CTRL}px; outline: 0;
}}
QSplitter::handle {{ background: transparent; }}

QCheckBox {{ color: {C.TEXT}; spacing: 7px; }}
"""


def apply(target) -> None:
    """Apply the global :data:`BASE_QSS` to a ``QApplication`` or top-level widget.

    Page modules add their object-name-specific rules with their own
    ``setStyleSheet`` on top of this (Qt merges ancestor + widget stylesheets).
    """
    target.setStyleSheet(BASE_QSS)
