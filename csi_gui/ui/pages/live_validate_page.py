"""Live-validate page: real-time position estimate from a trained model.

Listens for CSI on UDP :5500 (live hardware) — or replays a recorded session —
feeds the multiboard amp+rssi model, and draws the smoothed predicted position
on the floor-marker map. This is ``live_position.py`` brought inside the GUI
shell; the heavy lifting lives in the Qt-free :mod:`csi_gui.live_infer` core.
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from csi_gui import live_infer
from csi_gui.ui.theme import C

_QSS = f"""
#liveControls QLabel {{ color: {C.TEXT_DIM}; font-size: 12px; }}
#liveControls QComboBox {{
    background: {C.SURFACE_1}; border: 1px solid {C.BORDER};
    border-radius: 6px; padding: 5px 8px; color: {C.TEXT}; min-height: 18px;
}}
#liveStartBtn {{
    background: {C.ACCENT}; border: 1px solid {C.ACCENT_STRONG};
    color: {C.TEXT_ON_ACCENT}; font-weight: 600; border-radius: 6px;
    padding: 6px 18px;
}}
#liveStartBtn:hover {{ background: {C.ACCENT_STRONG}; }}
#liveStartBtn[running="true"] {{
    background: {C.BAD_SOFT}; border: 1px solid {C.BAD_LINE}; color: {C.BAD_TEXT};
}}
#liveMap {{
    background: {C.INSET}; border: 1px solid {C.BORDER}; border-radius: 8px;
}}
#liveStatus {{
    color: {C.TEXT_DIM}; font-family: "SF Mono", "Menlo", monospace; font-size: 12px;
    padding: 2px 2px;
}}
"""


class LiveValidatePage(QWidget):
    """Real-time inference view: model estimate drawn on the floor map."""

    SMOOTH = 5
    TRAIL = 80
    MAX_AGE_S = 0.6

    def __init__(self, context=None, parent=None) -> None:
        super().__init__(parent)
        self._ctx = context
        self._root = getattr(context, "root", os.getcwd())
        self._sessions_dir = os.path.join(self._root, "sessions")

        self._state = live_infer.BoardState()
        self._predictor: live_infer.Predictor | None = None
        self._listener: live_infer.UdpCsiListener | None = None
        self._replay: live_infer.ReplaySource | None = None
        self._raw: deque = deque(maxlen=self.SMOOTH)
        self._trail: deque = deque(maxlen=self.TRAIL)
        self._smoothed: tuple[float, float] | None = None
        self._anchors = self._load_anchors()

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 fps
        self._timer.timeout.connect(self._tick)
        self._render_idle()

    # -- setup -----------------------------------------------------------------

    def _load_anchors(self):
        try:
            return live_infer.load_layout(os.path.join(self._root, "marker_layout.json"))
        except Exception:
            return {}

    def _build_ui(self) -> None:
        self.setStyleSheet(_QSS)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        controls = QWidget()
        controls.setObjectName("liveControls")
        row = QHBoxLayout(controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self._model_combo = QComboBox()
        for label, path in live_infer.find_model_bundles(self._sessions_dir):
            self._model_combo.addItem(label, path)

        self._source_combo = QComboBox()
        self._source_combo.addItems(["Live (UDP :5500)", "Replay session"])
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)

        self._replay_combo = QComboBox()
        for label, path in live_infer.find_replay_sessions(self._sessions_dir):
            self._replay_combo.addItem(label, path)
        self._replay_combo.setVisible(False)

        self._start_btn = QPushButton("Start")
        self._start_btn.setObjectName("liveStartBtn")
        self._start_btn.clicked.connect(self._toggle)

        row.addWidget(QLabel("Model"))
        row.addWidget(self._model_combo, 3)
        row.addWidget(QLabel("Source"))
        row.addWidget(self._source_combo, 1)
        row.addWidget(self._replay_combo, 3)
        row.addStretch(1)
        row.addWidget(self._start_btn)
        outer.addWidget(controls)

        self._map = QLabel()
        self._map.setObjectName("liveMap")
        self._map.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._map.setMinimumHeight(360)
        outer.addWidget(self._map, 1)

        self._status = QLabel("Pick a model and press Start.")
        self._status.setObjectName("liveStatus")
        outer.addWidget(self._status)

        if self._model_combo.count() == 0:
            self._status.setText(
                "No trained model found under sessions/. Train one with "
                "train_final.py, then return here."
            )

    # -- controls --------------------------------------------------------------

    def _on_source_changed(self, _idx: int) -> None:
        is_replay = self._source_combo.currentText().startswith("Replay")
        self._replay_combo.setVisible(is_replay)

    @property
    def _running(self) -> bool:
        return self._timer.isActive()

    def _toggle(self) -> None:
        self._stop() if self._running else self._start()

    def _start(self) -> None:
        model_path = self._model_combo.currentData()
        if not model_path:
            self._status.setText("No trained model selected.")
            return
        try:
            self._predictor = live_infer.Predictor.from_path(model_path)
        except Exception as exc:  # noqa: BLE001 - surface any load error to the user
            self._status.setText(f"Could not load model: {exc}")
            return

        self._raw.clear()
        self._trail.clear()
        self._smoothed = None

        if self._source_combo.currentText().startswith("Replay"):
            csv = self._replay_combo.currentData()
            if not csv:
                self._status.setText("No recorded session available to replay.")
                return
            try:
                self._replay = live_infer.ReplaySource(self._state, csv).start()
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"Replay failed: {exc}")
                return
        else:
            self._listener = live_infer.UdpCsiListener(self._state).start()

        self._set_running(True)
        self._timer.start()

    def _stop(self) -> None:
        self._timer.stop()
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._replay is not None:
            self._replay.stop()
            self._replay = None
        self._set_running(False)
        self._render_idle()

    def _set_running(self, running: bool) -> None:
        self._start_btn.setText("Stop" if running else "Start")
        self._start_btn.setProperty("running", "true" if running else "false")
        self._start_btn.style().unpolish(self._start_btn)
        self._start_btn.style().polish(self._start_btn)
        self._model_combo.setEnabled(not running)
        self._source_combo.setEnabled(not running)
        self._replay_combo.setEnabled(not running)

    # -- render loop -----------------------------------------------------------

    def _tick(self) -> None:
        if self._predictor is None:
            return
        feat, ages = self._state.snapshot(self._predictor.n_boards, self.MAX_AGE_S)
        if feat is not None:
            x, y = self._predictor.predict_xy(feat)
            self._raw.append((x, y))
            self._smoothed = (
                float(np.mean([p[0] for p in self._raw])),
                float(np.mean([p[1] for p in self._raw])),
            )
            self._trail.append(self._smoothed)
        self._render(ages)

    def _status_lines(self, ages) -> list[str]:
        need = self._predictor.n_boards if self._predictor else 3
        if not ages:
            return [f"Waiting for CSI from {need} RX boards on UDP :5500 ..."]
        fresh = sorted(b for b, a in ages.items() if a <= self.MAX_AGE_S)
        if len(fresh) < need:
            return [f"Waiting for {need} boards  (fresh: {fresh or '-'})"]
        age_str = "  ".join(f"b{b}:{ages[b] * 1000:.0f}ms" for b in fresh[:need])
        return [f"CSI ages  {age_str}", f"smoothing window {self.SMOOTH}"]

    def _render(self, ages) -> None:
        lines = self._status_lines(ages)
        img = live_infer.render_floor_map(self._anchors, list(self._trail),
                                          self._smoothed, lines)
        self._set_pixmap(img)
        if self._smoothed is not None:
            self._status.setText(
                f"Estimated position  ({self._smoothed[0]:.0f}, {self._smoothed[1]:.0f}) cm"
            )
        else:
            self._status.setText(lines[0])

    def _render_idle(self) -> None:
        hint = "Pick a model and press Start." if self._anchors else \
            "marker_layout.json not found in the project root."
        img = live_infer.render_floor_map(self._anchors, [], None, [hint])
        self._set_pixmap(img)

    def _set_pixmap(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(qimg.copy())
        target = self._map.size()
        if target.width() > 4 and target.height() > 4:
            pix = pix.scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        self._map.setPixmap(pix)

    # -- lifecycle -------------------------------------------------------------

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if not self._running:
            self._render_idle()

    def hideEvent(self, event):  # noqa: N802 - Qt override
        self._stop()
        super().hideEvent(event)
