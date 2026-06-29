"""Tests for csi_gui.viz — headless Agg plot rendering.

Each render_* function must return a non-empty, geometry-consistent RGBA buffer
for a small synthetic session, WITHOUT opening any pyplot window (we assert the
Agg backend is active and that pyplot opened no figures). Also covers the
robustness requirements: the 7-col vs 13-col camera schema and degenerate /
empty sessions returning a placeholder image rather than raising.
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib

from csi_gui import viz
from csi_gui.viz import RenderedImage


def test_backend_is_agg():
    # The module forces Agg; no interactive/Qt backend should be active.
    assert matplotlib.get_backend().lower() == "agg"


def _synth_csi(n_per_board=40, boards=(1, 4, 5)):
    rows = []
    t = 0.0
    rng = np.random.default_rng(0)
    for k in range(n_per_board):
        for b in boards:
            iq = rng.integers(-30, 30, size=128)
            row = {"wall_time_s": t, "board_id": b, "rssi": -40 - b,
                   "channel": 6, "timestamp_us": int(t * 1e6),
                   "rx_seq": k, "csi_len": 128, "mac": "aa:bb"}
            for i in range(128):
                row[f"csi_{i}"] = int(iq[i])
            rows.append(row)
            t += 0.01
    return pd.DataFrame(rows)


def _synth_camera(n=60, cols13=False):
    t = np.linspace(0.0, n * 0.02, n)
    x = 100 + 80 * np.sin(np.linspace(0, 6.28, n))
    y = 100 + 80 * np.cos(np.linspace(0, 6.28, n))
    gx = (x // 50 * 50).astype(int)
    gy = (y // 50 * 50).astype(int)
    data = {
        "frame": np.arange(n),
        "timestamp_s": t,
        "wall_time_s": t,
        "x_cm": x, "y_cm": y,
        "grid_x_cm": gx, "grid_y_cm": gy,
        "detected": np.ones(n, dtype=int),
    }
    if cols13:
        data.update({
            "n_markers": np.full(n, 2),
            "method": ["both"] * n,
            "right_x": x + 5, "right_y": y + 5,
            "left_x": x - 5, "left_y": y - 5,
        })
    return pd.DataFrame(data)


def _synth_clap():
    return pd.DataFrame({
        "wall_time_s": [0.0, 2.0],
        "event": [0, 1],
        "event_name": ["start", "stop"],
        "seq": [0, 1],
        "timestamp_us": [0, 2_000_000],
    })


def _assert_valid_image(img: RenderedImage, expect_empty=None):
    assert isinstance(img, RenderedImage)
    assert img.width > 0 and img.height > 0
    assert img.stride == img.width * 4
    assert len(img.buffer) == img.height * img.stride
    assert len(img.buffer) > 0
    if expect_empty is not None:
        assert img.empty is expect_empty


@pytest.fixture
def session_7col():
    csi = _synth_csi()
    cam = _synth_camera(cols13=False)
    clap = _synth_clap()
    return viz.load_for_viz(csi, cam, clap)


@pytest.fixture
def session_13col():
    csi = _synth_csi()
    cam = _synth_camera(cols13=True)
    clap = _synth_clap()
    return viz.load_for_viz(csi, cam, clap)


def test_no_pyplot_figures_opened(session_7col):
    # Render every plot; assert pyplot never opened a managed figure (we use the
    # object API only, so its figure registry must stay empty).
    import matplotlib.pyplot as plt
    plt.close("all")
    for _label, fn in viz.PLOTS:
        fn(session_7col)
    assert plt.get_fignums() == []


def test_load_for_viz_trims_and_detects_boards(session_7col):
    assert session_7col.board_ids == (1, 4, 5)
    assert len(session_7col.csi) > 0


def test_csi_heatmap_7col(session_7col):
    _assert_valid_image(viz.render_csi_heatmap(session_7col), expect_empty=False)


def test_coverage_map_7col(session_7col):
    _assert_valid_image(viz.render_coverage_map(session_7col), expect_empty=False)


def test_walked_path_7col(session_7col):
    _assert_valid_image(viz.render_walked_path(session_7col), expect_empty=False)


def test_rate_timeline_7col(session_7col):
    _assert_valid_image(viz.render_rate_timeline(session_7col), expect_empty=False)


def test_all_plots_13col_schema(session_13col):
    # The 13-col camera schema (method/right_*/left_*) must render the same.
    for _label, fn in viz.PLOTS:
        _assert_valid_image(fn(session_13col), expect_empty=False)


def test_plots_robust_to_empty_camera():
    csi = _synth_csi()
    empty_cam = pd.DataFrame(columns=["frame", "timestamp_s", "x_cm", "y_cm",
                                      "grid_x_cm", "grid_y_cm", "detected"])
    bundle = viz.load_for_viz(csi, empty_cam, None)
    # CSI heatmap + rate timeline still render (CSI present)…
    _assert_valid_image(viz.render_csi_heatmap(bundle), expect_empty=False)
    _assert_valid_image(viz.render_rate_timeline(bundle), expect_empty=False)
    # …but path + coverage have no camera data -> placeholder (still valid img).
    _assert_valid_image(viz.render_walked_path(bundle), expect_empty=True)
    _assert_valid_image(viz.render_coverage_map(bundle), expect_empty=True)


def test_plots_robust_to_empty_csi():
    cam = _synth_camera()
    empty_csi = pd.DataFrame(columns=["wall_time_s", "board_id", "rssi"]
                             + [f"csi_{i}" for i in range(128)])
    bundle = viz.load_for_viz(empty_csi, cam, None)
    _assert_valid_image(viz.render_csi_heatmap(bundle), expect_empty=True)
    _assert_valid_image(viz.render_rate_timeline(bundle), expect_empty=True)


def test_large_session_is_sampled():
    # A big session must still render quickly (and the heatmap sub-samples).
    csi = _synth_csi(n_per_board=600)  # 1800 rows
    cam = _synth_camera(n=400)
    bundle = viz.load_for_viz(csi, cam, None)
    img = viz.render_csi_heatmap(bundle)
    _assert_valid_image(img, expect_empty=False)


def test_render_buffer_is_bytes(session_7col):
    img = viz.render_walked_path(session_7col)
    assert isinstance(img.buffer, (bytes, bytearray))
