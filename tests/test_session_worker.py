"""Tests for csi_gui.session_worker — the picklable subprocess entry points.

These are the functions a ``ProcessPoolExecutor`` runs in a worker PROCESS so the
GUI thread keeps its own GIL. The unit tests DO NOT spawn a real subprocess: they
call the functions DIRECTLY in-process (deterministic, fast) and assert:

  * both functions (and their results) are picklable / top-level (so a real pool
    under the macOS ``spawn`` start method could ship them across the boundary);
  * ``render_session_plot`` returns a non-empty RGBA dict for EACH plot index;
  * ``compute_report`` returns a picklable rows dict with a verdict.

The module is also asserted import-safe (no work / pool at import).
"""

import os
import pickle

import numpy as np

from csi_gui import session_worker
from csi_gui.viz import PLOTS


def _write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _make_session(root, name="20260601_01_alpha"):
    """A small but valid session on disk: 3 boards, detected camera path, clap."""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(1)

    header = (["wall_time_s", "board_id", "mac", "rssi", "channel",
               "timestamp_us", "rx_seq", "csi_len"]
              + [f"csi_{i}" for i in range(128)])
    rows = [",".join(header)]
    t = 0.0
    for k in range(40):
        for b in (1, 4, 5):
            iq = rng.integers(-30, 30, size=128)
            vals = [f"{t:.3f}", str(b), "aa:bb", str(-40 - b), "6",
                    str(int(t * 1e6)), str(k), "128"] + [str(int(v)) for v in iq]
            rows.append(",".join(vals))
            t += 0.005
    _write(os.path.join(d, "csi.csv"), rows)

    n = 80
    ct = np.linspace(0.0, 0.55, n)
    x = 100 + 60 * np.sin(np.linspace(0, 6.28, n))
    y = 100 + 60 * np.cos(np.linspace(0, 6.28, n))
    cam = ["frame,timestamp_s,x_cm,y_cm,grid_x_cm,grid_y_cm,detected"]
    for i in range(n):
        cam.append(f"{i},{ct[i]:.3f},{x[i]:.1f},{y[i]:.1f},"
                   f"{int(x[i] // 50 * 50)},{int(y[i] // 50 * 50)},1")
    _write(os.path.join(d, "camera.csv"), cam)

    _write(os.path.join(d, "clap.csv"),
           ["wall_time_s,event,event_name,seq,timestamp_us",
            "0.0,0,start,0,0",
            "0.60,1,stop,1,600000"])
    return d


def test_functions_are_picklable_top_level():
    # A spawn-based ProcessPoolExecutor ships these by qualified name.
    pickle.dumps(session_worker.render_session_plot)
    pickle.dumps(session_worker.compute_report)
    assert session_worker.render_session_plot.__module__ == "csi_gui.session_worker"
    assert session_worker.compute_report.__module__ == "csi_gui.session_worker"


def test_module_import_is_pool_free():
    # No process pool / heavy state created at import.
    assert not hasattr(session_worker, "_executor")
    assert not hasattr(session_worker, "executor")


def test_render_each_plot_returns_nonempty_rgba(tmp_path):
    d = _make_session(str(tmp_path))
    for idx in range(len(PLOTS)):
        result = session_worker.render_session_plot(d, idx)
        # Picklable + correct shape.
        pickle.dumps(result)
        assert isinstance(result, dict)
        assert isinstance(result["buffer"], (bytes, bytearray))
        assert result["width"] > 0 and result["height"] > 0
        assert result["stride"] == result["width"] * 4
        assert len(result["buffer"]) == result["height"] * result["stride"]
        assert len(result["buffer"]) > 0
        # A valid (non-placeholder) session renders real, non-empty plots.
        assert result["empty"] is False


def test_render_bad_index_raises(tmp_path):
    d = _make_session(str(tmp_path))
    import pytest
    with pytest.raises(IndexError):
        session_worker.render_session_plot(d, len(PLOTS))


def test_compute_report_returns_picklable_rows(tmp_path):
    d = _make_session(str(tmp_path))
    result = session_worker.compute_report(d)
    pickle.dumps(result)
    assert isinstance(result, dict)
    assert "rows" in result and "verdict" in result
    assert isinstance(result["rows"], list) and result["rows"]
    for row in result["rows"]:
        assert set(row) == {"status", "label", "message"}
        assert row["status"] in ("OK", "WARN", "FAIL")
        assert isinstance(row["label"], str)
        assert isinstance(row["message"], str)
    assert result["verdict"] in ("OK", "WARN", "FAIL")
