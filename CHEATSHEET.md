# Command Cheatsheet

Every command to run the system straight from the terminal, in workflow order.
Run all Python from the repo root with the virtualenv active.

---

## 1. Setup & run (start here)

```bash
# one-time: create the environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# every session: activate the environment
source venv/bin/activate

# launch the GUI (guided: calibrate -> record -> sessions -> live-validate)
python -m csi_gui.app

# run the test suite (~370 tests)
python -m pytest tests/ -q
```

Optional Apple-Silicon backend (only for `mlx_net.py` / `mlx_vs_torch.py`):

```bash
pip install mlx
```

---

## 2. Live position demo

The **Live-validate** page in the GUI is the easiest demo: launch
`python -m csi_gui.app`, open **Live-validate**, pick a model, choose a source,
and press **Start**.

- **Source = Live (UDP :5500)** — real-time from the RX boards (needs hardware).
- **Source = Replay session** — animate the estimate on a recorded session, no
  hardware needed. Pair a model with the session it was trained on so the dot
  tracks sensibly, e.g. model `20260602_0228_First_33Hz_Run/model_final.joblib`
  with the session of the same name.

Terminal-only (OpenCV window, live UDP):

```bash
python live_position.py --model sessions/<name>/model_final.joblib
# options: --task regression|classification --smooth 5 --trail 80 --max-age 0.5
# press 'q' in the window to quit
```

---

## 3. Firmware (ESP-IDF, target esp32c6)

Source the ESP-IDF environment first, then from each firmware directory:

```bash
# TX anchor
cd csi_tx && idf.py set-target esp32c6 && idf.py build flash monitor

# RX receiver (PING_INTERVAL_MS=30 -> ~33 Hz; 100 is wrong)
cd csi_rx && idf.py -DPING_INTERVAL_MS=30 flash monitor

# clapper (time-sync slate)
cd csi_clapper && idf.py build flash monitor
```

One-time per RX board, in the `rx>` serial console (persists in NVS):

```text
setid 4              # board id: 1, 4, or 5
setip 192.168.4.200  # laptop static IP
cfg                  # show config, then reboot to apply
```

RX status LED: **yellow** connecting · **blue** streaming CSI · **purple**
connected but no CSI · **red** link dropped.

---

## 4. Network setup (macOS) — required before capture

```bash
# 1. Unplug the ethernet cable (macOS silently drops the no-internet CSI_TX Wi-Fi otherwise)

# 2. Join the TX board's network
networksetup -setairportnetwork en0 CSI_TX 23456789

# 3. Give the laptop the static IP the RX boards unicast to
sudo networksetup -setmanual "Wi-Fi" 192.168.4.200 255.255.255.0 192.168.4.1
#    NOTE: -setmanual / -setdhcp take the SERVICE NAME ("Wi-Fi"), not en0.
#    List names with: networksetup -listallnetworkservices

# 4. iPhone camera tunnel (then open AnyCamStream, tap Start Streaming)
iproxy 8080 8080 &

# ...when done collecting for the day, revert to DHCP and re-plug ethernet:
sudo networksetup -setdhcp "Wi-Fi"
```

---

## 5. Calibrate (only when the setup changes)

```bash
# camera lens intrinsics (chessboard) -> lens_profile.json
python lens_calibrate.py --camera http://127.0.0.1:8080/video

# floor marker layout from pairwise measurements -> marker_layout.json
python marker_layout.py

# per-position floor homography (re-run if the phone moved) -> floor_calibration.json
python aruco_setup.py --camera http://127.0.0.1:8080/video
```

---

## 6. Record a session

Full step-by-step lives in **SESSION_CHECKLIST.md**. The core two terminals:

```bash
export SESSION=20260519_01_walk_grid

# terminal A — CSI + clap collector
python csi_collector.py --session $SESSION
python csi_collector.py --no-csv          # preflight: see boards without writing

# terminal B — camera ground truth
python aruco_track.py --camera http://127.0.0.1:8080/video \
    --log sessions/$SESSION/camera.csv --display-scale 0.25

# add metadata after recording
python session_metadata.py sessions/$SESSION --interactive
```

Press the clapper button for START, walk the grid, press again for STOP, then
Ctrl+C both scripts.

---

## 7. Inspect & validate a session

```bash
python validate_session.py sessions/<name>   # quality report (hard checks + warnings)
python plot_session.py sessions/<name>        # plot tracked path on the marker map
```

---

## 8. Train a model

```bash
# in-domain trainer + honest eval (saves model_final.joblib)
python train_final.py sessions/<name>

# leak-proof nested-protocol trainer (saves model_v3.joblib)
python train_v3.py sessions/<name>

# cross-session leave-one-session-out drift evaluation (the headline metric)
python ml_drift.py
```

---

## 9. Plots & analysis (thesis figures)

```bash
python exp_figures.py        # main result figures
python exp_ablation.py       # ablation bars
python exp_grid_density.py   # grid-density study
python occupancy_eval.py     # occupancy detection
python repro_v3.py sessions/<name>   # reproducibility report for a train_v3 card
```

---

## 10. Tests

```bash
python -m pytest tests/ -q                       # whole suite
python -m pytest tests/test_live_infer.py -q      # live inference core
python -m pytest tests/test_two_leg_geometry.py -v
```
