# WiFi CSI Indoor Localization

Device-free indoor localization from WiFi **Channel State Information (CSI)**. A
person walking through a room perturbs the multipath between cheap ESP32-C6 radios,
and a model trained on those perturbations infers where they are — **the person
carries nothing**. Wearable ArUco markers and an overhead camera are used only to
generate ground-truth labels during data collection; they are not needed at
inference time.

This repository accompanies my bachelor's thesis (Faculty of Automatic Control and
Computers). It contains the firmware, the data pipeline, the model training and
evaluation code, and the data-collection GUI.

> Indoor positioning that needs no wearable, no extra infrastructure beyond a few
> ESP32 boards, and no line of sight — just the WiFi that is already in the room.

## How it works

A fixed transmitter floods the room with WiFi frames. Several receivers measure the
per-subcarrier channel response (CSI) of every frame they hear from it. As a person
moves, they change the multipath, and the CSI changes with them. The receivers
stream their measurements to a laptop, which aligns them in time with camera-derived
ground-truth positions and trains a model to map CSI → position.

```
                 TX anchor  (ESP32-C6 Soft-AP, SSID "CSI_TX", ch.6)
                          board 3
                              |
            +-----------------+-----------------+
            |                 |                 |
          RX 1              RX 4              RX 5        (ESP32-C6 stations)
            \                 |                 /
             \                |                /
              \   unicast UDP CSI :5500       /
               +------------- | -------------+
                              v
                   Laptop  (static IP 192.168.4.200)
                   csi_collector.py  ->  sessions/<name>/csi.csv
                              ^
                   board 2 "clapper"  ->  clap.csv      (time-sync slate)
                              |
                   iPhone overhead camera (ultrawide)
                   AnyCamStream -> iproxy -> OpenCV
                   aruco_track.py  ->  sessions/<name>/camera.csv
```

All three CSV streams share a common `wall_time_s` column and are joined in pandas
(`merge_asof`). CSI is transported by **unicast** (not broadcast) so it is not held
back to the ~100 ms WiFi beacon interval.

## Repository layout

### Firmware (ESP-IDF, target `esp32c6`)

- **`csi_tx/`** — Soft-AP transmitter. Creates the `CSI_TX` network that the
  receivers and the laptop all associate with. This is the anchor everyone measures.
- **`csi_rx/`** — Receiver. Associates with the AP, captures CSI for each frame, and
  forwards it to the laptop over unicast UDP. Board identity (ID and laptop IP) is
  stored in NVS, so the same image is flashed to every receiver and configured once
  over the serial console.
- **`csi_clapper/`** — Sync "slate". A button press marks START/STOP by flashing an
  LED (seen by the camera) and sending a UDP packet (logged alongside the CSI), so
  the camera and CSI clocks can be aligned.

### Python pipeline (laptop side)

- **`csi_collector.py`** — UDP listener; writes per-session `csi.csv` and `clap.csv`.
- **`aruco_track.py`** — camera ground truth; tracks two foot markers and writes
  `camera.csv` (continuous position and snapped grid cell).
- **`aruco_setup.py`**, **`lens_calibrate.py`**, **`marker_layout.py`** — camera
  intrinsics, floor homography, and floor-marker geometry calibration.
- **`dataset.py`** — load and time-align CSI with camera labels; build feature matrices.
- **`mlpipe.py`** — unit-tested transforms and split/adaptation helpers.
- **`ml_drift.py`** — cross-session leave-one-session-out (LOSO) evaluation harness.
- **`train_v3.py`** — in-domain nested-protocol trainer (test scored once; all model
  selection on inner splits).
- **`torch_net.py`** / **`mlx_net.py`** — neural estimators (PyTorch / Apple MLX).
- **`live_position.py`** — real-time inference demo.
- **`validate_session.py`**, **`plot_session.py`**, **`session_metadata.py`** — session
  QA, plotting, and metadata tooling.
- **`csi_gui/`** — native PySide6/Qt application for guided data collection
  (preflight checks, recording, calibration, live view).
- **`tests/`** — pytest suite for the geometry, dedup, split, and pipeline helpers.

## Hardware

- 5 × ESP32-C6-DevKitC-1U with external 2.4 GHz antennas
  (1 transmitter, 3 receivers, 1 sync clapper)
- An overhead camera (used here: an iPhone ultrawide lens streamed over USB)
- Printed ArUco markers (`DICT_4X4_50`) for the floor grid and the foot labels
- A laptop (developed on Apple Silicon macOS)

The iOS camera-streamer app (AnyCamStream) lives in its own repository and is not
required to run the analysis — any overhead camera that OpenCV can open will do.

## Quick start (analysis pipeline)

```bash
git clone https://github.com/<your-username>/wifi-csi-indoor-localization.git
cd wifi-csi-indoor-localization

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run the test suite
python -m pytest tests/ -q

# launch the data-collection GUI
python -m csi_gui.app
```

Recorded sessions are written under `sessions/` (git-ignored). The repository ships
the analysis code and firmware; collect your own sessions, or replay the pipeline on
sessions you record, following [`SESSION_CHECKLIST.md`](SESSION_CHECKLIST.md). See
[`CHEATSHEET.md`](CHEATSHEET.md) for a full terminal command reference.

## Building the firmware

With [ESP-IDF](https://docs.espressif.com/projects/esp-idf/) installed and sourced
(target `esp32c6`):

```bash
cd csi_rx
idf.py set-target esp32c6
idf.py -DPING_INTERVAL_MS=30 flash monitor
# one-time per receiver, in the serial console:
#   setid 4              # board id
#   setip 192.168.4.200  # laptop IP
```

Build `csi_tx/` and `csi_clapper/` the same way. The receiver status LED reports link
state: yellow = connecting, blue = streaming CSI, purple = associated but no CSI,
red = link dropped.

## Machine-learning evaluation

The honest, headline metric is **cross-session generalization**: train on some
recording sessions, test on a held-out session recorded at a different time
(leave-one-session-out). This is what `ml_drift.py` measures. The in-domain split in
`train_v3.py` reports model *capacity* on a single session and is labelled as such —
it is not a generalization claim.

A random stratified split over CSI frames leaks temporally adjacent, near-duplicate
frames across train and test and badly inflates accuracy; the dataset loader
de-duplicates reused-CSI frames by default to avoid this. Cross-session median error
is sub-metre and improves further with a short same-day calibration. Exact figures,
ablations, and the full protocol are documented in the thesis.

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers the geometry/trilateration math, CSI de-duplication, the
per-cell temporal split, calibration helpers, the collector, and the GUI components.

## Citation

If you use this work, please cite the thesis:

```bibtex
@thesis{gherghisan2026csi,
  title  = {Device-free Indoor Localization using WiFi Channel State Information},
  author = {Gherghisan, Andrei},
  year   = {2026},
  type   = {Bachelor's thesis}
}
```

## License

Released under the [MIT License](LICENSE).
