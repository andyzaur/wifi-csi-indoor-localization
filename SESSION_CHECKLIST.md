# Data Collection Session Checklist

Run through this every time. Skipping steps loses sessions.

## A — Pre-flight (do once at start of the day)

```bash
# 1. Unplug ethernet cable from laptop
# 2. Verify ESP boards are powered on (TX + 3 RX + clapper on battery)
# 3. Verify iPhone on tripod, Camo unused, AnyCamStream installed

# 4. Connect laptop WiFi to CSI_TX
networksetup -setairportnetwork en0 CSI_TX 23456789

# 4b. Set the laptop's STATIC IP on CSI_TX (REQUIRED — RX boards unicast CSI here).
#     The RX firmware sends CSI to 192.168.4.200; the laptop MUST hold that IP.
sudo networksetup -setmanual "Wi-Fi" 192.168.4.200 255.255.255.0 192.168.4.1
#     NOTE: -setmanual/-setdhcp take the SERVICE NAME ("Wi-Fi"), not en0
#           (unlike -setairportnetwork, which takes en0). Verify the name with
#           `networksetup -listallnetworkservices` if "Wi-Fi" errors.
#     ⚠️ This sticks across networks. You MUST revert to DHCP in wrap-up
#        (section E) or your normal WiFi will have no internet afterward.

# 5. Verify TX is reachable
ping -c 3 192.168.4.1
# Must see "3 packets received, 0% loss". If not, power-cycle TX board.

# 6. Verify all 3 RX boards are sending CSI
source venv/bin/activate
python3 csi_collector.py --no-csv
# Should see all 3 RX boards (IDs 1, 4, 5) reporting within seconds
# Ctrl+C when verified

# 7. Start iPhone USB tunnel
iproxy 8080 8080 &

# 8. Open AnyCamStream on iPhone, tap Start Streaming
# 9. Verify camera stream
curl -s --max-time 3 http://127.0.0.1:8080/video -o /dev/null -w "HTTP %{http_code}\n"
# Must see "HTTP 200"

# 10. Refresh floor calibration (if camera moved since last setup)
python3 aruco_setup.py --camera http://127.0.0.1:8080/video
# SPACE when >=4 markers detected, 's' to save
```

**RX board status LEDs** (glance to confirm all 3 boards are healthy):
**blue** = streaming CSI ✓ · **purple** = connected but no CSI · **yellow** = connecting · **red** = link dropped.

First-time board setup (identity now lives in NVS — flash one image to all boards, no per-board reflash):
`idf.py -DPING_INTERVAL_MS=30 flash monitor`, then in the `rx>` console: `setid <1|4|5>`, `setip 192.168.4.200`, `cfg`, reboot.

## B — Per session (repeat for each capture)

Pick a session name following the convention: `YYYYMMDD_NN_purpose`
Examples:
- `20260519_01_empty_baseline`     (no person in room)
- `20260519_02_static_center`      (stand still at one cell, 60s)
- `20260519_03_walk_grid_slow`     (30s per grid cell, walk between)
- `20260519_04_walk_freeform`      (continuous walking)

```bash
# 1. Set session name (export so both terminals see it)
export SESSION=20260519_NN_purpose

# 2. TERMINAL A — CSI + clap collector
source venv/bin/activate
python3 csi_collector.py --session $SESSION

# 3. TERMINAL B — camera ground truth
source venv/bin/activate
python3 aruco_track.py --camera http://127.0.0.1:8080/video \
    --log sessions/$SESSION/camera.csv \
    --display-scale 0.25
# Two-leg defaults already correct: right=marker 0, left=marker 9,
# offsets 20/-15 cm, orient-smooth 0.3 — no need to pass them.
# See section "F — aruco_track.py arguments" below for what each flag does.

# 4. Make sure BOTH foot markers are on:
#    ID 0 = RIGHT foot, ID 9 = LEFT foot, each with the top edge toward the toes.

# 5. CLAP START — press clapper button (green flash). Hold in camera view briefly.

# 6. Conduct session
#    - Walk to grid cell, stand still 20-30s, walk to next
#    - Try to avoid covering >2 floor markers at once with your body
#    - Stay slow and steady

# 7. CLAP STOP — press clapper button (red flash). Hold in camera view briefly.

# 8. Ctrl+C both terminals (CSI first, then camera — order doesn't really matter)

# 9. Record metadata WHILE YOU REMEMBER (room, furniture, people, walk style, board placement):
python3 session_metadata.py sessions/$SESSION --interactive
#    (corners.csv + keyframes/ are auto-saved next to camera.csv as the label audit trail)
```

## F — aruco_track.py arguments (reference)

Run `python3 aruco_track.py --help` for the live list. Common flags:

| Flag | Default | What it does / how it acts |
|---|---|---|
| `--camera URL` | — | Live source. Use the AnyCamStream URL `http://127.0.0.1:8080/video` (or an OpenCV index like `0`). |
| `--log PATH` | none | Write the per-frame CSV here. **Omit `--log` → live preview only, nothing saved** (use for testing). |
| `--marker-right N` | 0 | ArUco ID on the RIGHT foot. `-1` disables that foot (single-marker mode). |
| `--marker-left N` | 9 | ArUco ID on the LEFT foot. `-1` disables. |
| `--offset-side CM` | 20 | Sideways distance (magnitude) from each foot marker to body center; auto-mirrored L/R. |
| `--offset-back CM` | -15 | Distance along the foot's facing direction; negative = toward the heel. |
| `--orient-smooth A` | 0.3 | Marker-orientation smoothing (EMA weight, 0<A≤1). **Lower = smoother/less jitter but more lag; 1 = off.** Tune live if the body dot jitters. |
| `--display-scale F` | 0.5 | Shrinks the PREVIEW WINDOW only (detection stays full-res). **Lower = smaller window + a bit more fps.** Try `0.25` if far from screen. |
| `--no-display` | off | Headless — no window at all. **Highest fps / lowest CPU**; use once you trust the tracking. |
| `--workers N` | 6 | Decode+detect thread-pool size. |
| `--video FILE` | none | Process a recorded video instead of live camera (offline labeling). |
| `--keyframe-stride N` | 30 | Audit: save 1 JPEG every N frames to `<session>/keyframes/` (`0` = off). ~70-150 MB/session at default. Raw ArUco corners always logged to `corners.csv` regardless of this. |

Notes on how they interact:
- **Accuracy is set by detection + offsets**, which always run at full resolution — `--display-scale` and `--no-display` only affect what you *see*, never the saved labels.
- To raise fps: lower `--display-scale` (cuts render cost) or use `--no-display`. To cut the *real* CPU hog (decode + ArUco detection), lower the **camera capture** resolution in the AnyCamStream app — that's not a flag here.
- With CSI now at ~33 Hz and the camera at ~25 fps, CSI is faster than the camera, so every frame gets fresh CSI and dedup removes ~nothing — the low camera fps is fine, not a problem.

## C — Verify before the next session

```bash
# Check the files exist and have data
ls -la sessions/$SESSION/
wc -l sessions/$SESSION/csi.csv sessions/$SESSION/camera.csv sessions/$SESSION/clap.csv
```

Expected sizes:
- `csi.csv`: thousands of rows (>= 2 lines for header + first packet, ideally 600+ rows per minute)
- `camera.csv`: hundreds to thousands of rows (depends on camera FPS)
- `clap.csv`: exactly 3 lines (header + START + STOP)

**If `csi.csv` is just the header** → WiFi dropped, redo session. Check `ping 192.168.4.1`.
**If `clap.csv` has only header** → clapper didn't connect. Power-cycle clapper.
**If `camera.csv` is small** → iproxy died or AnyCamStream stopped. Restart both.

## D — Behavior tips during sessions

- Same outfit across same-day sessions (clothing affects RF reflection)
- Doors closed, consistent room configuration
- Walk slowly between cells — transitions are noisy and will be discarded later
- Stand still at each cell for at least 20s for clean training data
- No phone/BT devices in the room (except the iPhone-as-camera, which has BT/WiFi off)
- No microwave running anywhere in the apartment

## E — Wrap up

```bash
# 0. ⚠️ FIRST: revert the static IP or your normal WiFi has no internet!
sudo networksetup -setdhcp "Wi-Fi"

# When done collecting for the day, plug ethernet back in
# Stop iproxy and AnyCamStream
pkill iproxy
# Tap Stop Streaming on iPhone

# Quick backup of session data
git status
# If you want to push session data (sessions/ is gitignored by default, edit .gitignore if needed)
```

## Common failure modes (one-line fixes)

| Symptom | Fix |
|---|---|
| `ping 192.168.4.1` fails | Power-cycle TX board, re-run `networksetup -setairportnetwork en0 CSI_TX 23456789` |
| Mac WiFi "not associated" but has IP | Plug/unplug ethernet to kick the routing table |
| `iproxy` says address in use | `pkill iproxy && iproxy 8080 8080 &` |
| AnyCamStream "Cannot verify app" | Plug iPhone in, Xcode → Cmd+R (7-day provisioning expired) |
| Clapper not flashing on press | Battery dead or button stuck; replace battery / try a different USB-C power bank |
| `csi.csv` only header | WiFi dropped to CSI_TX mid-session; redo with ethernet unplugged |
| No boards in `--no-csv` preflight (but WiFi up) | Laptop static IP not set — RX boards unicast to 192.168.4.200; run the step-4b `setmanual`. Check `ifconfig en0` shows `192.168.4.200`. |
| An RX board's LED is red / won't turn blue | Red = link dropped → power-cycle that board. Purple = connected but no CSI (check TX is up). Collector summary also prints `MALFORMED: n` if packets arrive corrupted. |
