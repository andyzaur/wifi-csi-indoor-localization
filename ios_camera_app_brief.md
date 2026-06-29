# iOS Camera Streaming App — Build Brief

## What this app is for

This is part of a bachelor's thesis on **WiFi-based indoor localization**. The system uses ESP32 boards to collect WiFi Channel State Information (CSI), and a **camera mounted overhead** to provide ground truth position labels. The camera watches the floor, detects ArUco markers placed on the floor and on the user, and maps the user's position to a 50cm grid. This labeled data trains an ML model to predict position from CSI alone.

**The camera is the ground truth labeling system.** Without it, we can't generate training data. Without training data, the thesis doesn't work.

## The problem this app solves

We need to stream the **iPhone 11 Pro Max ultrawide (0.5x) camera** to a MacBook Pro over USB, at the **highest resolution possible** (ideally 4K 3840x2160, minimum 1920x1440).

**Why ultrawide specifically:** The phone is mounted on a tripod at ~2.5m height looking down at a small room (~3m x 4m). The ultrawide's 120° FOV is the only lens that covers the entire walkable floor area. The standard 1x wide lens was tested and only captures about half the room — not enough.

**Why we can't use existing apps:**
- **macOS Continuity Camera**: caps at 1920x1440, only exposes the wide (1x) lens, not ultrawide. Desk View uses ultrawide but Apple crops it to a small desk-sized rectangle.
- **Camo (by Reincubate)**: supports ultrawide lens selection, but locks resolution above 720p behind a €50/year subscription.
- **Iriun Webcam**: free, gives 4K, but has NO ultrawide lens selection at all — only the main lens.
- **Other apps** (mmhmm, DroidCam, EpocCam, etc.): either don't support iPhone ultrawide, or paywall it.

**Bottom line:** No free app on the market gives us ultrawide + high resolution streaming from iPhone to Mac. So we're building our own.

## What the app needs to do

### Core functionality
1. **Open the ultrawide camera** using AVFoundation — specifically `AVCaptureDevice.DeviceType.builtInUltraWideCamera` on the back-facing position
2. **Capture frames** at the highest resolution the ultrawide supports — on iPhone 11 Pro Max, the ultrawide sensor is 12MP (4032x3024). For video streaming, 4K (3840x2160) or 1920x1440 at 30fps is the target.
3. **Stream the frames over HTTP as MJPEG** on a configurable port (default: 8080). The Mac-side Python script consumes the stream like this:
   ```python
   cap = cv2.VideoCapture("http://<iphone-ip>:8080/video")
   ```
   OpenCV reads MJPEG streams natively — no special protocol needed.
4. **Work over USB** — when the iPhone is connected via Lightning cable to the Mac, iOS creates a USB-tethered network interface. The Mac can reach the iPhone on a local IP. The app should display this IP on screen so the user knows where to connect.

### UI (minimal)
- Show the live camera preview so the user can verify framing
- Display the streaming status: "Streaming on http://x.x.x.x:8080/video"
- Display the current resolution and FPS
- A button to start/stop streaming
- Optionally: a resolution picker (4K / 1080p / 720p) in case 4K causes performance issues

### Nice to have (not critical)
- Auto-exposure lock and auto-focus lock toggle (for consistent lighting during long sessions)
- A "torch" / flashlight toggle (for poorly lit rooms)
- Frame counter / uptime display

## Technical details

### Camera setup (AVFoundation)
```swift
// Pseudocode — the actual implementation is up to you
let session = AVCaptureSession()
session.sessionPreset = .photo  // or a specific format for max resolution

let device = AVCaptureDevice.default(
    .builtInUltraWideCamera,
    for: .video,
    position: .back
)

// Configure for highest resolution
// The ultrawide on iPhone 11 PM supports up to 4032x3024 (photo) or 3840x2160 (video)
// Pick the highest video format available at 30fps

let input = try AVCaptureDeviceInput(device: device!)
session.addInput(input)

let output = AVCaptureVideoDataOutput()
output.setSampleBufferDelegate(delegate, queue: queue)
session.addOutput(output)

session.startRunning()
```

### MJPEG streaming
MJPEG over HTTP is the simplest video streaming protocol:
- The server responds with `Content-Type: multipart/x-mixed-replace; boundary=frame`
- Each frame is sent as a JPEG with the boundary separator
- OpenCV's VideoCapture handles this natively

For the HTTP server, you can use:
- **GCDWebServer** (CocoaPod/SPM — lightweight, well-known)
- **Vapor** (heavier but full-featured)
- **Raw NWListener** from Apple's Network.framework (no dependencies, more manual)
- A simple custom TCP server using `CFSocket` or `NWConnection`

The frame pipeline: `AVCaptureVideoDataOutput` → convert `CMSampleBuffer` to JPEG → push to HTTP response stream

### JPEG compression
- Quality 0.7–0.8 is a good balance of size vs quality for ArUco detection
- At 4K 30fps with quality 0.7, expect ~15-30 MB/s throughput. USB 2.0 (Lightning) handles 480 Mbps = 60 MB/s, so this is fine.
- If bandwidth is an issue, dropping to 1920x1440 or lowering JPEG quality are easy fallbacks

### Deployment
- **No Apple Developer Program needed** — use free Xcode provisioning to deploy to the user's own iPhone 11 Pro Max
- Free provisioning requires re-signing every 7 days, which is fine for a thesis project
- Target: iOS 16+ (the iPhone 11 PM supports up to iOS 18)
- The app only needs to run when actively collecting data — not a consumer product

## What the Mac side looks like

On the Mac, the existing Python scripts (already built) consume the stream:

```python
import cv2

# Connect to iPhone camera stream
cap = cv2.VideoCapture("http://172.20.10.1:8080/video")  # typical USB-tethered IP

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # ArUco marker detection, position tracking, grid overlay, etc.
    # ... (already implemented in aruco_setup.py and aruco_track.py)
    
    cv2.imshow("Ground Truth", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
```

The Python scripts that consume this are already built:
- `lens_calibrate.py` — chessboard lens calibration
- `aruco_setup.py` — floor marker detection + camera-to-floor homography
- `aruco_track.py` — live wearable marker tracking + grid snapping + CSV logging

These scripts currently use `cv2.VideoCapture(index)` for local cameras. They'll be updated to accept an HTTP URL instead — trivial change.

## Network setup over USB

When iPhone is connected to Mac via Lightning:
1. On iPhone: Settings → Personal Hotspot → ON (or it might just work via USB networking)
2. The iPhone typically gets IP `172.20.10.1` on the USB interface
3. The Mac can reach it at that IP
4. The app should also work over WiFi (same network), but USB is preferred for latency and reliability

**Important:** During actual data collection, the Mac's WiFi is connected to the ESP32 transmitter's Soft-AP network (for CSI data). So the camera connection MUST work over USB, not WiFi. This is the whole reason we need a wired connection.

## Summary

| Requirement | Detail |
|------------|--------|
| Camera | builtInUltraWideCamera (0.5x, 13mm) |
| Resolution | 4K preferred, 1920x1440 minimum |
| FPS | 30 |
| Protocol | MJPEG over HTTP |
| Connection | USB (Lightning cable) |
| Consumer | OpenCV `cv2.VideoCapture(url)` on Mac |
| Device | iPhone 11 Pro Max, iOS 16+ |
| Xcode | Free provisioning, no Developer Program |
| UI | Minimal — preview + status + start/stop |
| Dependencies | As few as possible — prefer Apple frameworks |
