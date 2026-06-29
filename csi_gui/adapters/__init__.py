"""Thread-boundary adapters between the ArUco tracker worker and the Qt GUI.

- ``frame_provider.LiveFrameProvider`` — a QQuickImageProvider holding exactly
  one current QImage, swapped under a lock (latest-frame-only).
- ``signal_bridge.CameraBridge`` — a QObject that receives the tracker's
  worker-thread ``on_frame`` callback, throttles it to ~15 fps, pushes the
  RGB frame into the provider, and re-emits queued cross-thread signals so the
  GUI thread can refresh.
"""
