"""Native PySide6/Qt data-collection GUI for the WiFi-CSI localization rig.

Phase 3 scope: the live-video shell only. The heavy decode/undistort/ArUco
detect pipeline stays in ``aruco_track.ArucoTracker`` (its worker thread); this
package only renders the small downscaled preview it already emits, throttled to
~15 fps, and overlays the current position. Preflight, record controls and
monitor charts arrive in later phases.
"""
