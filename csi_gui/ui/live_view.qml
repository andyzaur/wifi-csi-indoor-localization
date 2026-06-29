import QtQuick

// Live-video scene for the CSI data-collection GUI (Phase 3).
//
// The MainWindow drives two plain QML properties from the GUI thread:
//   * frameCounter  — bumped on every frameReady; appended to the Image source
//                     so the cache-busted URL forces a reload of the provider's
//                     latest QImage.
//   * position*     — the current PositionState fields, for the overlay.
// No JS timers, no polling: the C++/Python side pushes; QML just binds.
Rectangle {
    id: root
    color: "#0c0e13"   // theme C.INSET — the video well
    implicitWidth: 960
    implicitHeight: 540

    // ---- driven from MainWindow (GUI thread) ---------------------------------
    property int frameCounter: 0
    property bool posDetected: false
    property real posX: 0.0
    property real posY: 0.0
    property real gridX: 0.0
    property real gridY: 0.0
    property string posMethod: "-"
    property real posFps: 0.0
    // False for empty-room (CSI-only) sessions: the idle hint changes and the
    // tracking overlay stays hidden (there is no camera at all).
    property bool cameraEnabled: true

    // True once at least one preview frame has arrived this session.
    readonly property bool hasFrames: frameCounter > 0

    Image {
        id: live
        anchors.fill: parent
        cache: false
        asynchronous: false
        visible: root.hasFrames
        fillMode: Image.PreserveAspectFit
        // Cache-busting: a fresh URL each frame forces requestImage() to run.
        // Empty until the first frame so startup never queries the provider
        // (which would log "Failed to get image" before any session runs).
        source: root.hasFrames ? "image://live/frame/" + root.frameCounter : ""
    }

    // ---- idle state (before any frame arrives) -------------------------------
    Column {
        anchors.centerIn: parent
        spacing: 8
        visible: !root.hasFrames

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: root.cameraEnabled ? "◉" : "▦"
            color: "#313845"
            font.pixelSize: 44
        }
        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: root.cameraEnabled
                  ? "Live camera preview"
                  : "Empty-room session — no camera"
            color: "#99a2b2"
            font.pixelSize: 16
            font.bold: true
        }
        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: root.cameraEnabled
                  ? "The feed appears when you start a session."
                  : "Only CSI + clapper are recorded; watch the monitor on the right."
            color: "#626b79"
            font.pixelSize: 13
        }
    }

    // ---- overlay: position read-out -----------------------------------------
    Rectangle {
        id: overlay
        visible: root.hasFrames && root.cameraEnabled
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.margins: 12
        radius: 8
        color: Qt.rgba(0, 0, 0, 0.55)
        border.color: root.posDetected ? "#3ad492" : "#ff6b6b"
        border.width: 1
        width: info.implicitWidth + 24
        height: info.implicitHeight + 16

        Column {
            id: info
            anchors.centerIn: parent
            spacing: 2

            Text {
                text: root.posDetected ? "TRACKING" : "no marker"
                color: root.posDetected ? "#3ad492" : "#ff6b6b"
                font.pixelSize: 14
                font.bold: true
            }
            Text {
                visible: root.posDetected
                text: "x: " + root.posX.toFixed(1) + " cm   y: " + root.posY.toFixed(1) + " cm"
                color: "#e0e0e0"
                font.pixelSize: 13
                font.family: "Menlo"
            }
            Text {
                visible: root.posDetected
                text: "cell: " + root.gridX.toFixed(0) + ", " + root.gridY.toFixed(0) + " cm"
                color: "#a0a0a0"
                font.pixelSize: 12
                font.family: "Menlo"
            }
            Text {
                visible: root.posDetected
                text: "method: " + root.posMethod
                color: "#a0a0a0"
                font.pixelSize: 12
                font.family: "Menlo"
            }
            Text {
                text: "fps: " + root.posFps.toFixed(1)
                color: "#808080"
                font.pixelSize: 12
                font.family: "Menlo"
            }
        }
    }
}
