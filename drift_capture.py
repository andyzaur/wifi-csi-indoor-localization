#!/usr/bin/env python3
"""Open-ended empty-room drift capture — starts on the 1st clapper press,
stops on the 2nd. For overnight / multi-hour static-channel drift recordings.

No camera / iproxy needed (empty room, CSI only). The collector writes
sessions/<name>/csi.csv + clap.csv exactly like a normal session, so clap.csv
holds the two press timestamps as clean t=0 / t=end anchors for the drift
analysis. Stop is keyed off the PRESS COUNT (2nd press ends it), not the
START/STOP label, so a clapper reboot mid-run can't confuse it; Ctrl-C is
always a fallback.

Wrap in caffeinate so the Mac doesn't sleep (system sleep is set to 1 min idle):

    caffeinate -is python3 drift_capture.py --session 12h_drift_overnight

Then: press the clapper once (recording begins / t=0 anchor), leave; on return
press it again -> the capture ends cleanly and the CSVs are flushed and closed.
"""
import argparse
import threading

from csi_collector import CsiCollector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", "-s", default=None,
                    help="session name (default: timestamp)")
    ap.add_argument("--stop-after-presses", type=int, default=2,
                    help="end the capture after this many clapper presses "
                         "(default 2 = press to start, press to stop)")
    args = ap.parse_args()

    done = threading.Event()
    presses = [0]

    def on_clap(ev):
        presses[0] += 1
        print(f"\n[clap] press #{presses[0]} — {ev.event_name} at {ev.wall_time_s:.3f}")
        if presses[0] >= args.stop_after_presses:
            print(f"[clap] press #{presses[0]} -> ending capture.")
            done.set()

    collector = CsiCollector(session_name=args.session, on_clap=on_clap)
    collector.start()
    print(f"Recording to sessions/{collector.session_name}/ — "
          f"press the clapper to START, press again to STOP (Ctrl-C also works).")
    try:
        while not done.is_set():
            done.wait(timeout=5.0)
    except KeyboardInterrupt:
        print("\nCtrl-C — stopping.")
    finally:
        collector.stop()
        print(f"Saved sessions/{collector.session_name}/")


if __name__ == "__main__":
    main()
