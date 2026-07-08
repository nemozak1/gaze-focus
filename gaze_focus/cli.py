import argparse


def _camera_list(s):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(
        prog="gaze-focus",
        description="Look at a window, double-blink, type there: webcam gaze -> X11 focus.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_cameras(p):
        p.add_argument("--cameras", type=_camera_list, default=[0], metavar="N[,N...]",
                       help="V4L2 camera indices, e.g. 0,4 to fuse two cameras (default 0)")

    c = sub.add_parser("calibrate", help="show targets on each monitor and fit the gaze model")
    add_cameras(c)
    c.add_argument("--grid", type=int, default=3, help="targets per monitor axis (default 3 -> 9/monitor)")
    c.add_argument("--append", action="store_true",
                   help="add to the existing calibration instead of replacing it "
                        "(do this sitting in a different posture to build movement tolerance)")

    r = sub.add_parser("run", help="track gaze and switch window focus (double-blink by default)")
    add_cameras(r)
    r.add_argument("--trigger", choices=["dwell", "blink"], default="dwell",
                   help="dwell: auto-focus after the gaze rests on a window (default); "
                        "blink: double-blink at a window to focus it")
    r.add_argument("--no-clicks", action="store_true",
                   help="disable face-gesture clicks")
    r.add_argument("--click-gesture", choices=["smirk", "pucker", "cheeks", "jaw", "none"],
                   default="smirk",
                   help="left-click gesture (default smirk = shift mouth sideways); "
                        "mouth-open right-clicks unless taken; test channels in `preview`")
    r.add_argument("--click-key", default="F8", metavar="KEY",
                   help="global hotkey that left-clicks at the gaze point (default F8; 'none' disables)")
    r.add_argument("--dwell", type=float, default=0.4, help="[dwell] seconds gaze must rest on a window (default 0.4)")
    r.add_argument("--idle", type=float, default=0.6, help="[dwell] seconds since last keypress before switching (default 0.6)")
    r.add_argument("--cooldown", type=float, default=None, help="min seconds between switches (default 0.5 blink / 1.2 dwell)")
    r.add_argument("--smooth", type=float, default=0.25,
                   help="smoothing cutoff in Hz; LOWER = steadier but laggier gaze point (default 0.25)")
    r.add_argument("--dry-run", action="store_true", help="print would-be switches, don't focus")
    r.add_argument("--verbose", action="store_true", help="print gaze point twice a second")
    r.add_argument("--overlay", action="store_true",
                   help="show a translucent click-through blob at the predicted gaze point")

    p = sub.add_parser("preview", help="live webcam overlay: landmarks + predicted gaze on a monitor map")
    add_cameras(p)

    sub.add_parser("refit", help="refit the model from stored samples (no recalibration needed)")

    sub.add_parser("windows", help="list the windows/geometry the tracker sees (sanity check)")

    t = sub.add_parser("camera-test", help="check cameras + face detection without any UI")
    add_cameras(t)

    args = ap.parse_args()

    if args.cmd == "calibrate":
        from .calibrate import run_calibration
        run_calibration(cameras=args.cameras, grid=args.grid, append=args.append)
    elif args.cmd == "run":
        from .run import run
        run(cameras=args.cameras, trigger=args.trigger, dwell=args.dwell,
            idle=args.idle, cooldown=args.cooldown, smooth=args.smooth,
            clicks=not args.no_clicks, click_gesture=args.click_gesture,
            click_key=args.click_key,
            dry_run=args.dry_run, verbose=args.verbose, overlay=args.overlay)
    elif args.cmd == "preview":
        from .preview import preview
        preview(cameras=args.cameras)
    elif args.cmd == "refit":
        from .calibrate import refit
        refit()
    elif args.cmd == "windows":
        from .x11 import X11, get_monitors
        for m in get_monitors():
            print(f"monitor {m.name}: {m.width}x{m.height} at +{m.x}+{m.y}")
        x11 = X11()
        active = x11.active_window()
        for w in x11.list_windows():
            mark = "*" if w.id == active else " "
            print(f" {mark} [{w.x:5d},{w.y:5d} {w.width:4d}x{w.height:4d}] {w.title[:60]}")
        print("(* = currently focused; topmost first)")
    elif args.cmd == "camera-test":
        from .features import MultiCamera
        multi = MultiCamera(args.cameras)
        frames = dict.fromkeys(args.cameras, 0)
        faces = dict.fromkeys(args.cameras, 0)
        for _ in range(45):
            feats_list, frame_list = multi.read()
            for cam, fe, fr in zip(args.cameras, feats_list, frame_list):
                frames[cam] += fr is not None
                faces[cam] += fe is not None
        multi.close()
        for cam in args.cameras:
            print(f"cam{cam}: {frames[cam]}/45 frames captured, face detected in {faces[cam]}")
        if not any(frames.values()):
            raise SystemExit("no camera produced frames — try other --cameras indices")
