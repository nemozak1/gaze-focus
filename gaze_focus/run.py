"""Main loop: predict gaze point, hit-test windows, trigger, switch focus."""
import time

import numpy as np

from .blink import BlinkDetector
from .features import MultiCamera
from .filter import FixationFilter
from .gestures import GestureDetector
from .model import FusionModel, saved_error_px
from .paths import CALIB_PATH
from .x11 import X11, get_monitors


# click-gesture options: (score extractor, firing delta above baseline);
# deltas sit well above measured resting/talking noise per channel
GESTURE_OPTIONS = {
    "smirk": (lambda g: max(g["mouthLeft"], g["mouthRight"]), 0.30),
    "pucker": (lambda g: g["mouthPucker"], 0.45),
    "cheeks": (lambda g: g["cheekPuff"], 0.25),
    "jaw": (lambda g: g["jawOpen"], 0.30),
}


def run(cameras=(0,), trigger="dwell", dwell=0.4, idle=0.6, cooldown=None,
        smooth=0.25, clicks=True, click_gesture="smirk", click_key="F8",
        dry_run=False, verbose=False, overlay=False):
    if cooldown is None:
        cooldown = 0.5 if trigger == "blink" else 1.2
    if not CALIB_PATH.exists():
        raise SystemExit(f"no calibration at {CALIB_PATH} — run `gaze-focus calibrate` first")
    model = FusionModel.load(CALIB_PATH)
    if model.cameras != list(cameras):
        raise SystemExit(f"calibration was made with --cameras "
                         f"{','.join(map(str, model.cameras))} — pass the same, "
                         "or recalibrate")
    ext = MultiCamera(cameras)
    x11 = X11()
    blink = BlinkDetector()
    blob = None
    if overlay:
        from .overlay import GazeOverlay
        blob = GazeOverlay()
    err = saved_error_px()
    gaze_filter = FixationFilter(snap_dist=1.4 * err, min_cutoff=smooth)
    mons = get_monitors()
    seam_stick = 0.5 * err  # must land this far inside another monitor to switch
    cur_mon = None

    clickers = []
    if clicks and click_gesture != "none":
        extract, delta = GESTURE_OPTIONS[click_gesture]
        clickers.append(_GestureClick(extract, GestureDetector(delta_on=delta),
                                      1, click_gesture))
        if click_gesture != "jaw":  # mouth-open right click unless taken
            jaw_extract, jaw_delta = GESTURE_OPTIONS["jaw"]
            clickers.append(_GestureClick(jaw_extract, GestureDetector(delta_on=jaw_delta),
                                          3, "mouth-open"))
    click_code = x11.grab_key(click_key) if click_key and click_key != "none" else None

    smoothed = None
    candidate_id = None
    candidate_since = 0.0
    last_switch = 0.0
    last_key = 0.0
    windows = []
    windows_at = 0.0
    last_report = 0.0

    how = ("double-blink at a window to focus it" if trigger == "blink"
           else f"auto-focus after {dwell}s gaze dwell + {idle}s keyboard idle")
    extras = []
    if clickers:
        extras.append(f"{click_gesture} = left click"
                      + (", mouth-open = right click" if len(clickers) > 1 else ""))
    if click_code is not None:
        extras.append(f"{click_key} = left click at gaze")
    extras = ("; " + "; ".join(extras)) if extras else ""
    print(f"gaze-focus running ({'dry-run; ' if dry_run else ''}{how}{extras}); Ctrl-C to stop.")
    ext.start()  # cameras process in the background; this loop runs at UI rate
    last_stamp = 0.0
    try:
        while True:
            if blob is not None:
                blob.tick()  # glide the overlay every UI frame (~60Hz)
            time.sleep(0.015)
            feats_list, blink_score, gestures, stamp = ext.latest()
            now = time.time()
            if x11.any_key_down():
                last_key = now
            if click_code is not None and smoothed is not None:
                if any(c == click_code for c in x11.grabbed_key_presses()):
                    _click(x11, smoothed, 1, dry_run, blob, click_key)
            if stamp == last_stamp:  # no new camera data yet
                continue
            last_stamp = stamp
            if all(f is None for f in feats_list):
                candidate_id = None
                continue

            if smoothed is not None:
                for clicker in clickers:
                    point = clicker.feed(clicker.extract(gestures), now, smoothed)
                    if point is not None:
                        _click(x11, point, clicker.button, dry_run, blob, clicker.label)

            event = blink.update(blink_score, now)
            if not blink.closed:
                # Only track gaze while eyes are open: iris features are
                # garbage mid-blink, and freezing keeps the pre-blink target.
                point = model.predict(feats_list)
                if point is not None:
                    smoothed = gaze_filter.update(point, now)
            if blob is not None and smoothed is not None:
                blob.set_target(*smoothed)
            if smoothed is None:
                continue

            if now - windows_at > 0.2:  # refresh window list at ~5Hz
                windows = x11.list_windows()
                windows_at = now

            # seam hysteresis: gaze can't physically rest on the bezel, so a
            # point that only just crosses onto another monitor stays counted
            # on the current one until it lands decisively inside the new one
            hit_point = smoothed
            mon = next((m for m in mons
                        if m.x <= smoothed[0] < m.x + m.width
                        and m.y <= smoothed[1] < m.y + m.height), None)
            if cur_mon is None:
                cur_mon = mon
            elif mon is not cur_mon:
                clamped = np.array([
                    min(max(smoothed[0], cur_mon.x), cur_mon.x + cur_mon.width - 1),
                    min(max(smoothed[1], cur_mon.y), cur_mon.y + cur_mon.height - 1)])
                if mon is None or np.linalg.norm(smoothed - clamped) < seam_stick:
                    hit_point = clamped
                else:
                    cur_mon = mon
            hit = next((w for w in windows if w.contains(*hit_point)), None)
            active = x11.active_window()

            if verbose and now - last_report > 0.5:
                where = f"over [{hit.title[:45]}]" if hit else "over nothing"
                print(f"gaze ({smoothed[0]:5.0f},{smoothed[1]:5.0f}) "
                      f"blink={ext.blink_score:.2f} {where}")
                last_report = now

            if trigger == "blink":
                if (event == "double" and hit is not None and hit.id != active
                        and now - last_switch >= cooldown):
                    _switch(x11, hit, dry_run, blob)
                    last_switch = now
                continue

            # dwell trigger
            if hit is None or hit.id == active:
                candidate_id = None
                continue
            if hit.id != candidate_id:
                candidate_id = hit.id
                candidate_since = now
                continue
            if (now - candidate_since >= dwell and now - last_key >= idle
                    and now - last_switch >= cooldown):
                _switch(x11, hit, dry_run, blob)
                last_switch = now
                candidate_id = None
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if blob is not None:
            blob.close()
        ext.close()


def _switch(x11, win, dry_run, blob=None):
    if blob is not None:
        blob.flash()
    if dry_run:
        print(f"[dry-run] would focus: {win.title[:70]}")
    else:
        x11.activate(win.id)
        print(f"focused: {win.title[:70]}")


class _GestureClick:
    """Pairs a gesture with a click, latching a PRE-gesture gaze point —
    the frame that first shows the gesture already shows the perturbed
    face, so the click target comes from ~250ms before onset."""

    LOOKBACK_S = 0.25

    def __init__(self, extract, detector, button, label):
        from collections import deque
        self.extract = extract
        self.detector = detector
        self.button = button
        self.label = label
        self._latch = None
        self._history = deque(maxlen=32)  # (t, gaze point)

    def feed(self, score, now, smoothed):
        """Returns the point to click, or None."""
        self._history.append((now, np.array(smoothed, copy=True)))
        was_active = self.detector.active
        fired = self.detector.update(score, now)
        if self.detector.active and not was_active:
            self._latch = self._pre_gesture_point(now)
        if fired:
            point = self._latch if self._latch is not None else smoothed
            self._latch = None
            return point
        if not self.detector.active:
            self._latch = None
        return None

    def _pre_gesture_point(self, now):
        for t, p in reversed(self._history):
            if now - t >= self.LOOKBACK_S:
                return p
        return self._history[0][1]


def _click(x11, point, button, dry_run, blob, source):
    if blob is not None:
        blob.flash()
    name = {1: "left", 3: "right"}[button]
    if dry_run:
        print(f"[dry-run] would {name}-click at ({point[0]:.0f},{point[1]:.0f}) [{source}]")
    else:
        x11.click(point[0], point[1], button)
        print(f"{name}-click at ({point[0]:.0f},{point[1]:.0f}) [{source}]")
