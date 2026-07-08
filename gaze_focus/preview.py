"""Live webcam overlay: landmarks, feature readout, gaze point on a monitor map."""
import math
import time

import cv2
import numpy as np

from . import features as F
from .blink import BlinkDetector
from .features import MultiCamera
from .filter import FixationFilter
from .model import FusionModel, saved_error_px
from .paths import CALIB_PATH
from .x11 import X11, get_monitors

EYE_IDX = [F.L_IRIS, F.R_IRIS, F.L_EYE_OUTER, F.L_EYE_INNER, F.R_EYE_INNER,
           F.R_EYE_OUTER, F.L_LID_TOP, F.L_LID_BOT, F.R_LID_TOP, F.R_LID_BOT]
MAP_H = 170
CAM_H = 360  # each camera pane is scaled to this height
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, GREY, RED, YELLOW = (0, 220, 0), (150, 150, 150), (60, 60, 230), (0, 255, 255)


def _camera_pane(ext, feats, frame):
    if frame is None:
        pane = np.zeros((CAM_H, 480, 3), np.uint8)
        cv2.putText(pane, f"cam{ext.camera}: no frame", (10, 30), FONT, 0.7, RED, 2)
        return pane
    h, w = frame.shape[:2]
    vis = cv2.flip(frame, 1)  # mirror, so it moves like a mirror does
    if feats is not None:
        lm = ext.last_landmarks
        for i in EYE_IDX:
            cv2.circle(vis, (int((1 - lm[i].x) * w), int(lm[i].y * h)), 2, GREEN, -1)
        deg = math.degrees
        cv2.putText(vis,
                    f"cam{ext.camera} iris L({feats[0]:.2f},{feats[1]:.2f}) "
                    f"R({feats[2]:.2f},{feats[3]:.2f}) blink {ext.last_blink_score:.2f}",
                    (10, 22), FONT, 0.5, GREEN, 1)
        cv2.putText(vis,
                    f"rot({deg(feats[4]):+3.0f},{deg(feats[5]):+3.0f},"
                    f"{deg(feats[6]):+3.0f})deg "
                    f"pos({feats[7]:+4.1f},{feats[8]:+4.1f},{feats[9]:+5.1f})cm "
                    f"smirk {max(ext.last_gestures['mouthLeft'], ext.last_gestures['mouthRight']):.2f} "
                    f"pkr {ext.last_gestures['mouthPucker']:.2f} "
                    f"chk {ext.last_gestures['cheekPuff']:.2f} "
                    f"jaw {ext.last_gestures['jawOpen']:.2f}",
                    (10, 44), FONT, 0.5, GREEN, 1)
    else:
        cv2.putText(vis, f"cam{ext.camera}: NO FACE", (10, 30), FONT, 0.8, RED, 2)
    return cv2.resize(vis, (int(w * CAM_H / h), CAM_H))


def preview(cameras=(0,), smooth=0.25):
    model = FusionModel.load(CALIB_PATH) if CALIB_PATH.exists() else None
    if model is None:
        print("no calibration found — showing landmarks only "
              "(run `gaze-focus calibrate` to get the gaze map)")
    elif model.cameras != list(cameras):
        print(f"note: calibration is for --cameras "
              f"{','.join(map(str, model.cameras))}, previewing "
              f"{','.join(map(str, cameras))} — no gaze prediction")
        model = None
    mons = get_monitors()
    x11 = X11()
    multi = MultiCamera(cameras)

    ox = min(m.x for m in mons)
    oy = min(m.y for m in mons)
    total_w = max(m.x + m.width for m in mons) - ox
    total_h = max(m.y + m.height for m in mons) - oy

    smoothed = None
    gaze_filter = FixationFilter(snap_dist=1.4 * saved_error_px(), min_cutoff=smooth)
    windows, windows_at = [], 0.0
    blink = BlinkDetector()
    flash_until = 0.0
    print("preview running — press q or ESC in the window to quit; "
          "try a double-blink, it should flash")
    try:
        while True:
            feats_list, frames = multi.read()
            now = time.time()
            if all(f is None for f in frames):
                print("no camera produced frames")
                break
            panes = [_camera_pane(ext, feats, frame)
                     for ext, feats, frame in zip(multi.exts, feats_list, frames)]
            vis = panes[0] if len(panes) == 1 else np.hstack(panes)
            h, w = vis.shape[:2]

            if any(f is not None for f in feats_list):
                if blink.update(multi.blink_score, now) == "double":
                    flash_until = now + 0.8
            if now < flash_until:
                cv2.putText(vis, "DOUBLE BLINK", (w // 2 - 130, h // 2),
                            FONT, 1.1, YELLOW, 3)

            # monitor mini-map strip under the video
            canvas = np.zeros((h + MAP_H, w, 3), np.uint8)
            canvas[:h] = vis
            scale = min((w - 20) / total_w, (MAP_H - 45) / total_h)

            def mx(gx):
                return int(10 + (gx - ox) * scale)

            def my(gy):
                return int(h + 10 + (gy - oy) * scale)

            for m in mons:
                cv2.rectangle(canvas, (mx(m.x), my(m.y)),
                              (mx(m.x + m.width), my(m.y + m.height)), GREY, 1)
                cv2.putText(canvas, m.name, (mx(m.x) + 5, my(m.y) + 16),
                            FONT, 0.4, GREY, 1)

            if model is not None and not blink.closed:
                point = model.predict(feats_list)
                if point is not None:
                    smoothed = gaze_filter.update(point, now)
            if model is not None and smoothed is not None:
                if now - windows_at > 0.2:
                    windows = x11.list_windows()
                    windows_at = now
                hit = next((win for win in windows if win.contains(*smoothed)), None)
                if hit:
                    cv2.rectangle(canvas, (mx(hit.x), my(hit.y)),
                                  (mx(hit.x + hit.width), my(hit.y + hit.height)),
                                  GREEN, 1)
                cv2.circle(canvas, (mx(smoothed[0]), my(smoothed[1])), 5, GREEN, -1)
                label = hit.title[:55] if hit else "(nothing)"
                cv2.putText(canvas,
                            f"gaze ({smoothed[0]:5.0f},{smoothed[1]:5.0f}) -> {label}",
                            (10, h + MAP_H - 12), FONT, 0.5, GREEN, 1)
            elif model is None:
                cv2.putText(canvas, "uncalibrated: no gaze prediction",
                            (10, h + MAP_H - 12), FONT, 0.5, GREY, 1)

            cv2.imshow("gaze-focus preview", canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()
        multi.close()
