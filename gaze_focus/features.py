"""Webcam capture and gaze feature extraction via MediaPipe FaceLandmarker.

The feature vector combines iris position within each eye (fine gaze
direction) with head rotation and metric head position (coarse direction
+ distance) from the face transformation matrix. A ridge regression maps
this to screen coordinates.
"""
import math
import pathlib
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# FaceLandmarker landmark indices (478 points; 468-477 are the irises)
L_IRIS = 468
R_IRIS = 473
L_EYE_OUTER, L_EYE_INNER = 33, 133
R_EYE_INNER, R_EYE_OUTER = 362, 263
L_LID_TOP, L_LID_BOT = 159, 145
R_LID_TOP, R_LID_BOT = 386, 374

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

N_FEATURES = 10  # 2x(iris h,v) + 3 head rotation + 3 head translation


def _ensure_model():
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading face landmarker model (~4MB) to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class FeatureExtractor:
    def __init__(self, camera=0, width=1280, height=720):
        _ensure_model()
        self.camera = camera
        self.cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
        # MJPG first: raw YUYV at 720p exceeds USB2 bandwidth (~10fps);
        # MJPG keeps 30fps. Iris precision scales with pixels on the eye.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera index {camera} (/dev/video{camera})")
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_facial_transformation_matrixes=True,
                output_face_blendshapes=True))
        self._last_ts = 0
        self.last_landmarks = None    # set by read(); used by the preview overlay
        self.last_blink_score = 0.0   # min(eyeBlinkLeft, eyeBlinkRight), 0..1

    def close(self):
        self.cap.release()
        self.landmarker.close()

    def read(self):
        """Grab a frame. Returns (features, frame); features is None if no face."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:  # VIDEO mode needs strictly increasing timestamps
            ts = self._last_ts + 1
        self._last_ts = ts
        res = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.face_landmarks:
            self.last_landmarks = None
            self.last_blink_score = 0.0
            return None, frame
        self.last_landmarks = res.face_landmarks[0]
        if res.face_blendshapes:
            scores = {c.category_name: c.score for c in res.face_blendshapes[0]}
            self.last_blink_score = min(scores.get("eyeBlinkLeft", 0.0),
                                        scores.get("eyeBlinkRight", 0.0))
        else:
            self.last_blink_score = 0.0
        mat = (np.array(res.facial_transformation_matrixes[0])
               if res.facial_transformation_matrixes else np.eye(4))
        h, w = frame.shape[:2]
        return self._features(self.last_landmarks, mat, w, h), frame

    def _features(self, lm, mat, w, h):
        def px(i):
            return np.array([lm[i].x * w, lm[i].y * h])

        f = []
        # Iris position normalized within each eye's corner/lid box
        for iris, ci, co, lt, lb in (
            (L_IRIS, L_EYE_INNER, L_EYE_OUTER, L_LID_TOP, L_LID_BOT),
            (R_IRIS, R_EYE_INNER, R_EYE_OUTER, R_LID_TOP, R_LID_BOT),
        ):
            axis = px(co) - px(ci)
            denom = float(axis @ axis)
            hx = float((px(iris) - px(ci)) @ axis) / denom if denom > 1e-6 else 0.5
            top, bot = px(lt), px(lb)
            eye_h = bot[1] - top[1]
            vy = float(px(iris)[1] - top[1]) / eye_h if eye_h > 1e-3 else 0.5
            f += [hx, vy]

        # Head rotation (euler) + metric translation from the transformation matrix
        r = mat[:3, :3]
        sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
        f += [math.atan2(r[2, 1], r[2, 2]),
              math.atan2(-r[2, 0], sy),
              math.atan2(r[1, 0], r[0, 0])]
        f += list(mat[:3, 3])
        return np.array(f)


class MultiCamera:
    """Several FeatureExtractors read in lockstep, one per physical camera.

    Reads run in parallel threads — sequential reads would stack the
    cameras' frame waits and halve the sample rate the smoothing filter
    gets to work with.
    """

    def __init__(self, cameras):
        from concurrent.futures import ThreadPoolExecutor
        self.cameras = list(cameras)
        self.exts = [FeatureExtractor(c) for c in self.cameras]
        self._pool = (ThreadPoolExecutor(max_workers=len(self.exts))
                      if len(self.exts) > 1 else None)
        self._running = False

    def read(self):
        """Returns (feats_list, frames), both aligned with self.cameras."""
        if self._pool is None:
            results = [self.exts[0].read()]
        else:
            results = list(self._pool.map(lambda ext: ext.read(), self.exts))
        return [r[0] for r in results], [r[1] for r in results]

    @property
    def blink_score(self):
        """Best view wins: the camera facing you reports the real blink."""
        return max(ext.last_blink_score for ext in self.exts)

    def start(self):
        """Continuous background capture; poll with latest(). Lets the UI
        loop run at display rate instead of camera+inference rate."""
        import threading
        self._running = True
        self._lock = threading.Lock()
        self._latest = [(None, 0.0, 0.0)] * len(self.exts)  # feats, blink, stamp
        self._threads = []
        for i, ext in enumerate(self.exts):
            th = threading.Thread(target=self._capture_loop, args=(i, ext), daemon=True)
            th.start()
            self._threads.append(th)

    def _capture_loop(self, i, ext):
        import time as _time
        while self._running:
            feats, _ = ext.read()
            with self._lock:
                self._latest[i] = (feats, ext.last_blink_score, _time.monotonic())

    def latest(self):
        """(feats_list, fused_blink_score, newest_stamp) from background capture."""
        with self._lock:
            snap = list(self._latest)
        return ([s[0] for s in snap],
                max(s[1] for s in snap),
                max(s[2] for s in snap))

    def close(self):
        self._running = False
        for th in getattr(self, "_threads", []):
            th.join(timeout=2.0)
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        for ext in self.exts:
            ext.close()
