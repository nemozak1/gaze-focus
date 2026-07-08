"""Calibration: show gaze targets on every monitor, collect features, fit model."""
import json
import time
import tkinter as tk

import numpy as np

from .features import MultiCamera, N_FEATURES
from .model import FusionModel
from .paths import CALIB_PATH, SAMPLES_PATH
from .x11 import get_monitors

SETTLE_S = 1.0          # time to move eyes to the target before sampling starts
MIN_COLLECT_S = 1.0     # sample for at least this long
MIN_SAMPLES = 12        # ...and until this many face samples (or timeout)
COLLECT_TIMEOUT_S = 3.5


class _Aborted(Exception):
    pass


class _TargetScreen:
    """One borderless window spanning ALL monitors.

    Override-redirect so the WM can't decorate or resize it, and a single
    static window because moving a fullscreen window between monitors made
    GNOME flicker and starved the sampling loop with X events.
    """

    def __init__(self, mons):
        self.ox = min(m.x for m in mons)
        self.oy = min(m.y for m in mons)
        width = max(m.x + m.width for m in mons) - self.ox
        height = max(m.y + m.height for m in mons) - self.oy
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.configure(bg="black", cursor="none")
        self.root.geometry(f"{width}x{height}+{self.ox}+{self.oy}")
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.aborted = False
        for widget in (self.root, self.canvas):
            widget.bind("<Escape>", self._abort)
            widget.bind("<Button>", self._abort)
        self.root.lift()
        self.update()
        self.root.focus_force()  # so the ESC binding gets key events

    def _abort(self, _event):
        self.aborted = True

    def draw_target(self, gx, gy, mon, collecting, face_ok, label, n_samples):
        x, y = gx - self.ox, gy - self.oy  # global -> window coords
        c = self.canvas
        c.delete("all")
        color = "#00dc00" if collecting else "#ffa000"
        c.create_oval(x - 26, y - 26, x + 26, y + 26, outline=color, width=3)
        if collecting:
            c.create_oval(x - 10, y - 10, x + 10, y + 10, fill=color, outline="")
        status = f"target {label} — look at the circle (click or Ctrl-C quits)"
        if collecting:
            status += f"  [{n_samples} samples]"
        bx = mon.x - self.ox + 40
        by = mon.y - self.oy + mon.height - 60
        c.create_text(bx, by, anchor="w", text=status,
                      fill="#b4b4b4", font=("sans-serif", 16))
        if not face_ok:
            c.create_text(bx, by - 40, anchor="w", text="NO FACE DETECTED",
                          fill="#d03030", font=("sans-serif", 18, "bold"))

    def update(self):
        self.root.update()
        if self.aborted:
            raise _Aborted

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def _grid_points(mon, grid, margin=0.08):
    span = 1 - 2 * margin
    pts = []
    for gy in range(grid):
        for gx in range(grid):
            lx = int(mon.width * (margin + span * gx / (grid - 1)))
            ly = int(mon.height * (margin + span * gy / (grid - 1)))
            pts.append((lx, ly))
    return pts


def _load_prior(mons, cameras):
    """Saved samples from earlier sessions, if still compatible."""
    if not (SAMPLES_PATH.exists() and CALIB_PATH.exists()):
        print("no existing calibration to append to — starting fresh")
        return None
    meta = json.loads(CALIB_PATH.read_text())
    if meta.get("monitors") != [vars(m) for m in mons]:
        raise SystemExit("monitor layout changed since the saved calibration — "
                         "run a fresh `calibrate` (without --append)")
    if meta.get("cameras", [0]) != list(cameras):
        raise SystemExit(f"saved calibration used --cameras "
                         f"{','.join(map(str, meta.get('cameras', [0])))} — "
                         "append with the same cameras, or recalibrate fresh")
    data = np.load(SAMPLES_PATH)
    return data["X"], data["Y"], meta.get("sessions", 1)


def _fit_and_save(X, Y, mons, cameras, prior=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    sessions = 1
    if prior is not None:
        X0, Y0, prev_sessions = prior
        print(f"pooling with {len(X0)} samples from {prev_sessions} earlier session(s)")
        X = np.vstack([X0, X])
        Y = np.vstack([Y0, Y])
        sessions = prev_sessions + 1
    if len(X) < 30:
        print(f"only {len(X)} samples collected — too few to fit; not saving")
        return False
    model, rmses = FusionModel.fit(cameras, X, Y)
    if model is None:
        print("no camera collected enough face samples to fit a model; not saving")
        return False
    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SAMPLES_PATH, X=X, Y=Y)
    model.save(CALIB_PATH, extra={
        "rmse_px": rmses, "samples": len(X), "sessions": sessions,
        "cameras": list(cameras), "monitors": [vars(m) for m in mons],
    })
    err = "  ".join(f"{k} ~{v:.0f}px" for k, v in rmses.items())
    print(f"calibration saved to {CALIB_PATH}")
    print(f"  {len(X)} samples over {sessions} session(s); training error: {err}")
    print(f"  (monitor is {mons[0].width}px wide — window-sized zones need <~1/3 of that)")
    return True


def refit():
    """Refit the model from stored samples (after model-code changes)."""
    if not (SAMPLES_PATH.exists() and CALIB_PATH.exists()):
        raise SystemExit("no stored calibration samples to refit from")
    from .x11 import Monitor
    meta = json.loads(CALIB_PATH.read_text())
    mons = [Monitor(**m) for m in meta["monitors"]]
    data = np.load(SAMPLES_PATH)
    prior = (data["X"][:0], data["Y"][:0], meta.get("sessions", 1) - 1)
    return _fit_and_save(data["X"], data["Y"], mons, meta.get("cameras", [0]),
                         prior if meta.get("sessions", 1) > 1 else None)


def run_calibration(cameras=(0,), grid=3, append=False):
    mons = get_monitors()
    prior = _load_prior(mons, cameras) if append else None
    ext = MultiCamera(cameras)
    screen = _TargetScreen(mons)
    X, Y = [], []
    total = len(mons) * grid * grid
    done = 0
    try:
        for mon in mons:
            for lx, ly in _grid_points(mon, grid):
                done += 1
                gx, gy = mon.x + lx, mon.y + ly
                samples = _collect_point(ext, screen, gx, gy, mon, f"{done}/{total}")
                if len(samples) < MIN_SAMPLES:
                    print(f"  warning: only {len(samples)} face samples at "
                          f"{mon.name} ({lx},{ly}) — check lighting/camera angle")
                X += samples
                Y += [(gx, gy)] * len(samples)
    except (_Aborted, KeyboardInterrupt):
        print("aborted")
        return False
    finally:
        screen.close()
        ext.close()

    return _fit_and_save(X, Y, mons, cameras, prior)


def _collect_point(ext, screen, gx, gy, mon, label):
    """Show one target; returns collected sample rows. Raises _Aborted."""
    nan_block = np.full(N_FEATURES, np.nan)
    samples = []
    start = time.time()
    while True:
        t = time.time() - start
        collecting = t > SETTLE_S
        if collecting and ((t > SETTLE_S + MIN_COLLECT_S and len(samples) >= MIN_SAMPLES)
                           or t > SETTLE_S + COLLECT_TIMEOUT_S):
            return samples
        feats_list, _ = ext.read()  # read every loop so no stale frames buffer up
        face_ok = any(f is not None for f in feats_list)
        if collecting and face_ok:
            samples.append(np.concatenate(
                [f if f is not None else nan_block for f in feats_list]))
        screen.draw_target(gx, gy, mon, collecting, face_ok, label, len(samples))
        screen.update()
