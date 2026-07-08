"""Gaze point stabilization: median spike rejection + One Euro filter.

The One Euro filter (Casiez et al. 2012) adapts its cutoff to speed:
near-stationary gaze gets heavy smoothing (no jitter), fast saccades get
light smoothing (no lag). The median-of-5 in front kills single-frame
landmark glitches before they reach the filter.
"""
import math
from collections import deque

import numpy as np


class _OneEuroAxis:
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx_prev = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = self._alpha(cutoff, dt)
        self.x_prev = a * x + (1 - a) * self.x_prev
        return self.x_prev


class GazeFilter:
    """update(point, t) -> stabilized point. Lower min_cutoff = steadier."""

    # defaults tuned on synthetic 300px-jitter data: stationary noise drops
    # ~5.5x while a cross-screen saccade still lands in ~0.23s
    def __init__(self, min_cutoff=0.25, beta=0.0002, d_cutoff=0.3, median=5):
        self._buf = deque(maxlen=median)
        self._axes = (_OneEuroAxis(min_cutoff, beta, d_cutoff),
                      _OneEuroAxis(min_cutoff, beta, d_cutoff))

    def update(self, point, t):
        self._buf.append(np.asarray(point, dtype=float))
        med = np.median(np.stack(self._buf), axis=0)
        return np.array([ax(float(v), t) for ax, v in zip(self._axes, med)])


class FixationFilter:
    """Fixation-aware stabilizer on top of the One Euro filter.

    Gaze is fixations punctuated by ballistic saccades, so this behaves
    like dead reckoning in games: while samples stay near the current
    fixation the output barely moves; when `confirm` consecutive samples
    agree on a distant location, the output snaps there instantly and
    re-stabilizes. Lone outliers (landmark glitches, camera-handoff
    jumps, mid-saccade frames) never move the output at all.
    """

    def __init__(self, snap_dist=350.0, confirm=3,
                 min_cutoff=0.25, beta=0.0002, d_cutoff=0.3):
        from collections import deque as _deque
        self.snap_dist = snap_dist
        self.confirm = confirm
        self._params = (min_cutoff, beta, d_cutoff)
        self._euro = GazeFilter(*self._params, median=3)
        self._pending = _deque(maxlen=confirm)
        self._out = None

    def update(self, point, t):
        p = np.asarray(point, dtype=float)
        if self._out is None:
            self._out = self._euro.update(p, t)
            return self._out
        if np.linalg.norm(p - self._out) < self.snap_dist:
            self._pending.clear()
            self._out = self._euro.update(p, t)
            return self._out
        self._pending.append(p)
        if len(self._pending) == self.confirm:
            pts = np.stack(self._pending)
            med = np.median(pts, axis=0)
            if np.linalg.norm(pts - med, axis=1).max() < self.snap_dist * 0.6:
                self._euro = GazeFilter(*self._params, median=3)  # fresh state: jump, don't glide
                self._out = self._euro.update(med, t)
                self._pending.clear()
        return self._out
