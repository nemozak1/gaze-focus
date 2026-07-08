"""Gaze overlay: translucent heatmap blob at the predicted gaze point.

A full-desktop ARGB Qt window with true per-pixel alpha — the mechanism
screen-annotation tools use. (The previous XShape approach left stale
textures behind under GNOME's compositor and its opacity was ignored.)
The head blob is a soft radial heat gradient (hot center, cool fade-out)
whose radius is the calibration error margin; the trail is smaller,
fainter blobs behind it. Click-through via Qt.WindowTransparentForInput.
"""
import math
import os
import time
from collections import deque

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)  # cv2 poisons this on import
from PyQt5 import QtCore, QtGui, QtWidgets

from .model import saved_error_px
from .x11 import get_monitors

TRAIL_LEN = 8
PEAK_ALPHA = 100   # head-blob center alpha, 0-255: well see-through
FLASH_S = 0.4
GLIDE_TAU = 0.08   # display eases toward the target: ~63% of the gap per tau
TRAIL_MIN_STEP = 30  # px the display must move before a new trail dot drops


def error_radius(default=150.0):
    """Blob radius (px) = the calibration's error margin."""
    return float(min(max(saved_error_px(default), 60.0), 500.0))


class _HeatWindow(QtWidgets.QWidget):
    def __init__(self, geo):
        super().__init__(None, QtCore.Qt.FramelessWindowHint
                         | QtCore.Qt.WindowStaysOnTopHint
                         | QtCore.Qt.X11BypassWindowManagerHint
                         | QtCore.Qt.WindowTransparentForInput)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setGeometry(*geo)
        self.circles = []   # (x, y, radius, strength 0..1), head last
        self.flashing = False

    def paintEvent(self, _event):
        if not self.circles:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        for x, y, r, strength in self.circles:
            a = PEAK_ALPHA * strength
            grad = QtGui.QRadialGradient(QtCore.QPointF(x, y), r)
            if self.flashing:
                grad.setColorAt(0.0, QtGui.QColor(255, 255, 255, int(a)))
                grad.setColorAt(0.4, QtGui.QColor(80, 230, 110, int(a * 0.8)))
                grad.setColorAt(1.0, QtGui.QColor(80, 230, 110, 0))
            else:
                grad.setColorAt(0.0, QtGui.QColor(255, 70, 40, int(a)))
                grad.setColorAt(0.35, QtGui.QColor(255, 190, 0, int(a * 0.7)))
                grad.setColorAt(0.7, QtGui.QColor(70, 200, 90, int(a * 0.4)))
                grad.setColorAt(1.0, QtGui.QColor(70, 200, 90, 0))
            p.setBrush(QtGui.QBrush(grad))
            p.setPen(QtCore.Qt.NoPen)
            p.drawEllipse(QtCore.QPointF(x, y), r, r)
        hx, hy = self.circles[-1][:2]
        p.setBrush(QtGui.QColor(40, 40, 40, 180))
        p.drawEllipse(QtCore.QPointF(hx, hy), 4, 4)


class GazeOverlay:
    def __init__(self, radius=None):
        self.radius = radius if radius is not None else error_radius()
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        mons = get_monitors()
        self.ox = min(m.x for m in mons)
        self.oy = min(m.y for m in mons)
        width = max(m.x + m.width for m in mons) - self.ox
        height = max(m.y + m.height for m in mons) - self.oy
        self.win = _HeatWindow((self.ox, self.oy, width, height))
        self.win.show()
        self.app.processEvents()
        self.trail = deque(maxlen=TRAIL_LEN)
        self._flash_until = 0.0
        self._prev_rect = None
        self._target = None
        self._disp = None
        self._t = None

    def set_target(self, gx, gy):
        """Where the blob should head; tick() glides the display there."""
        self._target = (gx - self.ox, gy - self.oy)
        if self._disp is None:
            self._disp = list(self._target)

    def tick(self):
        """Advance the glide one display frame and repaint. Call often
        (~60Hz) — the logic may snap, but the visible blob sweeps."""
        now = time.time()
        dt = min(now - self._t, 0.05) if self._t else 0.016
        self._t = now
        if self._target is not None and self._disp is not None:
            k = 1.0 - math.exp(-dt / GLIDE_TAU)
            self._disp[0] += (self._target[0] - self._disp[0]) * k
            self._disp[1] += (self._target[1] - self._disp[1]) * k
            if (not self.trail or math.dist(self.trail[-1], self._disp) > TRAIL_MIN_STEP):
                self.trail.append(tuple(self._disp))
            self._render()
        self.app.processEvents()

    def _render(self):
        x, y = self._disp
        n = len(self.trail)
        circles = [(tx, ty,
                    self.radius * (0.25 + 0.3 * (i + 1) / n),
                    0.15 + 0.45 * (i + 1) / n)
                   for i, (tx, ty) in enumerate(self.trail)]
        circles.append((x, y, self.radius, 1.0))
        self.win.circles = circles
        self.win.flashing = time.time() < self._flash_until

        # repaint only where the comet was + where it is now
        pad = 6
        x0 = min(c[0] - c[2] for c in circles) - pad
        y0 = min(c[1] - c[2] for c in circles) - pad
        x1 = max(c[0] + c[2] for c in circles) + pad
        y1 = max(c[1] + c[2] for c in circles) + pad
        rect = QtCore.QRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        self.win.update(rect.united(self._prev_rect) if self._prev_rect else rect)
        self._prev_rect = rect

    def flash(self):
        self._flash_until = time.time() + FLASH_S

    def close(self):
        self.win.close()
        self.app.processEvents()
