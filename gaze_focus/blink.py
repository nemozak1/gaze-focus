"""Double-blink gesture detection from FaceLandmarker blink blendshapes.

Thresholds are adaptive: blendshape amplitudes vary a lot with camera
angle and face-crop quality (measured on this setup: open-eye baseline
~0.05-0.09, real blinks peaking anywhere from 0.23 to 0.6), so closure is
detected as an excursion above a slowly-adapting open-eye baseline rather
than a fixed level. The score fed in is min(eyeBlinkLeft, eyeBlinkRight)
per camera, max across cameras — both eyes must close, best view wins.
"""

DELTA_ON = 0.16        # rise this far above baseline -> eyes counted closed
DELTA_OFF = 0.06       # back within this of baseline -> blink complete
BASELINE_ALPHA = 0.02  # open-eye baseline EMA rate (~1s at 50Hz)
MIN_CLOSED_S = 0.04    # shorter spikes are single-frame noise, not blinks
MAX_CLOSED_S = 0.5     # longer closures (squints, eye rubs) aren't blinks
DOUBLE_WINDOW_S = 0.9  # second blink must complete within this of the first


class BlinkDetector:
    def __init__(self):
        self.closed = False
        self.baseline = None
        self._closed_at = 0.0
        self._last_blink_at = None

    def update(self, score, now):
        """Feed the blink score each frame; returns 'blink', 'double', or None."""
        if self.baseline is None:
            self.baseline = score
            return None
        if not self.closed:
            if score > self.baseline + DELTA_ON:
                self.closed = True
                self._closed_at = now
            else:
                # only learn the baseline from open eyes, so a blink can't
                # drag it upward and mask itself
                self.baseline += BASELINE_ALPHA * (score - self.baseline)
            return None
        if score > self.baseline + DELTA_OFF:
            return None  # still closed
        self.closed = False
        held = now - self._closed_at
        if held < MIN_CLOSED_S:
            return None
        if held > MAX_CLOSED_S:
            self._last_blink_at = None  # long closure breaks the chain
            return None
        if (self._last_blink_at is not None
                and now - self._last_blink_at < DOUBLE_WINDOW_S):
            self._last_blink_at = None  # consume, so a triple doesn't fire twice
            return "double"
        self._last_blink_at = now
        return "blink"
