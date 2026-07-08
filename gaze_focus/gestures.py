"""Face-gesture actions from blendshape scores.

Same adaptive-baseline approach as blink detection: absolute blendshape
levels vary with camera angle and face, so each gesture tracks its own
resting baseline and fires on a deliberate, sustained excursion above it.
Fires on release, so holding a gesture can't repeat-fire.
"""


class GestureDetector:
    def __init__(self, delta_on, delta_off=None, min_hold_s=0.12,
                 max_hold_s=1.2, cooldown_s=0.8, baseline_alpha=0.02):
        self.delta_on = delta_on
        self.delta_off = delta_off if delta_off is not None else delta_on * 0.4
        self.min_hold_s = min_hold_s
        self.max_hold_s = max_hold_s
        self.cooldown_s = cooldown_s
        self.baseline_alpha = baseline_alpha
        self.baseline = None
        self.active = False
        self._active_at = 0.0
        self._fired_at = 0.0

    def update(self, score, now):
        """Feed the blendshape score each frame; returns True when the
        gesture completes (held long enough, then released)."""
        if self.baseline is None:
            self.baseline = score
            return False
        if not self.active:
            if score > self.baseline + self.delta_on:
                self.active = True
                self._active_at = now
            else:
                # learn the resting level only while inactive, so the
                # gesture itself can't drag the baseline up and mask itself
                self.baseline += self.baseline_alpha * (score - self.baseline)
            return False
        if score > self.baseline + self.delta_off:
            return False  # still held
        self.active = False
        held = now - self._active_at
        if (self.min_hold_s <= held <= self.max_hold_s
                and now - self._fired_at >= self.cooldown_s):
            self._fired_at = now
            return True
        return False
