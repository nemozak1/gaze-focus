# gaze-focus

Look at a window, start typing there. Webcam-based gaze tracking that
switches X11 window focus to whatever you're looking at.

No eye-tracking hardware needed — it uses MediaPipe FaceMesh (iris position +
head pose) from a normal webcam, a one-time per-seat calibration, and EWMH to
switch focus. Accuracy is coarse (roughly 3–5 cm on screen), which is enough
to tell monitors and large windows apart, not enough for small overlapping
ones.

**X11 only.** On a GNOME Wayland session this approach can't work (the
compositor won't let external processes read window geometry or set focus);
you're currently on GNOME Xorg, which is fine.

## Setup

```sh
cd ~/Projects/Personal/gaze-focus
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python -r requirements.txt
```

## Usage

All commands via the launcher (add `bin/` to PATH if you like):

```sh
bin/gaze-focus camera-test        # 1. camera + face detection sanity check
bin/gaze-focus windows            # 2. check it sees your windows correctly
bin/gaze-focus calibrate          # 3. look at 9 dots per monitor (~40s)
bin/gaze-focus preview            # 4. live overlay: landmarks + gaze on a monitor map
bin/gaze-focus run --dry-run --verbose   # 5. watch what it WOULD do
bin/gaze-focus run                # 6. go live
```

Calibration blacks out all monitors at once and walks a target across them;
sit as you normally do and follow the circle — it samples while the circle
is green and waits until it has enough face samples per target. Click
anywhere (or Ctrl-C in the terminal) to abort. Saved to
`~/.config/gaze-focus/calibration.json`.

`preview` shows the webcam with eye/iris landmarks, the raw feature numbers,
and — once calibrated — the predicted gaze point on a mini-map of your
monitors plus the window it would pick. Use it to judge calibration quality
before going live.
Recalibrate if you move the camera or rearrange monitors.

**Movement tolerance:** one calibration pass captures one sitting posture.
To make the model tolerate you leaning back, slouching etc., run
`calibrate --append` while sitting in those other postures — samples pool
across sessions and the model learns to compensate using its head-position
features. Appending is refused if the monitor layout changed (stale
coordinates would poison the fit); recalibrate fresh in that case.

## Triggers and tuning (`run` flags)

Default trigger is **dwell**: focus follows your gaze automatically once it
rests on a window. Tuned by `--dwell 0.4` (gaze rest time) and `--idle 0.6`
(seconds since your last keypress — this is what stops mid-typing yanks;
focus moves in the pause after you stop typing).

- `--trigger blink` — gesture mode instead: look at a window and blink
  twice quickly (both eyes — winks don't count) to focus it. The gaze
  point freezes while your eyes are closed, so the switch targets what
  you were looking at just before the blink.

## Clicking

Three ways to click at the gaze point, all active by default during `run`:

- **Smirk — shift your mouth left or right** (hold ~150ms, release) — left click
- **Open your mouth** (hold ~150ms, close) — right click
- **Press F8** — left click (`--click-key KEY` to rebind, `none` to disable)

`--click-gesture smirk|pucker|cheeks|jaw` picks the left-click gesture;
test which of your expressions register strongly in `preview` (it shows
all channel scores live). Gestures live below the nose on purpose:
brow-raising was tried and rejected because lifting the eyelids perturbs
the gaze features, and tongue-out because MediaPipe's blendshape model
doesn't actually emit a tongue channel. The click targets the gaze from
~250ms before gesture onset, so even a gesture that wiggles tracking
clicks where you were looking before your face moved. Face gestures use
adaptive baselines with a hold-then-release requirement and a cooldown,
so ordinary expressions don't fire them. `--no-clicks` disables them. Honest
accuracy note: clicks land wherever the gaze estimate is, which is good
to roughly the error margin (~250px) — great for big buttons, videos,
window chrome; not for small links. The pointer warps to the click
point, so you always see where it landed.
- `--cooldown` — minimum gap between switches (default 0.5s blink / 1.2s dwell).
- `--cameras N[,N...]` — V4L2 indices (default `0`).
- `--overlay` — show a translucent heatmap blob at the predicted gaze
  point (eye-tracking-style: hot center fading to a cool transparent
  edge), sized to your calibration's error margin, with smaller fading
  blobs trailing recent motion. Flashes green when a switch fires. True
  per-pixel transparency (ARGB window), click-through, and invisible to
  the hit-test; combine with `--dry-run` to watch without switching.
  Caveat: don't *stare at the blob* — it shows where you look, so chasing
  it drags it around. Look at things; see if it follows.
- `--smooth 0.25` — within-fixation stabilization cutoff (Hz); lower =
  steadier but laggier. The gaze point runs through a fixation-aware
  filter: pinned while you fixate, lone outlier frames discarded, and a
  confirmed cluster of samples at a new location snaps the estimate there
  instantly (dead-reckoning style) instead of gliding.

Monitor changes have seam hysteresis: since you can't physically look at
the bezel, a prediction that only just crosses onto the other monitor
keeps counting on the current one until it lands decisively inside the
new one (half the error margin deep). Cameras capture at 1280x720 MJPG
for iris precision; if double-blinks start getting missed (the loop runs
~15fps at 720p), drop the resolution in `features.py`.

## Multiple cameras

If you have a camera per monitor (e.g. laptop cam + a monitor's built-in
camera), pass all of them to every command: `--cameras 0,4`. Calibration
then trains a joint model over both views plus a per-camera fallback; at
runtime the joint model is used when both cameras see your face, otherwise
whichever one does. The win is coverage: whichever screen you turn to,
some camera has a good view. Blink detection uses the camera that sees the
blink best. The camera list is baked into the calibration — `run`/`preview`
must be given the same `--cameras`, and `--append` refuses a mismatch.

## How it works

1. `features.py` — MediaPipe FaceMesh gives iris centres, eye corners, and
   head pose (solvePnP); ~10 numbers describing where your head+eyes point.
2. `model.py` — ridge regression from those features to a global screen
   coordinate, fitted on the calibration samples.
3. `run.py` — smooths the predicted point (EMA), hit-tests it against the
   EWMH window list (topmost first, current workspace only), and after
   dwell + keyboard-idle + cooldown sends `_NET_ACTIVE_WINDOW`.

## Known limitations

- Windows smaller than ~1/3 of a monitor's width are hit-or-miss; works best
  with 2–4 big windows per screen.
- Glasses with strong reflections, backlight, or a camera far off-axis
  degrade iris tracking — `camera-test` reports the face-detection rate.
- Calibration is per sitting-position; a laptop that moves between desks
  wants a recalibrate (or later: named calibration profiles).
