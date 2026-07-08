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

Default trigger is **double-blink**: look at a window and blink twice
quickly (both eyes — winks don't count) to focus it. Nothing happens on
normal blinking: a "double" needs the second blink to complete within 0.9s
of the first, and closures longer than 0.5s (squints, eye rubs) reset the
gesture. The gaze point freezes while your eyes are closed, so the switch
targets what you were looking at just before the blink. Practice in
`preview` — it shows the live blink score and flashes when a double-blink
registers.

- `--trigger dwell` — old behavior: auto-focus after your gaze rests on a
  window, no gesture needed. Tuned by `--dwell 0.4` (gaze rest time) and
  `--idle 0.6` (seconds since last keypress; stops mid-typing yanks).
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
