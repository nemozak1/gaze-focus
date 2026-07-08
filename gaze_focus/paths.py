import os
import pathlib

CONFIG_DIR = pathlib.Path(
    os.environ.get("XDG_CONFIG_HOME", pathlib.Path.home() / ".config")) / "gaze-focus"
CALIB_PATH = CONFIG_DIR / "calibration.json"
SAMPLES_PATH = CONFIG_DIR / "samples.npz"  # raw training samples, for --append
