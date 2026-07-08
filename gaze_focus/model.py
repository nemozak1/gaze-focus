"""Ridge regression mapping gaze features to global screen coordinates."""
import json
import pathlib

import numpy as np

from .paths import CALIB_PATH


def saved_error_px(default=250.0):
    """Best available error estimate (px) from the saved calibration."""
    try:
        meta = json.loads(pathlib.Path(CALIB_PATH).read_text())
    except (FileNotFoundError, ValueError):
        return default
    rmse = meta.get("rmse_px")
    if isinstance(rmse, dict) and rmse:
        return float(rmse.get("joint") or min(rmse.values()))
    if isinstance(rmse, (int, float)):
        return float(rmse)
    return default


class GazeModel:
    def __init__(self, mean, std, weights):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)  # (d+1, 2)

    @classmethod
    def fit(cls, X, Y, ridge=None, penalty=None):
        """Ridge regression with optional per-feature penalty scaling.

        `penalty` multiplies the ridge strength per column. Needed because
        in a single-sitting calibration the head naturally turns toward
        each target, so posture features look informative in-session and
        the fit gives them huge weights — which then blow up when posture
        actually changes. Penalizing those columns harder keeps their
        coefficients near physical scale. ridge=None cross-validates the
        base strength.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        Xn = (X - mean) / std
        Xb = np.hstack([Xn, np.ones((len(Xn), 1))])
        d = Xb.shape[1]
        pen = np.ones(d) if penalty is None else np.append(np.asarray(penalty, float), 0.0)
        if ridge is None:
            ridge = cls._cv_ridge(Xb, Y, pen)
        w = np.linalg.solve(Xb.T @ Xb + ridge * np.diag(pen), Xb.T @ Y)
        model = cls(mean, std, w)
        model.ridge = ridge
        rmse = float(np.sqrt(((model.predict_batch(X) - Y) ** 2).sum(axis=1).mean()))
        return model, rmse

    @staticmethod
    def _cv_ridge(Xb, Y, pen, folds=5):
        n, d = Xb.shape
        idx = np.random.default_rng(0).permutation(n)
        parts = np.array_split(idx, folds)
        best, best_err = 1.0, np.inf
        for lam in np.logspace(0, 5, 11):
            err = 0.0
            for k in range(folds):
                te = parts[k]
                tr = np.concatenate([parts[j] for j in range(folds) if j != k])
                w = np.linalg.solve(Xb[tr].T @ Xb[tr] + lam * np.diag(pen),
                                    Xb[tr].T @ Y[tr])
                err += ((Xb[te] @ w - Y[te]) ** 2).sum()
            if err < best_err:
                best_err, best = err, float(lam)
        return best

    def predict_batch(self, X):
        Xn = (np.asarray(X, dtype=np.float64) - self.mean) / self.std
        Xb = np.hstack([Xn, np.ones((len(Xn), 1))])
        return Xb @ self.weights

    def predict(self, feats):
        return self.predict_batch(feats.reshape(1, -1))[0]

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "weights": self.weights.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(data["mean"], data["std"], data["weights"])


# Per-feature ridge multipliers for one camera block; layout matches
# features.py: [iris x4, head rotation x3, head translation x3].
# Translation is penalized hard: swept on real data, 1000 cut posture
# sensitivity ~50x (3116 -> 62 px per cm of lean) for ~19% held-out cost.
BLOCK_PENALTY = [1.0] * 4 + [1.0] * 3 + [1000.0] * 3


class FusionModel:
    """Per-camera gaze models plus a joint model over all cameras.

    Predicts with the joint model when every camera sees a face (most
    accurate), otherwise falls back to the first camera that does.
    """

    def __init__(self, cameras, joint, per_camera):
        self.cameras = list(cameras)
        self.joint = joint            # GazeModel over concatenated features
        self.per_camera = per_camera  # list of GazeModel or None

    @classmethod
    def fit(cls, cameras, X, Y):
        """X rows are per-camera feature blocks concatenated; a camera that
        didn't see a face contributes a NaN block. Returns (model, rmses);
        model is None if nothing had enough samples."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n_cams = len(cameras)
        nf = X.shape[1] // n_cams
        joint, rmses = None, {}
        if n_cams > 1:
            full = ~np.isnan(X).any(axis=1)
            if full.sum() >= 30:
                joint, r = GazeModel.fit(X[full], Y[full],
                                         penalty=BLOCK_PENALTY * n_cams)
                rmses["joint"] = r
        per = []
        for i, cam in enumerate(cameras):
            block = X[:, i * nf:(i + 1) * nf]
            ok = ~np.isnan(block).any(axis=1)
            if ok.sum() >= 30:
                m, r = GazeModel.fit(block[ok], Y[ok], penalty=BLOCK_PENALTY)
                rmses[f"cam{cam}"] = r
            else:
                m = None
            per.append(m)
        if joint is None and all(m is None for m in per):
            return None, rmses
        return cls(cameras, joint, per), rmses

    def predict(self, feats_list):
        """feats_list: per-camera feature vector or None. Returns point or None."""
        if self.joint is not None and all(f is not None for f in feats_list):
            return self.joint.predict(np.concatenate(feats_list))
        for feats, m in zip(feats_list, self.per_camera):
            if feats is not None and m is not None:
                return m.predict(feats)
        return None

    def save(self, path, extra=None):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"cameras": self.cameras,
                "joint": self.joint.to_dict() if self.joint else None,
                "per_camera": [m.to_dict() if m else None for m in self.per_camera]}
        data.update(extra or {})
        path.write_text(json.dumps(data, indent=1))

    @classmethod
    def load(cls, path):
        data = json.loads(pathlib.Path(path).read_text())
        if "weights" in data:  # pre-fusion single-camera format
            m = GazeModel.from_dict(data)
            return cls([0], None, [m])
        return cls(data["cameras"],
                   GazeModel.from_dict(data["joint"]) if data["joint"] else None,
                   [GazeModel.from_dict(d) if d else None for d in data["per_camera"]])
