"""scikit-learn per-pixel suction classifier on hand-crafted geometric + colour features."""
import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pickvision.grasp import classical_item_mask
from pickvision.model import height_channel


def pixel_features(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    h = height_channel(depth)
    mean = cv2.blur(h, (7, 7))
    std = np.sqrt(np.clip(cv2.blur(h * h, (7, 7)) - mean ** 2, 0, None))
    gx, gy = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
    m = classical_item_mask(depth).astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255
    rel = h - cv2.dilate(h, np.ones((15, 15), np.uint8))  # 0 at the locally highest surface
    feats = [h, mean, std, np.hypot(gx, gy), dt, rel, (depth <= 0).astype(np.float32)] + [hsv[..., i] for i in range(3)]
    return np.stack(feats, -1).reshape(-1, len(feats))


def fit(rgbs, depths, sucs, px_per_image=1500, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for rgb, d, s in zip(rgbs, depths, sucs, strict=True):
        f = pixel_features(rgb, d.astype(np.float32))
        lab = s.reshape(-1)
        pos, neg = np.flatnonzero(lab), np.flatnonzero(lab == 0)
        n_pos = min(len(pos), px_per_image // 3)
        pick_pos = rng.choice(pos, n_pos, replace=False) if n_pos else np.array([], int)
        take = np.concatenate([pick_pos, rng.choice(neg, px_per_image - n_pos, replace=False)]).astype(int)
        X.append(f[take])
        y.append(lab[take])
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, random_state=seed)
    clf.fit(np.concatenate(X), np.concatenate(y))
    return clf


def predict_map(clf, rgb, depth) -> np.ndarray:
    H, W = depth.shape
    return clf.predict_proba(pixel_features(rgb, depth))[:, 1].reshape(H, W).astype(np.float32)
