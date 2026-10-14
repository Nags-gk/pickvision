"""Turn perception maps into suction pick poses, plus non-learned baselines.

A pick is (x, y) in pixels, the 3-D point under it and the approach direction (surface normal from
the depth image). Candidates are local maxima of the suction-affordance map, separated by
non-maximum suppression and grouped by item instance.
"""
from __future__ import annotations

import dataclasses as dc

import cv2
import numpy as np

from pickvision.tote import CAM_HEIGHT

FOCAL_PX = 300.0  # pinhole intrinsics of the simulated top-down camera (256 px image)


@dc.dataclass
class Pick:
    x: int
    y: int
    score: float
    instance: int
    point_m: tuple   # (X, Y, Z) in the camera frame, metres
    normal: tuple    # unit approach direction (surface normal, camera frame)

    def as_dict(self):
        return dc.asdict(self)


def instances_from_semantic(sem_prob: np.ndarray, min_px: int = 40) -> np.ndarray:
    """Interior pixels (item, not boundary) -> connected components; boundary pixels join the nearest."""
    interior = (sem_prob.argmax(0) == 1).astype(np.uint8)
    n, lab = cv2.connectedComponents(interior, connectivity=4)
    out = np.full(lab.shape, -1, np.int32)
    k = 0
    for i in range(1, n):
        m = lab == i
        if m.sum() >= min_px:
            out[m] = k
            k += 1
    item = sem_prob[1:].sum(0) > 0.5
    if k:
        # grow labels into boundary pixels
        dist, idx = cv2.distanceTransformWithLabels((out < 0).astype(np.uint8), cv2.DIST_L2, 3,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(out >= 0)
        lut = np.full(idx.max() + 1, -1, np.int32)
        lut[idx[ys, xs]] = out[ys, xs]
        grown = lut[idx]
        out = np.where(item & (out < 0) & (dist < 4), grown, out)
    return out


def surface_normal(depth: np.ndarray, x: int, y: int, r: int = 4):
    """Least-squares plane fit to the depth patch around (x, y) -> unit normal pointing at the camera."""
    H, W = depth.shape
    ys, xs = np.mgrid[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)]
    z = depth[ys, xs]
    ok = z > 0
    if ok.sum() < 6:
        return (0.0, 0.0, -1.0)
    X = (xs[ok] - W / 2) * z[ok] / FOCAL_PX
    Y = (ys[ok] - H / 2) * z[ok] / FOCAL_PX
    A = np.c_[X, Y, np.ones_like(X)]
    (a, b, _), *_ = np.linalg.lstsq(A, z[ok], rcond=None)
    n = np.array([a, b, -1.0])
    n /= np.linalg.norm(n)
    return tuple(float(v) for v in n)


def to_pick(x, y, score, inst_map, depth) -> Pick:
    H, W = depth.shape
    if depth[y, x] > 0:
        z = float(depth[y, x])
    else:  # sensor dropout under the cup: fall back to the scene median
        z = float(np.median(depth[depth > 0])) if (depth > 0).any() else CAM_HEIGHT
    P = ((x - W / 2) * z / FOCAL_PX, (y - H / 2) * z / FOCAL_PX, z)
    return Pick(int(x), int(y), float(score), int(inst_map[y, x]) if inst_map is not None else -1, P,
                surface_normal(depth, x, y))


def rank_picks(suction: np.ndarray, inst_map, depth, k: int = 5, nms_px: int = 12, blur: int = 5):
    """Top-k suction candidates with non-maximum suppression."""
    s = cv2.GaussianBlur(suction.astype(np.float32), (blur, blur), 0)
    picks = []
    work = s.copy()
    for _ in range(k):
        y, x = np.unravel_index(int(work.argmax()), work.shape)
        if work[y, x] <= 0:
            break
        picks.append(to_pick(x, y, s[y, x], inst_map, depth))
        cv2.circle(work, (int(x), int(y)), nms_px, 0, -1)
    return picks


# ---------------------------------------------------------------- non-learned baselines
def classical_item_mask(depth: np.ndarray, floor_tol: float = 0.006) -> np.ndarray:
    """Anything measurably above the tote floor (largest-mode depth) and inside the walls."""
    valid = depth > 0
    floor = np.median(depth[valid]) if valid.any() else CAM_HEIGHT
    hist, edges = np.histogram(depth[valid], bins=200)
    floor = edges[hist.argmax()] + (edges[1] - edges[0]) / 2
    m = valid & (depth < floor - floor_tol)
    H, W = depth.shape
    wall = 12 * H // 256
    m[:wall], m[-wall:], m[:, :wall], m[:, -wall:] = False, False, False, False
    return m


def heuristic_highest_flat(depth: np.ndarray) -> np.ndarray:
    """Classic bin-picking heuristic: prefer the highest surface, far from its edges, where the depth
    patch is flat. Returns a score map (no learning)."""
    m = classical_item_mask(depth)
    d = depth.copy()
    d[d <= 0] = np.median(depth[depth > 0])
    gx, gy = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    edges = (grad > 0.004).astype(np.uint8)
    interior = (m & (edges == 0)).astype(np.uint8)
    dt = cv2.distanceTransform(interior, cv2.DIST_L2, 3)
    height = np.clip(CAM_HEIGHT - d, 0, None)
    score = np.where(interior > 0, height / max(height.max(), 1e-6) + 0.05 * np.minimum(dt, 10), 0)
    return score.astype(np.float32)
