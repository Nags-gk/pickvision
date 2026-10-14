"""Procedural RGB-D tote scenes with exact ground truth.

A top-down camera looks into a tote holding a random pile of items. Items are stacked in
z-order: each one rests on the highest surface under its footprint. The renderer produces

* ``rgb``       uint8  HxWx3   shaded colour image (domain-randomised)
* ``depth``     float32 HxW   metres from the camera (0 = sensor dropout)
* ``instance``  int16  HxW    visible item id, -1 = tote
* ``semantic``  uint8  HxW    0 tote, 1 item interior, 2 item boundary
* ``suction``   uint8  HxW    1 where a suction cup would seal *and* lift the item
* ``items``     per-item metadata (kind, pinned/occluded, visible area)

The suction label is the physics oracle used both for training and for scoring pick attempts.
"""
from __future__ import annotations

import dataclasses as dc
import math

import numpy as np

KINDS = ("box", "envelope", "bottle", "can_lying", "pouch", "mesh_bag", "ball")
# which surfaces can a vacuum cup seal on at all
SEALABLE = {"box": True, "envelope": True, "bottle": True, "can_lying": True, "pouch": True,
            "mesh_bag": False, "ball": True}
CAM_HEIGHT = 1.0      # m, camera above tote floor
CUP_RADIUS_PX = 4     # suction cup footprint radius in pixels
# surface tilt a compliant bellows cup tolerates, per item kind (curved surfaces need a softer cup)
MAX_TILT_DEG = {"box": 14.0, "envelope": 14.0, "bottle": 14.0, "can_lying": 22.0, "pouch": 12.0,
                "mesh_bag": 0.0, "ball": 30.0}


@dc.dataclass
class Item:
    kind: str
    cx: float
    cy: float
    angle: float
    w: float          # px
    h: float          # px
    height: float     # m
    color: tuple
    accent: tuple
    seed: int


@dc.dataclass
class Scene:
    items: list
    size: int
    seed: int
    light: tuple       # unit vector toward light (x, y, z)
    tote_color: tuple
    gain: float
    noise: float
    blur: bool


def _hsv(rng, s_lo=0.25, s_hi=0.9, v_lo=0.35, v_hi=0.95):
    h, s, v = rng.random(), rng.uniform(s_lo, s_hi), rng.uniform(v_lo, v_hi)
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def sample_scene(seed: int, size: int = 256, n_items: int | None = None) -> Scene:
    rng = np.random.default_rng(seed)
    n = int(n_items if n_items is not None else rng.integers(3, 13))
    margin = size * 0.12
    items = []
    for _ in range(n):
        kind = KINDS[rng.choice(len(KINDS), p=[0.3, 0.12, 0.12, 0.1, 0.16, 0.1, 0.1])]
        s = size / 256
        if kind == "box":
            w, h, ht = rng.uniform(28, 70) * s, rng.uniform(22, 55) * s, rng.uniform(0.03, 0.12)
        elif kind == "envelope":
            w, h, ht = rng.uniform(45, 80) * s, rng.uniform(30, 50) * s, rng.uniform(0.004, 0.012)
        elif kind == "bottle":
            w = h = rng.uniform(18, 30) * s
            ht = rng.uniform(0.12, 0.22)
        elif kind == "can_lying":
            w, h, ht = rng.uniform(45, 75) * s, rng.uniform(18, 28) * s, 0.0
            ht = h / size * 0.45  # radius ~ half the width, in metres
        elif kind == "pouch":
            w, h, ht = rng.uniform(35, 70) * s, rng.uniform(28, 55) * s, rng.uniform(0.015, 0.04)
        elif kind == "mesh_bag":
            w, h, ht = rng.uniform(40, 70) * s, rng.uniform(32, 55) * s, rng.uniform(0.03, 0.06)
        else:  # ball
            w = h = rng.uniform(24, 40) * s
            ht = w / size * 0.45
        items.append(Item(kind, rng.uniform(margin, size - margin), rng.uniform(margin, size - margin),
                          rng.uniform(0, math.pi), w, h, ht, _hsv(rng), _hsv(rng, 0.0, 0.4, 0.75, 1.0),
                          int(rng.integers(1 << 31))))
    elev = rng.uniform(math.radians(35), math.radians(80))
    az = rng.uniform(0, 2 * math.pi)
    light = (math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev))
    tote = _hsv(rng, 0.05, 0.35, 0.3, 0.6)
    return Scene(items, size, seed, light, tote, rng.uniform(0.8, 1.2), rng.uniform(2, 7), bool(rng.random() < 0.3))


def _local(item: Item, xx, yy):
    c, s = math.cos(item.angle), math.sin(item.angle)
    dx, dy = xx - item.cx, yy - item.cy
    return (dx * c + dy * s) / (item.w / 2), (-dx * s + dy * c) / (item.h / 2)


def _profile(item: Item, xx, yy, rng):
    """Footprint mask and surface height (m) above the item's resting base."""
    u, v = _local(item, xx, yy)
    k = item.kind
    if k in ("box", "envelope"):
        m = (np.abs(u) <= 1) & (np.abs(v) <= 1)
        bevel = np.clip((1 - np.maximum(np.abs(u), np.abs(v))) * 12, 0, 1)
        z = item.height * (0.92 + 0.08 * bevel)
    elif k in ("bottle", "ball"):
        r2 = u * u + v * v
        m = r2 <= 1
        if k == "bottle":  # cap on top: flat disc in the middle, rounded shoulder outside
            z = item.height * np.where(r2 < 0.45, 1.0, 1.0 - 0.6 * (np.sqrt(r2) - 0.67) / 0.33)
        else:
            z = item.height * np.sqrt(np.clip(1 - r2, 0, 1))
    elif k == "can_lying":
        m = (np.abs(u) <= 1) & (np.abs(v) <= 1)
        z = item.height * np.sqrt(np.clip(1 - v * v, 0, 1))
    else:  # pouch / mesh bag: lumpy blob
        ang = np.arctan2(v, u)
        prng = np.random.default_rng(item.seed)
        ph = prng.uniform(0, 2 * np.pi, 3)
        rad = 1 + 0.12 * np.sin(3 * ang + ph[0]) + 0.08 * np.sin(5 * ang + ph[1])
        r = np.sqrt(u * u + v * v) / rad
        m = r <= 1
        wr = 0.12 * np.sin(u * 9 + ph[2]) * np.sin(v * 7) if k == "pouch" else 0.35 * np.sin(u * 14) * np.sin(v * 14)
        z = item.height * np.clip((1 - r ** 2) * (1 + wr), 0, None)
    return m, np.where(m, z, 0.0)


def _normals(height_px):
    gy, gx = np.gradient(height_px)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _texture(item: Item, xx, yy, m):
    u, v = _local(item, xx, yy)
    base = np.array(item.color)[None, None, :] * np.ones(xx.shape + (3,))
    acc = np.array(item.accent)
    prng = np.random.default_rng(item.seed + 1)
    if item.kind == "box":
        # cardboard grain + a printed shipping label with "text" lines
        grain = 0.06 * np.sin(u * 40 + prng.uniform(0, 6)) * np.sin(v * 3)
        base = base * (1 + grain[..., None])
        lx, ly = prng.uniform(-0.5, 0.2), prng.uniform(-0.5, 0.2)
        lab = (u > lx) & (u < lx + 0.55) & (v > ly) & (v < ly + 0.5)
        base[lab] = acc
        lines = lab & (np.sin((v - ly) * 60) > 0.6) & (u < lx + 0.45)
        base[lines] = acc * 0.25
    elif item.kind == "envelope":
        flap = np.abs(v + 0.2 - 0.6 * np.abs(u)) < 0.04
        base[flap] *= 0.75
    elif item.kind == "bottle":
        r = np.sqrt(u * u + v * v)
        base[r < 0.55] = acc
    elif item.kind == "can_lying":
        band = np.abs(v) < 0.45
        base[band] = acc
    elif item.kind == "mesh_bag":
        grid = (np.sin(u * 25) > 0.75) | (np.sin(v * 25) > 0.75)
        base[grid] = base[grid] * 0.35
    elif item.kind == "ball":
        seam = np.abs(np.sin(u * 3.1 + v * 1.7)) < 0.08
        base[seam] = acc
    return np.clip(base, 0, 1)


def render(scene: Scene, removed: frozenset = frozenset()):
    S = scene.size
    rng = np.random.default_rng(scene.seed + 7)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    height = np.zeros((S, S), np.float32)
    inst = np.full((S, S), -1, np.int16)
    albedo = np.zeros((S, S, 3), np.float32)
    glossy = np.zeros((S, S), bool)
    foot = {}
    above = {}
    order = [i for i in range(len(scene.items)) if i not in removed]
    for i in order:
        it = scene.items[i]
        m, z = _profile(it, xx, yy, rng)
        if not m.any():
            continue
        base = float(height[m].max())
        for j in np.unique(inst[m]):
            if j >= 0:
                above.setdefault(int(j), set()).add(i)
        height = np.where(m, base + z, height)
        inst[m] = i
        tex = _texture(it, xx, yy, m)
        albedo[m] = tex[m]
        if it.kind in ("pouch", "bottle", "ball"):
            glossy |= m
        foot[i] = int(m.sum())

    # tote floor and walls
    wall = 10 * S // 256
    edge = np.minimum.reduce([xx, yy, S - 1 - xx, S - 1 - yy])
    tote_rgb = np.array(scene.tote_color)
    floor = inst < 0
    albedo[floor] = tote_rgb * (0.9 + 0.1 * rng.random((int(floor.sum()), 1)))
    wall_m = (edge < wall) & floor
    height = np.where(wall_m, 0.25, height)
    albedo[wall_m] = tote_rgb * 0.7

    # shading: lambert + specular on glossy items, cast shadows by ray-marching the height map
    hpx = height / CAM_HEIGHT * S * 1.2
    n = _normals(hpx)
    L = np.array(scene.light, np.float32)
    lam = np.clip((n * L).sum(-1), 0, 1)
    shadow = np.zeros((S, S), bool)
    lx, ly, lz = L
    tan = lz / max(1e-3, math.hypot(lx, ly))
    for k in range(2, 40, 2):
        sx, sy = int(round(lx / math.hypot(lx, ly) * k)), int(round(ly / math.hypot(lx, ly) * k))
        shifted = np.roll(np.roll(hpx, -sy, 0), -sx, 1)
        shadow |= shifted > hpx + k * tan
    shade = 0.38 + 0.62 * lam * np.where(shadow, 0.45, 1.0)
    H = (L + np.array([0, 0, 1])) / np.linalg.norm(L + np.array([0, 0, 1]))
    spec = np.where(glossy, np.clip((n * H).sum(-1), 0, 1) ** 40 * 0.8, 0)
    rgb = albedo * shade[..., None] * scene.gain + spec[..., None]
    rgb += rng.normal(0, scene.noise / 255, rgb.shape)
    if scene.blur:
        rgb = (rgb + np.roll(rgb, 1, 0) + np.roll(rgb, 1, 1) + np.roll(rgb, -1, 0) + np.roll(rgb, -1, 1)) / 5
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    # depth with sensor noise and specular dropout
    depth = (CAM_HEIGHT - height + rng.normal(0, 0.002, height.shape)).astype(np.float32)
    drop = glossy & (spec > 0.25) | (rng.random(height.shape) < 0.003)
    depth[drop] = 0.0

    # ground truth
    semantic = np.zeros((S, S), np.uint8)
    item_px = inst >= 0
    semantic[item_px] = 1
    diff = np.zeros((S, S), bool)
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        diff |= np.roll(inst, sh, ax) != inst
    boundary = diff & item_px
    semantic[boundary] = 2

    tilt = np.degrees(np.arccos(np.clip(n[..., 2], -1, 1)))
    items_meta = []
    suction = np.zeros((S, S), np.uint8)
    from scipy import ndimage  # local import keeps the module light for the API service

    for i in order:
        it = scene.items[i]
        vis = inst == i
        va = int(vis.sum())
        pinned = bool(above.get(i))
        meta = dict(id=i, kind=it.kind, visible_px=va, footprint_px=foot.get(i, 0), pinned=pinned)
        items_meta.append(meta)
        if va == 0 or pinned or not SEALABLE[it.kind]:
            continue
        inner = ndimage.binary_erosion(vis, iterations=CUP_RADIUS_PX)
        flat = tilt < MAX_TILT_DEG[it.kind]
        # the whole cup disc must be flat enough: erode the flat mask by (almost) the cup radius
        flat_disc = ndimage.binary_erosion(flat, iterations=CUP_RADIUS_PX - 1)
        ok = inner & flat_disc & vis
        suction[ok] = 1
    return dict(rgb=rgb, depth=depth, instance=inst, semantic=semantic, suction=suction, items=items_meta)


def suction_success(gt: dict, x: int, y: int) -> bool:
    """Oracle: does a pick at pixel (x, y) seal on a free item and lift it?"""
    S = gt["suction"].shape[0]
    if not (0 <= x < S and 0 <= y < S):
        return False
    return bool(gt["suction"][y, x])


def item_at(gt: dict, x: int, y: int) -> int:
    return int(gt["instance"][y, x])
