import numpy as np

from pickvision.grasp import instances_from_semantic, rank_picks, surface_normal


def test_flat_plane_normal_faces_camera():
    d = np.full((32, 32), 0.9, np.float32)
    n = surface_normal(d, 16, 16)
    assert np.allclose(n, (0, 0, -1), atol=1e-6)


def test_tilted_plane_normal():
    xs = np.arange(32, dtype=np.float32)
    d = np.tile(0.9 + 0.001 * xs, (32, 1))
    n = surface_normal(d, 16, 16)
    assert n[0] > 0.05 and abs(n[1]) < 1e-3 and n[2] < 0


def test_two_blobs_two_instances():
    p = np.zeros((3, 40, 40), np.float32)
    p[0] = 1
    for x0 in (5, 25):
        p[:, 10:30, x0:x0 + 10] = 0
        p[1, 10:30, x0:x0 + 10] = 1
    inst = instances_from_semantic(p, min_px=10)
    assert inst.max() == 1


def test_rank_picks_respects_nms():
    s = np.zeros((64, 64), np.float32)
    s[20, 20], s[22, 22], s[50, 50] = 1.0, 0.95, 0.9
    picks = rank_picks(s, None, np.full((64, 64), 0.9, np.float32), k=3, nms_px=8, blur=1)
    pts = [(p.x, p.y) for p in picks]
    assert (20, 20) in pts and (50, 50) in pts and (22, 22) not in pts
