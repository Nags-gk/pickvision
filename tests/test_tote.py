import numpy as np

from pickvision.tote import SEALABLE, render, sample_scene, suction_success


def test_render_is_deterministic():
    a, b = render(sample_scene(5)), render(sample_scene(5))
    assert np.array_equal(a["rgb"], b["rgb"]) and np.array_equal(a["suction"], b["suction"])


def test_shapes_and_ranges():
    g = render(sample_scene(1))
    assert g["rgb"].shape == (256, 256, 3) and g["rgb"].dtype == np.uint8
    assert g["depth"].shape == (256, 256)
    assert set(np.unique(g["semantic"])) <= {0, 1, 2}
    assert ((g["depth"] == 0) | ((g["depth"] > 0.6) & (g["depth"] < 1.02))).all()


def test_suction_only_on_free_sealable_items():
    for s in range(20):
        g = render(sample_scene(s))
        for m in g["items"]:
            on_item = g["suction"][g["instance"] == m["id"]]
            if m["pinned"] or not SEALABLE[m["kind"]]:
                assert on_item.sum() == 0, m
        assert g["suction"][g["instance"] < 0].sum() == 0


def test_oracle_and_removal():
    sc = sample_scene(11)
    g = render(sc)
    ys, xs = np.nonzero(g["suction"])
    assert suction_success(g, int(xs[0]), int(ys[0]))
    assert not suction_success(g, 2, 2)  # tote wall
    i = int(g["instance"][ys[0], xs[0]])
    g2 = render(sc, frozenset({i}))
    assert (g2["instance"] == i).sum() == 0
