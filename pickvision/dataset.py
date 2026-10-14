"""Render train/val/test shards of tote scenes (multiprocess) to compressed .npz files."""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pickvision.tote import render, sample_scene


def _one(seed):
    g = render(sample_scene(seed))
    return g["rgb"], g["depth"].astype(np.float16), g["semantic"], g["suction"]


def build(seeds, path, workers=2):
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_one, seeds, chunksize=16))
    rgb, depth, sem, suc = (np.stack(x) for x in zip(*out, strict=True))
    np.savez_compressed(path, rgb=rgb, depth=depth, semantic=sem, suction=suc, seeds=np.array(seeds))
    return len(seeds)


SPLITS = {"train": range(0, 1500), "val": range(100_000, 100_250), "test": range(200_000, 200_250)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of each split to render")
    a = ap.parse_args()
    import pathlib
    pathlib.Path(a.out).mkdir(exist_ok=True)
    for name, r in SPLITS.items():
        seeds = list(r)[: max(8, int(len(r) * a.scale))]
        print(name, build(seeds, f"{a.out}/{name}.npz"))
