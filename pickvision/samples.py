"""Write held-out demo totes for the web page: RGB PNG, 16-bit depth PNG (mm), ground-truth suction PNG."""
import json
import pathlib
import sys

import numpy as np
from PIL import Image

from pickvision.model import height_channel
from pickvision.tote import render, sample_scene


def main(out="web/samples", n=16, start=400_000):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for k in range(n):
        seed = start + k
        g = render(sample_scene(seed))
        Image.fromarray(g["rgb"]).save(out / f"{seed}_rgb.png")
        Image.fromarray((g["depth"] * 1000).round().astype(np.uint16)).save(out / f"{seed}_depth.png")
        Image.fromarray((g["suction"] * 255).astype(np.uint8)).save(out / f"{seed}_suction.png")
        # 8-bit normalised height for the browser (canvas cannot decode 16-bit PNG losslessly)
        Image.fromarray((height_channel(g["depth"]) * 255).round().astype(np.uint8)).save(out / f"{seed}_height.png")
        free = [m for m in g["items"] if m["visible_px"] and not m["pinned"]]
        index.append(dict(seed=seed, items=len(g["items"]), free=len(free),
                          kinds=sorted({m["kind"] for m in g["items"]})))
    (out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:2])
