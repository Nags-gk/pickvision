"""Held-out test-set perception metrics for the trained model (RGB-D and RGB-only)."""
import json
import pathlib

import torch

from pickvision.model import PickNet
from pickvision.train import evaluate, load


def main(weights="artifacts/picknet.pt", data="data/test.npz", out="results/seg_metrics.json"):
    m = PickNet(pretrained=False)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    X, S, U = load(data)
    rep = {"test_totes": len(X), "rgbd": evaluate(m, X, S, U), "rgb_only": evaluate(m, X, S, U, rgb_only=True),
           "params": sum(p.numel() for p in m.parameters())}
    pathlib.Path(out).parent.mkdir(exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
