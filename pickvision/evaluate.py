"""Closed-loop "clear the tote" benchmark.

Each policy sees only RGB-D. It proposes a suction point; the physics oracle decides success; on
success the item is removed and the scene re-rendered. Failed points are remembered for the rest of
the tote (a standard retry rule shared by all policies). Episode budget: 2 attempts per item. The
episode ends early once nothing in the tote is pickable at all, so success rates measure the policy,
not unpickable leftovers.
"""
import argparse
import json
import pathlib
import pickle
import time

import cv2
import numpy as np
import torch

from pickvision import baselines
from pickvision.grasp import classical_item_mask, heuristic_highest_flat
from pickvision.model import PickNet
from pickvision.tote import item_at, render, sample_scene, suction_success
from pickvision.train import to_input


def make_policies(model, clf, rng):
    @torch.no_grad()
    def picknet(obs, rgb_only=False):
        x = to_input(obs["rgb"], obs["depth"])[None]
        if rgb_only:
            x[:, 3] = 0
        return model(x)[0, 3].numpy()

    def random_on_mask(obs):
        m = classical_item_mask(obs["depth"])
        return np.where(m, rng.random(m.shape), 0).astype(np.float32)

    return {
        "oracle (ground-truth suction map)": lambda o: o["suction"].astype(np.float32),
        "random point on item": random_on_mask,
        "highest-flat heuristic (OpenCV)": lambda o: heuristic_highest_flat(o["depth"]),
        "gradient boosting (scikit-learn)": lambda o: baselines.predict_map(clf, o["rgb"], o["depth"]),
        "PickNet RGB only": lambda o: picknet(o, rgb_only=True),
        "PickNet RGB-D": picknet,
        # hand the tote to a person instead of trying low-confidence points
        "PickNet RGB-D, confidence-gated": lambda o: np.where((s := picknet(o)) >= 0.6, s, 0).astype(np.float32),
    }


def episode(seed, policy):
    scene = sample_scene(seed)
    removed, failed = set(), []
    n = len(scene.items)
    attempts = successes = 0
    first = None
    while attempts < 2 * n:
        obs = render(scene, frozenset(removed))
        if not obs["suction"].any():
            # nothing left that any suction cup could lift (empty, or only mesh bags and items pinned
            # under them): the tote goes to a human. Scoring stops here for every policy alike.
            break
        score = cv2.GaussianBlur(policy(obs).astype(np.float32), (5, 5), 0)
        for (fx, fy) in failed:
            cv2.circle(score, (fx, fy), 10, 0, -1)
        y, x = np.unravel_index(int(score.argmax()), score.shape)
        if score[y, x] <= 1e-3:
            break  # policy sees nothing worth trying: hand the tote to a human
        attempts += 1
        ok = suction_success(obs, x, y)
        if first is None:
            first = ok
        if ok:
            successes += 1
            removed.add(item_at(obs, x, y))
        else:
            failed.append((int(x), int(y)))
    return dict(items=n, attempts=attempts, picked=successes, first_ok=bool(first))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="artifacts/picknet.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--totes", type=int, default=60)
    ap.add_argument("--out", default="results/clear_tote.json")
    ap.add_argument("--skip-sklearn", action="store_true", help="CI smoke test: no training data needed")
    ap.add_argument("--only", default="", help="comma-separated policy names to run (default: all)")
    a = ap.parse_args()
    model = PickNet(pretrained=False)
    if pathlib.Path(a.weights).exists():
        model.load_state_dict(torch.load(a.weights, map_location="cpu"))
    model.eval()
    clf_path = pathlib.Path("artifacts/hgb.pkl")
    if a.skip_sklearn:
        clf = None
    elif clf_path.exists():
        clf = pickle.loads(clf_path.read_bytes())
    else:
        d = np.load(f"{a.data}/train.npz")
        t0 = time.time()
        clf = baselines.fit(d["rgb"][:300], d["depth"][:300], d["suction"][:300])
        clf_path.write_bytes(pickle.dumps(clf))
        print(f"fit sklearn baseline in {time.time() - t0:.0f}s")
    rng = np.random.default_rng(0)
    pols = make_policies(model, clf, rng)
    if clf is None:
        pols.pop("gradient boosting (scikit-learn)")
    if a.only:
        keep = [k.strip() for k in a.only.split(",")]
        pols = {k: v for k, v in pols.items() if any(k.startswith(x) for x in keep)}
    seeds = list(range(300_000, 300_000 + a.totes))
    report = {}
    for name, pol in pols.items():
        t0 = time.time()
        eps = [episode(s, pol) for s in seeds]
        att = sum(e["attempts"] for e in eps)
        pk = sum(e["picked"] for e in eps)
        items = sum(e["items"] for e in eps)
        report[name] = dict(
            pick_success_rate=pk / max(att, 1), first_pick_success=float(np.mean([e["first_ok"] for e in eps])),
            items_cleared=pk / items, attempts=att, picked=pk, items=items, totes=len(eps),
            seconds=round(time.time() - t0, 1))
        print(name, json.dumps(report[name]), flush=True)
    out = pathlib.Path(a.out)
    out.parent.mkdir(exist_ok=True)
    if a.only and out.exists():  # merge partial runs into the existing report
        report = {**json.loads(out.read_text()), **report}
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
