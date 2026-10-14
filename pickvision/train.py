"""Train PickNet on rendered totes; report segmentation and suction metrics; export ONNX."""
import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from pickvision.model import PickNet, height_channel


def to_input(rgb, depth):
    x = np.concatenate([rgb.astype("float32") / 255, height_channel(depth.astype("float32"))[..., None]], -1)
    return torch.from_numpy(x).permute(2, 0, 1)


def load(path):
    d = np.load(path)
    X = torch.stack([to_input(r, dp) for r, dp in zip(d["rgb"], d["depth"], strict=True)])
    return X, torch.from_numpy(d["semantic"].astype("int64")), torch.from_numpy(d["suction"].astype("float32"))


def augment(x, sem, suc, crop):
    n, _, H, W = x.shape
    i, j = np.random.randint(0, H - crop + 1), np.random.randint(0, W - crop + 1)
    x, sem, suc = x[..., i:i + crop, j:j + crop], sem[..., i:i + crop, j:j + crop], suc[..., i:i + crop, j:j + crop]
    k = np.random.randint(4)
    x, sem, suc = (torch.rot90(t, k, (-2, -1)) for t in (x, sem, suc))
    if np.random.rand() < 0.5:
        x, sem, suc = x.flip(-1), sem.flip(-1), suc.flip(-1)
    x = x.clone()
    # photometric jitter
    x[:, :3] = (x[:, :3] * torch.empty(n, 1, 1, 1).uniform_(0.75, 1.25)
                + torch.empty(n, 3, 1, 1).uniform_(-0.06, 0.06)).clamp(0, 1)
    # depth-sensor failure: drop the whole height channel for 30% of samples (RGB-only robustness)
    drop = torch.rand(n) < 0.3
    x[drop, 3] = 0
    return x, sem, suc


def loss_fn(z, sem, suc):
    ce = F.cross_entropy(z[:, :3], sem, weight=torch.tensor([1.0, 1.0, 3.0]))
    bce = F.binary_cross_entropy_with_logits(z[:, 3], suc, pos_weight=torch.tensor(2.0))
    return ce + bce


@torch.no_grad()
def evaluate(model, X, sem, suc, bs=16, rgb_only=False):
    model.eval()
    P = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].clone()
        if rgb_only:
            x[:, 3] = 0
        P.append(model(x))
    P = torch.cat(P)
    pred = P[:, :3].argmax(1)
    ious = {}
    for c, name in ((1, "item"), (2, "boundary")):
        inter = ((pred == c) & (sem == c)).sum().item()
        union = ((pred == c) | (sem == c)).sum().item()
        ious[name] = inter / max(union, 1)
    s = P[:, 3]
    rng = np.random.default_rng(0)
    idx = rng.choice(s.numel(), size=min(400_000, s.numel()), replace=False)
    ap = average_precision_score(suc.flatten()[idx].numpy(), s.flatten()[idx].numpy())
    # top-1 pick precision: is the single most confident pixel actually pickable?
    flat = s.flatten(1).argmax(1)
    top1 = suc.flatten(1).gather(1, flat[:, None]).mean().item()
    return {"iou_item": ious["item"], "iou_boundary": ious["boundary"], "suction_ap": float(ap),
            "top1_pick_precision": top1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    Xtr, Str, Utr = load(f"{a.data}/train.npz")
    Xva, Sva, Uva = load(f"{a.data}/val.npz")
    print("train", tuple(Xtr.shape), "val", tuple(Xva.shape), "suction+ rate", Utr.mean().item())
    model = PickNet()
    model.freeze_stem(4)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    steps = a.epochs * (len(Xtr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.15)
    best, log = -1, []
    out = pathlib.Path(a.out)
    out.mkdir(exist_ok=True)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0
        for i in range(0, len(Xtr) - a.bs + 1, a.bs):
            b = perm[i:i + a.bs]
            x, s, u = augment(Xtr[b], Str[b], Utr[b], a.crop)
            loss = loss_fn(model.logits(x), s, u)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        m = evaluate(model, Xva, Sva, Uva)
        m.update(epoch=ep, train_loss=tot / (len(Xtr) // a.bs), minutes=round((time.time() - t0) / 60, 1))
        log.append(m)
        print(json.dumps(m), flush=True)
        score = m["suction_ap"] + m["iou_item"]
        if score > best:
            best = score
            torch.save(model.state_dict(), out / "picknet.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
