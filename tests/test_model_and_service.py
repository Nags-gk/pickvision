import io
import pathlib

import numpy as np
import pytest
import torch
from PIL import Image

from pickvision.model import PickNet
from pickvision.tote import render, sample_scene

ONNX = pathlib.Path(__file__).resolve().parents[1] / "model" / "picknet.onnx"


def test_picknet_outputs_probability_maps():
    m = PickNet(pretrained=False).eval()
    with torch.no_grad():
        y = m(torch.rand(2, 4, 64, 96))
    assert y.shape == (2, 4, 64, 96)
    assert torch.allclose(y[:, :3].sum(1), torch.ones(2, 64, 96), atol=1e-5)
    assert float(y[:, 3].min()) >= 0 and float(y[:, 3].max()) <= 1


@pytest.mark.skipif(not ONNX.exists(), reason="exported model not present")
def test_service_end_to_end():
    from fastapi.testclient import TestClient

    from pickvision.service import app

    g = render(sample_scene(400_123))
    rgb = io.BytesIO()
    Image.fromarray(g["rgb"]).save(rgb, format="PNG")
    dep = io.BytesIO()
    Image.fromarray((g["depth"] * 1000).astype(np.uint16)).save(dep, format="PNG")
    c = TestClient(app)
    assert c.get("/healthz").json()["ok"]
    r = c.post("/v1/picks", files={"rgb": ("t.png", rgb.getvalue()), "depth": ("d.png", dep.getvalue())})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "rgbd" and len(body["picks"]) >= 1
    p = body["picks"][0]
    assert 0 <= p["x"] < 256 and 0 <= p["y"] < 256 and abs(np.linalg.norm(p["normal"]) - 1) < 1e-6
    bad = c.post("/v1/picks", files={"rgb": ("x.png", b"not an image")})
    assert bad.status_code == 422 and bad.json()["detail"] == "'rgb' is not a readable PNG or JPEG image"
    # browser path: 8-bit normalised height instead of 16-bit depth gives the same top pick
    from pickvision.preprocess import height_channel
    hb = io.BytesIO()
    Image.fromarray((height_channel(g["depth"]) * 255).round().astype(np.uint8)).save(hb, format="PNG")
    rgb.seek(0)
    r2 = c.post("/v1/picks", files={"rgb": ("t.png", rgb.getvalue()), "height": ("h.png", hb.getvalue())})
    assert r2.status_code == 200 and r2.json()["mode"] == "rgbd"
    p2 = r2.json()["picks"][0]
    assert abs(p2["x"] - p["x"]) <= 3 and abs(p2["y"] - p["y"]) <= 3
    assert c.get("/metrics").json()["requests"] == 3
