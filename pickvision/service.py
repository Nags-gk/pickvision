"""Perception microservice: RGB(-D) image in, ranked suction picks out.

    uvicorn pickvision.service:app --port 8080
    curl -F rgb=@tote.png -F depth=@tote_depth.png localhost:8080/v1/picks

Runs the exported ONNX model with ONNX Runtime (no PyTorch at serving time). Depth is an optional
16-bit PNG in millimetres; without it the model runs in RGB-only mode.
"""
from __future__ import annotations

import io
import os
import time
from collections import Counter

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from pickvision.grasp import instances_from_semantic, rank_picks
from pickvision.preprocess import CAM_HEIGHT, HEIGHT_RANGE, height_channel

MODEL_PATH = os.environ.get("PICKNET_ONNX", os.path.join(os.path.dirname(__file__), "..", "model", "picknet.onnx"))
app = FastAPI(title="PickVision", version="0.1.0")
_session = None
STATS = Counter()
LAT_MS: list[float] = []


def session():
    global _session
    if _session is None:
        import onnxruntime as ort
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def infer(rgb: np.ndarray, depth: np.ndarray | None, k: int = 5):
    H, W, _ = rgb.shape
    if H % 8 or W % 8:
        raise ValueError("image sides must be multiples of 8")
    hgt = height_channel(depth) if depth is not None else np.zeros((H, W), np.float32)
    x = np.concatenate([rgb.astype(np.float32) / 255, hgt[..., None]], -1).transpose(2, 0, 1)[None]
    maps = session().run(None, {"rgbd": x.astype(np.float32)})[0][0]
    inst = instances_from_semantic(maps[:3])
    d = depth if depth is not None else np.full((H, W), 1.0, np.float32)
    return maps, inst, rank_picks(maps[3], inst, d, k=k)


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": os.path.basename(MODEL_PATH)}


@app.get("/metrics")
def metrics():
    lat = np.array(LAT_MS[-1000:]) if LAT_MS else np.array([0.0])
    return {"requests": STATS["requests"], "errors": STATS["errors"],
            "latency_ms_p50": float(np.percentile(lat, 50)), "latency_ms_p95": float(np.percentile(lat, 95))}


def _read_image(data: bytes, field: str, mode: str | None = None) -> np.ndarray:
    try:
        im = Image.open(io.BytesIO(data))
        return np.asarray(im.convert(mode) if mode else im)
    except (OSError, ValueError) as e:  # PIL.UnidentifiedImageError is an OSError
        raise ValueError(f"'{field}' is not a readable PNG or JPEG image") from e


@app.post("/v1/picks")
async def picks(rgb: UploadFile = File(...), depth: UploadFile | None = File(None),
                height: UploadFile | None = File(None), k: int = 5):
    """rgb: colour image. Optional geometry, either `depth` (16-bit PNG, millimetres) or `height`
    (8-bit PNG, 0-255 = 0-0.3 m above the tote floor, 0 = unknown; what a browser can produce)."""
    t0 = time.perf_counter()
    STATS["requests"] += 1
    try:
        img = _read_image(await rgb.read(), "rgb", "RGB")
        dep = None
        if depth is not None:
            dep = _read_image(await depth.read(), "depth").astype(np.float32) / 1000.0
        elif height is not None:
            h = _read_image(await height.read(), "height", "L").astype(np.float32) / 255.0
            dep = np.where(h > 0, CAM_HEIGHT - h * HEIGHT_RANGE, 0.0).astype(np.float32)
        if dep is not None and dep.shape != img.shape[:2]:
            raise ValueError("geometry image and rgb sizes differ")
        _, inst, ranked = infer(img, dep, k=max(1, min(k, 20)))
    except ValueError as e:
        STATS["errors"] += 1
        raise HTTPException(status_code=422, detail=str(e)) from e
    ms = (time.perf_counter() - t0) * 1000
    LAT_MS.append(ms)
    return {"picks": [p.as_dict() for p in ranked], "items_detected": int(inst.max() + 1),
            "mode": "rgbd" if dep is not None else "rgb", "latency_ms": round(ms, 1)}
