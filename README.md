# PickVision

**RGB-D perception and suction-grasp planning for robotic picking from cluttered totes.**
A MobileNetV3 LR-ASPP network, fine-tuned in PyTorch, segments items and scores every pixel for
whether a vacuum cup would seal *and* lift the item free. OpenCV turns those maps into ranked 3-D
pick poses. The same ONNX model is served by a FastAPI microservice and runs in the browser with
ONNX Runtime Web. It is deployed on AWS Lambda behind API Gateway.

**Live demo: https://nags-gk.github.io/pickvision/** (switch "Run on" to *AWS Lambda API* to use the deployed service)

[![CI / CD](https://github.com/Nags-gk/pickvision/actions/workflows/ci.yml/badge.svg)](https://github.com/Nags-gk/pickvision/actions/workflows/ci.yml)

## Results

**Perception** on 250 held-out simulated totes (`results/seg_metrics.json`):

| Metric | RGB-D | RGB only (depth camera failed) |
|---|---:|---:|
| Top-1 pick precision: is the most confident pixel actually pickable? | **96.8%** | 94.4% |
| Suction-affordance average precision | 94.0% | 93.3% |
| Item mask IoU | 88.9% | 88.6% |

**Closed loop: clear the tote** (`results/clear_tote.json`). This covers 60 unseen totes and 460 items.
The policy sees only the camera. On a successful pick the item is removed and the scene is
re-rendered. Each tote allows two attempts per item, and failed points are not retried. Scoring stops
once nothing left can be lifted.

| Policy | Pick success | First-pick success | Items cleared |
|---|---:|---:|---:|
| *Ceiling: perfect ground-truth knowledge* | *100%* | *100%* | *71%* |
| Random point on an item | 16% | 33% | 31% |
| Highest-flat-surface heuristic (OpenCV, no learning) | 24% | 67% | 45% |
| Gradient boosting on hand-built features (scikit-learn) | 39% | 87% | 58% |
| PickNet, RGB only | 39% | 100% | 59% |
| **PickNet, RGB-D** | **42%** | **100%** | **59%** |
| **PickNet, RGB-D, confidence-gated (stops below 0.6)** | **70%** | **100%** | **56%** |

What the numbers say:
* **First picks:** the learned model takes the right first pick in every tote. The classic geometric heuristic manages two in three.
* **Clearing:** PickNet clears 84% of what is clearable (59% of 71%), against 63% for the heuristic.
* **Later picks are the hard part.** The easy items go first; what remains are pouches with only small sealable patches, and items partly buried under others. Gating on confidence trades 3 points of clearing for **1.7× the success rate (42% to 70%) with 43% fewer attempts**. In a real cell, each failed attempt costs cycle time, so the gated policy is the deployable one.
* **Depth helps but is not critical.** With the depth camera off, top-1 precision drops only from 96.8% to 94.4%, because depth was dropped from 30% of training images on purpose.

## How it works

```mermaid
flowchart LR
  G[Procedural RGB-D tote renderer<br/>domain randomisation] -->|1,500 train / 250 val / 250 test| T[PyTorch fine-tuning<br/>MobileNetV3 LR-ASPP, 4-ch input]
  T -->|ONNX opset 17| O[picknet.onnx]
  O --> W[ONNX Runtime Web<br/>browser demo]
  O --> F[FastAPI service<br/>POST /v1/picks<br/>AWS Lambda arm64 + API Gateway]
  O --> E[Closed-loop benchmark]
  C[RGB-D frame] --> O --> M[4 maps: tote, item, boundary, suction]
  M -->|OpenCV: components, NMS,<br/>plane-fit normal| P[ranked 3-D pick poses]
```

**Simulator** (`pickvision/tote.py`)
* Stacks boxes, envelopes, bottles, lying cans, pouches, mesh bags and balls in z-order; each item rests on the highest surface beneath it.
* Randomises lighting direction, cast shadows, specular glare, colour and texture, sensor noise, blur, and depth dropout on shiny surfaces.
* A physics oracle labels a pixel pickable only if all of these hold:
  * the cup's whole footprint lies on one item, clear of its edges;
  * that patch's tilt is within the tolerance for that material;
  * the material can seal at all (mesh bags never do);
  * nothing rests on top of the item.

**Model** (`pickvision/model.py`, `train.py`)
* COCO-pretrained LR-ASPP MobileNetV3-Large (3.2M parameters). The first conv is widened to accept a depth/height channel, initialised from the RGB filters.
* The heads output tote, item and boundary classes plus a suction score.
* Training uses 192 px random crops, rotations and flips, photometric jitter, and depth-channel dropout on 30% of samples.
* The loss is class-weighted cross-entropy plus BCE.
* 12 epochs on CPU take about 31 minutes.

**Grasp planning** (`pickvision/grasp.py`)
* Items are connected components of the item-interior map.
* Pick candidates come from Gaussian-smoothed suction peaks with non-maximum suppression.
* The approach direction is a least-squares plane fit to the depth patch under the cup, back-projected through the camera intrinsics.

**Baselines** (`pickvision/baselines.py`, `grasp.py`)
* An OpenCV "highest flat surface, far from edges" heuristic using depth gradients and a distance transform.
* A scikit-learn HistGradientBoosting per-pixel classifier on 10 hand-built geometric and colour features.

**Serving** (`pickvision/service.py`)
* A FastAPI service with no PyTorch at runtime, only ONNX Runtime.
* `POST /v1/picks` takes an RGB PNG plus optional geometry and returns ranked picks with 3-D points and normals. Geometry is either a 16-bit depth PNG in millimetres, or the 8-bit height image a browser can produce.
* Unreadable uploads get a 422 with a plain message, not a stack trace.
* `/healthz` and `/metrics` report request count, errors, and p50/p95 latency.

## Deployed on AWS

`infra/` is an AWS CDK (TypeScript) stack in us-east-2:

* **Lambda:** Python 3.12 on arm64 (Graviton), 2 GB. The FastAPI app runs through the Mangum adapter, with ONNX Runtime only and no PyTorch, in a 220 MB package.
* **API Gateway HTTP API:** routes `POST /v1/picks`, `GET /healthz` and `GET /metrics`. CORS is restricted to the demo page, and stage throttling is 10 rps with a burst of 20.
* **Logs and alerting:** CloudWatch logs are kept for 2 weeks. A function-error alarm notifies an SNS email topic.

Measured against the live endpoint:
* About 50 ms of model and grasp-planning time per 256×256 image.
* 0.25–0.4 s warm round trip from a browser, and about 3 s on a cold start.
* On held-out tote 400000, the top 5 picks returned by the API all seal and lift in the simulator.

```bash
./scripts/build_lambda.sh                              # fetch arm64 wheels, add code + model into build/lambda
cd infra && npm ci && npm test                         # CDK assertion tests
npx cdk deploy PickVision -c alertEmail=you@example.com
curl -F rgb=@web/samples/400000_rgb.png -F height=@web/samples/400000_height.png <ApiUrl>/v1/picks
npx cdk destroy PickVision                             # tear down
```

## Run it

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
python -m pytest -q                                   # 11 tests, including end-to-end API and Lambda-adapter tests
python -m pickvision.dataset --out data               # render 2,000 totes (about 5 minutes on 2 cores)
python -m pickvision.train --data data --out artifacts
python -m pickvision.export && python -m pickvision.report
python -m pickvision.evaluate --totes 60              # closed-loop benchmark (about 15 minutes)
uvicorn pickvision.service:app --port 8080
```

## Limitations

* Everything is simulated: top-down camera, procedural items, an analytic suction model. Real totes add transparent and deformable items, reflections and calibration error, and need real data or sim-to-real adaptation.
* Instance separation relies on a thin boundary class, which the network finds hard (boundary IoU 26%). Touching items of similar colour can merge. The suction map does not depend on it.
* Suction only, with no two-finger grasps, and no arm motion planning or collision checking.
* The browser demo's "try your own photo" runs in RGB-only mode and has no ground truth to check against.
* The API ranks peaks across the whole tote, so its top 5 can include several points on one item. The browser demo lists one best point per item.
* The API authenticates nobody; throttling and CORS are its only guards. A production cell would sit on a private network or behind IAM auth.

## License

MIT
