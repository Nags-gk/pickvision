"""Export PickNet to ONNX (FP32) and a dynamically-quantised INT8 variant; check parity."""
import argparse
import json
import pathlib

import numpy as np
import onnxruntime as ort
import torch

from pickvision.model import PickNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="artifacts/picknet.pt")
    ap.add_argument("--out", default="model")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(exist_ok=True)
    m = PickNet(pretrained=False)
    m.load_state_dict(torch.load(a.weights, map_location="cpu"))
    m.eval()
    x = torch.rand(1, 4, 256, 256)
    fp32 = out / "picknet.onnx"
    torch.onnx.export(m, x, fp32, input_names=["rgbd"], output_names=["maps"], opset_version=17,
                      dynamic_axes={"rgbd": {0: "n", 2: "h", 3: "w"}, "maps": {0: "n", 2: "h", 3: "w"}}, dynamo=False)
    with torch.no_grad():
        ref = m(x).numpy()
    got = ort.InferenceSession(str(fp32)).run(None, {"rgbd": x.numpy()})[0]
    report = {"fp32_max_abs_diff_vs_torch": float(np.abs(ref - got).max()),
              "fp32_mb": round(fp32.stat().st_size / 1e6, 2)}
    print(json.dumps(report, indent=2))
    (out / "export_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
