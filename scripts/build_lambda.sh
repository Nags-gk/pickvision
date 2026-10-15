#!/usr/bin/env bash
# Builds the AWS Lambda package (Python 3.12, arm64/Graviton) into build/lambda.
# Wheels are fetched for the Lambda platform, so this works from any host OS. No PyTorch: ONNX Runtime only.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/lambda
rm -rf "$OUT" && mkdir -p "$OUT"
pip install --quiet --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 \
  --implementation cp --python-version 3.12 --only-binary=:all: --target "$OUT" \
  onnxruntime numpy "opencv-python-headless<4.12" pillow fastapi mangum python-multipart
# Trim what Lambda never imports (keeps the unzipped size well under the 250 MB limit).
find "$OUT" -depth -type d \( -name __pycache__ -o -name tests -o -name "*.dist-info" \) -exec rm -rf {} +
mkdir -p "$OUT/pickvision" "$OUT/model"
cp pickvision/{__init__,service,grasp,tote,preprocess,lambda_handler}.py "$OUT/pickvision/"
cp model/picknet.onnx "$OUT/model/"
du -sh "$OUT"
