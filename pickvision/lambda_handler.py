"""AWS Lambda entry point: the FastAPI service behind API Gateway (HTTP API, payload v2) via Mangum.

The ONNX session is created once per execution environment (module scope via service.session()),
so warm invocations only pay for inference.
"""
from mangum import Mangum

from pickvision.service import app

handler = Mangum(app, lifespan="off")
