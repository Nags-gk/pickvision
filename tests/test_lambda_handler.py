"""The Lambda entry point answers an API Gateway HTTP API (payload v2) event like the deployed stack sends."""
import json

from pickvision.lambda_handler import handler


def test_healthz_through_lambda_adapter():
    event = {
        "version": "2.0", "routeKey": "GET /healthz", "rawPath": "/healthz", "rawQueryString": "",
        "headers": {"host": "example.execute-api.us-east-2.amazonaws.com"},
        "requestContext": {"http": {"method": "GET", "path": "/healthz", "sourceIp": "1.2.3.4", "protocol": "HTTP/1.1"},
                           "stage": "$default", "requestId": "r1", "accountId": "123", "apiId": "a1"},
        "isBase64Encoded": False,
    }
    out = handler(event, None)
    assert out["statusCode"] == 200
    assert json.loads(out["body"])["ok"] is True
