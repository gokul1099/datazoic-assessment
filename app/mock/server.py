import json
import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app.ingest.collection import iter_requests, load_collection
from app.ingest.metadata import SUCCESS_PREFIX, get_path_template


COLLECTION_PATH = "data/paypal_collection.postman_collection.json"
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]


def path_pattern(template):
    parts = []
    for segment in template.strip("/").split("/"):
        if re.fullmatch(r"\{[^}]+\}", segment):
            parts.append("[^/]+")
        else:
            parts.append(re.escape(segment))
    return re.compile("^/" + "/".join(parts) + "$")


def get_success_example(item):
    for response in item.get("response", []):
        if str(response.get("code", "")).startswith(SUCCESS_PREFIX):
            return int(response["code"]), response.get("body") or None
    return 200, None


def build_routes(collection_path):
    routes = []
    for folders, item in iter_requests(load_collection(collection_path)["item"]):
        request = item.get("request", {})
        method = request.get("method", "GET")
        template = get_path_template(request.get("url", {}).get("path", []))
        status, body = get_success_example(item)
        routes.append((method, path_pattern(template), template.count("{"), status, body))
    routes.sort(key=lambda route: route[2])
    return routes


ROUTES = build_routes(COLLECTION_PATH)
app = FastAPI(title="paypal-mock")


@app.api_route("/{full_path:path}", methods=METHODS)
async def mock_operation(full_path: str, request: Request):
    path = "/" + full_path
    for method, pattern, _, status, body in ROUTES:
        if method == request.method and pattern.match(path):
            if body is None:
                return Response(status_code=status)
            return JSONResponse(status_code=status, content=json.loads(body))
    return JSONResponse(
        status_code=404,
        content={"name": "RESOURCE_NOT_FOUND", "message": "No mock example for this operation"},
    )
