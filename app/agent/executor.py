import json
import os

import httpx

from app.agent.tools import top_level


DEFAULT_BASE_URL = "http://127.0.0.1:4010"


def get_base_url():
    return os.environ.get("TOOL_BASE_URL", DEFAULT_BASE_URL)


def build_path(record, arguments):
    path = record["path"]
    for param in record["path_params"]:
        path = path.replace("{" + param["name"] + "}", str(arguments.get(param["name"], "")))
    return path


def build_query(record, arguments):
    names = {param["name"] for param in record["query_params"]}
    return {name: value for name, value in arguments.items() if name in names}


def build_body(record, arguments):
    names = {param["name"] for param in top_level(record["body_params"])}
    path_names = {param["name"] for param in record["path_params"]}
    body = {name: value for name, value in arguments.items() if name in names and name not in path_names}
    return body or None


def execute_tool(record, arguments, base_url=None):
    url = (base_url or get_base_url()).rstrip("/") + build_path(record, arguments)
    response = httpx.request(
        record["method"],
        url,
        params=build_query(record, arguments) or None,
        json=build_body(record, arguments),
        timeout=30,
    )
    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = response.text
    return {"status": response.status_code, "body": payload}
