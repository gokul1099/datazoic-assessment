import json
import re


WRITE_METHODS = {"POST", "PATCH", "PUT", "DELETE"}
REQUIRED_MARKER = re.compile(r"\(required\)", re.IGNORECASE)
SUCCESS_PREFIX = "2"


def get_text(value):
    if isinstance(value, dict):
        return value.get("content", "") or ""
    return value or ""


def get_action_type(method):
    if method == "GET":
        return "read"
    if method in WRITE_METHODS:
        return "write"
    return "unknown"


def get_path_template(segments):
    parts = []
    for segment in segments:
        if segment.startswith(":"):
            parts.append("{" + segment[1:] + "}")
        else:
            parts.append(segment)
    return "/" + "/".join(parts)


def get_path_params(segments, variables):
    descriptions = {var.get("key"): get_text(var.get("description")) for var in variables}
    params = []
    for segment in segments:
        if segment.startswith(":"):
            name = segment[1:]
            params.append({
                "name": name,
                "type": "string",
                "required": True,
                "description": descriptions.get(name, ""),
            })
    return params


def get_scalar_type(value):
    text = str(value).strip()
    if text.lower() in ("true", "false"):
        return "boolean"
    if re.fullmatch(r"-?\d+", text):
        return "integer"
    if re.fullmatch(r"-?\d+\.\d+", text):
        return "number"
    return "string"


def get_json_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def flatten_schema(value, prefix=""):
    fields = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else key
            fields.append({"name": name, "type": get_json_type(child)})
            fields.extend(flatten_schema(child, name))
    elif isinstance(value, list) and value:
        name = f"{prefix}[]"
        fields.append({"name": name, "type": get_json_type(value[0])})
        fields.extend(flatten_schema(value[0], name))
    return fields


def get_query_params(query):
    params = []
    for param in query:
        description = get_text(param.get("description"))
        params.append({
            "name": param.get("key", ""),
            "type": get_scalar_type(param.get("value", "")),
            "required": bool(REQUIRED_MARKER.search(description)),
            "description": description,
        })
    return params


def get_body_params(body):
    if not body:
        return []
    mode = body.get("mode")
    if mode == "raw":
        try:
            return flatten_schema(json.loads(body.get("raw", "")))
        except json.JSONDecodeError:
            return []
    if mode in ("formdata", "urlencoded"):
        return [
            {"name": field.get("key", ""), "type": "string", "required": None}
            for field in body.get(mode, [])
        ]
    return []


def get_responses(responses):
    status_codes = sorted({response.get("code") for response in responses if response.get("code")})
    success = []
    for response in responses:
        code = str(response.get("code", ""))
        if not code.startswith(SUCCESS_PREFIX) or not response.get("body"):
            continue
        try:
            payload = json.loads(response["body"])
        except json.JSONDecodeError:
            continue
        success.append({"status": response["code"], "fields": flatten_schema(payload)})
    return status_codes, success


def extract_metadata(folders, item):
    request = item.get("request", {})
    url = request.get("url", {})
    method = request.get("method", "GET")
    segments = url.get("path", [])
    folder = "/".join(folders)
    status_codes, responses = get_responses(item.get("response", []))
    return {
        "id": f"{method} {folder}/{item['name']}",
        "tool_name": item["name"],
        "domain": folders[0] if folders else "",
        "folder": folder,
        "method": method,
        "action_type": get_action_type(method),
        "path": get_path_template(segments),
        "description": get_text(request.get("description")) or get_text(item.get("description")),
        "path_params": get_path_params(segments, url.get("variable", [])),
        "query_params": get_query_params(url.get("query", [])),
        "body_params": get_body_params(request.get("body")),
        "status_codes": status_codes,
        "responses": responses,
    }
