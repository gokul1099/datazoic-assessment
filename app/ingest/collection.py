import json


def load_collection(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_description(value):
    if isinstance(value, dict):
        return value.get("content", "")
    return value or ""


def get_url(url):
    if isinstance(url, dict):
        return url.get("raw", "")
    return url or ""


def flatten_requests(items, folders=None):
    folders = folders or []
    tools = []
    for item in items:
        if "item" in item:
            tools.extend(flatten_requests(item["item"], folders + [item["name"]]))
            continue
        request = item.get("request", {})
        method = request.get("method", "GET")
        folder = "/".join(folders)
        tools.append({
            "id": f"{method} {folder}/{item['name']}",
            "name": item["name"],
            "folder": folder,
            "method": method,
            "url": get_url(request.get("url")),
            "description": get_description(request.get("description")) or get_description(item.get("description")),
        })
    return tools
