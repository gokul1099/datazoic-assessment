import json


def load_collection(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_requests(items, folders=()):
    for item in items:
        if "item" in item:
            yield from iter_requests(item["item"], folders + (item["name"],))
        else:
            yield folders, item
