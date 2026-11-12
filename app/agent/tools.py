import re


PAYMENT_DOMAINS = {
    "Invoices",
    "Orders",
    "Payments",
    "Payouts",
    "Disputes",
    "Subscriptions",
    "Payment Method Tokens",
}
JSON_TYPES = {"string", "integer", "number", "boolean", "object", "array"}


def get_function_name(tool_name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)[:64]


def get_json_type(value_type):
    return value_type if value_type in JSON_TYPES else "string"


def get_description(record):
    enrichment = record.get("enrichment") or {}
    return enrichment.get("description") or record["description"] or record["tool_name"]


def top_level(params):
    return [param for param in params if "." not in param["name"] and "[" not in param["name"]]


def build_tool_schema(record):
    properties = {}
    required = []
    for param in record["path_params"]:
        properties[param["name"]] = {"type": "string", "description": param["description"]}
        required.append(param["name"])
    for param in record["query_params"]:
        properties[param["name"]] = {"type": get_json_type(param["type"]), "description": param["description"]}
        if param["required"]:
            required.append(param["name"])
    for param in top_level(record["body_params"]):
        properties.setdefault(param["name"], {"type": get_json_type(param["type"])})
    return {
        "type": "function",
        "function": {
            "name": get_function_name(record["tool_name"]),
            "description": get_description(record),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def needs_approval(record):
    if record["action_type"] != "write":
        return False
    enrichment = record.get("enrichment") or {}
    return bool(enrichment.get("moves_money")) or record["domain"] in PAYMENT_DOMAINS
