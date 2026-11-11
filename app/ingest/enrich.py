import json
from concurrent.futures import ThreadPoolExecutor

from groq import Groq


DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_ATTEMPTS = 3
MAX_WORKERS = 4
MAX_BODY_FIELDS = 60
MAX_CATALOG_FIELDS = 15
ENRICHMENT_TOOL_NAME = "record_enrichment"

ENRICHMENT_TOOL = {
    "type": "function",
    "function": {
        "name": ENRICHMENT_TOOL_NAME,
        "description": "Record the enrichment for one PayPal API operation.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Plain English explanation of what this request does, for an agent choosing tools.",
                },
                "example_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 5,
                    "description": "User queries that should lead to this operation.",
                },
                "requires": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "produced_by": {
                                "type": ["string", "null"],
                                "description": "Exact id of the catalog operation that produces this field, or null if the caller must supply it.",
                            },
                        },
                        "required": ["field", "produced_by"],
                    },
                },
                "produces": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["field", "description"],
                    },
                },
                "moves_money": {"type": "boolean"},
                "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                "risk_reason": {"type": "string"},
            },
            "required": [
                "description",
                "example_queries",
                "requires",
                "produces",
                "moves_money",
                "risk_level",
                "risk_reason",
            ],
        },
    },
}


def get_top_level_fields(tool):
    if not tool["responses"]:
        return []
    return [
        field["name"]
        for field in tool["responses"][0]["fields"]
        if "." not in field["name"] and "[" not in field["name"]
    ]


def build_catalog(tools):
    lines = []
    for tool in tools:
        fields = get_top_level_fields(tool)[:MAX_CATALOG_FIELDS]
        lines.append(f"{tool['id']} | {tool['method']} {tool['path']} | response fields: {', '.join(fields) or 'none'}")
    return "\n".join(lines)


def summarize_tool(tool):
    return {
        "id": tool["id"],
        "tool_name": tool["tool_name"],
        "method": tool["method"],
        "path": tool["path"],
        "action_type": tool["action_type"],
        "spec_description": tool["description"],
        "path_params": [param["name"] for param in tool["path_params"]],
        "query_params": [
            {"name": param["name"], "type": param["type"], "required": param["required"]}
            for param in tool["query_params"]
        ],
        "body_params": [
            {"name": param["name"], "type": param["type"]}
            for param in tool["body_params"][:MAX_BODY_FIELDS]
        ],
        "response_fields": get_top_level_fields(tool),
    }


def build_messages(tool, catalog):
    system = (
        "You enrich PayPal REST API operations for an agent tool catalog. "
        "The spec already gives method, path, params and response fields, so do not restate them. "
        "Fill only the enrichment fields by calling record_enrichment. "
        "For requires, list each input the operation needs that another operation in the catalog produces, "
        "and set produced_by to that operation's exact id from the catalog. Use null when the caller must supply it. "
        "For produces, list only identifiers or fields that another operation would need as an input, not every response field. "
        "Set moves_money to true only if the operation moves, sends, refunds, captures, pays out or cancels real funds. "
        "Full operation catalog:\n" + catalog
    )
    user = json.dumps(summarize_tool(tool), indent=2)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_model(client, model, messages):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        tools=[ENRICHMENT_TOOL],
        tool_choice={"type": "function", "function": {"name": ENRICHMENT_TOOL_NAME}},
    )
    call = response.choices[0].message.tool_calls[0]
    return json.loads(call.function.arguments)


def validate(result, valid_ids):
    examples = result.get("example_queries", [])
    if not 2 <= len(examples) <= 5:
        raise ValueError("example_queries must contain 2 to 5 items")
    for requirement in result.get("requires", []):
        if requirement.get("produced_by") not in valid_ids:
            requirement["produced_by"] = None
    return result


def enrich_tool(client, model, tool, catalog, valid_ids):
    messages = build_messages(tool, catalog)
    last_error = None
    for _ in range(MAX_ATTEMPTS):
        try:
            return validate(call_model(client, model, messages), valid_ids)
        except Exception as error:
            last_error = error
    raise RuntimeError(f"Failed to enrich {tool['id']}: {last_error}")


def enrich_tools(tools, model=DEFAULT_MODEL, limit=None):
    client = Groq()
    catalog = build_catalog(tools)
    valid_ids = {tool["id"] for tool in tools}
    targets = tools[:limit] if limit else tools
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(lambda tool: enrich_tool(client, model, tool, catalog, valid_ids), targets))
    for tool, enrichment in zip(targets, results):
        tool["enrichment"] = enrichment
    return targets
