import os

import psycopg
from psycopg.types.json import Jsonb


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tools (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    folder TEXT NOT NULL,
    method TEXT NOT NULL,
    action_type TEXT NOT NULL,
    path TEXT NOT NULL,
    description TEXT NOT NULL,
    path_params JSONB NOT NULL,
    query_params JSONB NOT NULL,
    body_params JSONB NOT NULL,
    status_codes INTEGER[] NOT NULL,
    responses JSONB NOT NULL,
    enrich_description TEXT,
    example_queries JSONB,
    requires JSONB,
    produces JSONB,
    moves_money BOOLEAN,
    risk_level TEXT,
    risk_reason TEXT,
    record JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS tools_domain_idx ON tools (domain)",
    "CREATE INDEX IF NOT EXISTS tools_action_type_idx ON tools (action_type)",
    "CREATE INDEX IF NOT EXISTS tools_method_idx ON tools (method)",
    "CREATE INDEX IF NOT EXISTS tools_moves_money_idx ON tools (moves_money)",
]

COLUMNS = [
    "id",
    "tool_name",
    "domain",
    "folder",
    "method",
    "action_type",
    "path",
    "description",
    "path_params",
    "query_params",
    "body_params",
    "status_codes",
    "responses",
    "enrich_description",
    "example_queries",
    "requires",
    "produces",
    "moves_money",
    "risk_level",
    "risk_reason",
    "record",
]

UPSERT_SQL = (
    f"INSERT INTO tools ({', '.join(COLUMNS)}) "
    f"VALUES ({', '.join('%(' + column + ')s' for column in COLUMNS)}) "
    f"ON CONFLICT (id) DO UPDATE SET "
    f"{', '.join(column + ' = EXCLUDED.' + column for column in COLUMNS if column != 'id')}, "
    f"updated_at = now()"
)


def get_connection():
    return psycopg.connect(os.environ["DATABASE_URL"])


def create_schema(conn):
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        for statement in INDEX_SQL:
            cur.execute(statement)
    conn.commit()


def jsonb(value):
    return Jsonb(value) if value is not None else None


def build_row(tool):
    enrichment = tool.get("enrichment") or {}
    return {
        "id": tool["id"],
        "tool_name": tool["tool_name"],
        "domain": tool["domain"],
        "folder": tool["folder"],
        "method": tool["method"],
        "action_type": tool["action_type"],
        "path": tool["path"],
        "description": tool["description"],
        "path_params": jsonb(tool["path_params"]),
        "query_params": jsonb(tool["query_params"]),
        "body_params": jsonb(tool["body_params"]),
        "status_codes": tool["status_codes"],
        "responses": jsonb(tool["responses"]),
        "enrich_description": enrichment.get("description"),
        "example_queries": jsonb(enrichment.get("example_queries")),
        "requires": jsonb(enrichment.get("requires")),
        "produces": jsonb(enrichment.get("produces")),
        "moves_money": enrichment.get("moves_money"),
        "risk_level": enrichment.get("risk_level"),
        "risk_reason": enrichment.get("risk_reason"),
        "record": jsonb(tool),
    }


def store_tools(conn, tools):
    rows = [build_row(tool) for tool in tools]
    with conn.cursor() as cur:
        cur.executemany(UPSERT_SQL, rows)
        cur.execute("DELETE FROM tools WHERE id <> ALL(%s)", ([tool["id"] for tool in tools],))
    conn.commit()
    return len(rows)
