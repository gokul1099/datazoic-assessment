from psycopg.types.json import Jsonb

from app.ingest.sql_store import get_connection


LOG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agent_logs (
    id BIGSERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    query TEXT,
    event TEXT NOT NULL,
    tool_name TEXT,
    arguments JSONB,
    status TEXT,
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

LOG_INDEX_SQL = "CREATE INDEX IF NOT EXISTS agent_logs_thread_idx ON agent_logs (thread_id, created_at)"


def create_log_table():
    with get_connection() as conn:
        conn.execute(LOG_SCHEMA_SQL)
        conn.execute(LOG_INDEX_SQL)


def log_event(thread_id, event, query=None, tool_name=None, arguments=None, status=None, result=None):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO agent_logs (thread_id, query, event, tool_name, arguments, status, result) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                thread_id,
                query,
                event,
                tool_name,
                Jsonb(arguments) if arguments is not None else None,
                status,
                Jsonb(result) if result is not None else None,
            ),
        )


def fetch_logs(thread_id, text, limit):
    sql = (
        "SELECT event, query, tool_name, status, result, created_at FROM agent_logs WHERE thread_id = %s"
    )
    params = [thread_id]
    if text:
        sql += " AND (query ILIKE %s OR tool_name ILIKE %s)"
        pattern = f"%{text}%"
        params.extend([pattern, pattern])
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    with get_connection() as conn:
        return conn.execute(sql, params).fetchall()


def search_logs(thread_id, text="", limit=20):
    rows = fetch_logs(thread_id, text, limit)
    if not rows and text:
        rows = fetch_logs(thread_id, "", limit)
    return [
        {
            "event": event,
            "query": query,
            "tool_name": tool_name,
            "status": status,
            "result": result,
            "created_at": created_at.isoformat(),
        }
        for event, query, tool_name, status, result, created_at in rows
    ]
