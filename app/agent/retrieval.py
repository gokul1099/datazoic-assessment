from app.ingest.embed import embed_texts
from app.ingest.sql_store import get_connection
from app.ingest.store import COLLECTION_NAME, get_client, hybrid_search


DEFAULT_DB_PATH = "data/milvus_tools.db"


def retrieve_tools(query, top_k=10, db_path=DEFAULT_DB_PATH):
    query_vector = embed_texts([query])[0]
    hits = hybrid_search(get_client(db_path), query, query_vector, limit=top_k, name=COLLECTION_NAME)
    names = [hit["tool_name"] for hit in hits]
    if not names:
        return []
    with get_connection() as conn:
        rows = conn.execute("SELECT tool_name, record FROM tools WHERE tool_name = ANY(%s)", (names,)).fetchall()
    records = {tool_name: record for tool_name, record in rows}
    return [records[name] for name in names if name in records]
