from app.agent.logs import search_logs
from app.agent.retrieval import retrieve_tools
from app.ingest.embed import embed_texts
from app.knowledge.store import KNOWLEDGE_COLLECTION, search_chunks
from app.ingest.store import get_client


DEFAULT_DB_PATH = "data/milvus_tools.db"
RAG_TOP_K = 5
BUILTIN_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the product documentation and guides knowledge base. Use it for conceptual, how-to or policy questions. It does not call PayPal.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look up in the documentation."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_search",
            "description": "Search this system. kind 'capabilities' finds which PayPal operations are available for a topic. kind 'logs' finds what happened earlier in this session, such as the status of the last request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["capabilities", "logs"]},
                    "query": {"type": "string", "description": "Topic or keywords to search for."},
                },
                "required": ["kind", "query"],
            },
        },
    },
]
BUILTIN_NAMES = {schema["function"]["name"] for schema in BUILTIN_SCHEMAS}


def rag_search(query, db_path=DEFAULT_DB_PATH):
    vector = embed_texts([query])[0]
    hits = search_chunks(get_client(db_path), query, vector, limit=RAG_TOP_K, name=KNOWLEDGE_COLLECTION)
    return {"results": hits}


def system_search(kind, query, thread_id):
    if kind == "capabilities":
        records = retrieve_tools(query)
        return {
            "tools": [
                {
                    "tool_name": record["tool_name"],
                    "domain": record["domain"],
                    "method": record["method"],
                    "path": record["path"],
                    "description": (record.get("enrichment") or {}).get("description") or record["description"],
                }
                for record in records
            ]
        }
    if kind == "logs":
        return {"events": search_logs(thread_id, query)}
    return {"error": f"unknown kind '{kind}', use 'capabilities' or 'logs'"}


def run_builtin(name, arguments, thread_id):
    if name == "rag_search":
        return rag_search(arguments.get("query", ""))
    if name == "system_search":
        return system_search(arguments.get("kind", ""), arguments.get("query", ""), thread_id)
    return {"error": f"unknown builtin tool {name}"}
