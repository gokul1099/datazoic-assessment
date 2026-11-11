import argparse

from dotenv import load_dotenv

from app.ingest.embed import EMBEDDING_DIMENSION
from app.ingest.enrich import DEFAULT_MODEL
from app.ingest.pipeline import run_embedding, run_enrichment, run_pipeline
from app.ingest.sql_store import create_schema, get_connection, store_tools as store_rows
from app.ingest.store import COLLECTION_NAME, create_collection, get_client, store_tools


def main():
    parser = argparse.ArgumentParser(prog="ingest")
    parser.add_argument("--collection", default="data/paypal_collection.postman_collection.json")
    parser.add_argument("--db", default="data/milvus_tools.db")
    parser.add_argument("--enrich", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preview", type=int, default=5)
    args = parser.parse_args()

    load_dotenv()

    tools = run_pipeline(args.collection)
    print(f"Parsed {len(tools)} tools from {args.collection}")

    if args.enrich:
        tools = run_enrichment(tools, args.model, args.limit)
        print(f"Enriched {len(tools)} tools with {args.model}")

    with get_connection() as conn:
        create_schema(conn)
        print(f"Stored {store_rows(conn, tools)} tool rows in PostgreSQL (source of truth)")

    texts, vectors = run_embedding(tools)
    print(f"Embedded {len(vectors)} tools")

    client = get_client(args.db)
    create_collection(client, EMBEDDING_DIMENSION)
    stored = store_tools(client, tools, texts, vectors)
    print(f"Stored {stored} tools in {args.db} (collection {COLLECTION_NAME}, dense + BM25)")

    for tool in tools[: args.preview]:
        print(f"- [{tool['action_type']}] {tool['method']} {tool['path']} ({tool['domain']})")


if __name__ == "__main__":
    main()
