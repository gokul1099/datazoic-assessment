import argparse
import json

from dotenv import load_dotenv

from app.ingest.enrich import DEFAULT_MODEL
from app.ingest.pipeline import run_embedding, run_enrichment, run_pipeline
from app.ingest.store import COLLECTION_NAME, create_collection, get_client, store_tools
from app.ingest.embed import EMBEDDING_DIMENSION


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

    vectors = run_embedding(tools)
    print(f"Embedded {len(vectors)} tools")

    client = get_client(args.db)
    create_collection(client, EMBEDDING_DIMENSION)
    stored = store_tools(client, tools, vectors)
    print(f"Stored {stored} tools in {args.db} (collection {COLLECTION_NAME})")

    for tool in tools[: args.preview]:
        print(f"- [{tool['action_type']}] {tool['method']} {tool['path']} ({tool['domain']})")


if __name__ == "__main__":
    main()
