import argparse

from dotenv import load_dotenv

from app.ingest.embed import EMBEDDING_DIMENSION, embed_texts
from app.ingest.store import get_client
from app.knowledge.ingest import build_chunks
from app.knowledge.store import KNOWLEDGE_COLLECTION, create_collection, store_chunks


def main():
    parser = argparse.ArgumentParser(prog="knowledge")
    parser.add_argument("--source", default="data/knowledge")
    parser.add_argument("--db", default="data/milvus_tools.db")
    args = parser.parse_args()

    load_dotenv()

    chunks = build_chunks(args.source)
    if not chunks:
        print(f"No .md or .txt documents found in {args.source}; knowledge base not changed")
        return

    vectors = embed_texts([chunk["text"] for chunk in chunks])
    client = get_client(args.db)
    create_collection(client, EMBEDDING_DIMENSION)
    stored = store_chunks(client, chunks, vectors)
    print(f"Stored {stored} chunks from {args.source} in {args.db} (collection {KNOWLEDGE_COLLECTION})")


if __name__ == "__main__":
    main()
