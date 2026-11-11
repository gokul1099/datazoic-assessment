from app.ingest.collection import load_collection, iter_requests
from app.ingest.embed import build_embedding_text, embed_texts
from app.ingest.enrich import enrich_tools
from app.ingest.metadata import extract_metadata


def run_pipeline(collection_path):
    collection = load_collection(collection_path)
    tools = [extract_metadata(folders, item) for folders, item in iter_requests(collection["item"])]
    return tools


def run_enrichment(tools, model, limit=None):
    return enrich_tools(tools, model, limit)


def run_embedding(tools):
    texts = [build_embedding_text(tool) for tool in tools]
    return embed_texts(texts)
