from app.ingest.collection import load_collection, flatten_requests


def run_pipeline(collection_path):
    collection = load_collection(collection_path)
    tools = flatten_requests(collection["item"])
    return tools
