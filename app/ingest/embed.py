from openai import OpenAI
from tqdm import tqdm


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
EMBED_BATCH_SIZE = 100


def build_embedding_text(tool):
    enrichment = tool.get("enrichment") or {}
    description = enrichment.get("description") or tool["description"]
    parts = [
        tool["tool_name"],
        f"domain: {tool['domain']}",
        description,
        *enrichment.get("example_queries", []),
    ]
    return "\n".join(part for part in parts if part)


def embed_texts(texts, model=EMBEDDING_MODEL, progress=False):
    client = OpenAI()
    vectors = []
    bar = tqdm(total=len(texts), desc="Embedding", unit="text", disable=not progress)
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        response = client.embeddings.create(model=model, input=texts[start:start + EMBED_BATCH_SIZE])
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        bar.update(len(response.data))
    bar.close()
    return vectors
