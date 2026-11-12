from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, RRFRanker


KNOWLEDGE_COLLECTION = "knowledge_docs"
TEXT_MAX_LENGTH = 8192
INSERT_BATCH_SIZE = 500


def create_collection(client, dimension, name=KNOWLEDGE_COLLECTION):
    if client.has_collection(name):
        client.drop_collection(name)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=512, is_primary=True)
    schema.add_field("source", DataType.VARCHAR, max_length=512)
    schema.add_field("text", DataType.VARCHAR, max_length=TEXT_MAX_LENGTH, enable_analyzer=True)
    schema.add_field("text_sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_function(Function(
        name="bm25",
        input_field_names=["text"],
        output_field_names=["text_sparse"],
        function_type=FunctionType.BM25,
    ))
    index_params = client.prepare_index_params()
    index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index("text_sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(name, schema=schema, index_params=index_params)


def store_chunks(client, chunks, vectors, name=KNOWLEDGE_COLLECTION):
    rows = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk["text"],
            "vector": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    for start in range(0, len(rows), INSERT_BATCH_SIZE):
        client.insert(name, rows[start:start + INSERT_BATCH_SIZE])
    return len(rows)


def search_chunks(client, query_text, query_vector, limit=5, name=KNOWLEDGE_COLLECTION):
    client.load_collection(name)
    dense = AnnSearchRequest(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=limit,
    )
    sparse = AnnSearchRequest(
        data=[query_text],
        anns_field="text_sparse",
        param={"metric_type": "BM25"},
        limit=limit,
    )
    results = client.hybrid_search(
        name,
        reqs=[dense, sparse],
        ranker=RRFRanker(),
        limit=limit,
        output_fields=["chunk_id", "source", "text"],
    )
    return [
        {
            "chunk_id": hit["entity"]["chunk_id"],
            "source": hit["entity"]["source"],
            "text": hit["entity"]["text"],
        }
        for hit in results[0]
    ]
