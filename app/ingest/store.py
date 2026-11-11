from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker


COLLECTION_NAME = "paypal_tools"
INSERT_BATCH_SIZE = 500
TEXT_MAX_LENGTH = 8192


def get_client(db_path):
    return MilvusClient(db_path)


def create_collection(client, dimension, name=COLLECTION_NAME):
    if client.has_collection(name):
        client.drop_collection(name)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("tool_name", DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=TEXT_MAX_LENGTH, enable_analyzer=True)
    schema.add_field("text_sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field("domain", DataType.VARCHAR, max_length=256)
    schema.add_field("action_type", DataType.VARCHAR, max_length=16)
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


def store_tools(client, tools, texts, vectors, name=COLLECTION_NAME):
    names = [tool["tool_name"] for tool in tools]
    if len(set(names)) != len(names):
        raise ValueError("tool_name must be unique to be used as the vector store key")
    rows = [
        {
            "tool_name": tool["tool_name"],
            "text": text,
            "vector": vector,
            "domain": tool["domain"],
            "action_type": tool["action_type"],
        }
        for tool, text, vector in zip(tools, texts, vectors)
    ]
    for start in range(0, len(rows), INSERT_BATCH_SIZE):
        client.insert(name, rows[start:start + INSERT_BATCH_SIZE])
    return len(rows)


def build_filter(domain=None, action_type=None):
    clauses = []
    if domain:
        clauses.append(f'domain == "{domain}"')
    if action_type:
        clauses.append(f'action_type == "{action_type}"')
    return " and ".join(clauses)


def hybrid_search(client, query_text, query_vector, limit=10, domain=None, action_type=None, name=COLLECTION_NAME):
    expr = build_filter(domain, action_type)
    dense = AnnSearchRequest(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=limit,
        expr=expr,
    )
    sparse = AnnSearchRequest(
        data=[query_text],
        anns_field="text_sparse",
        param={"metric_type": "BM25"},
        limit=limit,
        expr=expr,
    )
    results = client.hybrid_search(
        name,
        reqs=[dense, sparse],
        ranker=RRFRanker(),
        limit=limit,
        output_fields=["tool_name", "domain", "action_type"],
    )
    return [
        {"tool_name": hit["entity"]["tool_name"], "score": hit["distance"]}
        for hit in results[0]
    ]
