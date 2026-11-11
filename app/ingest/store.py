from pymilvus import DataType, MilvusClient


COLLECTION_NAME = "paypal_tools"
INSERT_BATCH_SIZE = 500


def get_client(db_path):
    return MilvusClient(db_path)


def create_collection(client, dimension, name=COLLECTION_NAME):
    if client.has_collection(name):
        client.drop_collection(name)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field("domain", DataType.VARCHAR, max_length=256)
    schema.add_field("action_type", DataType.VARCHAR, max_length=16)
    schema.add_field("record", DataType.JSON)
    index_params = client.prepare_index_params()
    index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(name, schema=schema, index_params=index_params)


def store_tools(client, tools, vectors, name=COLLECTION_NAME):
    rows = [
        {
            "id": tool["id"],
            "vector": vector,
            "domain": tool["domain"],
            "action_type": tool["action_type"],
            "record": tool,
        }
        for tool, vector in zip(tools, vectors)
    ]
    for start in range(0, len(rows), INSERT_BATCH_SIZE):
        client.insert(name, rows[start:start + INSERT_BATCH_SIZE])
    return len(rows)
