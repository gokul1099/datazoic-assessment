import argparse

from app.ingest.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(prog="ingest")
    parser.add_argument("--collection", default="data/paypal_collection.postman_collection.json")
    parser.add_argument("--preview", type=int, default=5)
    args = parser.parse_args()

    tools = run_pipeline(args.collection)
    print(f"Loaded {len(tools)} tools from {args.collection}")
    for tool in tools[: args.preview]:
        print(f"- {tool['id']}")


if __name__ == "__main__":
    main()
