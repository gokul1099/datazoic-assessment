import argparse
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.agent.graph import build_graph
from app.agent.logs import create_log_table, log_event


RECURSION_LIMIT = 30


def ask_approvals(pending):
    decisions = {}
    for item in pending:
        print(f"\nApproval needed: {item['method']} {item['path']} ({item['tool']})")
        print(f"  arguments: {item['arguments']}")
        answer = input("Approve this call? [y/N] ").strip().lower()
        decisions[item["id"]] = answer == "y"
    return decisions


def run_query(graph, query, thread_id):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    state = {
        "thread_id": thread_id,
        "query": query,
        "messages": [HumanMessage(content=query)],
        "candidates": [],
        "approvals": {},
    }
    result = graph.invoke(state, config)
    while "__interrupt__" in result:
        pending = result["__interrupt__"][0].value["pending_approvals"]
        result = graph.invoke(Command(resume=ask_approvals(pending)), config)
    answer = result["messages"][-1].content
    log_event(thread_id, "turn", query=query, result={"answer": answer})
    return answer


def main():
    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument("query", nargs="*")
    parser.add_argument("--session", default=None)
    args = parser.parse_args()

    load_dotenv()
    create_log_table()
    graph = build_graph()
    thread_id = args.session or str(uuid.uuid4())
    query = " ".join(args.query) or input("You: ").strip()
    print(f"[session {thread_id}]")
    print(run_query(graph, query, thread_id))


if __name__ == "__main__":
    main()
