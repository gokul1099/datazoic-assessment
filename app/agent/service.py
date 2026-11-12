import json

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.agent.logs import log_event


RECURSION_LIMIT = 30


def get_config(thread_id):
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def build_initial_state(thread_id, query):
    return {
        "thread_id": thread_id,
        "query": query,
        "messages": [HumanMessage(content=query)],
        "candidates": [],
        "approvals": {},
    }


def get_pending_approvals(graph, thread_id):
    state = graph.get_state(get_config(thread_id))
    for task in state.tasks:
        for item in task.interrupts:
            return item.value["pending_approvals"]
    return []


def get_tool_status(result):
    if isinstance(result, dict) and "error" in result:
        return "rejected" if "rejected by human reviewer" in str(result["error"]) else "error"
    if isinstance(result, dict) and "status" in result:
        return f"HTTP {result['status']}"
    return "ok"


def parse_tool_result(message):
    try:
        return json.loads(message.content)
    except (TypeError, json.JSONDecodeError):
        return message.content


def summarize_candidate(record):
    return {"tool_name": record["tool_name"], "method": record["method"], "path": record["path"]}


def stream_events(graph, graph_input, thread_id, first_step):
    """Run the graph and yield one event per step, so a client can show what the agent is doing."""
    config = get_config(thread_id)
    yield {"type": "started", "step": first_step}
    try:
        for chunk in graph.stream(graph_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    yield {"type": "approval_required", "pending": update[0].value["pending_approvals"]}
                    return
                if node == "retrieve":
                    yield {"type": "retrieved", "candidates": [summarize_candidate(r) for r in update["candidates"]]}
                    yield {"type": "started", "step": "agent"}
                elif node == "agent":
                    message = update["messages"][-1]
                    calls = getattr(message, "tool_calls", None)
                    if calls:
                        yield {"type": "tool_calls", "calls": [{"name": c["name"], "arguments": c["args"]} for c in calls]}
                        yield {"type": "started", "step": "review"}
                    else:
                        query = graph.get_state(config).values.get("query")
                        log_event(thread_id, "turn", query=query, result={"answer": message.content})
                        yield {"type": "answer", "answer": message.content}
                elif node == "review":
                    yield {"type": "reviewed", "approvals": update.get("approvals") or {}}
                    yield {"type": "started", "step": "tools"}
                elif node == "tools":
                    results = []
                    for message in update["messages"]:
                        result = parse_tool_result(message)
                        results.append({"name": message.name, "status": get_tool_status(result), "result": result})
                    yield {"type": "tool_results", "results": results}
                    yield {"type": "started", "step": "agent"}
    except Exception as error:
        yield {"type": "error", "message": f"{type(error).__name__}: {error}"}


def stream_chat(graph, thread_id, query):
    return stream_events(graph, build_initial_state(thread_id, query), thread_id, "retrieve")


def stream_resume(graph, thread_id, decisions):
    return stream_events(graph, Command(resume=decisions), thread_id, "review")
