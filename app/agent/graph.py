import json
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from app.agent.builtin_tools import BUILTIN_NAMES, BUILTIN_SCHEMAS, run_builtin
from app.agent.executor import execute_tool
from app.agent.logs import log_event
from app.agent.retrieval import retrieve_tools
from app.agent.tools import build_tool_schema, get_function_name, needs_approval


MODEL = "openai/gpt-oss-120b"
MAX_WORKERS = 8
SYSTEM_PROMPT = (
    "You are a PayPal API assistant. You are given the API operations relevant to the user's request as tools, "
    "plus two system tools: rag_search for product documentation and guides, and system_search for finding which "
    "operations exist for a topic (kind 'capabilities') or what happened earlier in this session (kind 'logs'). "
    "Before calling tools, decide whether the request needs sequential calls. If a call needs a value that only an "
    "earlier call returns, such as an id created by a previous call, call only the prerequisite now and wait for its "
    "result before calling the dependent operation. If calls do not depend on each other, request them together in "
    "one turn. Use only the tools provided. Never invent ids or argument values; if a required argument is missing, "
    "ask the user. If a tool result says it was rejected by a human reviewer, do not retry that operation; report that "
    "it was not performed. When the needed calls are finished, answer the user concisely using the tool results."
)


class AgentState(TypedDict):
    thread_id: str
    query: str
    messages: Annotated[list, add_messages]
    candidates: list
    approvals: dict


def get_records_by_function(records):
    return {get_function_name(record["tool_name"]): record for record in records}


def retrieve_node(state):
    return {"candidates": retrieve_tools(state["query"])}


def agent_node(state):
    schemas = [build_tool_schema(record) for record in state["candidates"]] + BUILTIN_SCHEMAS
    llm = ChatGroq(model=MODEL, temperature=0).bind_tools(schemas)
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


def route_after_agent(state):
    last = state["messages"][-1]
    return "review" if getattr(last, "tool_calls", None) else END


def review_node(state):
    records = get_records_by_function(state["candidates"])
    pending = []
    for call in state["messages"][-1].tool_calls:
        record = records.get(call["name"])
        if record and needs_approval(record):
            pending.append({
                "id": call["id"],
                "tool": record["tool_name"],
                "method": record["method"],
                "path": record["path"],
                "arguments": call["args"],
            })
    if not pending:
        return {"approvals": {}}
    decisions = interrupt({"pending_approvals": pending})
    return {"approvals": {item["id"]: bool(decisions.get(item["id"], False)) for item in pending}}


def run_call(call, records, approvals, thread_id):
    if call["name"] in BUILTIN_NAMES:
        try:
            return run_builtin(call["name"], call["args"], thread_id), "ok"
        except Exception as error:
            return {"error": str(error)}, "error"
    record = records.get(call["name"])
    if record is None:
        return {"error": f"unknown tool {call['name']}"}, "error"
    if approvals.get(call["id"]) is False:
        return {"error": "rejected by human reviewer; this operation was not performed and must not be retried"}, "rejected"
    try:
        return execute_tool(record, call["args"]), "ok"
    except Exception as error:
        return {"error": str(error)}, "error"


def tools_node(state):
    calls = state["messages"][-1].tool_calls
    records = get_records_by_function(state["candidates"])
    approvals = state.get("approvals", {})
    thread_id = state["thread_id"]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        outcomes = list(pool.map(lambda call: run_call(call, records, approvals, thread_id), calls))
    messages = []
    for call, (result, status) in zip(calls, outcomes):
        if call["name"] not in BUILTIN_NAMES:
            log_event(
                thread_id,
                "tool_call",
                tool_name=call["name"],
                arguments=call["args"],
                status=status,
                result=result,
            )
        messages.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"], name=call["name"]))
    return {"messages": messages}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("review", review_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"review": "review", END: END})
    graph.add_edge("review", "tools")
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())
