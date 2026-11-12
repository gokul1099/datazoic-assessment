import json
import os
import time
import uuid

import gradio as gr
import httpx


DEFAULT_API_URL = "http://127.0.0.1:8000"
OUTPUT_PREVIEW_CHARS = 2000
STEP_TITLES = {
    "retrieve": "Retrieving relevant operations…",
    "agent": "Agent is deciding what to do…",
    "review": "Checking whether any call needs approval…",
    "tools": "Running tool call(s)…",
}
EXAMPLES = [
    "List my billing plans",
    "Show the details of order 5O190127TN364715T",
    "What operations are available for disputes?",
    "Create a product called Premium Video with type SERVICE",
]


def get_api_url():
    return os.environ.get("AGENT_API_URL", DEFAULT_API_URL).rstrip("/")


def new_session():
    return {"thread_id": str(uuid.uuid4()), "pending": []}


def as_json_block(value):
    text = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
    if len(text) > OUTPUT_PREVIEW_CHARS:
        text = text[:OUTPUT_PREVIEW_CHARS] + f"\n… truncated, {len(text) - OUTPUT_PREVIEW_CHARS} more characters"
    return f"```json\n{text}\n```"


def describe_pending(item):
    arguments = json.dumps(item["arguments"], default=str)
    return f"{item['method']} {item['path']} — {item['tool']} — {arguments}"


def describe_candidates(candidates):
    if not candidates:
        return "No matching operations found."
    return "\n".join(f"- `{item['method']} {item['path']}` — {item['tool_name']}" for item in candidates)


def describe_calls(calls):
    return "\n\n".join(f"**{call['name']}**\n{as_json_block(call['arguments'])}" for call in calls)


def step_message(title, content="", status="done", duration=None):
    metadata = {"title": title, "status": status}
    if duration is not None:
        metadata["duration"] = round(duration, 1)
    return {"role": "assistant", "content": content, "metadata": metadata}


def render(history, session, busy=False):
    waiting = bool(session["pending"])
    choices = [(describe_pending(item), item["id"]) for item in session["pending"]]
    return (
        history,
        session,
        gr.update(value="", interactive=not (waiting or busy)),
        gr.update(visible=waiting),
        gr.update(choices=choices, value=[]),
        f"Session `{session['thread_id']}` · agent API `{get_api_url()}`",
    )


class Steps:
    """Keeps one in-progress step at the end of the chat and replaces it when the step finishes."""

    def __init__(self, history):
        self.history = history
        self.started = None

    def begin(self, title):
        self.finish()
        self.started = time.monotonic()
        self.history.append(step_message(title, status="pending"))

    def finish(self, *messages):
        elapsed = None
        if self.started is not None:
            elapsed = time.monotonic() - self.started
            self.history.pop()
            self.started = None
        for index, (title, content) in enumerate(messages):
            self.history.append(step_message(title, content, duration=elapsed if index == 0 else None))


def read_events(path, payload):
    with httpx.stream("POST", get_api_url() + path, json=payload, timeout=None) as response:
        if response.status_code >= 400:
            response.read()
            raise RuntimeError(f"Agent API returned {response.status_code}: {response.text}")
        for line in response.iter_lines():
            if line.strip():
                yield json.loads(line)


def apply_event(event, steps, history, session):
    kind = event["type"]
    if kind == "started":
        steps.begin(STEP_TITLES[event["step"]])
    elif kind == "retrieved":
        candidates = event["candidates"]
        steps.finish((f"Retrieval done — {len(candidates)} candidate operations", describe_candidates(candidates)))
    elif kind == "tool_calls":
        steps.finish((f"Agent requested {len(event['calls'])} tool call(s)", describe_calls(event["calls"])))
    elif kind == "reviewed":
        approvals = event["approvals"]
        approved = sum(approvals.values())
        summary = f"Approval recorded — {approved} approved, {len(approvals) - approved} rejected" if approvals else "No approval needed"
        steps.finish((summary, ""))
    elif kind == "tool_results":
        steps.finish(*[
            (f"Tool output — {item['name']} — {item['status']}", as_json_block(item["result"]))
            for item in event["results"]
        ])
    elif kind == "approval_required":
        session["pending"] = event["pending"]
        steps.finish((
            "Paused — approval needed before these calls run",
            "\n".join(f"- {describe_pending(item)}" for item in session["pending"]),
        ))
    elif kind == "answer":
        steps.finish(("Agent wrote the answer", ""))
        history.append({"role": "assistant", "content": event["answer"]})
    elif kind == "error":
        steps.finish((
            "Something went wrong in the agent",
            f"{event['message']}\n\nIf this repeats, start a new session to clear the conversation.",
        ))


def run_request(path, payload, history, session):
    steps = Steps(history)
    steps.begin("Sending request to the agent API…")
    yield render(history, session, busy=True)
    try:
        for event in read_events(path, payload):
            apply_event(event, steps, history, session)
            yield render(history, session, busy=True)
    except httpx.TransportError as error:
        steps.finish((
            "Cannot reach the agent API",
            f"{get_api_url()} did not respond ({type(error).__name__}).\n\n"
            "Start it with `uvicorn app.main:app --port 8000`, then try again.",
        ))
    except Exception as error:
        steps.finish(("Something went wrong", f"{type(error).__name__}: {error}"))
    steps.finish()
    yield render(history, session)


def send(message, history, session):
    session = session or new_session()
    query = (message or "").strip()
    if not query or session["pending"]:
        yield render(history, session)
        return
    history = history + [{"role": "user", "content": query}]
    yield from run_request("/chat", {"query": query, "session_id": session["thread_id"]}, history, session)


def decide(approved_ids, history, session):
    if not session or not session["pending"]:
        yield render(history, session or new_session())
        return
    decisions = {item["id"]: item["id"] in approved_ids for item in session["pending"]}
    approved = sum(decisions.values())
    history = history + [{"role": "user", "content": f"Approved {approved} of {len(decisions)} call(s)."}]
    session["pending"] = []
    payload = {"session_id": session["thread_id"], "decisions": decisions}
    yield from run_request("/chat/resume", payload, history, session)


def reset():
    return render([], new_session())


def build_ui():
    with gr.Blocks(title="PayPal API Assistant") as demo:
        gr.Markdown(
            "# PayPal API Assistant\n"
            "Ask in plain English; the agent picks and calls the matching PayPal operations. "
            "Each step appears as it happens — click a step to see its details."
        )
        session = gr.State(None)
        session_label = gr.Markdown()
        chatbot = gr.Chatbot(height=560, show_label=False, placeholder="Ask about orders, invoices, payouts, subscriptions…")
        with gr.Group(visible=False) as approval_box:
            approvals = gr.CheckboxGroup(
                label="Approval needed — tick the calls to allow; unticked calls are rejected",
                choices=[],
            )
            submit_approvals = gr.Button("Submit decision", variant="primary")
        with gr.Row():
            textbox = gr.Textbox(placeholder="Type a request and press Enter", show_label=False, scale=8, autofocus=True)
            send_button = gr.Button("Send", variant="primary", scale=1)
            reset_button = gr.Button("New session", scale=1)
        gr.Examples(EXAMPLES, inputs=textbox)

        outputs = [chatbot, session, textbox, approval_box, approvals, session_label]
        textbox.submit(send, [textbox, chatbot, session], outputs)
        send_button.click(send, [textbox, chatbot, session], outputs)
        submit_approvals.click(decide, [approvals, chatbot, session], outputs)
        reset_button.click(reset, None, outputs)
        demo.load(reset, None, outputs)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1")
