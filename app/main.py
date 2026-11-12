import json
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import build_graph
from app.agent.logs import create_log_table
from app.agent.service import get_pending_approvals, stream_chat, stream_resume


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = None


class ResumeRequest(BaseModel):
    session_id: str
    decisions: dict[str, bool]


@asynccontextmanager
async def lifespan(app):
    load_dotenv()
    create_log_table()
    app.state.graph = build_graph()
    yield


app = FastAPI(title="datazoic-assesment", lifespan=lifespan)


def ndjson(session_id, events):
    def lines():
        yield json.dumps({"type": "session", "session_id": session_id}) + "\n"
        for event in events:
            yield json.dumps(event, default=str) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/")
def read_root():
    return {"message": "Hello, World"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatRequest, request: Request):
    """Start an agent run. Streams one JSON event per line as each step happens."""
    graph = request.app.state.graph
    session_id = body.session_id or str(uuid.uuid4())
    if get_pending_approvals(graph, session_id):
        raise HTTPException(status_code=409, detail="This session is waiting for an approval decision; call /chat/resume")
    return ndjson(session_id, stream_chat(graph, session_id, body.query.strip()))


@app.post("/chat/resume")
def resume(body: ResumeRequest, request: Request):
    """Continue a run that paused for approval. decisions maps each pending call id to true or false."""
    graph = request.app.state.graph
    if not get_pending_approvals(graph, body.session_id):
        raise HTTPException(status_code=409, detail="This session has no pending approval")
    return ndjson(body.session_id, stream_resume(graph, body.session_id, body.decisions))
