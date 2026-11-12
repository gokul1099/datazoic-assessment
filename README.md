# PayPal API Assistant

An agent that answers plain-English requests by finding the right PayPal API operations, calling them, and summarising the result. The operations come from a Postman collection (115 of them), and the calls go to a local mock server, so nothing touches real PayPal.

## How it fits together

| Part | What it does | Where |
|---|---|---|
| Ingestion | Parses the Postman collection into a tool catalogue, stored in PostgreSQL (full records) and Milvus Lite (search index) | `app/ingest` |
| Mock server | Serves the example responses from the collection, standing in for PayPal | `app/mock` |
| Agent API | FastAPI server that runs the agent when `/chat` is called | `app/main.py`, `app/agent` |
| UI | Gradio chat page that calls the agent API and shows each step live | `app/ui.py` |

For each request the agent retrieves candidate operations, lets the model choose which to call, pauses for human approval on payment-related write calls, runs the calls against the mock server, and writes the answer.

## Prerequisites

- Python 3.11
- Docker, for PostgreSQL (or a PostgreSQL server you already have)
- A Groq API key (the agent's model) and an OpenAI API key (embeddings)

## Setup

### 1. Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Create `.env`

Create a file named `.env` in the project root:

```
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://<user>:<password>@localhost:5432/api_collection
```

Optional settings:

| Variable | Default | Purpose |
|---|---|---|
| `TOOL_BASE_URL` | `http://127.0.0.1:4010` | Where the agent sends PayPal calls (the mock server) |
| `AGENT_API_URL` | `http://127.0.0.1:8000` | Where the UI finds the agent API |

### 3. Start PostgreSQL

Use the same user, password and database name as in `DATABASE_URL`:

```bash
docker run -d --name datazoic-postgres --restart unless-stopped \
  -e POSTGRES_USER=<user> -e POSTGRES_PASSWORD=<password> -e POSTGRES_DB=api_collection \
  -p 127.0.0.1:5432:5432 -v datazoic_pgdata:/var/lib/postgresql/data \
  postgres:16-alpine
```

The tables are created automatically. Later, `docker stop datazoic-postgres` and `docker start datazoic-postgres` stop and restart it.

### 4. Ingest the tool catalogue

Run this once before first use, and again whenever the collection changes. Stop the agent API first if it is running (see [One process at a time](#one-process-at-a-time)).

```bash
python -m app.ingest
```

It parses the collection, writes 115 rows to PostgreSQL, embeds them with OpenAI, and stores them in `data/milvus_tools.db`. Progress bars show the embedding and storing steps; the whole run takes under a minute.

| Option | Default | Meaning |
|---|---|---|
| `--collection` | `data/paypal_collection.postman_collection.json` | Postman collection to ingest |
| `--db` | `data/milvus_tools.db` | Milvus Lite location |
| `--enrich` | off | Add model-written metadata to each tool (next section) |
| `--model` | `openai/gpt-oss-120b` | Groq model used for enrichment |
| `--limit` | none | Only with `--enrich`; see the warning below |
| `--preview` | `5` | How many tools to print at the end |

## Enrichment (optional)

```bash
python -m app.ingest --enrich
```

Enrichment asks the Groq model to write extra metadata for every tool before it is stored:

- a clearer description, and 2–5 example user requests (both go into the search text, so retrieval improves)
- `requires` and `produces`: which inputs a tool needs and which other tool supplies them
- `moves_money`, `risk_level` and `risk_reason`

The agent works without it. Without enrichment, the original Postman descriptions are used, and approval is required for write calls in the payment domains (Invoices, Orders, Payments, Payouts, Disputes, Subscriptions, Payment Method Tokens). With enrichment, any write call flagged `moves_money` also requires approval.

Things to know before running it:

- **It is slow on a small Groq plan.** There is one model call per tool (115), and every call includes the whole catalogue, about 5,400 tokens. On Groq's on-demand tier (8,000 tokens per minute for this model) roughly one call per minute gets through, so a full run takes around two hours and can fail partway. A higher tier removes the problem. The "Enriching" progress bar shows how far it has got.
- **It is all or nothing.** Each tool gets three attempts; if one still fails, the run stops and nothing is stored.
- **`--limit N` shrinks the catalogue.** It enriches only the first N tools and then stores only those N, replacing the full catalogue in both PostgreSQL and Milvus. Use it to try enrichment out, then run `python -m app.ingest` again to restore all 115 tools.

## Knowledge base (optional)

The agent's `rag_search` tool answers documentation questions from a knowledge base. To fill it, put `.md` or `.txt` files in `data/knowledge` and run:

```bash
python -m app.knowledge
```

Without it, the agent still works, but `rag_search` returns an error saying the collection does not exist.

## Running

Use three terminals, each with the virtualenv activated (`source .venv/bin/activate`):

```bash
# 1. Mock PayPal API. The port matters: the agent calls 4010.
uvicorn app.mock.server:app --port 4010

# 2. Agent API
uvicorn app.main:app --port 8000

# 3. UI
python -m app.ui
```

Open the URL the UI prints (normally http://127.0.0.1:7860). Each step appears in the chat as it happens: retrieval, the tool calls the agent chose, the approval check, each tool's output, and the answer. When a call needs approval, a box appears; tick the calls to allow and submit.

### Calling the API directly

```bash
curl -N -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"query": "Show the details of order 5O190127TN364715T"}'
```

| Endpoint | Body | Purpose |
|---|---|---|
| `POST /chat` | `{"query": "...", "session_id": "optional"}` | Start an agent run |
| `POST /chat/resume` | `{"session_id": "...", "decisions": {"<call id>": true}}` | Continue a run that paused for approval |
| `GET /health` | – | Liveness check |

Both `POST` endpoints stream one JSON object per line. The `type` field is one of `session`, `started`, `retrieved`, `tool_calls`, `reviewed`, `tool_results`, `approval_required`, `answer` or `error`. Reuse the `session_id` from the `session` event to continue a conversation.

### Command-line agent

```bash
python -m app.agent "list my billing plans"
```

This runs the agent in its own process and asks for approvals in the terminal. It needs the mock server, and it cannot run while the agent API is up.

## One process at a time

Milvus Lite lets only one process open `data/milvus_tools.db`. The agent API holds it while running, so stop the agent API before running `python -m app.ingest`, `python -m app.knowledge` or `python -m app.agent`. Ingest checks this first and stops with a message if the store is locked.

For the same reason, run the agent API with a single worker. Conversation history and paused approvals are kept in its memory, so they are lost when it restarts.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `DataDirLockedError: another process holds the lock` | Another process has the Milvus store open. Stop the agent API (or the command-line agent) and retry. |
| The agent says the PayPal request failed; the tool output shows `Connection refused` | The mock server is not on port 4010. Start it with `--port 4010`. Without the flag, uvicorn uses 8000. |
| The UI shows "Cannot reach the agent API" | Start `uvicorn app.main:app --port 8000`, or set `AGENT_API_URL`. |
| Groq error 413, "Request too large … tokens per minute" | The request exceeded the Groq plan's per-minute limit, usually because a tool returned a large response. Start a new session and ask for something narrower, or use a higher Groq tier. |
| `code for hash blake2b was not found` on every start | The Python build is missing a library; it is harmless here. On macOS with pyenv, `brew install libb2` should remove it. |
| `AllocTimestamp … Method not implemented` during ingest | Milvus Lite log noise; ingest still completes. |
