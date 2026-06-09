# Agentic RAG Document Q&A — Interview Practice Project

## Context

Built to practice for a Software Engineer interview at **Moody's Analytics** (Agentic AI / LangChain role).
The goal is production-quality code I can walk through and defend line-by-line in a technical interview.

---

## What it does

An agentic document Q&A chatbot. Upload a PDF → it gets ingested, chunked, embedded, and stored in ChromaDB. Ask a question → a LangChain ReAct agent decides whether it needs to retrieve document context or can answer from general knowledge. If retrieval is needed, the agent calls a FastMCP tool (`query_document`) that fetches relevant chunks from ChromaDB and feeds them to the LLM for a grounded, cited answer.

---

## Architecture

```
User question → FastAPI → LangChain ReAct Agent
                                    ↓
                        Needs document context?
                        YES → calls MCP Tool: query_document()
                                    ↓
                             ChromaDB retrieves top-k chunks
                                    ↓
                             OpenAI generates grounded answer
                        NO  → Agent answers from own knowledge
                                    ↓
                        LangSmith traces the entire execution graph
                                    ↓
                        FastAPI returns { answer, sources, used_retrieval }
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| REST API | FastAPI | Async-native, automatic OpenAPI docs, production-standard |
| Agent | LangChain + LangGraph `create_react_agent` | ReAct loop: reason → act → observe until done |
| Tool protocol | FastMCP (HTTP transport) | MCP is the emerging standard for agent↔tool communication |
| LLM / Embeddings | OpenAI `gpt-4o-mini` / `text-embedding-3-small` | Cost-efficient, strong quality |
| Vector store | ChromaDB `PersistentClient` | Local, zero-infra, persistent across restarts |
| Observability | LangSmith | Auto-traces every LLM call and tool call in the LangGraph |
| Config | `pydantic-settings` | Type-safe env vars, fails loudly on missing keys |
| Tests | pytest + pytest-asyncio | Mocked ChromaDB — no external services needed |

---

## Project Structure

```
chatbot-agent-langchain/
├── app/
│   ├── main.py              # FastAPI entry point + lifespan (auto-starts MCP server)
│   ├── config.py            # pydantic-settings — all env vars in one place
│   ├── models.py            # Pydantic request/response models
│   ├── api/
│   │   └── routes.py        # POST /upload, POST /chat, GET /health
│   ├── core/
│   │   ├── ingestion.py     # PDF load → chunk → embed → store in ChromaDB
│   │   ├── retrieval.py     # ChromaDB similarity search
│   │   └── agent.py         # LangGraph ReAct agent + MCP tool wiring
│   └── mcp/
│       └── server.py        # FastMCP server exposing query_document tool
├── tests/
│   └── test_pipeline.py     # Unit tests (mocked ChromaDB)
├── chroma_db/               # Persisted vector data (gitignored)
├── .env                     # Your actual keys (gitignored)
├── .env.example             # Template — copy to .env and fill in
├── requirements.txt
└── README.md
```

---

## API Endpoints

### `POST /upload`
Accepts a PDF file. Ingests it into ChromaDB and returns the collection name.

```json
// Response
{
  "collection_name": "my_document",
  "chunks_stored": 42,
  "message": "Successfully ingested 42 chunks into 'my_document'."
}
```

### `POST /chat`
Sends a question to the LangChain agent. Pass `collection` to enable retrieval.

```json
// Request
{
  "question": "What are the key risk factors?",
  "collection": "my_document"
}

// Response
{
  "answer": "The key risk factors are...",
  "sources": [{ "content": "...", "metadata": { ... } }],
  "used_retrieval": true
}
```

### `GET /health`
Returns `{"status": "ok"}`.

---

## How to Run

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env — add OPENAI_API_KEY (required) and LANGCHAIN_API_KEY (optional)

# 4. Start the app (MCP server auto-starts as a subprocess)
uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

> **How the two-service architecture works locally:** `app/main.py` uses a FastAPI `lifespan` context manager that spawns `python -m app.mcp.server` as a subprocess on startup and shuts it down on exit. No Docker needed — one command runs everything.

---

## Key Implementation Decisions (for the interview)

| Decision | Reasoning |
|---|---|
| Each PDF → its own ChromaDB collection | Avoids cross-document retrieval noise; collection name is returned to the caller so the agent knows exactly where to look |
| ReAct agent over a simple chain | The agent decides *whether* to retrieve — a chain would always retrieve, even for general-knowledge questions |
| MCP over direct function call | Demonstrates the MCP protocol (tool registry, JSON schema, transport) — the same interface Claude Desktop and other agents use |
| `MultiServerMCPClient` as async context manager | Opens and closes an HTTP session per request; keeps the API stateless |
| LangSmith env vars set before any `langchain` import | LangChain reads tracing config at import time — setting them after the import is too late |
| `delete=False` on Windows tempfiles | Windows holds an OS-level lock on open files; PyPDFLoader needs the file on disk while parsing |
| `pydantic-settings` singleton in `config.py` | One `.env` read at startup — not per-request — so secrets aren't re-read on every call |
| MCP server as a lifespan subprocess | Keeps the two-process architecture (identical to the Docker design) without requiring Docker |

---

## Running Tests

```powershell
pytest tests/ -v
```

Tests mock `chromadb.PersistentClient` — no OpenAI key or running server needed.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `LANGCHAIN_API_KEY` | No | `""` | LangSmith API key |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | No | `rag-agent` | LangSmith project name |
| `CHAT_MODEL` | No | `gpt-4o-mini` | OpenAI chat model |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | OpenAI embedding model |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | ChromaDB storage path |
| `MCP_SERVER_URL` | No | `http://localhost:9001/mcp` | FastMCP server URL |
| `CHUNK_SIZE` | No | `1000` | Text splitter chunk size |
| `CHUNK_OVERLAP` | No | `200` | Text splitter overlap |
