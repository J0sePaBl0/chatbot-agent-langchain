import asyncio
import os
import socket
import subprocess
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

os.environ.setdefault("LANGSMITH_TRACING", settings.LANGSMITH_TRACING)
os.environ.setdefault("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)
os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)
os.environ.setdefault("LANGSMITH_ENDPOINT", settings.LANGSMITH_ENDPOINT)

from app.api.routes import router  # noqa: E402 — intentionally after env setup


def _mcp_port_ready(host: str = "localhost", port: int = 9001) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the FastMCP server as a sidecar process; tear it down on shutdown."""
    mcp_proc = subprocess.Popen(
        [sys.executable, "-m", "app.mcp.server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 15 s for the MCP HTTP port to accept connections
    for _ in range(30):
        if _mcp_port_ready():
            break
        await asyncio.sleep(0.5)
    else:
        mcp_proc.kill()
        raise RuntimeError("MCP server did not start within 15 seconds.")

    yield

    mcp_proc.terminate()
    try:
        mcp_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        mcp_proc.kill()


app = FastAPI(
    title="Agentic RAG Document Q&A",
    description=(
        "Upload PDFs and ask questions. "
        "A LangChain ReAct agent decides when to call the FastMCP query_document "
        "tool to retrieve relevant chunks from ChromaDB, then generates a grounded answer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Vite dev server to call the API. In production, restrict to the actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
