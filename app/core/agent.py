"""LangChain agent wired to the FastMCP query_document tool.

Architecture overview:
  The agent is a LangGraph ReAct graph (create_react_agent).
  It receives a user question, decides autonomously whether it needs
  document context, and if so calls the query_document MCP tool.

MCP integration:
  langchain-mcp-adapters' MultiServerMCPClient connects to the running
  FastMCP server over HTTP and loads its tools as LangChain BaseTool objects.
  In Docker this URL is http://mcp:9001/mcp (Docker DNS resolves 'mcp').
  Locally it's http://localhost:9001/mcp (set MCP_SERVER_URL in .env).

  The context manager opens an SSE/HTTP session for the duration of one
  agent invocation, then closes it. This is fine for request/response APIs;
  a long-lived connection pool would be an optimisation for high throughput.

LangSmith tracing:
  When LANGSMITH_TRACING=true, every node in the LangGraph execution
  (LLM call, tool call, routing decision) is recorded as a span in LangSmith.
  No additional code is needed — the integration is automatic via env vars
  set in main.py before any langchain import.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from app.config import settings
from app.models import ChatResponse, SourceChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about uploaded documents. "
    "When the user's question requires specific information from a document, use the "
    "query_document tool with the appropriate collection name. "
    "Always cite the source file and page number when referencing document content. "
    "If no collection is specified or the question is general knowledge, answer directly."
)


async def run_agent(question: str, collection: str | None = None) -> ChatResponse:
    """Run the ReAct agent and return a structured ChatResponse.

    The function is async because MultiServerMCPClient requires an async
    context manager — it opens an HTTP session to the MCP server.
    FastAPI routes are async, so `await run_agent(...)` works natively.
    """
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
    )

    # As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient is no longer
    # a context manager — use client.get_tools() directly instead.
    client = MultiServerMCPClient(
        {
            "rag-agent": {
                "url": settings.MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()

    graph = create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    user_content = question
    if collection:
        user_content = f"{question}\n\n[Document collection: {collection}]"

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_content)]}
    )

    # ── Parse the output messages ──────────────────────────────────────────
    # result["messages"] is the full conversation: HumanMessage, then
    # alternating AIMessage / ToolMessage until the final AIMessage answer.
    messages = result.get("messages", [])

    answer = ""
    sources: list[SourceChunk] = []
    used_retrieval = False

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not answer:
            # The last AIMessage is the final answer
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)

    for msg in messages:
        if isinstance(msg, ToolMessage):
            # Each ToolMessage represents one MCP tool execution
            used_retrieval = True
            raw = msg.content if isinstance(msg.content, str) else str(msg.content)
            sources.append(
                SourceChunk(
                    content=raw[:800],  # truncate to avoid bloated responses
                    metadata={
                        "tool_call_id": getattr(msg, "tool_call_id", ""),
                        "tool_name": getattr(msg, "name", "query_document"),
                    },
                )
            )

    return ChatResponse(
        answer=answer,
        sources=sources,
        used_retrieval=used_retrieval,
    )
