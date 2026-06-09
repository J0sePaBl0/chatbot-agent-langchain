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
  When LANGCHAIN_TRACING_V2=true, every node in the LangGraph execution
  (LLM call, tool call, routing decision) is recorded as a span in LangSmith.
  No additional code is needed — the integration is automatic via env vars
  set in main.py before any langchain import.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
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

    # Open a session to the MCP server for the lifetime of this request.
    # The 'streamable_http' transport matches FastMCP's HTTP server mode.
    async with MultiServerMCPClient(
        {
            "rag-agent": {
                "url": settings.MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        }
    ) as client:
        tools = client.get_tools()

        # create_react_agent builds a LangGraph StateGraph with two nodes:
        #   1. "agent" — calls the LLM with the current messages
        #   2. "tools" — executes any tool calls the LLM requested
        # It loops until the LLM produces a final answer with no tool calls.
        graph = create_react_agent(
            llm,
            tools=tools,
            # messages_modifier prepends a SystemMessage to every invocation
            messages_modifier=SYSTEM_PROMPT,
        )

        # Inject the collection name into the question so the agent knows
        # which ChromaDB collection to pass to query_document.
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
