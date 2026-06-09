"""FastMCP server — exposes query_document as an MCP tool.

Run standalone:
    python -m app.mcp.server

In Docker this is the `mcp` service's entrypoint command.
External clients (langchain-mcp-adapters, Claude Desktop, etc.) connect via HTTP.
"""
from fastmcp import FastMCP

from app.core.retrieval import query_collection

mcp = FastMCP("rag-agent")


@mcp.tool()
def query_document(question: str, collection: str) -> str:
    """Search an uploaded document collection stored in ChromaDB.

    Args:
        question: The question or search query to find relevant content for.
        collection: The ChromaDB collection name returned by POST /upload.

    Returns:
        Formatted string of the most relevant chunks with source citations.
        Returns a 'not found' message if the collection is empty or missing.
    """
    docs = query_collection(question=question, collection_name=collection, k=4)

    if not docs:
        return f"No relevant documents found in collection '{collection}'."

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i}] Source: {source} | Page: {page}\n{doc.page_content}")

    # Chunks separated by a horizontal rule so the LLM can clearly distinguish
    # where one chunk ends and the next begins
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    # host="0.0.0.0" is required in Docker — binding to 127.0.0.1 would make
    # the port unreachable from other containers or the host machine
    mcp.run(transport="http", host="0.0.0.0", port=9001)
