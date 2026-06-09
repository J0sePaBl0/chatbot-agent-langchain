"""Unit tests for the RAG pipeline.

These tests cover individual components and never hit external services
(OpenAI, ChromaDB, MCP server) — all IO is mocked. This makes them fast,
deterministic, and safe to run in CI without credentials.

Run with:
    pytest tests/ -v
or inside Docker:
    docker compose run --rm api pytest tests/ -v
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import ChatResponse, SourceChunk, UploadResponse


# ── Test 1: Text splitter produces correctly bounded chunks ──────────────────

def test_text_splitter_chunk_size():
    """RecursiveCharacterTextSplitter must keep all chunks within bounds.

    We allow up to chunk_size + chunk_size * 0.2 to account for the splitter
    preferring natural boundaries (newlines, sentences) over hard cuts.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    long_text = "This is a test sentence about financial risk and regulatory frameworks. " * 200
    doc = Document(page_content=long_text, metadata={"source": "test.pdf", "page": 0})
    chunks = splitter.split_documents([doc])

    assert len(chunks) > 1, "A 14k-char document must be split into multiple chunks"
    for chunk in chunks:
        # Hard limit: no chunk should be more than 20% above chunk_size
        assert len(chunk.page_content) <= 1200, (
            f"Chunk exceeds expected size: {len(chunk.page_content)} chars"
        )


# ── Test 2: Retrieval converts ChromaDB results to LangChain Documents ───────

def test_query_returns_documents():
    """query_collection must convert ChromaDB query results into Document objects.

    We mock PersistentClient so no real database is needed — this tests the
    transformation logic, not ChromaDB itself.
    """
    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.query.return_value = {
        "documents": [
            [
                "Quarterly revenue rose 12% to $2.4B driven by institutional demand.",
                "Risk factors include exposure to interest rate volatility.",
            ]
        ],
        "metadatas": [
            [
                {"source": "annual-report.pdf", "page": 0},
                {"source": "annual-report.pdf", "page": 5},
            ]
        ],
        "distances": [[0.12, 0.22]],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection

    with patch("app.core.retrieval.chromadb.PersistentClient", return_value=mock_client):
        from app.core.retrieval import query_collection

        results = query_collection("What was quarterly revenue?", "annual-report", k=4)

    assert len(results) == 2
    assert isinstance(results[0], Document)
    assert "2.4B" in results[0].page_content
    assert results[0].metadata["source"] == "annual-report.pdf"
    assert results[0].metadata["page"] == 0
    assert results[1].metadata["page"] == 5


# ── Test 3: ChatResponse Pydantic model serializes correctly ─────────────────

def test_chat_response_schema():
    """ChatResponse must serialize all fields correctly via model_dump().

    Tests that Pydantic accepts the expected types and that the serialized
    dict has the contract the frontend/API consumers depend on.
    """
    response = ChatResponse(
        answer="Revenue rose 12% in Q3 to $2.4B (Source: annual-report.pdf, Page 1).",
        sources=[
            SourceChunk(
                content="Q3 earnings were $2.4B, up 12% year-over-year.",
                metadata={"source": "annual-report.pdf", "page": 1},
            )
        ],
        used_retrieval=True,
    )

    assert isinstance(response.answer, str)
    assert isinstance(response.sources, list)
    assert isinstance(response.used_retrieval, bool)
    assert response.used_retrieval is True
    assert response.sources[0].metadata["page"] == 1

    serialized = response.model_dump()
    assert set(serialized.keys()) == {"answer", "sources", "used_retrieval"}
    assert serialized["sources"][0]["content"].startswith("Q3 earnings")


# ── Test 4: Collection name sanitization matches ChromaDB constraints ────────

def test_sanitize_collection_name():
    """sanitize_collection_name must produce valid ChromaDB collection names.

    ChromaDB enforces: 3–63 chars, [a-zA-Z0-9_-], no leading/trailing hyphens.
    """
    from app.core.ingestion import sanitize_collection_name

    assert sanitize_collection_name("Annual Report 2024.pdf") == "annual-report-2024"
    assert sanitize_collection_name("My Document.pdf") == "my-document"

    # Names shorter than 3 chars must be padded
    short = sanitize_collection_name("ab.pdf")
    assert len(short) >= 3, f"Sanitized name too short: '{short}'"

    # Names longer than 63 chars must be truncated
    long_name = "a" * 100 + ".pdf"
    result = sanitize_collection_name(long_name)
    assert len(result) <= 63, f"Sanitized name too long: {len(result)} chars"

    # Must not have leading or trailing hyphens
    assert not result.startswith("-"), "Collection name must not start with hyphen"
    assert not result.endswith("-"), "Collection name must not end with hyphen"
