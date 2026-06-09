import chromadb
from langchain_core.documents import Document

from app.config import settings


def query_collection(
    question: str,
    collection_name: str,
    k: int = 4,
) -> list[Document]:
    """Query a ChromaDB collection and return matching chunks as LangChain Documents.

    Returns an empty list if the collection doesn't exist or is empty —
    the caller decides how to handle missing documents rather than raising here.
    """
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        # ChromaDB raises ValueError if the collection doesn't exist
        return []

    count = collection.count()
    if count == 0:
        return []

    # Guard: n_results must not exceed the number of stored documents.
    # ChromaDB raises an error rather than clamping silently.
    n_results = min(k, count)

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents: list[Document] = []
    if results["documents"] and results["documents"][0]:
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            documents.append(
                Document(page_content=text, metadata=meta or {})
            )

    return documents
