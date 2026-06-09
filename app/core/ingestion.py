import os
import tempfile

import chromadb
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def sanitize_collection_name(filename: str) -> str:
    """Convert a filename to a valid ChromaDB collection name.

    ChromaDB rules: 3–63 chars, [a-zA-Z0-9_-], no leading/trailing hyphens.
    """
    stem = os.path.splitext(filename)[0]
    safe = "".join(c if c.isalnum() else "-" for c in stem.lower())
    safe = safe.strip("-")
    # Pad short names — ChromaDB rejects names shorter than 3 characters
    if len(safe) < 3:
        safe = (safe + "---")[:3]
    return safe[:63]


async def ingest_pdf(upload_file: UploadFile) -> tuple[str, int]:
    """Load a PDF, chunk it, embed it, and store it in ChromaDB.

    Returns the collection name and the number of chunks stored.
    Each PDF goes into its own collection so queries are always scoped
    to a single document — this prevents cross-document retrieval noise.
    """
    collection_name = sanitize_collection_name(upload_file.filename or "document")

    # PyPDFLoader requires a real filesystem path, not a file-like object.
    # delete=False is mandatory on Windows: the OS holds an exclusive lock on
    # NamedTemporaryFile while it's open, which blocks PyPDFLoader from opening
    # the same path. We close the handle first, then let the loader open it.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await upload_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )

    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    # cosine distance is standard for semantic similarity with OpenAI embeddings
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"{collection_name}-chunk-{i}" for i in range(len(texts))]

    # Embed via OpenAI, then store vectors + text + metadata together.
    # ChromaDB can embed on insert but using our own embeddings model gives us
    # explicit control over the model version and cost.
    embedded_vectors = embeddings.embed_documents(texts)
    collection.add(
        documents=texts,
        embeddings=embedded_vectors,
        metadatas=metadatas,
        ids=ids,
    )

    return collection_name, len(chunks)
