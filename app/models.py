from pydantic import BaseModel


class UploadResponse(BaseModel):
    collection_name: str
    chunks_stored: int
    message: str


class ChatRequest(BaseModel):
    question: str
    # Optional: if omitted, the agent answers from its own knowledge.
    # When provided, the agent knows which ChromaDB collection to search.
    collection: str | None = None


class SourceChunk(BaseModel):
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    answer: str
    # Populated only when the agent actually called the query_document tool
    sources: list[SourceChunk]
    # Lets the client know whether retrieval was triggered — useful for UI
    # indicators ("answered from document" vs "answered from LLM knowledge")
    used_retrieval: bool
