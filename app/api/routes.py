from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.agent import run_agent
from app.core.ingestion import ingest_pdf
from app.models import ChatRequest, ChatResponse, UploadResponse

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a PDF, ingest it into ChromaDB, and return the collection name.

    The collection name is derived from the filename and must be passed to
    POST /chat so the agent knows which ChromaDB collection to query.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        collection_name, chunks_stored = await ingest_pdf(file)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return UploadResponse(
        collection_name=collection_name,
        chunks_stored=chunks_stored,
        message=f"Successfully ingested {chunks_stored} chunks into '{collection_name}'.",
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a question to the LangChain agent and receive a grounded answer.

    If `collection` is provided, the agent will call the query_document MCP tool
    to retrieve relevant chunks before answering.  If omitted, the agent answers
    from its own knowledge.

    The response includes the answer, any retrieved source chunks, and a boolean
    indicating whether retrieval was used — useful for building UI indicators.
    """
    try:
        response = await run_agent(question=request.question, collection=request.collection)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return response
