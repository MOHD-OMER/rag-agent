"""
FastAPI Backend for the Multi-Document RAG Chatbot.

Endpoints:
  POST /upload       — Upload and ingest documents (PDF, TXT, DOCX)
  POST /chat         — Send message, get streamed SSE response
  GET  /history      — Retrieve chat history for a session
  DELETE /clear      — Clear memory and optionally wipe vector store
  GET  /health       — Health check
  GET  /stats        — System stats (doc count, sessions, etc.)
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Ensure backend/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings
from memory import get_memory_manager, ChatMessage
from rag_pipeline import run_agent, run_agent_stream, clear_checkpoint
from retriever import get_retriever

# ============================================================
# Logging
# ============================================================
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Document Ingestion
# ============================================================

def ingest_file(file_path: str, filename: str) -> int:
    """Load, chunk, and index a single file. Returns chunk count."""
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredWordDocumentLoader,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    ext = Path(filename).suffix.lower()
    loader_map = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
    }

    loader_cls = loader_map.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")

    loader = loader_cls(file_path)
    raw_docs = loader.load()

    # Attach clean source name
    for doc in raw_docs:
        doc.metadata["source"] = filename
        doc.metadata.setdefault("page", doc.metadata.get("page", 0))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    retriever = get_retriever()
    count = retriever.add_documents(chunks)
    logger.info("Ingested %s → %d chunks", filename, count)
    return count


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Multi-Document RAG Chatbot API",
    description="Agentic RAG system with hybrid retrieval, LangGraph orchestration, and streaming",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Schemas
# ============================================================

class UploadResponse(BaseModel):
    status: str
    filename: str
    chunks_added: int
    total_chunks: int


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[dict]
    tool_calls: List[str]


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    turn_count: int


class StatsResponse(BaseModel):
    total_documents: int
    active_sessions: int
    llm_provider: str
    embedding_model: str


class ClearResponse(BaseModel):
    status: str
    cleared_memory: bool
    cleared_vectorstore: bool


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Return system statistics."""
    retriever = get_retriever()
    memory_manager = get_memory_manager()
    return StatsResponse(
        total_documents=retriever.document_count,
        active_sessions=memory_manager.active_sessions,
        llm_provider=settings.llm_provider,
        embedding_model=settings.embedding_model,
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, or DOCX file"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Upload a document and ingest it into the vector store.
    Supports PDF, TXT, and DOCX formats.
    """
    allowed_extensions = {".pdf", ".txt", ".docx", ".doc"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}",
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks_added = ingest_file(tmp_path, file.filename)
        retriever = get_retriever()

        return UploadResponse(
            status="success",
            filename=file.filename,
            chunks_added=chunks_added,
            total_chunks=retriever.document_count,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Upload error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/chat", tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message and receive a response from the RAG agent.

    Supports both streaming (SSE) and non-streaming modes.
    - stream=True: Returns Server-Sent Events for real-time token streaming
    - stream=False: Returns complete response as JSON
    """
    session_id = request.session_id or str(uuid.uuid4())
    memory_manager = get_memory_manager()
    session = memory_manager.get_or_create(session_id)

    if request.stream:
        async def event_generator():
            import json
            # First send the session_id
            yield f"data: {json.dumps({'type': 'session_id', 'session_id': session_id})}\n\n"
            async for chunk in run_agent_stream(request.message, session, session_id):
                yield chunk

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Session-ID": session_id,
            },
        )
    else:
        # FIX: await the async run_agent directly — no asyncio.run()
        result = await run_agent(request.message, session, session_id)
        return ChatResponse(
            answer=result["answer"],
            session_id=session_id,
            sources=result["sources"],
            tool_calls=result["tool_calls"],
        )


@app.get("/history", response_model=HistoryResponse, tags=["Chat"])
async def get_history(
    session_id: str = Query(..., description="Session ID to retrieve history for")
):
    """Retrieve the full conversation history for a session."""
    memory_manager = get_memory_manager()
    session = memory_manager.get(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return HistoryResponse(
        session_id=session_id,
        messages=session.get_history(),
        turn_count=session.turn_count,
    )


@app.delete("/clear", response_model=ClearResponse, tags=["Chat"])
async def clear(
    session_id: Optional[str] = Query(None, description="Session to clear (omit to clear all)"),
    clear_vectorstore: bool = Query(False, description="Also wipe the vector store"),
):
    """
    Clear conversation memory and optionally the vector store.

    - session_id=None: clears all sessions
    - clear_vectorstore=True: also deletes all indexed documents
    """
    memory_manager = get_memory_manager()

    cleared_memory = False
    if session_id:
        cleared_memory = memory_manager.delete_session(session_id)
        clear_checkpoint(session_id)
    else:
        for sid in memory_manager.list_sessions():
            memory_manager.delete_session(sid)
        clear_checkpoint(None)  # Clear all
        cleared_memory = True

    cleared_vectorstore = False
    if clear_vectorstore:
        get_retriever().clear()
        cleared_vectorstore = True

    return ClearResponse(
        status="success",
        cleared_memory=cleared_memory,
        cleared_vectorstore=cleared_vectorstore,
    )


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )