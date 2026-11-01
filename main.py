"""Private RAG agent. Run with: python -m uvicorn main:app --port 3000"""
import settings  # Load .env before creating model clients.
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from pypdf.errors import PdfReadError
from uploads import MAX_FILE_BYTES, upload_name, save_upload
from fastapi.responses import RedirectResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag import Agent


ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Local RAG Agent", version="0.1.0")


class IngestRequest(BaseModel):
    path: str = Field(min_length=1, description="Local file or folder; relative paths use the project directory.")


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=6)
    min_score: float = Field(default=0.25, ge=-1, le=1)
    model_mode: Literal['auto', 'fast', 'thorough'] = 'auto'


@contextmanager
def agent_session():
    db = Path(os.getenv("RAG_DB", "data/index.sqlite3"))
    agent = None
    try:
        agent = Agent(db if db.is_absolute() else ROOT / db)
        yield agent
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if agent is not None:
            agent.db.close()


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    """Service liveness only; does not check Ollama readiness."""
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest):
    path = Path(request.path)
    with agent_session() as agent:
        count = agent.ingest(path if path.is_absolute() else ROOT / path)
    return {"indexed_documents": count}



@app.post('/upload')
async def upload(request: Request):
    # Node parses browser multipart data and forwards the bounded file as raw bytes.
    if request.headers.get('content-type', '').split(';')[0].lower() != 'application/octet-stream':
        raise HTTPException(status_code=415, detail='Use application/octet-stream.')
    try:
        name = upload_name(request.headers.get('x-filename'))
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    declared = request.headers.get('content-length')
    if declared and declared.isdigit() and int(declared) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail='File exceeds the 10 MB limit.')
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail='File exceeds the 10 MB limit.')
        content.extend(chunk)
    if not content:
        raise HTTPException(status_code=400, detail='The attached file is empty.')

    def index_upload():
        with agent_session() as agent:
            target = save_upload(ROOT / 'data' / 'uploads', name, content)
            try:
                count = agent.ingest(target)
            except PdfReadError as exc:
                raise HTTPException(status_code=400, detail='The attached PDF is invalid or unreadable.') from exc
            return {'indexed_documents': count, 'filename': name}

    return await run_in_threadpool(index_upload)


@app.post("/search")
def search(request: QuestionRequest):
    with agent_session() as agent:
        return {"sources": agent.retrieve(request.question, request.top_k, request.min_score)}


@app.post("/ask")
def ask(request: QuestionRequest):
    with agent_session() as agent:
        try:
            result = agent.ask(request.question, request.top_k, request.min_score, request.model_mode)
            return {**result, "token_usage": agent.client.usage}
        except (ValueError, RuntimeError, OSError) as exc:
            return JSONResponse(status_code=502 if isinstance(exc, RuntimeError) else 400,
                                content={"detail": str(exc), "token_usage": agent.client.usage})
