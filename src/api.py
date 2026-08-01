"""FastAPI backend: wraps Retriever -> Generator behind an HTTP endpoint, and
serves the static frontend from the same origin (avoids CORS entirely).

Retriever and the production generator are expensive to construct (loads all 739
chunks, builds the BM25 index, connects to Pinecone/Cohere/OpenRouter) so they're
built once at startup via the lifespan handler, not per-request.

Rate limiting (slowapi, in-memory) guards /ask specifically -- this is a public
demo with no auth, and each question costs real money (Cohere embed + Pinecone
query + OpenRouter generation), so an unrestricted endpoint is a real cost risk,
not just an abuse-prevention nicety.

Run locally: uvicorn src.api:app --reload --port 8000
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.generation import get_production_generator
from src.retrieval import Retriever

PROJECT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT / "frontend"

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["retriever"] = Retriever()
    _state["generator"] = get_production_generator()
    yield
    _state.clear()


app = FastAPI(title="Jordan Labor & Social Security Law RAG API", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "عدد الأسئلة تجاوز الحد المسموح — يُرجى الانتظار قليلاً ثم المحاولة مجدداً."},
    )


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    # Keep in step with Retriever.retrieve()'s default -- this value shadows it on every
    # request, so leaving it at 10 would have silently cancelled the Phase 16 change.
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    article_number: str
    in_retrieved_context: bool


class Source(BaseModel):
    law: str
    language: str
    article_number: str
    clause: str
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute;100/day")
def ask(request: Request, req: AskRequest):
    retriever: Retriever = _state["retriever"]
    generator = _state["generator"]

    chunks = retriever.retrieve(req.question, top_k=req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="لم يتم العثور على مواد قانونية ذات صلة بهذا السؤال.")

    result = generator.generate(req.question, chunks)

    return AskResponse(
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        sources=[
            Source(law=c["law"], language=c["language"], article_number=c["article_number"],
                   clause=c["clause"], text=c["text"])
            for c in chunks
        ],
    )


# Mounted last -- StaticFiles would otherwise shadow the API routes above at "/".
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
