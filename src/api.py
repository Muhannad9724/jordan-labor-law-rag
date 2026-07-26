"""FastAPI backend: wraps Retriever -> Generator behind an HTTP endpoint.

Retriever and the production generator are expensive to construct (loads all 739
chunks, builds the BM25 index, connects to Pinecone/Cohere/OpenRouter) so they're
built once at startup via the lifespan handler, not per-request.

Run locally: uvicorn src.api:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.generation import get_production_generator
from src.retrieval import Retriever

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["retriever"] = Retriever()
    _state["generator"] = get_production_generator()
    yield
    _state.clear()


app = FastAPI(title="Jordan Labor & Social Security Law RAG API", lifespan=lifespan)

# Permissive for now -- tighten to the actual frontend origin once Phase 10/11 pick one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=20)


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
def ask(req: AskRequest):
    retriever: Retriever = _state["retriever"]
    generator = _state["generator"]

    chunks = retriever.retrieve(req.question, top_k=req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant law articles found for this question.")

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
