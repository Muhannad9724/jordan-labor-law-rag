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
from typing import Literal

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


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    # Keep in step with Retriever.retrieve()'s default -- this value shadows it on every
    # request, so leaving it at 10 would have silently cancelled the Phase 16 change.
    top_k: int = Field(default=5, ge=1, le=20)
    # Prior turns, oldest first, sent by the client. The server stays stateless -- no
    # session store, no cookie, nothing to expire or leak between users -- which keeps
    # this deployable on a single Railway container with no database. The cost is that
    # history is client-supplied and therefore untrusted; it only ever reaches the model
    # as conversational text, never the retriever's index or a citation check, and it is
    # length-capped both per turn (above) and in total (below).
    history: list[Turn] = Field(default_factory=list, max_length=20)


class Citation(BaseModel):
    article_number: str
    in_retrieved_context: bool


class Source(BaseModel):
    chunk_id: str
    law: str
    language: str
    article_number: str
    clause: str
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[Source]
    # What was actually sent to the retriever. Equals `question` on a first turn or when
    # the rewrite was skipped/failed. Returned because when a follow-up retrieves badly,
    # the rewrite is the first thing to inspect, and without this it is invisible.
    retrieval_query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute;100/day")
def ask(request: Request, req: AskRequest):
    retriever: Retriever = _state["retriever"]
    generator = _state["generator"]

    history = [m.model_dump() for m in req.history]

    # Rewrite BEFORE retrieving. A follow-up like "yes, I'm covered" carries no semantic
    # signal about pensions, so retrieving on it directly returns unrelated articles and
    # the answer is grounded in the wrong documents. Falls back to the raw question on
    # any failure, i.e. to the previous single-turn behaviour.
    retrieval_query = generator.rewrite_for_retrieval(req.question, history) if history else req.question

    chunks = retriever.retrieve(retrieval_query, top_k=req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="لم يتم العثور على مواد قانونية ذات صلة بهذا السؤال.")

    # The user's own words go to the generator, not the rewrite -- the rewrite exists to
    # find documents, and answering a machine-reworded version of the question tends to
    # drift from what was actually asked. History gives it whatever context it needs.
    result = generator.generate(req.question, chunks, history=history)

    return AskResponse(
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        sources=[
            Source(chunk_id=c["chunk_id"], law=c["law"], language=c["language"],
                   article_number=c["article_number"], clause=c["clause"], text=c["text"])
            for c in chunks
        ],
        retrieval_query=retrieval_query,
    )


class FollowupRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=8000)
    # The chunks /ask already returned. Sent as ids rather than as text so the payload
    # stays small and, more importantly, so the client cannot inject article text the
    # corpus does not contain -- the server resolves every id against its own index.
    chunk_ids: list[str] = Field(min_length=1, max_length=20)


@app.post("/followups")
@limiter.limit("15/minute;150/day")
def followups(request: Request, req: FollowupRequest):
    """Suggested next questions for an answer the user already has.

    Deliberately a second call rather than part of /ask: the answer renders the moment
    it is ready and the chips arrive a beat later, so suggestions cost the user no
    waiting. It also means a failure here cannot take the answer down with it.

    Always 200. An empty list is a valid, expected response -- the frontend simply
    shows no chips, and a suggestion feature is never worth surfacing an error for.
    """
    retriever: Retriever = _state["retriever"]
    generator = _state["generator"]

    by_id = retriever.chunks.set_index("chunk_id")
    chunks = []
    for cid in req.chunk_ids:
        if cid in by_id.index:
            row = by_id.loc[cid]
            chunks.append({"chunk_id": cid, "law": row["law"], "language": row["language"],
                           "article_number": str(row["article_number"]), "text": row["text"]})
    if not chunks:
        return {"suggestions": []}

    return {"suggestions": generator.suggest_followups(req.question, chunks, req.answer)}


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles sends etag/last-modified but no Cache-Control at all. Without an
    explicit directive, browsers apply RFC 7234 heuristic caching -- they may serve a
    file, including index.html itself, straight from disk with ZERO network request,
    not merely skip revalidation. That defeated the ?v=2 query-string cache-busting on
    app.js/style.css entirely: if the *HTML document containing the new ?v=2 references*
    is itself served from a heuristically-fresh cache, the browser never sees them and
    keeps running whatever JS it loaded last. `no-cache` still allows the etag/
    last-modified conditional GET (a fast 304 when nothing changed) -- it only forbids
    skipping the round trip, which is the failure mode this closes.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Mounted last -- StaticFiles would otherwise shadow the API routes above at "/".
app.mount("/", NoCacheStaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
