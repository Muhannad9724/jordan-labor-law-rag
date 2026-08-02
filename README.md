# Jordan Labor & Social Security Law Assistant

A bilingual (Arabic/English) RAG application answering questions about Jordan's
Labor Law (1996), its 2019 Professional Work Law amendment, and the Social
Security Law — built end-to-end from raw PDFs to a deployed web app, with every
architectural decision made empirically rather than by default.

## What it does

Ask a question in English or Arabic ("How much notice does my employer need to
give before ending my contract?" / "كم يوم إجازة سنوية مدفوعة الأجر أستحق؟") and
get an answer grounded in the actual law articles, with citations back to the
specific law and article number, and the retrieved source text shown alongside
the answer.

## Architecture

```
PDFs → extraction/OCR → chunking → embeddings → vector store
                                                       │
question ──► hybrid retrieval (semantic + keyword) ◄──┘
                        │
                        ▼
              LLM generation (grounded, cited)
                        │
                        ▼
              FastAPI backend ──► static frontend
```

| Stage | Choice | Why |
|---|---|---|
| Embeddings | Cohere `embed-v4.0` | Best same-language accuracy of 5 candidates tested (self-hosted and hosted), negligible cost at this scale |
| Vector store | Pinecone (serverless) | Hosted, free tier, no infra to manage |
| Retrieval | Hybrid: Pinecone semantic + BM25 keyword, fused via Reciprocal Rank Fusion (k=5, w=0.85) | Beat pure semantic and pure keyword on recall/MRR across 40 hand-written eval questions; chosen over a marginally-higher-scoring weighted-sum blend because RRF only needs rank position from independent top-k API calls, matching how retrieval actually runs in production |
| Generation | `gpt-4o-mini` via OpenRouter | Beat Cohere `command-a` and Qwen2.5-72B head-to-head on both citation accuracy and LLM-as-judge faithfulness/relevancy scoring, at ~17x lower cost |
| Backend | FastAPI | `Retriever`/generator built once at startup, not per-request; rate-limited (`slowapi`) since every question costs real money in API calls |
| Frontend | Static HTML/CSS/vanilla JS | Per-message RTL detection so English and Arabic messages render correctly in the same conversation; no framework |

## Results

Measured against a 40-question hand-labeled eval set spanning both laws, both
languages, and cross-lingual questions:

- **Retrieval**: recall@5 = 0.825, recall@10 = 0.900, MRR = 0.744
- **Generation**: ~87.5% correct-article citation rate, mean LLM-judge
  faithfulness 4.9/5, mean relevancy 4.8/5
- A hosted reranker (Cohere `rerank-v3.5`) was tested and *not* adopted — it
  fixed the known cross-lingual retrieval gap but measurably hurt same-language
  precision (the 80% majority case), traced to the reranker pattern-matching on
  boilerplate phrasing shared across unrelated law articles

Full methodology, including every option that was tested and rejected and why,
is in [`ROADMAP.md`](ROADMAP.md) and the notebooks in [`scripts/`](scripts/).

## Running locally

```bash
pip install -r requirements.txt
# .env with GEMINI_API_KEY, PINECONE_API_KEY, OPENROUTER_API_KEY
# (COHERE_API_KEY only if you roll retrieval back to the Cohere index)
uvicorn src.api:app --reload --port 8000
```

Open `http://127.0.0.1:8000` — the backend serves the frontend directly.

## Project structure

- `src/retrieval.py` — hybrid retriever (Pinecone + BM25 + RRF)
- `src/generation.py` — grounded answer generation with citation extraction
- `src/api.py` — FastAPI backend + static frontend serving
- `frontend/` — chat UI
- `scripts/` — every decision notebook, numbered in the order the work happened;
  see [`scripts/README.md`](scripts/README.md) for what each one settled
- `eval/` — the eval sets (v1 40 questions, v2/v3 60) and every run's results
- `0. cleaned_data/` — extracted article tables and the chunked corpus
- `1. data/` — the source PDFs
- `notes/` — working notes, not part of the pipeline
- `ROADMAP.md` — full phase-by-phase build log with reasoning for every choice

## Disclaimer

General legal information, not legal advice. Not affiliated with the Jordanian
government or any official body.
