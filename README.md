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
| Embeddings | Google `gemini-embedding-001` | Beat Cohere `embed-v4.0` (the original pick) 11 of 13 contested questions, +0.093 MRR, after a full phase-by-phase re-review — see `ROADMAP.md` Phase 19 |
| Vector store | Pinecone (serverless) | Hosted, free tier, no infra to manage |
| Retrieval | Hybrid: Pinecone semantic + BM25 keyword, fused via Reciprocal Rank Fusion (k=5, w=0.90), plus a small hand-checked list of "only if" articles that get pulled in alongside the ones they gate | Beat pure semantic and pure keyword on recall/MRR; RRF was kept over a marginally-higher-scoring weighted-sum blend because it only needs rank position from independent top-k API calls, matching how retrieval actually runs in production |
| Generation | `gpt-4o-mini` via OpenRouter | Beat Cohere `command-a` and Qwen2.5-72B head-to-head on both citation accuracy and LLM-as-judge faithfulness/relevancy scoring, at ~17x lower cost |
| Calculations | Plain Python (`src/calc.py`), not the LLM | Contribution totals, end-of-service pay, overtime and leave accrual are arithmetic, and asking the model to do arithmetic itself was measured to be wrong a meaningful fraction of the time; the rates are still read from the retrieved article text every time, never hardcoded |
| Backend | FastAPI | `Retriever`/generator built once at startup, not per-request; rate-limited (`slowapi`) since every question costs real money in API calls |
| Frontend | Static HTML/CSS/vanilla JS | Per-message RTL detection so English and Arabic messages render correctly in the same conversation; no framework |

## Results

Measured against a 60-question hand-labeled eval set (`eval_v3`) spanning both
laws, both languages, and phrased the way a person would actually ask rather than
echoing the statute's own wording:

- **Retrieval**: recall@1 = 0.883, recall@5 = 1.000, MRR = 0.926
- **Generation**: mean LLM-judge faithfulness 4.70/5, mean relevancy 4.72/5
- A hosted reranker (Cohere `rerank-v3.5`) was tested and *not* adopted — it
  fixed a known cross-lingual retrieval gap but measurably hurt same-language
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
- `src/calc.py` — deterministic calculators (contributions, end-of-service, overtime, leave, pension eligibility) that read rates from the retrieved text rather than doing arithmetic in the prompt
- `src/api.py` — FastAPI backend + static frontend serving
- `frontend/` — chat UI
- `scripts/` — every decision notebook, numbered in the order the work happened;
  see [`scripts/README.md`](scripts/README.md) for what each one settled
- `eval/` — the eval sets (v1 40 questions, v2/v3 60) and every run's results
- `0. cleaned_data/` — extracted article tables and the chunked corpus
- `1. data/` — the source PDFs
- `notes/` — working notes, not part of the pipeline
- `EXPLAINER.md` — how the whole system works, from scratch, in plain language
- `LESSONS.md` — the mistakes made along the way and what each one taught
- `ROADMAP.md` — full phase-by-phase build log with reasoning for every choice

## Disclaimer

General legal information, not legal advice. Not affiliated with the Jordanian
government or any official body.
