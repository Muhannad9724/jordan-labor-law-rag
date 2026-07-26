# Roadmap: source PDFs → live public RAG app

End goal: a publicly launched RAG app (live URL) answering questions on Jordan's Social Security Law and Work Law (1996 + Professional 2019), bilingual (Arabic/English). See `CLAUDE.md` for the deployment-goal constraint that should shape every choice below.

## Phase 0 — Source data ✅ done
6 PDFs acquired (SCL + Work Law 1996 + Professional Work Law 2019, each EN/AR).

## Phase 1 — Extraction & cleaning ✅ done
PyMuPDF (EN) + Gemini vision OCR (AR) → article-level parsing. Fixed several real defects along the way (duplicate-key content loss, false-positive headers from cross-references, chapter/footnote noise, wrong-language header word, a missing article header, an OCR digit misread). Output: 6 `Knowledgebase/*.xlsx` tables, all reproducible from `scripts/files_text_extraction.ipynb`.

**Open issue:** Professional Work Law EN (unamended 2019) and AR (amended-as-of-2024) aren't fully parallel — undecided, currently just documented.

## Phase 2 — Chunking ✅ done
`0. cleaned_data/chunks.parquet`, 739 chunks, clause-aware splitting for long articles, in `scripts/chunking.ipynb`. Caught and fixed a regex gap (EN hyphen-style clause markers) and a duplicate-chunk-id bug during review.

**Known limitation:** 3 "definitions" glossary articles have no lettered clauses to split on, so they stay oversized (>3000 chars).

## Phase 3 — Eval set ✅ done
`eval/eval_draft.jsonl` + `.csv`, 40 hand-drafted bilingual Q&A pairs sampled across all 3 laws, balanced 16 ar/ar + 16 en/en + 8 cross-lingual, with `question_category`/`difficulty`/`labeling_note`/`source_article_excerpt` fields. Reviewed and revised for tone (plain, how-a-worker-would-ask phrasing) and topic relevance (leave/pay/termination concerns over institutional trivia). This is the ground truth for measuring the pipeline in Phase 8.

## Phase 4 — Embeddings ✅ done
Tested 5 candidates against the real corpus (739 chunks) and eval set (40 questions) in `scripts/embeddings_evaluation.ipynb`: self-hosted `Qwen3-Embedding-0.6B`, `bge-m3`, `multilingual-e5-large`, plus hosted `cohere-embed-v4` and `openai-text-embedding-3-small`. **Decision: Cohere `embed-v4.0`** — clearly best same-language accuracy (Recall@10 = 1.000, Recall@1 = 0.812) and best overall MRR; initial lean toward bge-m3 for cross-lingual strength didn't survive scrutiny (the two supporting content-gap examples were either wrong — SCL 85-bis exists in both languages, a stale note — or not actually a retrieval problem — the Professional Work Law EN/AR divergence is a stale-translation issue no embedding model fixes). Cost checked against the real corpus: ~$0.02 one-time to index, ~$0.000003/query ongoing — negligible at this project's scale. Full results table and reasoning are documented in the notebook itself.

## Phase 5 — Vector store & indexing ✅ done
**Decision: Pinecone** (serverless free tier, AWS `us-east-1`, cosine, dim=1536). Chosen over Qdrant Cloud primarily for resume value — it shows up most frequently by name in job postings for RAG/vector-DB experience; SaaS-only (no self-host) was a non-issue since this project deploys to the cloud anyway. All 739 chunks embedded with Cohere `embed-v4.0` and upserted with metadata (`law`, `language`, `article_number`, `clause`, `text`, `source_file`) in `scripts/vector_store_indexing.ipynb` — the full chunk text lives in Pinecone metadata, so no separate document store is needed. Sanity-checked against 3 real eval questions; correct articles came back in the top-5. One thing worth remembering: right after a bulk upsert, Pinecone serverless can return slightly out-of-order scores for a short window (indexing consistency lag) — settles within about a minute, not a retrieval bug.

## Phase 6 — Retrieval ✅ done
**Search-method decision made:** hybrid, beta=0.9 (90% semantic Cohere `embed-v4.0` cosine similarity, pulled straight from the Pinecone index by id, + 10% BM25 keyword score with light Arabic normalization, both min-max normalized per query before combining) — tested in `scripts/search_method_evaluation.ipynb` by sweeping beta from 0.0 (pure keyword) to 1.0 (pure semantic) over the same 40 eval questions. Beat pure semantic on every overall metric (recall@1 0.675 vs 0.650, recall@10 0.925 vs 0.900, MRR 0.748 vs 0.732) while matching it cross-lingual; pure keyword alone is far behind and cross-lingually blind (BM25 has zero token overlap across languages by construction). The cross-lingual recall@1 = 0.0 result across every beta is a pre-existing Phase 4 embedding limitation, not something the search method changes.

Also cross-checked against RRF (Reciprocal Rank Fusion, a rank-based fusion method that needs no score normalization), sweeping k over 10 values (1-100) × weight 0.0-1.0 in 0.05 steps (210 configs) — best RRF config (k=5, w=0.85) landed close to weighted-sum beta=0.9 (recall@1/5 identical at 0.675/0.825; weighted-sum ahead on recall@10 0.925 vs 0.900 and MRR 0.748 vs 0.744), confirming beta=0.9 wasn't a min-max-normalization artifact.

**Final production choice: RRF, k=5, w=0.85** — not the marginally-higher-scoring weighted-sum. Reason is architectural: the eval computed weighted-sum's normalization over the full 739-chunk score matrix, but the real retrieval function only gets a top-k list from Pinecone and a top-k list from BM25 (not all 739 scores), and min-max normalizing over a partial, per-query-varying slice is unstable. RRF only needs rank position within whatever list it got, which matches how two independent top-k API calls actually work. The eval gap (one question out of 40) doesn't outweigh that implementation-robustness argument.

Also tuned BM25's own `k1`/`b` parameters (25-combo grid, RRF held fixed at k=5/w=0.85): `rank_bm25`'s defaults (k1=1.5, b=0.75) were already optimal for this corpus — `k1` had zero measurable effect (BM25 is only 15% of the RRF weight, so its exact scores barely matter, only rank order does), `b=0.75` beat both extremes (full or no length normalization measurably hurt). No change made.

**Retrieval function built:** `src/retrieval.py` (`Retriever` class) — embeds the query with Cohere, pulls top-50 candidates each from Pinecone (semantic) and local BM25 (keyword), RRF-fuses them (k=5, w=0.85), returns top-k chunks with full metadata. Reusable/importable, not locked in a notebook, so Phase 9's FastAPI backend can call it directly. Validated in `scripts/retrieval.ipynb`: running it over all 40 eval questions reproduces the eval notebook's numbers exactly (recall@1 0.675, recall@5 0.825, recall@10 0.900, MRR 0.744) — confirms capping each ranker to its top 50 candidates before fusing (required since real Pinecone/BM25 queries return top-k, not full-corpus scores) doesn't lose anything versus the full-corpus eval.

The cross-lingual eval questions still matter for regression-checking retrieval quality — Cohere's cross-lingual Recall@10 (0.625) is real and worth keeping an eye on, not zero. Optional/deferred: reranking, metadata-scoped search.

## Phase 7 — Generation / answer synthesis ✅ done
Initial pick was Cohere `command-a-03-2025` (`scripts/generation.ipynb`, native `documents`/`citation_options` grounding, spot-checked on 4 cases, all correct) — but that choice was made without comparing it against anything, unlike every other phase in this project. Went back and actually tested it in `scripts/generation_llm_comparison.ipynb` against OpenAI `gpt-4o-mini` and Qwen2.5-72B (via OpenRouter/DeepInfra) on 10 real eval questions + 1 out-of-domain honesty trap, scored on correct-article-citation rate and a fabricated-citation check (does the answer mention an article number that wasn't even in the retrieved context):

| Model | Correct article (10 q) | Fabricated citations | Cost/question |
|---|---|---|---|
| Cohere `command-a-03-2025` | 9/10 | 1 | ~$0.007 |
| **OpenAI `gpt-4o-mini`** | **10/10** | **0** | **~$0.0004** |
| Qwen 2.5-72B | 9/10 | 1 | ~$0.0008 |

Citation-matching is a shallow proxy, not real faithfulness/relevancy — it only checks whether the right article *number* appears, not whether every claim in the answer is actually backed by the source text, and it says nothing about whether the answer addresses the question at all. Went back and added a proper **LLM-as-judge** pass (Claude Sonnet 5 via OpenRouter, deliberately independent of all 3 candidates to avoid self-preference bias) scoring Faithfulness and Relevancy 1-5 on every answer:

| Model | Mean faithfulness (min) | Mean relevancy (min) |
|---|---|---|
| Cohere `command-a-03-2025` | 4.82 (3) | 4.73 (2) |
| **OpenAI `gpt-4o-mini`** | **5.00 (5)** | 4.91 (4) |
| Qwen 2.5-72B | 4.64 (3) | 4.91 (4) |

This surfaced two real defects the citation check was blind to: Cohere's relevancy dropped to 2/5 on the out-of-domain trap question (judge: volunteering unrelated Jordan minimum-wage info "substitutes unrelated information" for what was actually asked, confirming that behavior was a real flaw, not just a style quirk); Qwen produced a stray Chinese sentence in one answer, an outright generation-quality bug unrelated to citations.

**Decision: `gpt-4o-mini`, routed through OpenRouter** (`openai/gpt-4o-mini` — same $0.15/$0.60-per-1M price as calling OpenAI directly, OpenRouter passes through provider rates with no token markup, so this just consolidates all chat-model traffic behind one key instead of splitting between OpenAI and OpenRouter for no benefit). Wins on every metric measured — perfect citation correctness, zero fabrication, best and most consistent faithfulness (never below 5), competitive relevancy, and ~17x cheaper than the original Cohere pick. Trade-off: loses Cohere's native structured-citation API in favor of manual citation-prompting (chunks formatted into the prompt, citations extracted from the answer text via regex) — empirically that cost nothing here, but it's a real design difference, not just a cost swap. Caveat: 10 questions is real signal, not proof; worth confirming on the full Phase 8 run.

Implementation: `src/generation.py` — `get_production_generator()` is the single entry point (GPT-4o-mini via OpenRouter); `Generator` (Cohere) and `OpenAICompatibleGenerator` (OpenAI-compatible, used for both GPT-4o-mini and Qwen) both still live in the file for the comparison. System prompt enforces: answer only from provided documents, admit honestly when they don't have the answer, cite law + article, match the question's language, close with a "not legal advice" disclaimer. One infra fix along the way: OpenRouter's automatic provider fallback for the Qwen test was routing to a broken provider after DeepInfra rate-limited — fixed by pinning the request to DeepInfra only and adding proper retry-with-backoff around both empty responses and thrown exceptions.

## Phase 8 — Evaluation harness ✅ done
Ran all 40 eval questions through the full production pipeline (`Retriever` → `get_production_generator()`) in `scripts/evaluation_harness.ipynb`. **Retrieval: exact match with Phase 6** — recall@1 0.675, recall@5 0.825, recall@10 0.900, MRR 0.744, no regression at full scale.

**Generation (regex proxy): 33/40 (82.5%) correct-article rate, 2/40 fabricated citations** — lower than the 10/10 in the Phase 7 sample. Traced every one of the 7 misses individually (re-queried the retriever directly rather than trusting a first manual read, which had errors — see notebook for the corrected version): only **1 is a genuine generation defect** (q07 — cited the wrong article despite the right chunk retrieved at rank 1) and **1 a genuine retrieval miss** (q22 — Article 90 never surfaced in top 10, model substituted a related-but-wrong article). The other 5 aren't real defects: 3 were retrieved but outside production `top_k=5` (the known recall@5-vs-10 gap, q25/q29/q31), and 2 are a **regex false negative** — `ARTICLE_NUM_RE` in `src/generation.py` doesn't match the "للمادة" elided-prefix form of "according to article" (q08/q40 both cited correctly; the model was right, the proxy metric missed it). True correct-article rate is ~87.5% once that's accounted for; not yet patched, since fixing it would mean re-validating the earlier Phase 7 comparison numbers too.

**LLM-as-judge (Claude Sonnet 5 via OpenRouter), full 40**: mean faithfulness 4.90 (min 3), mean relevancy 4.80 (min 2), consistent with the Phase 7 sample (5.00/4.91). 3 questions scored ≤3 — all Social Security Law pension questions (q22, q29, q35) where the model reached for a related-but-wrong provision among adjacent articles (62/71/80/85/90). Separately, **q09 and q21** each fabricated one extra, ungrounded citation despite correctly citing the right article from properly-retrieved context (q21: correctly cited Article 53 at rank 1, but also added Article 90, which was nowhere in its retrieved context) — the same fabrication pattern seen from Cohere/Qwen in Phase 7, now reproduced twice by the "winning" model.

Bottom line: Phase 6/7 decisions confirmed at full scale, no regressions. Three scoped follow-ups logged for Phase 12, none actioned: fix the `ARTICLE_NUM_RE` regex, look at why Article 90 didn't retrieve for q22, and check whether SSL pension articles need finer chunking or a disambiguation prompt nudge. Results in `eval/full_eval_results.csv` (per-question answers + citation checks) and `eval/full_eval_judge_scores.csv` (faithfulness/relevancy scores + reasons).

## Phase 8b — Generation config tuning ✅ done
Two defaults surfaced by the Phase 8 failure analysis had never actually been tested: `top_k=5` for the generator's context (inherited from the retrieval eval's recall@5 checkpoint, not chosen for generation quality) and `temperature` (never set, running at the provider default ~1.0). Ran a 2×2 grid (`top_k` ∈ {5, 10} × `temperature` ∈ {default, 0.0}) across all 40 questions in `scripts/generation_config_tuning.ipynb`, scored with the same regex proxy + LLM-judge as Phase 8:

| config | hit rate | fabrication rate | mean faithfulness (min) | mean relevancy (min) |
|---|---|---|---|---|
| top5, default temp (old default) | 0.825 | 0.025 | 4.83 (2) | 4.68 (2) |
| top5, temp=0 | 0.775 | 0.050 | 4.78 (1) | 4.80 (2) |
| **top10, default temp** | 0.825 | 0.025 | **4.90 (4)** | **4.85 (3)** |
| top10, temp=0 | 0.850 | 0.050 | 4.93 (4) | 4.90 (4) |

`temperature=0` didn't help — it raised the fabrication rate at both `top_k` settings and lowered hit rate at `top_k=5`, not adopted. `top_k=10` matched `top_k=5`'s hit/fabrication rate while clearing the worst-case failures: dug into the min-faithfulness=1 outlier (`top5_temp0`/q29, "which law takes precedence") and confirmed it's the same q29 Phase 8 found retrieved at rank 7 — missing at `top_k=5`, present at `top_k=10`, and the faithfulness failure (misapplying Article 107) disappears entirely once the model actually has the right article. **Decision: production default changed to `top_k=10`, temperature left unset** — `src/retrieval.py`'s `Retriever.retrieve()` default updated from 5 to 10.

## Phase 8c — Reranker evaluation ✅ done, not adopted
Tested whether a hosted reranker (Cohere `rerank-v3.5`, ~$0.002/query) beats RRF, in three configurations (`scripts/reranker_evaluation.ipynb`): RRF alone, reranking the raw pooled semantic+keyword candidates (bypassing RRF), and RRF-then-rerank (RRF top-20 → rerank).

| | RRF only (production) | Rerank raw pool | RRF → rerank |
|---|---|---|---|
| Same-language (32/40) recall@1 / @10 / MRR | **0.844 / 0.969 / 0.874** | 0.719 / 0.938 / 0.804 | 0.719 / 0.938 / 0.806 |
| Cross-lingual (8/40) recall@1 / @5 / @10 / MRR | 0.0 / 0.375 / 0.625 / 0.223 | 0.125 / **0.875 / 0.875 / 0.438** | 0.125 / 0.75 / 0.75 / 0.385 |

Mixed result, not a clean win: reranking substantially fixes the known Phase 4 cross-lingual weakness, but costs real same-language precision — RRF wins every same-language metric outright. RRF-then-rerank (hoping to get both) actually underperformed reranking the raw pool, because RRF's top-20 pre-filter already excludes the correct chunk for some cross-lingual questions (RRF's own cross-lingual recall@10 is only 0.625) before the reranker ever sees it — a reranker can only pick from what it's given.

Dug into why reranking hurts same-language specifically (q03: RRF ranked the correct Article 61 chunk #1; the reranker ranked it #5, below three *wrong* articles). Cause: several distinct leave provisions (annual/additional/sick leave) share nearly identical boilerplate opening phrasing ("a worker shall have the right to N days per year with full pay..."), and `rerank-v3.5` — a general-purpose model with no tuning on this corpus — pattern-matches on that shared surface phrasing rather than the specific clause that actually distinguishes the correct article. The embedding model (`embed-v4.0`) doesn't have this problem because it was chosen in Phase 4 specifically for its accuracy on this corpus/eval set; the reranker has no equivalent calibration.

**Decision: not adopted.** Same-language is 80% of the eval set and RRF is unambiguously better there; the cross-lingual fix, while real, doesn't outweigh that majority-case regression given the added cost/latency and no clean way to apply reranking selectively (cross-linguality isn't known at query time). `Retriever` in `src/retrieval.py` is unchanged from Phase 8b (RRF, `top_k=10`). Revisit if cross-lingual queries turn out to be common in real usage, or if a domain-tuned/larger reranker becomes worth testing.

## Phase 9 — Backend service ✅ done
FastAPI wrapper in `src/api.py`: `POST /ask` (`{question, top_k}` → `{answer, citations, sources}`) and `GET /health`. `Retriever` and the production generator are built once at startup via FastAPI's `lifespan` handler (both are expensive to construct — 739 chunks, BM25 index, Pinecone/Cohere/OpenRouter clients), not per-request. Pydantic validates `question` (1–1000 chars) and `top_k` (1–20) before any retrieval/generation cost is incurred. CORS currently wide open (`allow_origins=["*"]`) — tighten to the real frontend origin in Phase 11.

Demoed end-to-end in `scripts/api_demo.ipynb` — starts the real server as a subprocess, hits it over actual HTTP (not just calling the Python objects directly), for both an English and an Arabic question; both correctly retrieve and cite Labor Law 1996 Article 61. Added `requirements.txt` (pins `fastapi`, `uvicorn` alongside the existing deps) for Phase 11 deployment.

Not yet done: rate limiting / usage caps (needed before a public link goes live, see Phase 11), auth (not needed for a public demo), and the two smaller open issues from Phase 8 (the `ARTICLE_NUM_RE` "للمادة" regex bug, and investigating the q22 retrieval miss) are still outstanding.

## Phase 10 — Frontend ✅ built, not yet visually verified
Static HTML/CSS/vanilla JS chat UI in `frontend/` (`index.html`, `style.css`, `app.js`) — no framework, calls `POST /ask` on the Phase 9 backend directly via `fetch`. Chosen over Streamlit/Gradio deliberately: this is meant for the resume, and a real hand-built frontend demonstrates more than wrapping a Python demo tool.

Features: per-message RTL detection (if >30% of a message's letters are Arabic-script, that bubble renders `dir="rtl"`, so English and Arabic messages coexist correctly in the same conversation), collapsible sources panel per answer (law name, article, clause, language, snippet), pending/error states, Enter-to-send with Shift+Enter for newline.

Verified functionally: static files serve correctly, backend health check passes, and a real CORS preflight from the frontend's origin (`127.0.0.1:5500`) to the backend (`127.0.0.1:8000`) returns `access-control-allow-origin: *` — confirms the browser-side fetch will actually be allowed to reach the API. **Not yet visually verified in an actual browser** (no browser-automation tool available in this environment) — layout, RTL rendering, and the actual chat interaction need a manual look before calling this done.

`API_BASE_URL` in `app.js` is hardcoded to `http://127.0.0.1:8000` — needs updating once the backend has a real deployed URL (Phase 11).

## Phase 11 — Deployment & launch ⬜ not started
Wire backend + frontend + vector store together, attach a domain, add a usage/cost cap so a public link can't run up a surprise bill, write the README/resume blurb.

## Phase 12 — Polish ⬜ ongoing
Revisit the Professional Work Law EN/AR mismatch, glossary-article chunking, and expand the eval set once real pipeline failures show what's missing.
