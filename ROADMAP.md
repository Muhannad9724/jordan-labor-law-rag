# Roadmap: source PDFs → live public RAG app

End goal: a publicly launched RAG app (live URL) answering questions on Jordan's Social Security Law and Work Law (1996 + Professional 2019), bilingual (Arabic/English). See `CLAUDE.md` for the deployment-goal constraint that should shape every choice below.

## Phase 0 — Source data ✅ done
6 PDFs acquired (SCL + Work Law 1996 + Professional Work Law 2019, each EN/AR).

## Phase 1 — Extraction & cleaning ✅ done
PyMuPDF (EN) + Gemini vision OCR (AR) → article-level parsing. Fixed several real defects along the way (duplicate-key content loss, false-positive headers from cross-references, chapter/footnote noise, wrong-language header word, a missing article header, an OCR digit misread). Output: 6 `Knowledgebase/*.xlsx` tables, all reproducible from `scripts/01. files_text_extraction.ipynb`.

**Open issue:** Professional Work Law EN (unamended 2019) and AR (amended-as-of-2024) aren't fully parallel — undecided, currently just documented.

## Phase 2 — Chunking ✅ done, later revised
Originally: `0. cleaned_data/chunks.parquet`, 739 chunks, clause-aware splitting for long articles, in `scripts/02. chunking.ipynb`. Caught and fixed a regex gap (EN hyphen-style clause markers) and a duplicate-chunk-id bug during review.

**Superseded in Phase 12a — production now chunks at whole-article granularity** (`chunks_articles.parquet`, 531 units). Clause-splitting was tested against whole articles and lost on every retrieval metric; see Phase 12a below for the evidence and the reason. `chunks.parquet` and `scripts/02. chunking.ipynb` are kept as the historical record and the rollback path.

**Known limitation:** 3 "definitions" glossary articles have no lettered clauses to split on, so they stay oversized (>3000 chars). These are also the one article type where merging is genuinely unhelpful (a glossary really is multi-topic) — worth watching if definitional questions regress.

## Phase 3 — Eval set ✅ done
`eval/eval_draft.jsonl` + `.csv`, 40 hand-drafted bilingual Q&A pairs sampled across all 3 laws, balanced 16 ar/ar + 16 en/en + 8 cross-lingual, with `question_category`/`difficulty`/`labeling_note`/`source_article_excerpt` fields. Reviewed and revised for tone (plain, how-a-worker-would-ask phrasing) and topic relevance (leave/pay/termination concerns over institutional trivia). This is the ground truth for measuring the pipeline in Phase 8.

## Phase 4 — Embeddings ✅ done — **re-reviewed and re-confirmed in Phase 13, see below**
Tested 5 candidates against the real corpus (739 chunks) and eval set (40 questions) in `scripts/03. embeddings_evaluation.ipynb`: self-hosted `Qwen3-Embedding-0.6B`, `bge-m3`, `multilingual-e5-large`, plus hosted `cohere-embed-v4` and `openai-text-embedding-3-small`. **Decision: Cohere `embed-v4.0`** — clearly best same-language accuracy (Recall@10 = 1.000, Recall@1 = 0.812) and best overall MRR; initial lean toward bge-m3 for cross-lingual strength didn't survive scrutiny (the two supporting content-gap examples were either wrong — SCL 85-bis exists in both languages, a stale note — or not actually a retrieval problem — the Professional Work Law EN/AR divergence is a stale-translation issue no embedding model fixes). Cost checked against the real corpus: ~$0.02 one-time to index, ~$0.000003/query ongoing — negligible at this project's scale. Full results table and reasoning are documented in the notebook itself.

## Phase 5 — Vector store & indexing ✅ done
**Decision: Pinecone** (serverless free tier, AWS `us-east-1`, cosine, dim=1536). Chosen over Qdrant Cloud primarily for resume value — it shows up most frequently by name in job postings for RAG/vector-DB experience; SaaS-only (no self-host) was a non-issue since this project deploys to the cloud anyway. All 739 chunks embedded with Cohere `embed-v4.0` and upserted with metadata (`law`, `language`, `article_number`, `clause`, `text`, `source_file`) in `scripts/04. vector_store_indexing.ipynb` — the full chunk text lives in Pinecone metadata, so no separate document store is needed. Sanity-checked against 3 real eval questions; correct articles came back in the top-5. One thing worth remembering: right after a bulk upsert, Pinecone serverless can return slightly out-of-order scores for a short window (indexing consistency lag) — settles within about a minute, not a retrieval bug.

## Phase 6 — Retrieval ✅ done
**Search-method decision made:** hybrid, beta=0.9 (90% semantic Cohere `embed-v4.0` cosine similarity, pulled straight from the Pinecone index by id, + 10% BM25 keyword score with light Arabic normalization, both min-max normalized per query before combining) — tested in `scripts/05. search_method_evaluation.ipynb` by sweeping beta from 0.0 (pure keyword) to 1.0 (pure semantic) over the same 40 eval questions. Beat pure semantic on every overall metric (recall@1 0.675 vs 0.650, recall@10 0.925 vs 0.900, MRR 0.748 vs 0.732) while matching it cross-lingual; pure keyword alone is far behind and cross-lingually blind (BM25 has zero token overlap across languages by construction). The cross-lingual recall@1 = 0.0 result across every beta is a pre-existing Phase 4 embedding limitation, not something the search method changes.

Also cross-checked against RRF (Reciprocal Rank Fusion, a rank-based fusion method that needs no score normalization), sweeping k over 10 values (1-100) × weight 0.0-1.0 in 0.05 steps (210 configs) — best RRF config (k=5, w=0.85) landed close to weighted-sum beta=0.9 (recall@1/5 identical at 0.675/0.825; weighted-sum ahead on recall@10 0.925 vs 0.900 and MRR 0.748 vs 0.744), confirming beta=0.9 wasn't a min-max-normalization artifact.

**Final production choice: RRF, k=5, w=0.85** — not the marginally-higher-scoring weighted-sum. Reason is architectural: the eval computed weighted-sum's normalization over the full 739-chunk score matrix, but the real retrieval function only gets a top-k list from Pinecone and a top-k list from BM25 (not all 739 scores), and min-max normalizing over a partial, per-query-varying slice is unstable. RRF only needs rank position within whatever list it got, which matches how two independent top-k API calls actually work. The eval gap (one question out of 40) doesn't outweigh that implementation-robustness argument.

Also tuned BM25's own `k1`/`b` parameters (25-combo grid, RRF held fixed at k=5/w=0.85): `rank_bm25`'s defaults (k1=1.5, b=0.75) were already optimal for this corpus — `k1` had zero measurable effect (BM25 is only 15% of the RRF weight, so its exact scores barely matter, only rank order does), `b=0.75` beat both extremes (full or no length normalization measurably hurt). No change made.

**Retrieval function built:** `src/retrieval.py` (`Retriever` class) — embeds the query with Cohere, pulls top-50 candidates each from Pinecone (semantic) and local BM25 (keyword), RRF-fuses them (k=5, w=0.85), returns top-k chunks with full metadata. Reusable/importable, not locked in a notebook, so Phase 9's FastAPI backend can call it directly. Validated in `scripts/06. retrieval.ipynb`: running it over all 40 eval questions reproduces the eval notebook's numbers exactly (recall@1 0.675, recall@5 0.825, recall@10 0.900, MRR 0.744) — confirms capping each ranker to its top 50 candidates before fusing (required since real Pinecone/BM25 queries return top-k, not full-corpus scores) doesn't lose anything versus the full-corpus eval.

The cross-lingual eval questions still matter for regression-checking retrieval quality — Cohere's cross-lingual Recall@10 (0.625) is real and worth keeping an eye on, not zero. Optional/deferred: reranking, metadata-scoped search.

## Phase 7 — Generation / answer synthesis ✅ done
Initial pick was Cohere `command-a-03-2025` (`scripts/07. generation.ipynb`, native `documents`/`citation_options` grounding, spot-checked on 4 cases, all correct) — but that choice was made without comparing it against anything, unlike every other phase in this project. Went back and actually tested it in `scripts/08. generation_llm_comparison.ipynb` against OpenAI `gpt-4o-mini` and Qwen2.5-72B (via OpenRouter/DeepInfra) on 10 real eval questions + 1 out-of-domain honesty trap, scored on correct-article-citation rate and a fabricated-citation check (does the answer mention an article number that wasn't even in the retrieved context):

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
Ran all 40 eval questions through the full production pipeline (`Retriever` → `get_production_generator()`) in `scripts/09. evaluation_harness.ipynb`. **Retrieval: exact match with Phase 6** — recall@1 0.675, recall@5 0.825, recall@10 0.900, MRR 0.744, no regression at full scale.

**Generation (regex proxy): 35/40 (87.5%) correct-article rate, 2/40 fabricated citations.** (Originally reported as 33/40 — the scorer, not the model, was wrong; see the regex fix below.) Traced every one of the 7 originally-flagged misses individually (re-queried the retriever directly rather than trusting a first manual read, which had errors — see notebook for the corrected version): only **1 is a genuine generation defect** (q07 — cited the wrong article despite the right chunk retrieved at rank 1) and **1 a genuine retrieval miss** (q22 — Article 90 never surfaced in top 10, model substituted a related-but-wrong article). Of the rest, 3 were retrieved but outside production `top_k=5` (the known recall@5-vs-10 gap, q25/q29/q31) and 2 were pure scoring bugs (q08/q40).

**`ARTICLE_NUM_RE` fixed.** The old pattern anchored on the literal string `المادة`. Arabic attaches prefixes straight onto the definite article, and `ل` elides its alef (ل + ال → لل), so `المادة` does not occur inside `للمادة` — the ordinary way to write "according to article N". Every citation written that way scored as a miss. Measured on the Phase 8 answers: 23 citations used `المادة` (matched) and 6 used `للمادة` (silently missed). The pattern now anchors on the invariant stem `مادة` with optional prefixes: `[Aa]rticle\D{0,15}?(\d+)|[وف]?(?:ال|لل|[بك]ال)?مادة\D{0,5}?(\d+)`. Verified against real citation forms plus negative cases (`هذه المادة توضح` — article named with no number, and `مادة كيميائية` — the word in its "substance" sense) with no false positives and no regressions on previously-matching forms. Re-scoring the stored Phase 8 answers moved exactly q08 and q40 from miss to hit — the two Phase 8 had already identified by hand — confirming the predicted ~87.5%.

**Numbers elsewhere that are still understated:** only `eval/full_eval_results.csv` stored raw answer text, so only Phase 8 could be re-scored offline. The Phase 7 model comparison, Phase 8b `top_k`/temperature tuning, and Phase 12a chunk-granularity runs all saved extracted citations rather than answers, so their correct-article rates remain measured with the broken pattern and are biased low. Those comparisons stay internally valid — the same bug applied to every arm — but their absolute rates should not be quoted. Future eval notebooks should persist the answer text so a scorer fix never requires re-paying for generation.

**LLM-as-judge (Claude Sonnet 5 via OpenRouter), full 40**: mean faithfulness 4.90 (min 3), mean relevancy 4.80 (min 2), consistent with the Phase 7 sample (5.00/4.91). 3 questions scored ≤3 — all Social Security Law pension questions (q22, q29, q35) where the model reached for a related-but-wrong provision among adjacent articles (62/71/80/85/90). Separately, **q09 and q21** each fabricated one extra, ungrounded citation despite correctly citing the right article from properly-retrieved context (q21: correctly cited Article 53 at rank 1, but also added Article 90, which was nowhere in its retrieved context) — the same fabrication pattern seen from Cohere/Qwen in Phase 7, now reproduced twice by the "winning" model.

Bottom line: Phase 6/7 decisions confirmed at full scale, no regressions. Three scoped follow-ups logged for Phase 12, none actioned: fix the `ARTICLE_NUM_RE` regex, look at why Article 90 didn't retrieve for q22, and check whether SSL pension articles need finer chunking or a disambiguation prompt nudge. Results in `eval/full_eval_results.csv` (per-question answers + citation checks) and `eval/full_eval_judge_scores.csv` (faithfulness/relevancy scores + reasons).

## Phase 8b — Generation config tuning ✅ done
Two defaults surfaced by the Phase 8 failure analysis had never actually been tested: `top_k=5` for the generator's context (inherited from the retrieval eval's recall@5 checkpoint, not chosen for generation quality) and `temperature` (never set, running at the provider default ~1.0). Ran a 2×2 grid (`top_k` ∈ {5, 10} × `temperature` ∈ {default, 0.0}) across all 40 questions in `scripts/10. generation_config_tuning.ipynb`, scored with the same regex proxy + LLM-judge as Phase 8:

| config | hit rate | fabrication rate | mean faithfulness (min) | mean relevancy (min) |
|---|---|---|---|---|
| top5, default temp (old default) | 0.825 | 0.025 | 4.83 (2) | 4.68 (2) |
| top5, temp=0 | 0.775 | 0.050 | 4.78 (1) | 4.80 (2) |
| **top10, default temp** | 0.825 | 0.025 | **4.90 (4)** | **4.85 (3)** |
| top10, temp=0 | 0.850 | 0.050 | 4.93 (4) | 4.90 (4) |

`temperature=0` didn't help — it raised the fabrication rate at both `top_k` settings and lowered hit rate at `top_k=5`, not adopted. `top_k=10` matched `top_k=5`'s hit/fabrication rate while clearing the worst-case failures: dug into the min-faithfulness=1 outlier (`top5_temp0`/q29, "which law takes precedence") and confirmed it's the same q29 Phase 8 found retrieved at rank 7 — missing at `top_k=5`, present at `top_k=10`, and the faithfulness failure (misapplying Article 107) disappears entirely once the model actually has the right article. **Decision: production default changed to `top_k=10`, temperature left unset** — `src/retrieval.py`'s `Retriever.retrieve()` default updated from 5 to 10.

## Phase 8c — Reranker evaluation ✅ done, not adopted
Tested whether a hosted reranker (Cohere `rerank-v3.5`, ~$0.002/query) beats RRF, in three configurations (`scripts/11. reranker_evaluation.ipynb`): RRF alone, reranking the raw pooled semantic+keyword candidates (bypassing RRF), and RRF-then-rerank (RRF top-20 → rerank).

| | RRF only (production) | Rerank raw pool | RRF → rerank |
|---|---|---|---|
| Same-language (32/40) recall@1 / @10 / MRR | **0.844 / 0.969 / 0.874** | 0.719 / 0.938 / 0.804 | 0.719 / 0.938 / 0.806 |
| Cross-lingual (8/40) recall@1 / @5 / @10 / MRR | 0.0 / 0.375 / 0.625 / 0.223 | 0.125 / **0.875 / 0.875 / 0.438** | 0.125 / 0.75 / 0.75 / 0.385 |

Mixed result, not a clean win: reranking substantially fixes the known Phase 4 cross-lingual weakness, but costs real same-language precision — RRF wins every same-language metric outright. RRF-then-rerank (hoping to get both) actually underperformed reranking the raw pool, because RRF's top-20 pre-filter already excludes the correct chunk for some cross-lingual questions (RRF's own cross-lingual recall@10 is only 0.625) before the reranker ever sees it — a reranker can only pick from what it's given.

Dug into why reranking hurts same-language specifically (q03: RRF ranked the correct Article 61 chunk #1; the reranker ranked it #5, below three *wrong* articles). Cause: several distinct leave provisions (annual/additional/sick leave) share nearly identical boilerplate opening phrasing ("a worker shall have the right to N days per year with full pay..."), and `rerank-v3.5` — a general-purpose model with no tuning on this corpus — pattern-matches on that shared surface phrasing rather than the specific clause that actually distinguishes the correct article. The embedding model (`embed-v4.0`) doesn't have this problem because it was chosen in Phase 4 specifically for its accuracy on this corpus/eval set; the reranker has no equivalent calibration.

**Decision: not adopted.** Same-language is 80% of the eval set and RRF is unambiguously better there; the cross-lingual fix, while real, doesn't outweigh that majority-case regression given the added cost/latency and no clean way to apply reranking selectively (cross-linguality isn't known at query time). `Retriever` in `src/retrieval.py` is unchanged from Phase 8b (RRF, `top_k=10`). Revisit if cross-lingual queries turn out to be common in real usage, or if a domain-tuned/larger reranker becomes worth testing.

## Phase 9 — Backend service ✅ done
FastAPI wrapper in `src/api.py`: `POST /ask` (`{question, top_k}` → `{answer, citations, sources}`) and `GET /health`. `Retriever` and the production generator are built once at startup via FastAPI's `lifespan` handler (both are expensive to construct — 739 chunks, BM25 index, Pinecone/Cohere/OpenRouter clients), not per-request. Pydantic validates `question` (1–1000 chars) and `top_k` (1–20) before any retrieval/generation cost is incurred. CORS currently wide open (`allow_origins=["*"]`) — tighten to the real frontend origin in Phase 11.

Demoed end-to-end in `scripts/12. api_demo.ipynb` — starts the real server as a subprocess, hits it over actual HTTP (not just calling the Python objects directly), for both an English and an Arabic question; both correctly retrieve and cite Labor Law 1996 Article 61. Added `requirements.txt` (pins `fastapi`, `uvicorn` alongside the existing deps) for Phase 11 deployment.

Not yet done: rate limiting / usage caps (needed before a public link goes live, see Phase 11), auth (not needed for a public demo), and the two smaller open issues from Phase 8 (the `ARTICLE_NUM_RE` "للمادة" regex bug, and investigating the q22 retrieval miss) are still outstanding.

## Phase 10 — Frontend ✅ built, not yet visually verified
Static HTML/CSS/vanilla JS chat UI in `frontend/` (`index.html`, `style.css`, `app.js`) — no framework, calls `POST /ask` on the Phase 9 backend directly via `fetch`. Chosen over Streamlit/Gradio deliberately: this is meant for the resume, and a real hand-built frontend demonstrates more than wrapping a Python demo tool.

Features: per-message RTL detection (if >30% of a message's letters are Arabic-script, that bubble renders `dir="rtl"`, so English and Arabic messages coexist correctly in the same conversation), collapsible sources panel per answer (law name, article, clause, language, snippet), pending/error states, Enter-to-send with Shift+Enter for newline.

Verified functionally: static files serve correctly, backend health check passes, and a real CORS preflight from the frontend's origin (`127.0.0.1:5500`) to the backend (`127.0.0.1:8000`) returns `access-control-allow-origin: *` — confirms the browser-side fetch will actually be allowed to reach the API. **Not yet visually verified in an actual browser** (no browser-automation tool available in this environment) — layout, RTL rendering, and the actual chat interaction need a manual look before calling this done.

`API_BASE_URL` in `app.js` is hardcoded to `http://127.0.0.1:8000` — needs updating once the backend has a real deployed URL (Phase 11).

## Phase 11 — Deployment & launch 🔄 in progress
Progress so far:
- **Git/GitHub**: repo initialized, pushed to `github.com/Muhannad9724/jordan-labor-law-rag` (private for now, plan to flip public once the live deploy is confirmed working).
- **Frontend + backend combined into one deployment**: `src/api.py` now mounts `frontend/` as static files on the FastAPI app itself (`StaticFiles(..., html=True)` mounted last so it doesn't shadow the API routes) — one service instead of two, and CORS is no longer needed since everything's same-origin. `frontend/app.js`'s `API_BASE_URL` changed from a hardcoded `localhost:8000` to `""` (relative).
- **Rate limiting added** (`slowapi`, in-memory, keyed by IP): `/ask` capped at 10/minute and 100/day. Every question costs real money (Cohere + Pinecone + OpenRouter calls), so this was needed before any public link, not optional polish. Verified locally: 11th request in a minute correctly returns 429 with a proper `{"detail": ...}` body the frontend already knows how to render as an error message.
- **README.md** written — architecture table, results summary, local run instructions, resume-facing.
- **Deploy config added**: `Procfile` (`web: uvicorn src.api:app --host 0.0.0.0 --port $PORT`), `.python-version` pinned to 3.11.
- **Host chosen: Railway** — picked over Render/Fly.io specifically because the user prioritized dashboard/interface quality over cost and is fine paying; not evaluated empirically like other project decisions, just a stated preference.

Not yet done: actually deploying to Railway (requires the user to log in via browser — can't be done from here), setting the three API keys as Railway environment variables, confirming the live URL works end-to-end, then flipping the GitHub repo public.

## Phase 12a — Chunk granularity: whole articles ✅ done, adopted
Phase 8's q22 failure ("will my pension go up over the years?", ground truth SSL Article 90) turned out not to be a ranking problem at all. Article 90 sat at semantic rank 11 — it was **crowded out**: seven of the ten slots were separate clauses of SSL Article 71, all saying "the pension shall be increased by X%". Across the eval set 5 of 40 questions lose ≥3 slots to same-article duplicates, and two of those (q22, q33) were outright retrieval failures — half of all "never retrieved" cases traced to this one cause.

Three options were tested on the 32 same-language questions in `scripts/14. chunk_granularity_evaluation.ipynb`, scored on retrieval **and** on real generated answers graded by the independent LLM judge:

| | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| A — clause chunks (previous production) | 0.844 | 0.938 | 0.969 | 0.874 |
| **B — whole articles (adopted)** | **0.875** | **0.969** | **1.000** | **0.904** |
| C — clauses merged at retrieval time | 0.844 | 0.938 | 1.000 | 0.879 |

**Why the original clause-splitting reasoning was wrong.** The Phase 2 assumption was that a whole article makes a blurrier embedding, since one vector must represent several provisions. Reading the text disproved it: these articles are *single-topic*. SSL 71 is entirely about military insureds' pensions (29 clauses), SSL 64 entirely about Early Retirement Pension (14), Labor Law 12 entirely about non-Jordanian workers (11). Clauses are sub-rules of one subject — that is simply how statutes are drafted — so merging restores one topic rather than blending many. Worse, the clause chunks were **semantically incomplete by construction**: *"b. Pension stipulated in Paragraph (a) of this Sub-paragraph shall be increased by (0.5%)…"* cannot be understood alone — "Paragraph (a)" dangles and the benefit is never named. Six such near-identical fragments of Article 71 are precisely why they all looked alike to the retriever and piled up together.

Generation-side scores (hit rate, faithfulness, relevancy) differed by 1–2 questions and ≤0.09 of a 5-point scale across A/B/C — inside the noise floor for n=32, especially with the generator running at default temperature (non-deterministic) and only one run per config. **Retrieval metrics are deterministic and reproducible; the generation-side gaps are not**, so the decision rests on retrieval. Noted as a methodology gap: no repeat runs were done to establish that noise floor.

**Rebuilt in `scripts/15. rechunking_articles.ipynb`:** 739 clause chunks → 531 article units, all 477,249 characters preserved exactly, largest article 17,842 bytes (within Pinecone's 40 KB metadata cap). Only 21 articles are actually split, so 510 units are byte-identical to their old chunk. New index `jordan-labor-law-articles`; the clause-level `chunks.parquet` and `jordan-labor-law` index are both left intact, so rollback is a two-line edit in `src/retrieval.py`.

**Live production numbers after the switch (full 40 questions):**

| | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| overall — clause | 0.675 | 0.825 | 0.900 | 0.744 |
| **overall — article** | 0.675 | **0.900** | **0.950** | **0.763** |
| same-language — clause | 0.844 | 0.938 | 0.969 | 0.874 |
| **same-language — article** | 0.844 | **0.969** | **1.000** | **0.888** |
| cross-lingual — clause | 0.000 | 0.375 | 0.625 | 0.223 |
| **cross-lingual — article** | 0.000 | **0.625** | **0.750** | **0.262** |

Nothing regressed; same-language recall@10 is now perfect and cross-lingual recall@5 improved as a free side effect. The only two questions still outside the top 10 (q04, q08) are both cross-lingual. Verified end-to-end through the API: the q22 pension question now answers from Article 90 correctly, where clause-level production answered from the wrong article (Article 71) and scored faithfulness 3 / relevancy 2 in Phase 8.

**Full Phase 8 harness re-run on article granularity** (`scripts/09. evaluation_harness.ipynb`, results in `eval/full_eval_results.csv`; the clause-level run is preserved as `*_clauselevel.csv`):

| | clause-level (top_k=5) | article-level (top_k=10) |
|---|---|---|
| correct article rate | 35/40 | **38/40** |
| fabricated citations | 2 | **1** |
| mean faithfulness (min) | 4.90 (3) | 4.78 (3) |
| mean relevancy (min) | 4.80 (2) | 4.72 (2) |

Citations improved clearly. Judge means fell slightly, and that is **not** dismissable as noise: 12 questions changed score, 4 up and 8 down. The declines share one mechanism — whole articles put more material in front of both the model and the judge. q23 (5/4 → 5/2) and q38 (5/5 → 4/3) were both marked incomplete for omitting provisions that, under clause-chunking, were probably never in context for the judge to notice. q36 (5/5 → 3/4) is a genuine new failure: with Article 85's paragraphs (a) and (b) now merged into one unit, the model conflated two distinct provisions. Against that, the previously worst case q22 went 3/2 → 4/4, plus q29 (5/3 → 5/5) and q33 (5/4 → 5/5).

Confound to be aware of: the clause-level baseline ran at `top_k=5` (Phase 8 predates the Phase 8b switch to 10), so granularity *and* context size both changed — the judge comparison is suggestive, not clean. Retrieval and citation counts are the deterministic, trustworthy signals and both improved.

**Concrete follow-up, not yet done:** sharpen the system prompt for the larger context so the model names *which clause* of an article applies rather than blending adjacent provisions (the q36 failure mode).

Two further caveats worth carrying forward: the first validation run right after upsert produced badly wrong numbers (recall@10 0.938, q03 missing entirely) purely from Pinecone serverless indexing lag — the same behaviour documented in Phase 5; re-running minutes later gave the correct figures. And the RRF/BM25 constants (k=5, w=0.85, k1=1.5, b=0.75) were tuned in Phase 6 on clause-level chunks and have **not** been re-tuned for article granularity — they may no longer be optimal.

## Phase 12b — Prompt precision for clause-level detail 🔄 in progress
Phase 12a's follow-up: sharpen the system prompt so the model stops blending adjacent paragraphs now that whole articles arrive in one block. Targeted the three concrete regressions Phase 12a found — q36 (merged Article 85(a)'s rule with 85(b)'s ceiling, which only applies to a different pensioner category), q23 (answered from a narrow provision, Article 48/unemployment funding, while the general provision, Article 19, sat unused in context), q38 (reported Article 9's fine but dropped the sibling paragraphs' closure/imprisonment consequences) — with three candidate rules appended to `SYSTEM_PROMPT`:

> 6. Paragraphs within an article are separate rules with separate scope — state which one applies, never merge conditions/rates/ceilings across paragraphs or across categories of person/benefit.
> 7. *(original wording)* Cover every bearing paragraph/article; report every consequence and exception, not just the first one found.
> 8. Say explicitly when a provision governs only a sub-category, and prefer the provision that actually governs over an adjacent narrower one.

**First real noise-floor measurement in this project.** Every prior generation-side comparison (Phase 8b, Phase 12a's A/B/C) read a 1–2 question gap off a single run, at default (non-zero) temperature, with no idea how much of that gap was sampling noise. `scripts/17. prompt_clause_precision.ipynb` fixed that: the current prompt was run **twice** under identical conditions alongside the revised prompt, all three arms sharing identical retrieved context (retrieval executed once per question). Whatever changes between the two identical runs is the noise floor a real effect has to clear.

| | baseline mean | revised | noise floor (run1 vs run2) | revised vs run1 | revised vs run2 |
|---|---|---|---|---|---|
| correct-article hit rate | 0.962 | 0.975 | 3 | 3 | 0 |
| fabricated citation rate | 0.025 | 0.050 | 2 | 1 | 1 |
| faithfulness (judge) | 4.762 | 4.775 | **7** | 10 | 11 |
| relevancy (judge) | 4.812 | **4.950** | **4** | 8 | 8 |

Hit rate and fabrication rate stayed within noise — hit rate was already near ceiling post-12a. **Relevancy cleared the noise floor cleanly** (moved roughly double what identical runs move, mean up 4.81→4.95, no question scored worse under revised than under *both* baselines) — a real effect. **Faithfulness was a wash**: q36 broke a genuine reproducible failure (both baseline runs independently scored 2 — not noise — revised reached 4), but four questions scored *worse* under revised than both baselines, including a new defect on q29 (faithfulness 5→3) where the judge flagged an unsupported interpretive claim connecting two articles that neither baseline run added.

**Diagnosis:** rule 7's "cover all of them... report every consequence and exception" told the model *what* to be complete about but not *how*, and on at least one question it was read as license to add explanatory reasoning about how separate provisions relate — exactly the kind of unsupported claim rule 1 already forbids. Rules 6 and 8 show no comparable regression and get credit for the wins (q36, and the relevancy improvement generally).

**Decision:** keep rules 6 and 8 as tested, rewrite rule 7 to demand completeness without granting license to synthesize:

> 7. If more than one paragraph or article in the documents bears on the question, state what each one says, addressed separately. Do not explain how they relate to each other or why one takes precedence unless the documents say so explicitly — restate what is written, don't reason beyond it.

Re-running the same three-arm/noise-floor test with this wording before promoting anything into `src/generation.py`'s production `SYSTEM_PROMPT`. Production is unchanged pending that result.

**Infra note, worth keeping:** OpenRouter reserves credit against a request's `max_tokens` *ceiling*, not its actual usage — an unset `max_tokens` defaults to the model's max (65,536 for Claude Sonnet 5 here) and can 402 a run even with a healthy balance. Three runs of this notebook died before that was diagnosed and fixed; the notebook now also persists generations and judge scores to disk incrementally and resumes from what's already there, so a future crash costs only the unfinished calls, not the whole pass. Worth backporting the same persistence pattern into the other eval notebooks (`09. evaluation_harness.ipynb`, `10. generation_config_tuning.ipynb`, `11. reranker_evaluation.ipynb`, `14. chunk_granularity_evaluation.ipynb`) — none of them have it yet.

## Phase 13 — Embedding model re-review ✅ done, incumbent kept
First step of a deliberate phase-by-phase re-review of the whole project. `scripts/19. embeddings_evaluation_v2.ipynb`.

**Why re-run Phase 4 at all.** Every input to the original comparison had changed: it scored on 739 *clause* chunks (production has served 531 whole articles since Phase 12a), on the 40-question v1 set whose 8 cross-lingual questions *were* the entire Cohere-vs-bge-m3 argument (v2 has none), with bge-m3 and Qwen3 capped at `max_length=1024` despite supporting 8192+, and with OpenAI judged solely on its cheap `text-embedding-3-small`. Prior numbers were treated as void, not as a baseline.

**Scope: pure semantic retrieval only** — no BM25/RRF, no generation, no judge. Embeddings + cosine are deterministic, so unlike Phase 12b there is no noise floor to measure: one pass per model is the whole truth.

**Decision rule fixed before the run:** switch only if a challenger beats Cohere on *both* recall@5 and MRR by ≥ 0.05 (3 of 60 questions).

| model | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| gemini-embedding-001 | **0.967** | 1.000 | 1.000 | **0.981** |
| **cohere-embed-v4** *(incumbent)* | 0.950 | 1.000 | 1.000 | 0.975 |
| multilingual-e5-large-instruct | 0.917 | 1.000 | 1.000 | 0.951 |
| qwen3-embedding-0.6b | 0.900 | 1.000 | 1.000 | 0.935 |
| bge-m3 | 0.900 | 0.983 | 0.983 | 0.931 |
| openai-text-embedding-3-large | 0.867 | 0.983 | 1.000 | 0.919 |
| openai-text-embedding-3-small | 0.817 | 0.950 | 0.950 | 0.876 |

**Decision: keep Cohere `embed-v4.0`.** Gemini leads the table but clears neither half of the rule — recall@5 is tied at 1.000 and the MRR gap is 0.006, about 8× under the bar; its recall@1 lead is exactly one question in sixty. `src/retrieval.py` unchanged, no re-index. This is the case the pre-committed threshold existed for: without it, "Gemini wins" would have justified a new 3072-dim Pinecone index, an embedder abstraction and an RRF re-tune to move one question. Phase 4's *conclusion* survives its own methodology bugs — fixing the `max_length` handicap lifted bge-m3 and Qwen3 but did not reorder anything, and the never-tested `text-embedding-3-large` finished sixth of seven. What does not survive is Phase 4's stated *reasoning*: the Cohere-over-bge-m3 margin is 0.044 MRR, real but far narrower than that writeup implied.

**The bigger finding — `eval_v2` is at its ceiling.** Five of seven models hit recall@5 = 1.000; the total spread across seven architecturally different models is **four questions** (q07, q16, q30, q49), everything else unanimous. Likely cause: the v2 questions were written *by reading the articles*, so they reuse statute vocabulary and each question sits close to a paraphrase of its own answer. Corroborating evidence — `multilingual-e5-large-instruct` had **50 of 531 articles truncated** at its hard 512-token cap and still placed third, ahead of two models that read every article in full; if losing the tail of ~9% of the corpus costs nothing measurable, the signal is concentrated in the opening lines. Consequence: **do not trust `eval_v2` to resolve fine differences in the RRF `k`/`w` and BM25 `k1`/`b` re-tune still pending** — those gaps will be smaller than the ones it already cannot separate. Harder, paraphrased, scenario-shaped questions would restore its power.

**Voyage dropped mid-phase**, before producing a number: the justification was only "benchmarks well" against a `CLAUDE.md` rule that favours cheap/reliable deployment, and with the top models already at recall@5 = 1.000 there was no headroom left for it to win. (Its unpaid tier also caps at 3 RPM / 10K TPM, which would have forced ~14 minutes for this 135K-token corpus — an inconvenience, not the reason.)

**Infra notes.** Vectors are cached to `eval/emb_cache/` keyed by a hash of the corpus, so they self-invalidate when the corpus changes and every future re-scoring question — including the deferred hybrid-RRF check — is free. Two harness defects fixed after the first run: `_retry` logged only the exception *type* (so a Voyage billing limit was indistinguishable from a batch-size bug without hand-testing), and one model's failure aborted the whole loop, costing three *free* local models their run. Whole comparison cost under $0.09.

**Still open:** confirm the ranking holds under hybrid RRF (free, from cache); verify whether Gemini silently truncates at its 2,048-token cap — no API error fired, so it is unestablished either way.

## Phase 19 — Switched to Gemini embeddings ✅ **done and live**
`scripts/24. vector_store_gemini.ipynb`. Closes the decision that had been blocking everything since Phase 13.

**Why:** `gemini-embedding-001` beats Cohere `embed-v4.0` **11–2 on paired ranks (p = 0.022)** under production's exact hybrid config, at every RRF weight including Cohere's own optimum, worth **+0.093 MRR** — about 7× every fusion parameter combined. Phase 13's pre-committed rule technically didn't fire only because recall@5 was saturated at 1.000 for both, a flaw in the rule rather than a case for the incumbent.

**Cost: near zero.** The 531 Gemini corpus vectors were already cached in `eval/emb_cache/` from Phase 13, keyed by corpus hash — the corpus hadn't changed, so the hash matched and **no re-embedding was needed**. The switch cost Pinecone writes only. This is the concrete payoff from caching the expensive artefact rather than only the scores.

**What changed:** new index `jordan-labor-law-articles-gemini` (3072-dim vs 1536, so a new index was unavoidable); `src/retrieval.py` gains a small `_make_embedder(provider)` factory with lazy per-provider imports, `EMBED_PROVIDER = "gemini"`, and an index-name map keyed by provider so the index can never drift out of step with the embedder. Rollback is two constants — the 1536-dim Cohere index is untouched.

**Side effect: the pending watermark re-index is now closed.** The old index still held watermarked text for `labor96_ar_37` / `labor96_ar_65`; the cached Gemini vectors were computed *after* that cleanup, so the new index is clean from the start.

**Verified live, not just offline:** the built index reproduces the offline top-1 exactly on 6 spot-checked questions, and the full production path (live Gemini embedding → live Pinecone → BM25 → RRF) over all 60 eval_v3 questions gives **recall@5 1.000, MRR 0.922, worst rank 4, 0 misses, 0 API failures** at ~1.0 s/query. MRR matches the offline prediction exactly; recall@1 differs by one question (0.867 vs 0.883) because Pinecone uses approximate nearest-neighbour search while the offline calculation was exact cosine — expected, not a defect.

**Carried forward deliberately:** Gemini **silently truncates past 2,048 tokens** — no error, just a vector describing less text. 4 of 531 articles exceed it (`ssl_ar_71`, `ssl_ar_64`, `ssl_en_71`, `ssl_en_64`); only SSL 64 is ground truth for any question (q15) and Gemini ranks it #1 anyway. It won *while carrying* that handicap. Documented in the module docstring because nothing will ever raise to warn about it.

**Still needed before public launch:** set `GEMINI_API_KEY` in the Railway environment, and check Gemini's free-tier request limits against real traffic — the same failure mode that removed Voyage from the Phase 13 candidate list.

## Phase 23 — Multi-turn conversation ✅ done, verified live with a control
`src/generation.py` (`rewrite_for_retrieval`, history in `generate`), `src/api.py` (`history` on `/ask`, `retrieval_query` on the response), `frontend/` (history tracking, suggestion dock). Closes the limitation Phase 22's rule 11 shipped with.

**The core insight: this is a retrieval problem, not a generation one.** Rule 11 made the model ask for a missing personal fact, but the reply ("yes, I'm covered", "12 years") went to a *stateless semantic retriever* that embeds the query text and nothing else. A bare fragment embeds to nothing useful, so retrieval returned unrelated articles and the answer was grounded in the wrong documents. **Passing history to the generator alone would not have fixed this** -- by the time the generator runs, the wrong context has already been fetched.

**So the follow-up is rewritten into a standalone question *before* retrieval.** `rewrite_for_retrieval()` folds the last few turns in: "yes, I'm covered" becomes "What end-of-service compensation am I owed if I am subject to the Social Security Law?", which embeds to the right neighbourhood. Falls back to the raw query on any failure, i.e. degrades to the old single-turn behaviour rather than erroring. The rewrite prompt explicitly forbids adding legal terms nobody said -- inventing vocabulary would pull in articles the user never asked about.

**Measured, with a control:**
| | retrieval_query | articles retrieved |
|---|---|---|
| Turn 2 **with** history | "ما هو تعويض نهاية الخدمة الذي أستحقه إذا كنت خاضعًا لقانون الضمان الاجتماعي؟" | `labor96:32, ssl:70, ssl:102, ssl:103` |
| Same fragment, **no** history (old behaviour) | "نعم، أنا خاضع لقانون الضمان الاجتماعي" | `ssl:1, ssl:2, ssl:4, ssl:7` |

The control retrieves definitions and scope boilerplate -- exactly the failure this closes.

**Design decisions:** the user's *own words* go to the generator, not the rewrite (the rewrite exists to find documents; answering a machine-reworded question drifts from what was asked). History is **client-held and re-sent per request**, so the server stays stateless -- no session store, no cookie, nothing to expire or leak between users, still one container with no database. Cost is that history is untrusted input: it reaches the model only as conversational text, never the retriever's index or a citation check, and is capped per-turn and in total on both sides. Prior turns enter generation as real chat messages but **carry text only, never their old documents** -- each turn is grounded in its own freshly retrieved context, and replaying stale articles would invite citing something absent from *this* turn's context, the exact fabrication mode the citation check exists to catch. `retrieval_query` is returned on every response because when a follow-up retrieves badly, the rewrite is the first thing to inspect and would otherwise be invisible.

**Suggestion chips moved to a sticky dock above the question bar** (user request). They belong to what you'd type next, so they now live with the input rather than inside the answer. This **deleted the chip scroll-into-view logic entirely** -- every previous placement (end of answer, then top of answer) needed its own scrolling rules to be seen, and each still had a case where they weren't; a sticky dock is visible at any scroll position by construction. Dock height is measured into `--dock-clearance` since it changes as chips appear and the textarea grows, replacing the hardcoded 92px.

**Cost:** one extra `gpt-4o-mini` call per follow-up turn (~200 tokens in, ~30 out). First turns skip it entirely -- no history, nothing to rewrite.

**Not done:** no harness eval of multi-turn quality (the eval set is single-turn by construction); rate limits unchanged, still per-IP.

## Phase 22 — Ask instead of assume, neutral judgment answers, reset button ✅ done, not harness-evaluated
`src/generation.py` (rules 11 & 13) + `frontend/` (reset button). Live-launched and spot-checked with single real questions, not a full eval run — the harness pass is a follow-up, not done here.

**Rule 11 rewritten.** The Phase 20/21 version ("name the condition rather than assuming it either way") is correct for a short closed set of conditions the documents themselves lay out, but said nothing about facts only the person asking could supply -- age, years of service, a date. The model would either guess one case or dump every branch. It now asks for that fact in one short question instead, and only states conditions directly when the documents give a short complete set.

**IMPORTANT, stated in the code comment and here: this does not close the loop.** The API is single-turn — `/ask` takes only `question`, no conversation history, and retrieval runs fresh on whatever text arrives. If the model asks "how many years have you contributed?" and the person replies "12", that reply is a brand-new, context-free query to the retriever, which has no way to know it's answering the model's own question. **This rule makes the model ask a better question; the app still can't hear the answer.** Closing that gap needs real conversation memory (thread history folded into `/ask` and into retrieval) — a feature, not a prompt tweak, and explicitly not built here.

**New rule 13, "Judgment questions".** A question like "should I join the optional social security?" is a request for a personal decision, not a fact lookup, and the Phase 21 prompt had no instruction for that shape of question. Rule 2 ("say so if the documents don't answer") would fire wrong -- the documents *do* describe the optional-insurance provisions, they just don't say what any given person should do. The failure mode on the other side is giving an actual directive recommendation ("yes, join"), which is advice, not legal information -- the exact line the closing disclaimer exists to hold. The rule threads between the two: state the benefits and the obligations/costs the documents describe, neutrally, side by side, and stop there.

**Reset button, frontend-only.** The backend is stateless -- every `/ask` call is independent, there is nothing server-side to tell to forget -- so "reset" is purely a client-side DOM rebuild (`resetConversation()` in `app.js`). Placed in the composer, not the masthead: the page scrolls (not an inner pane), so a header button would be unreachable without scrolling all the way up in a long conversation, while the composer is `position: sticky` and stays on screen throughout. Hidden until a conversation exists (tied to the same `body.reading` class the collapsing masthead already uses), and disabled alongside the send button while a request is in flight -- resetting mid-answer would otherwise clear `#thread` out from under a pending fetch, which would then append its result into a thread that looks freshly reset.

**Verification:** balance/wiring checks on `app.js` (no JS engine on this machine, so this catches malformed edits, not logic bugs) and prompt-structure asserts on `SYSTEM_PROMPT` (13 rules, 5 sections, the two new behaviours present, the Phase 12b guardrail still intact) -- both free. Domain rename (`web-production-54d43` → `jordan-legal-guide.up.railway.app`) confirmed live serving the latest asset version. Prompt behaviour confirmed with single live questions against production, not a harness run.

**Not done:** a full eval pass on rules 11/13 (needs a fresh go-ahead per standing project practice on paid runs); the actual multi-turn fix for rule 11's stated limitation.

## Phase 21 — Coverage rules + bilingual context ✅ done — **adopted; relevancy up, faithfulness held**
`src/generation.py` (Coverage rules, bilingual `_format_context`, scorer fixes) + `scripts/26. prompt_coverage_eval.ipynb`.

**The problem, from Phase 20:** the dominant failure was omission, and it was *not* retrieval — 9 of 10 "omitted" articles were already in the context, the gold article was at rank 1 in 16 of 19 penalised questions, and answers named a mean of **1.22 of 3.87** articles offered (48 of 60 cited exactly one). The behaviour was **instructed**: "Keep it brief", "answer in a sentence or two", "do not reason about how two provisions relate" — three pushes toward short single-source answers, with nothing telling the model to check every document.

**Why a model swap was rejected as the first move:** a bigger model told to be brief is also brief. Changing the model while the prompt still caused the effect would have been the Phase 8b confound again. Fix the instruction, then a model comparison becomes a clean instruction-following test.

**Design: paired on identical context.** Retrieval didn't change, so the notebook replays Phase 20's saved retrieval rather than re-querying. Both arms see byte-identical documents (no Pinecone approximate-NN drift between calls) and retrieval costs nothing.

**Two changes.** (1) Prompt 8 → 12 rules, adding Coverage (4–6) — address every article that bears on the question, cover every lettered paragraph of one you cite, and *state each provision on its own terms without explaining how they combine*. That guardrail sits immediately after the coverage rules because Phase 12b's regression happened at exactly that spot. (2) `_format_context` is now bilingual, scaffolding included — English headings ("Law:", "Article:") were pulling Arabic answers into English citations, so localising only the law name would have left the pull in place.

### Coverage — the mechanism took
mean articles cited **1.22 → 1.55**; answers citing exactly one **48 → 34**; **19 questions cite more, 0 cite fewer**; mean length 463 → 831 chars.

### Judge, paired, ties excluded
| axis | P20 | P21 | delta | better/worse | sign test |
|---|---|---|---|---|---|
| faithfulness | 4.700 | 4.800 | +0.100 | 12 / 8 | **p = 0.50 — noise** |
| relevancy | 4.550 | **4.800** | **+0.250** | **14 / 3** | **p = 0.013 — real** |

**Relevancy is a real gain; faithfulness is a wash and is reported as "held", not "improved"** — +0.100 sits at the Phase 12b noise floor and 12–8 is a coin flip. Worst cases improved on both axes (faithfulness min 1 → 3, relevancy min 2 → 3; below-4 counts 2 → 1 and **7 → 2**). Perfect 5/5 on both axes **34/60 → 43/60**. Biggest movements are exactly where Phase 20 was weakest: **ssl relevancy 4.33 → 4.70**, **Arabic relevancy 4.43 → 4.83**.

### Citation defects
**Arabic answers citing in English 8/30 → 0/30.** Wrong law named **3/60 → 1/60** (q05 survives; it is the ambiguous case where labor96 Art 90 arguably does answer a question naming no law). **q16, Phase 20's worst answer, is fixed: faithfulness 1 → 4, relevancy 2 → 5** — though this run cannot say whether the prompt or the Arabic labels did it.

### Proxy, and two scorer bugs closed
correct article **55 → 58/60**, all correct **51 → 55/60**, **real fabrications 0 → 0**. Flags rose 1 → 4 and **all four are false positives** — numbers quoted from inside the retrieved text (Article 25's own text says "entitlements specified in articles (32) and (33) of this Law"). **The flag count rises with faithfulness, not against it**, since longer answers quote more source. Both scorer defects are now fixed in `src/generation.py` with asserts: `grounded_article_numbers()` treats in-text cross-references (singular, plural, Arabic dual) as legitimate, and `extract_cited_articles()` normalises Arabic-Indic digits — latent until Arabic citations started actually happening, since "المادة (٦٥)" never equals "65".

### The cost, stated plainly
8 questions lost faithfulness and **5 of the 8 are the judge describing the model relating two provisions**: q09, q08, q18, q17, q23. Phase 12b's failure mode is **attenuated, not eliminated** — rule 6 held it to a level where relevancy won clearly and faithfulness broke even, but "cover more provisions" and "start relating provisions" stay coupled.

**q09 is the lesson worth keeping.** Mid-run it looked like the flagship success: Phase 20 cited only Article 40 and the judge said Article 85 was missing; Phase 21 picked up Article 85(b) and named the paragraph. It then **misapplied** it — 85(b) governs combining a pension with *another pension*, not with *wages* — and faithfulness fell 4 → 3. More coverage is not automatically better coverage, and a partial result read mid-run is not a result.

**Limits:** one run per arm, so the noise floor is imported from Phase 12b rather than re-measured; two variables moved together (mostly separable by instrument, but q16 is not attributable); the judge still scores faithfulness *to the retrieved context*, so an answer citing the wrong law can still score 5. Residual work list — below 5 in **both** arms: q01, q05, q09, q10, q19, q29, q35 (cross-article omissions the rule didn't reach), q42, q49 (precision, not coverage).

## Phase 20 — Prompt rewrite + full end-to-end evaluation ✅ done — **the new reference point**
`src/generation.py` (`SYSTEM_PROMPT` rewritten) + `scripts/25. final_evaluation_v3.ipynb`. Every generation number this project had published was stale: faithfulness 4.90 / relevancy 4.80 / correct-article 35/40 were all measured on Cohere embeddings, clause chunks, `top_k=10`, `RRF_W=0.85` and the statute-worded 40-question v1 eval set. **Not one of those inputs still exists.**

**One arm, deliberately.** Running the old prompt as a control was proposed and rejected: it was written for clause fragments and statute-shaped questions, so it would characterise a configuration that will never ship. **The consequence is stated rather than buried — this run cannot claim the new prompt is better than the old one.** It establishes where the system stands; future changes get measured against *these* numbers.

**Prompt changes (Phase 7 five rules → Phase 20 eight).** New rule 4 (name the lettered paragraph, never merge two) targets the Phase 12a regression where whole-article chunks let the model fuse Article 85(a) and (b) into one rule — impossible under clause chunking, so the old prompt never needed it. New rule 6 (answer the person's situation first) targets eval_v3's situational phrasing. Rule 1's second clause — *do not reason about how two provisions relate unless a document says so* — is the survivor of Phase 12b, where a "cover every consequence and exception" rule **caused** a faithfulness regression (q29, 5→3, an invented connective claim between two articles).

### Retrieval (60/60)
recall@1 **0.867** · recall@5 **1.000** · MRR **0.922** · worst rank **4** · misses **0** · full coverage 0.950. Identical recall in both languages. Reproduces the Phase 19 live check exactly, and confirms `top_k=5` on the full set — nothing landed past rank 4.

**Retrieval is no longer the bottleneck, and that redirects the work.** In **16 of the 19** questions the judge penalised, the gold article was retrieved at **rank 1**. The text was in front of the model and the answer still lost points; no amount of further retrieval tuning fixes any of those.

### Generation — regex proxy
correct article cited **55/60 (91.7%)** · all correct articles cited **51/60** · flagged fabrications **1/60 → 0/60 after inspection**.

The lone flag (q34) is a **false positive**: the answer's phrase "unless the termination was a dismissal in accordance with Article 28" is quoted from the retrieved Article 26(a), whose own text contains that cross-reference. Reproducing a cross-reference inside the source is the most faithful behaviour available, and the check penalised it for naming an article not separately retrieved. **51 of 531 corpus units cross-reference another article**, so this is a systematic limit — future fabrication counts must exclude articles named inside retrieved text.

### Generation — LLM judge (Claude Sonnet, 1–5)
| | mean | median | min | 5/4/3/2/1 |
|---|---|---|---|---|
| faithfulness | **4.70** | 5.0 | 1 | 46/12/1/0/1 |
| relevancy | **4.55** | 5.0 | 2 | 41/12/6/1/0 |

By law: labor96 4.77/4.77 · prof19 4.75/4.75 · **ssl 4.63/4.33**. By language: faithfulness 4.70 both; relevancy ar 4.43 vs en 4.67. SSL is the weak arm on both axes — it is also the longest and most cross-referential of the three laws.

### The dominant failure is omission, not hallucination
26 of 60 scored below 5. **18 of the 19 relevancy penalties are omission complaints**, in near-identical judge wording ("omits Article 30", "omits the 6-month contribution requirement", "omits the 36-contribution condition"). 4 of the 14 faithfulness penalties say the same thing. Median answer length by relevancy score: 5→445 chars, 4→482, **3→324, 2→258** — the lowest-scoring answers are the shortest.

Rule 8 ends "Keep it brief." Phase 12b's completeness rule was *removed* for causing over-reach. So the prompt's two tested states are **brief and incomplete** or **complete and over-reaching**; it has never been tested asking for completeness *within* a provision while still forbidding inference *across* provisions. That is the next experiment, and it is a prompt change, not a retrieval change.

### Two defects that are ours, not the model's
1. **Arabic answers cite in English — 8 of 30 (27%)** (q16, q27, q44, q46, q47, q48, q53, q56). `_format_context` labels every document with the English-only `LAW_DISPLAY_NAMES`, so an Arabic question gets Arabic article text under an English heading and the model cites what it was shown. Bodies are correctly Arabic; only the citation is not. Fix: bilingual display names keyed on answer language.
2. **Wrong law named — 3 of 60** (q03, q05, q16), all SSL questions answered from the Labour Law, with both laws in context. **The regex proxy scores all three as correct** because it reads article numbers and never the law name. For q03/q05 the pick is defensible — SSL 27 and labor96 87 both impose injury reporting with a 48-hour deadline; SSL 30 and labor96 90 both set death/disability compensation — which points at ground-truth breadth in `eval_v3` as much as at the model, since the questions name no law but the ground truth admits only one.

**q16 is the one real failure** (faithfulness 1, relevancy 2, gold article at rank 1). Asked whether work-injury insurance is deducted from the worker's salary, it opens **"نعم، يُخصم من راتبك"** — yes, it is — then correctly quotes that the establishment alone pays the 2%. Self-contradictory, wrong part first, wrong law attributed, and **scored correct by the proxy**. Rule 6 tells the model to answer the situation "first, in a sentence or two"; committing to yes/no before working through the article is exactly this failure's shape, and that link is untested.

### Infrastructure lesson (cost a real run)
The judge cell first ran with `max_tokens` **unset**. OpenRouter reserves credit against the request's `max_tokens` *ceiling*, not actual usage, so each call demanded credit for the model's 65,536-token maximum to produce a ~200-token verdict and 402'd with ~60,000 tokens of balance available. It died at question 31 of 60, and because the eval set is ordered by law, **every judged question was an SSL question and the entire Labour Law block was unmeasured** — the aborted half was not a random half, and the partial numbers (faith 4.63 / rel 4.33) understated the finished ones. Retries were futile since the ceiling never changed. Both extremes are now on record: unset → 402, and 300 (Phase 12b) → empty content as Claude's reasoning consumed the budget. **`JUDGE_MAX_TOKENS = 2000`.** Incremental persistence meant resuming cost only the missing 30 calls.

### What this run does not measure
No comparison arm, so nothing here attributes the result to the prompt. The judge scores faithfulness **to the retrieved context, not correctness against ground truth** — for q03 it wrote that the answer "accurately reflects Article 87(a)" while the ground truth is SSL 27, so an answer grounded in the wrong retrieved article can still score 5. Single judge, single run: Phase 12b measured a real noise floor for judge scores and this run did not re-establish one, so small future differences are unreadable without it.

## Phase 18 — Semantic/keyword balance re-tuned ✅ done — **nothing adopted, deliberately**
`scripts/23. fusion_weight_tuning.ipynb`. Zero API cost (cached vectors + local BM25). Swept RRF `w` (0→1, step 0.01), RRF `k` (1→60), and weighted-sum `beta`, all at the new `top_k=5`, for both Gemini and Cohere.

**Finding 1 — re-tuning RRF `w` is not worth doing; the apparent gain is overfitting.** In-sample the sweep looked convincing: Cohere peaks at `w=0.83` (MRR 0.850) vs production `0.85` (0.838), and a 2,000-resample **bootstrap agreed the peak was stable** (median 0.82, 90% CI [0.75, 0.83], `w=0.85` chosen in 0/2000). But 500 split-half CVs (tune on 30, score the untouched 30) put **tuned RRF *below* untuned production for both models** — gemini 0.9237 vs 0.9297, cohere 0.8355 vs 0.8369; per split, tuned RRF beats production in **0%** of splits for gemini (mean −0.0071) and 56% for cohere (mean −0.0016, i.e. loses bigger than it wins). **Key methodological lesson: a stable bootstrap optimum is NOT evidence that moving to it helps** — every resample is drawn from the same 60 questions, so the bootstrap can only confirm the argmax is reliably located, never that it generalises. Only held-out data exposes that the curve near the top is flat and chasing the peak buys nothing. This is precisely the trap Phase 6 fell into. **`RRF_W=0.85` and `RRF_K=5` left alone.**

**Finding 2 — weighted sum genuinely beats RRF, reversing Phase 6 — but most of the gain can't be deployed.** Out-of-sample: gemini 0.9375 vs 0.9297 (71% of splits, +0.0084); **cohere 0.8766 vs 0.8369 (98% of splits, +0.0409)**, stable beta (median 0.92, CI [0.86, 0.92]). **However that version min–max normalises over all 531 articles, which production cannot do** — Pinecone returns only the top `CANDIDATE_TOP_K`, so the full score vector doesn't exist at request time. The deployable pool-normalised variant keeps ~a third of the benefit: gemini 58% of splits/+0.0056 (coin flip), **cohere 81%/+0.0134** (real but small, in-sample recall@1 0.750→0.800).

**Adopted: `RRF_W` 0.85 → 0.90** (90% semantic / 10% keyword), on the user's call after seeing the full analysis. This is a choice *inside the plateau*, not a measured improvement — on eval_v3 at top_k=5, 0.90 scores ~1 question below 0.85 for both models (cohere .829 vs .838, gemini .922 vs .931), and resolving a gap that size needs ~520 questions. Everything in [0.75, 0.90] is indistinguishable on the evidence available. `RRF_K=5`, `BM25_K1=1.5`, `BM25_B=0.75`, `CANDIDATE_TOP_K=50` unchanged; fusion method stays RRF.

**Not adopted:** the weighted-sum switch. If the project stays on Cohere, pool-normalised weighted sum at `beta≈0.80` is a modest genuine win and would need `_semantic_candidates` to stop discarding Pinecone's scores. If it switches to Gemini, leave fusion alone entirely.

**Reproducibility note:** `23. fusion_weight_tuning.ipynb`, `22. embeddings_hybrid_v3.ipynb` and `21. embeddings_rescore_v3.ipynb` all `from src.retrieval import RRF_W`, so a re-run now computes at 0.90 while their stored outputs were produced at 0.85. The stored outputs are the historical record of the decision; expect a small mismatch on re-execution and read the constant, not the prose, as current.

**Finding 3 — BM25's measurable contribution is ~nothing, but it is also ~free.** Asked directly: why not go 100% semantic, given eval_v3 deliberately avoids statute vocabulary? Tested `w=0.85` vs `w=1.00` per question: gemini **1 question helped, 0 hurt, 59 identical** (sign test p=1.00); cohere **2 helped, 0 hurt, 58 identical** (p=0.50). Not significant either way — but **0 questions were ever made worse, in either model**. The strongest hypothesis for keeping it was exact-match queries (article-number lookups), a type eval_v3 contains none of — **tested and refuted**: 12 probes like "ما نص المادة 65 من قانون العمل؟" / "What does Article 90 of the Labour Law say?" were found 12/12 at rank 1-2 by **pure semantic**, with BM25 changing not one of them. Cost measured on this corpus: **121 ms once at startup, 1.8 MB, 1.06 ms/query** — against a 200-500 ms embedding API call, i.e. ~0.3% of request latency. So the honest position is that the choice is near-free in both directions: keep it as cheap insurance against query types a 60-question eval cannot cover (and for the resume value of a named hybrid-retrieval technique, the same logic that chose Pinecone over Qdrant in Phase 5), or drop it for simplicity at a cost of ~1 question. **Kept at `w=0.85`, on robustness and CV grounds rather than on a claim that the data proves it helps.**

**Finding 4 — BM25 `k1`/`b` do not need re-tuning either, and this one is unambiguous.** They have been flagged stale since Phase 12a (fitted on clause chunks). Swept 56 combos (`k1` 0.5-3.0 × `b` 0.0-1.0) **in the production configuration** (hybrid `w=0.90`, `top_k=5`): the best combo gains **+0.0000 MRR in-sample** for both models, and beats the defaults in **0% of 500 held-out splits**. Gemini's in-sample "best" was `k1=0.5, b=0.0` — the *worst* BM25 config — which merely tied the default. Reason: measured **in isolation** these parameters matter enormously (`b=0` → MRR 0.181 vs `b=0.75` → 0.429, a 2.4× swing, and Arabic normalisation in `tokenize()` is worth another 0.404 → 0.429), but BM25 carries only 10% of the fusion weight, so a 140% improvement to it moves the final ranking by nothing. **Cautionary note for reading any component-level metric in this project: a large gain measured on an isolated component is not a gain in the shipped system.** `BM25_K1=1.5` / `BM25_B=0.75` left alone; the Phase 12a "re-tune BM25 for article granularity" item is now **closed as not worth doing**.

**When tuning IS worth it here** (so this isn't read as blanket anti-tuning): the wins in this project — chunk granularity (12a, recall@10 .969→1.000), `top_k` (17, half the context cost), and the embedding model (+0.093 MRR, p=0.02) — all satisfy three conditions the fusion knobs fail: the effect is large relative to what 60 questions can resolve (~0.05, not ~0.005); it is measured in the *shipping* configuration rather than on an isolated component; and it is validated on data that did not choose the value.

**The number that matters:** every fusion knob here is worth **≤0.013 MRR** out-of-sample, several worth zero or less. The undecided **embedding-model choice is worth ~0.093** — Gemini with untuned, unmodified fusion (0.9297) beats Cohere with the best fusion found anywhere in this notebook (0.8766). **The model decision dominates every fusion decision ~7:1**, so fusion work belongs behind it, and adopting Gemini makes most of this moot.

## Phase 17 — `top_k` cut to 5 ✅ done, **adopted into production**
First production code change of the re-review. `src/retrieval.py`'s `retrieve()` default **10 → 5**, and `src/api.py`'s `AskRequest.top_k` field default **10 → 5**.

**Why the old default existed, and why the reason is gone.** Phase 8b did not pick `top_k=10` for recall — it found 10 and 5 tied on hit rate and fabrication rate, and chose 10 only because it lifted LLM-judge worst cases (min faithfulness 4 vs 2, min relevancy 3 vs 2). That traced to a few questions, notably q29, whose correct article sat at **rank 6-7** — inside recall@10, outside recall@5. The extra slots existed to catch those.

**Re-measured on article chunks** (`scripts/22. embeddings_hybrid_v3.ipynb`, searching 30 deep across both eval sets = 120 question-runs): with Gemini or Cohere, **zero questions put the correct article in slots 6-10**. Worst rank 4 on v3, 2-3 on v2. The failure `top_k=10` was adopted to fix no longer occurs under article-level chunking.

**What it buys:** context handed to the generator roughly halves — mean **4,172 → 2,218** tokens, median 2,954 → 1,642, worst case **21,997 → 11,058**. That is input cost on every question, plus reduced crowding risk of the kind that drove the Phase 12a chunking change.

**Caveats recorded in the docstring, not just here:** (1) this is a *retrieval* measurement — Phase 8b's argument was about answer quality, which has not been re-tested at `top_k=5` under article chunking, since generation work stays parked until the retriever is settled; (2) worst rank drifted 2-3 → 4 as questions got harder (v2 → v3), so both sets stay inside 5 but the trend runs deeper and a harder set could change it; (3) **this is model-dependent, not universal** — qwen3-embedding-0.6b puts 9 of 60 correct articles in slots 6-10, so the check must be re-run if the embedding model changes.

**Gotcha worth remembering:** `AskRequest.top_k` shadows the `Retriever` default on every HTTP request, so changing only `retrieval.py` would have left the deployed app on 10 and produced no effect at all. Two places, not one. (The frontend never sends `top_k`, so it inherits the default.)

## Phase 16 — Hybrid confirmation ✅ done — the Gemini result holds
`scripts/22. embeddings_hybrid_v3.ipynb`. **Zero API calls** — corpus *and* query vectors for all 7 models were already cached from Phases 13/15, and BM25 runs locally. Pure arithmetic on cached arrays.

**Hybrid RRF at production constants (`k=5, w=0.85, k1=1.5, b=0.75`) does not change the ranking.** Hybrid MRR: gemini **0.931** · openai-3-large 0.838 · cohere 0.838 · bge-m3 0.835 · e5-instruct 0.751 · openai-3-small 0.692 · qwen3 0.674. Paired sign test on hybrid ranks: **gemini beats cohere 11–2, p = 0.0225** — slightly stronger than semantic-only. openai-3-large vs cohere **8–8, p = 1.000**, still a dead tie.

**The `RRF_W` confound was real and is now eliminated.** `w=0.85` was fitted in Phase 6 on clause chunks *with Cohere as the semantic ranker*, so it was not neutral. Swept w ∈ [0, 1]: **Gemini wins at every w from 0.2 to 1.0**, and the margin widens as the embedding gets more weight. Decisive comparison — at Cohere's *own* best setting (w=0.80, MRR 0.844), Gemini scores 0.931; Cohere's best anywhere in the sweep is below Gemini's worst from w≥0.6. (Note: at w=0 all 7 models tie at 0.437 since that arm is pure BM25; the "winner" there is an alphabetical `idxmax` artifact, not a result.)

**BM25 contributes very little.** Alone: recall@1 0.317, recall@5 0.600, MRR 0.437 — far below every embedding model. Fusing it moves MRR by −0.002 to +0.032 (gemini just +0.008, bge-m3 most at +0.032). So retrieval quality here is essentially *the embedding model's* quality, and hybrid is cheap insurance rather than a rescue. This also refutes the earlier worry that natural phrasing would favour BM25 — v3 avoids statute vocabulary, which is precisely what BM25 needs, so the keyword half got *less* useful.

**`w=0.85` is no longer optimal** — every model peaks at 0.80 or 0.70. The pending RRF re-tune is warranted but the prize is small (cohere 0.838→0.844, gemini unchanged), and it must come *after* the embedding model is settled since the optimum moves with the model.

**Conclusion: the evidence supports switching to `gemini-embedding-001`.** Every check that could have overturned Phase 15 has been run and none did — holds under production hybrid, holds across the whole weight range including Cohere's own optimum, significant on paired tests in both modes (p=0.035 semantic / p=0.0225 hybrid), and achieved *while* silently truncating 4 articles Cohere reads in full. The lone dissenting metric (recall@5, tied at 1.000) is saturated and cannot express a difference. **Not executed unilaterally** — `src/retrieval.py` is still on Cohere. Switching requires: a new 3072-dim Pinecone index, an embedder abstraction in `src/retrieval.py`, and a check of Gemini's production rate limits (the failure mode that removed Voyage in Phase 13).

## Phase 15 — Re-scoring on eval v3 ✅ done — Phase 13's answer does not survive
`scripts/21. embeddings_rescore_v3.ipynb`. Near-free: `eval/emb_cache/` already held all seven models' 531 corpus vectors, so only 60 queries per model needed embedding.

**Every model dropped.** Mean **recall@1 −0.233**, mean **MRR −0.155**, on identical ground truth with only the wording changed — so roughly a quarter of the v2 scores were an artifact of questions echoing their own answers. Measured, not inferred, because the design is paired.

**The eval set has its power back:** models at recall@5 = 1.000 went 4/7 → 2/7; MRR spread across the field 0.104 → **0.265 (2.5×)**; contested questions 4 → 10; models disagree on rank for **48 of 60**. Crucially **0 questions are missed by every model**, so the rewrite made things harder without breaking any ground truth.

| model | v2 r@1 | v3 r@1 | v2 MRR | v3 MRR |
|---|---|---|---|---|
| gemini-embedding-001 | 0.967 | **0.867** | 0.981 | **0.922** |
| openai-text-embedding-3-large | 0.867 | 0.733 | 0.919 | 0.837 |
| **cohere-embed-v4** *(incumbent)* | 0.950 | 0.717 | 0.975 | 0.828 |
| bge-m3 | 0.900 | 0.667 | 0.931 | 0.803 |
| multilingual-e5-large-instruct | 0.917 | 0.633 | 0.951 | 0.741 |
| openai-text-embedding-3-small | 0.817 | 0.550 | 0.876 | 0.694 |
| qwen3-embedding-0.6b | 0.900 | 0.517 | 0.935 | 0.657 |

**Paired sign tests** (deterministic scores, same 60 questions): **gemini beats cohere 12–3, p = 0.035**; gemini vs openai-3-large 12–4, p = 0.077; **openai-3-large vs cohere 9–8, p = 1.000 — a genuine tie despite the table order.** Don't read a sorted leaderboard as a ranking for close pairs. Qwen3-0.6B is the cautionary case: 4th on v2, last on v3, biggest drop in the field — its v2 placing was almost entirely phrasing.

**Decision status: unresolved, nothing changed.** Phase 13's pre-committed rule (beat Cohere on *both* recall@5 and MRR by ≥0.05) does **not** fire — MRR clears it at +0.094, but recall@5 is tied at 1.000. That tie exposes a flaw in the rule rather than a case for the incumbent: **recall@5 was saturated when it was chosen as a primary metric**, so it cannot express any difference between the top models. Deliberately not amended in-flight — a rule written to prevent post-hoc rationalisation must not be rewritten post-hoc to license a switch. `src/retrieval.py` still uses Cohere `embed-v4.0`.

**Gemini truncation caveat now closed:** `gemini-embedding-001` **silently truncates** — the longest article (3,279 of its tokens, over the 2,048 cap) and even a 6× longer input are both accepted with no error. Extent: **4 of 531 articles** (`ssl_ar_71`, `ssl_ar_64`, `ssl_en_71`, `ssl_en_64`); only SSL 64 is ground truth (q15) and Gemini ranks it #1 anyway. Real but immaterial here — and it won while carrying the handicap. Would matter if the corpus ever gains longer articles, since the failure is silent.

**Before any switch:** (1) confirm under hybrid RRF, not semantic-only — free from cache; (2) Gemini is 3072-dim vs the index's 1536, so a new Pinecone index plus an embedder abstraction in `src/retrieval.py`; (3) check Gemini's free-tier request limits against a live app — the same class of problem that removed Voyage in Phase 13.

**Process note:** this notebook was first written without the retry and per-model guard that `19. embeddings_evaluation_v2.ipynb` had *already been fixed* to include, and a transient `ConnectTimeout` on the first Cohere call killed the whole run. Same defect, repeated in a new file. Both are now in place here too.

## Phase 14 — Eval set v3: natural phrasing ✅ built and scored (see Phase 15)
`scripts/20. eval_set_v3.ipynb` → `eval/eval_v3.jsonl` + `.csv`. Direct response to Phase 13's finding that `eval_v2` had run out of discriminating power (5 of 7 models at recall@5 = 1.000, total spread 4 questions).

**Root cause:** v2's questions were written *by reading the articles*, so they reused the statute's vocabulary and each question sat close to a paraphrase of its own answer — "What are the conditions for entitlement to the unemployment allowance?" against SSL Art 50, which is headed exactly that.

**What changed:** only `question_text`. Every question now describes a *situation* in ordinary words — "I just lost my job. Do I qualify for unemployment payments?" The register stays **formal, not colloquial**: people type more formally than they speak, so the original formal decision was right; what was wrong was reusing the law's exact terms. All ground-truth fields (`answer_law`, `answer_article_numbers`, `question_category`, `difficulty`, `labeling_note`, …) are copied across and asserted identical field-by-field, so v2 and v3 scores are directly comparable and the gap between them measures the inflation rather than guessing at it.

**Verified, not assumed:** a lexical-echo metric (share of a question's content words that also appear in its own gold article, Arabic normalised as `src/retrieval.py` does it, stopwords removed both sides) fell **0.708 → 0.285 mean**, evenly across languages (ar 0.680→0.260, en 0.737→0.309). The notebook asserts the drop before saving. Two questions (`q44`, `q55`) score *higher* and were checked by hand — both genuinely dropped the statute term (`العطلة الأسبوعية` → `راحة في الأسبوع`), and the rise is an artifact of short questions plus unhandled Arabic clitics. Recorded in the notebook rather than optimised away: pushing the metric lower by avoiding words people actually use would make the set *less* realistic.

**Versioning:** `eval_v2.jsonl` left intact (as `eval_draft.jsonl` was when v2 was built) — Phase 13's published numbers were measured on it, and overwriting would silently make a documented result unreproducible. The notebook asserts v2 is unmodified. Each v3 row also carries the v2 wording in `question_text_statute`.

**Next:** re-score the seven Phase 13 embedding models on v3. Near-free — `eval/emb_cache/` already holds every model's 531 corpus vectors, so only 60 new query embeddings per model are needed. That answers two things: how much of Phase 13's 100% was phrasing, and whether the model ranking survives. Until then the pending RRF `k`/`w` and BM25 `k1`/`b` re-tune should stay parked — it needs a ruler that can still tell configurations apart. Only after *that* is it worth asking whether an LLM query parser earns its per-question cost, since v3 is the first version of the eval set that gives one something to improve and a way to prove it did.

## Phase 12 — Polish ⬜ ongoing
Revisit the Professional Work Law EN/AR mismatch, glossary-article chunking, and expand the eval set once real pipeline failures show what's missing.
