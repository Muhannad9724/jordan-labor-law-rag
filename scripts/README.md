# Notebooks

Numbered in the order the work actually happened, which is the order `ROADMAP.md`
narrates. The numbers are chronological, not a pipeline — later notebooks
deliberately revisit earlier decisions, and several of them overturn one.

Nothing here is dead weight. Notebooks that tested an option and **rejected** it
(11, 17, 23) are kept on purpose: they are the record that the decision was made
with data rather than taste.

Every notebook resolves the project root by walking up from its own working
directory, so they run from either the repo root or `scripts/`. They need `.env`
at the repo root for API keys.

| # | Notebook | Phase | What it settled |
|---|---|---|---|
| 01 | `files_text_extraction` | 1 | PDF → OCR → per-article `Knowledgebase/*.xlsx` |
| 02 | `chunking` | 2 | 739 clause-level chunks (later replaced by 15) |
| 03 | `embeddings_evaluation` | 4 | Picked Cohere `embed-v4.0` over 4 others (later overturned by 21) |
| 04 | `vector_store_indexing` | 5 | Pinecone serverless, chosen over Qdrant Cloud |
| 05 | `search_method_evaluation` | 6 | Hybrid RRF over pure semantic / pure BM25 / weighted sum |
| 06 | `retrieval` | 6 | `src/retrieval.py` reproduces the eval numbers exactly |
| 07 | `generation` | 7 | First generator, Cohere — picked without comparison |
| 08 | `generation_llm_comparison` | 7 | Head-to-head → `gpt-4o-mini`, ~17× cheaper and more faithful |
| 09 | `evaluation_harness` | 8 | First full 40-question end-to-end run |
| 10 | `generation_config_tuning` | 8b | `top_k` 5 vs 10, `temperature` 0 vs default |
| 11 | `reranker_evaluation` | 8c | **Rejected** Cohere `rerank-v3.5` — hurt same-language precision |
| 12 | `api_demo` | 9 | FastAPI backend proven over real HTTP, not direct calls |
| 13 | `retrieval_improvements` | 12a | Diagnosed the q22 failure as crowding, not bad ranking |
| 14 | `chunk_granularity_evaluation` | 12a | Whole articles beat clause chunks on every retrieval metric |
| 15 | `rechunking_articles` | 12a | Rebuilt the corpus: 739 chunks → 531 whole articles |
| 16 | `kb_watermark_cleanup` | 12a | Stripped OCR watermark text out of the knowledge base |
| 17 | `prompt_clause_precision` | 12b | First noise-floor-controlled prompt test; **rule 7 rejected** |
| 18 | `eval_set_v2` | 13 | 40 ad-hoc questions → 60 built to an explicit brief |
| 19 | `embeddings_evaluation_v2` | 13 | Re-ran the Phase 4 choice on the real corpus; Cohere kept |
| 20 | `eval_set_v3` | 14 | Rewrote questions in natural wording — v2 echoed statute vocabulary |
| 21 | `embeddings_rescore_v3` | 15 | On honest wording every model drops; **Gemini beats Cohere 12–3** |
| 22 | `embeddings_hybrid_v3` | 16 | Confirmed under hybrid RRF at every fusion weight |
| 23 | `fusion_weight_tuning` | 18 | **Nothing adopted** — tuned RRF loses out-of-sample (overfitting) |
| 24 | `vector_store_gemini` | 19 | Switched to `gemini-embedding-001`, new 3072-dim index, live |
| 25 | `final_evaluation_v3` | 20 | New reference point: recall@5 1.000, faith 4.70, rel 4.55 |
| 26 | `prompt_coverage_eval` | 21 | Coverage rules + bilingual context — relevancy 4.55 → 4.80 |

Phases without a notebook: 3 (the v1 eval set was hand-built), 10–11 (frontend and
deploy), 17 (measured inside 22), 22–23 (prompt rules and multi-turn, verified
live rather than through the harness).
