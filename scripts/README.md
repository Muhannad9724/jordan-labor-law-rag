# Notebooks

Numbered in the order the work actually happened, which is the order `ROADMAP.md`
narrates. The numbers are chronological, not a pipeline — later notebooks
deliberately revisit earlier decisions, and several of them overturn one.

Nothing here is dead weight. Notebooks that tested an option and **rejected** it
(10, 16, 20) are kept on purpose: they are the record that the decision was made
with data rather than taste.

Every notebook resolves the project root by walking up from its own working
directory, so they run from either the repo root or `scripts/`. They need `.env`
at the repo root for API keys. Shared scoring helpers live in
[`evallib/`](../evallib/) rather than being redefined in each notebook — see the
note at the bottom.

| # | Notebook | Phase | What it settled |
|---|---|---|---|
| 01 | `files_text_extraction` | 1 | PDF → OCR → per-article `Knowledgebase/*.xlsx` |
| 02 | `chunking` | 2 | 739 clause-level chunks (later replaced by 14) |
| 03 | `embeddings_evaluation` | 4 | Picked Cohere `embed-v4.0` over 4 others (later overturned by 19) |
| 04 | `vector_store_indexing` | 5 | Pinecone serverless, chosen over Qdrant Cloud |
| 05 | `search_method_evaluation` | 6 | Hybrid RRF over pure semantic / pure BM25 / weighted sum |
| 06 | `retrieval` | 6 | `src/retrieval.py` reproduces the eval numbers exactly |
| 07 | `generation_model_choice` | 7 | The convenient pick, then the head-to-head that overturned it → `gpt-4o-mini` |
| 08 | `evaluation_harness` | 8 | First full 40-question end-to-end run |
| 09 | `generation_config_tuning` | 8b | `top_k` 5 vs 10, `temperature` 0 vs default |
| 10 | `reranker_evaluation` | 8c | **Rejected** Cohere `rerank-v3.5` — hurt same-language precision |
| 11 | `api_demo` | 9 | FastAPI backend proven over real HTTP, not direct calls |
| 12 | `retrieval_improvements` | 12a | Diagnosed the q22 failure as crowding, not bad ranking |
| 13 | `chunk_granularity_evaluation` | 12a | Whole articles beat clause chunks on every retrieval metric |
| 14 | `rechunking_articles` | 12a | Rebuilt the corpus: 739 chunks → 531 whole articles |
| 15 | `kb_watermark_cleanup` | 12a | Stripped OCR watermark text out of the knowledge base |
| 16 | `prompt_clause_precision` | 12b | First noise-floor-controlled prompt test; **rule 7 rejected** |
| 17 | `eval_set_v2_and_v3` | 13, 14 | The measuring stick, rebuilt twice — see the chronology note inside |
| 18 | `embeddings_evaluation_v2` | 13 | Re-ran the Phase 4 choice on the real corpus; Cohere kept |
| 19 | `embeddings_rescore_and_hybrid_v3` | 15, 16 | On honest wording **Gemini beats Cohere**, and it survives hybrid RRF |
| 20 | `fusion_weight_tuning` | 18 | **Nothing adopted** — tuned RRF loses out-of-sample (overfitting) |
| 21 | `vector_store_gemini` | 19 | Switched to `gemini-embedding-001`, new 3072-dim index, live |
| 22 | `final_evaluation_v3` | 20 | New reference point: recall@5 1.000, faith 4.70, rel 4.55 |
| 23 | `prompt_coverage_eval` | 21 | Coverage rules + bilingual context — relevancy 4.55 → 4.80 |
| 24 | `deterministic_calculation` | 24 | Contribution arithmetic moved into code — 3/6 → 6/6 |

Phases without a notebook: 3 (the v1 eval set was hand-built), 10–11 (frontend and
deploy), 17 (measured inside 19), 22–23 (prompt rules and multi-turn, verified
live rather than through the harness).

## Merged notebooks

Three notebooks each combine two halves of one decision:

- **07** = the original Cohere pick + the comparison that overturned it
- **17** = eval set v2 + v3
- **19** = re-scoring on v3, semantic + hybrid

**17 has a chronology caveat, stated inside it.** Its two halves were *not*
consecutive: notebook 18 runs between them, and its result — five of seven models
tied at a perfect recall@5 — is the reason v3 was written at all. Read the bridge
cell before treating it as one sitting.

## Shared helpers

`metrics`, `rank_of_correct`, and the LLM-judge scaffolding used to be redefined in
each notebook — seven copies of recall@k/MRR alone. They now live in
[`evallib/`](../evallib/).

That is not tidying. `LESSONS.md` records a scoring bug fixed in one place while
other runs stayed wrong permanently; seven copies of a metric is seven places the
next one can hide.

Where a notebook's copy had a different interface (rounding, a Series argument, an
omitted key) a two-line shim preserves that notebook's exact output, so no stored
result changed. [`evallib/test_equivalence.py`](../evallib/test_equivalence.py)
proves it, against the seven original implementations frozen verbatim:

```bash
python evallib/test_equivalence.py
```
