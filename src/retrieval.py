"""Hybrid retrieval: Pinecone semantic search + BM25 keyword search, combined via RRF.

Config (k=5, w=0.90 for RRF; k1=1.5, b=0.75 for BM25). The k/k1/b values come from
scripts/search_method_evaluation.ipynb (Phase 6) -- see that notebook for the comparison
against pure semantic, pure keyword, weighted-sum hybrid, and the RRF/BM25 parameter
sweeps. Do not change them without re-running that eval.

RRF_W was 0.85 until Phase 18 and is now 0.90. scripts/fusion_weight_tuning.ipynb swept
w across 0.00-1.00 in steps of 0.01 on eval_v3 at top_k=5 and found the curve is steep up
to ~0.75 and then FLAT: every value in roughly [0.75, 0.90] performs the same. The
difference between 0.85 and 0.90 is ~1 question in 60, and a power calculation puts the
sample needed to resolve it at ~520 questions -- so 0.90 is a choice inside a plateau,
not a measured improvement over 0.85.

What that notebook did establish, and what should stop anyone re-tuning this:
  - Tuning w on this eval set does NOT transfer. Under 500 split-half cross-validations
    (tune on 30 questions, score the untouched 30), a tuned w scored *below* the untuned
    setting for both candidate embedding models. A 2000-resample bootstrap had said the
    optimum was stable -- it was, but stability of an argmax drawn from the same 60
    questions is not evidence it generalises. Only held-out data showed the plateau.
  - The keyword half contributes very little here. Against pure semantic (w=1.0) it helped
    1 question and hurt 0 (Gemini), or helped 2 and hurt 0 (Cohere) -- not significant, but
    never harmful. Exact-match queries were the best argument for keeping it and that was
    tested and refuted: 12 article-number lookups ("ما نص المادة 65...", "What does Article
    90 ... say?") were all found at rank 1-2 by pure semantic, with BM25 changing none.
    It is retained because it costs ~1 ms/query against a 200-500 ms embedding call, has
    never made a result worse, and insures against query shapes a 60-question eval cannot
    cover.

Granularity is one unit per *article*, not per clause. Phase 2 originally split long
articles at their clauses; scripts/chunk_granularity_evaluation.ipynb tested that against
whole articles and whole articles won every retrieval metric (recall@1 .875 vs .844,
recall@10 1.000 vs .969, MRR .904 vs .874 over the 32 same-language eval questions).
The reason is that these articles are single-topic -- SSL 71 is entirely about military
insureds' pensions, SSL 64 entirely about early retirement -- so a clause is a fragment
of one subject rather than a separable topic, and several were literally unreadable alone
("Pension stipulated in Paragraph (a) ... shall be increased by (0.5%)" never says which
pension). Six such near-identical fragments of Article 71 crowded the whole result list on
q22 and pushed the correct article out entirely. Rebuilt in scripts/rechunking_articles.ipynb.

Embedding model is Gemini `gemini-embedding-001` since Phase 19, replacing Cohere
`embed-v4.0` which had held the slot since Phase 4. Phase 4's original comparison was
re-run from scratch because every input to it had changed -- it scored clause chunks, on
an eval set whose cross-lingual questions carried the whole argument, with two candidates
capped at max_length=1024 despite supporting 8192. On the rebuilt setup Gemini wins
11-2 on paired ranks (p = 0.022) under this exact hybrid configuration, at every RRF
weight including Cohere's own optimum, and by +0.093 MRR -- about seven times what every
fusion parameter in this file is worth combined. Full trail: scripts/embeddings_evaluation_v2
.ipynb, embeddings_rescore_v3.ipynb, embeddings_hybrid_v3.ipynb, vector_store_gemini.ipynb.

Known Gemini property: it **silently truncates input past 2,048 tokens** -- no error, just
a vector describing less text. 4 of 531 articles exceed it (ssl_ar_71, ssl_ar_64, ssl_en_71,
ssl_en_64). It won the comparison while carrying that handicap, so it is accepted, but any
future corpus with longer articles needs this re-checked; nothing will raise to warn you.

Rollback is still possible but is no longer instant. The Cohere and clause-level Pinecone
indexes were deleted in Phase 19 once Gemini had been verified live, so switching
EMBED_PROVIDER back to "cohere" now requires REBUILDING that index first -- the constant
alone will point at an index that does not exist and queries will fail.

To roll back to Cohere:
  1. Rebuild the 1536-dim index. scripts/vector_store_gemini.ipynb is the template; the
     Cohere article vectors are still cached at
     eval/emb_cache/cohere-embed-v4__<corpus-hash>__corpus.npy, so this costs nothing if
     that cache survives. If it does not, re-embed the 531 articles (~$0.02).
  2. Set EMBED_PROVIDER = "cohere".
To roll back the Phase 12a granularity change as well, chunks.parquet (739 clause chunks)
is still on disk and would need re-embedding and a further index (~$0.02).
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

PROJECT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT / "0. cleaned_data" / "chunks_articles.parquet"

EMBED_PROVIDER = "gemini"  # "gemini" | "cohere" -- see rollback note above
GEMINI_MODEL = "gemini-embedding-001"
COHERE_MODEL = "embed-v4.0"

# The index must match the provider: dimensions differ (3072 vs 1536), so pointing at
# the wrong one fails at query time rather than silently returning nonsense.
# NOTE: only the gemini index currently exists. The cohere entry is kept so the rollback
# path stays documented in code, but it must be rebuilt before use -- see the module
# docstring.
PINECONE_INDEX_BY_PROVIDER = {
    "gemini": "jordan-labor-law-articles-gemini",
    "cohere": "jordan-labor-law-articles",  # DELETED in Phase 19 -- rebuild before use
}
PINECONE_INDEX_NAME = PINECONE_INDEX_BY_PROVIDER[EMBED_PROVIDER]

RRF_K = 5
RRF_W = 0.90  # weight on the semantic ranker; keyword gets (1 - RRF_W) = 10%
BM25_K1 = 1.5
BM25_B = 0.75
CANDIDATE_TOP_K = 50  # how many candidates each ranker contributes before fusing

# Restrict retrieval to the question's own language. Adopted after measuring it on
# eval_v3 (60 questions): recall@1 0.867 -> 0.900, MRR 0.922 -> 0.939, full coverage
# unchanged at 0.950, no question lost coverage, worst rank still 4.
#
# The reason it helps has nothing to do with answer language. Every article exists
# twice, once per language, so roughly half of the CANDIDATE_TOP_K candidates were
# the other-language copy of something already in the list. They consumed the
# candidate budget without adding an article, making the effective search depth
# about 25 rather than 50. Filtering doubles it: measured on a rates probe, one
# article moved from worse-than-rank-30 to rank 22, another from 25 to 13.
#
# What this gives up: exactly one article in the corpus (prof19 Article 13) exists
# in English only, so it is unreachable from an Arabic question. Everything else is
# perfectly parallel -- 111/111 ssl, 142/142 labor96 -- so cross-language retrieval
# was buying almost nothing. Note that eval_v2 and eval_v3 contain no cross-lingual
# questions, so neither set can measure that loss; the claim rests on the corpus
# audit above, not on the eval numbers.
FILTER_BY_LANGUAGE = True

# Sibling expansion. Some questions are answered by a *set* of parallel articles
# rather than one: "how much does the company pay into social security" is the sum
# of four separate insurances, each funded by its own article (SSL 24 work injury,
# 42 maternity, 48 unemployment, 59 old-age/disability/death, 72 public sector).
#
# Ranking alone cannot fix that. Measured on the rates probe, asking generally put
# those articles at ranks 11, 25 and worse-than-30 -- yet asking about one branch by
# name put the same article at rank 1. Each article is strongly about its own branch
# and only weakly about the general question, so a general query never gathers them.
#
# The families are not hand-listed. They are read off a structural phrase the statute
# uses to introduce a funding article, which independently yields the same five
# articles in both languages. If any member is retrieved, the rest are promoted to
# sit beside it.
#
# Query decomposition was tested as the alternative and scored exactly baseline while
# costing an LLM call per question: the model split "how much does the company pay"
# into "employer rate" and "employee rate" -- paraphrases -- because nothing in the
# question names the four branches and it has no way to know they exist. The structure
# lives in the corpus, not in the question.
EXPAND_FAMILIES = True
FAMILY_MARKERS = {"ar": "تتكون مصادر تمويل", "en": "shall be financed"}
# When expansion fires the window widens, because injecting siblings into a fixed 5
# would push out relevant non-family articles -- measured: it cost r05 half its
# coverage. Questions that touch no family keep the Phase 17 default of 5.
FAMILY_TOP_K = 12
# Phase 24b. A single "top-5" trigger missed exactly the case the feature exists for:
# stating a salary ("راتبي 1000 دينار كم ادفع للضمان الاجتماعي") DILUTES the query enough
# that the funding articles drop out of the top 5-8 -- sometimes much deeper than
# expected. Firing requires >=2 family members within FAMILY_TRIGGER_DEPTH, not "the
# single best one within top_k": one member showing up is incidental, several
# clustered together is the topic.
#
# The depth constant below was picked from the TRUE unexpanded RRF ranking -- i.e. the
# ranking _expand_families itself receives, before it runs. An earlier attempt measured
# positions by calling retrieve(top_k=30), which internally calls _expand_families and
# so returns an ALREADY-REORDERED list; that produced clustered-looking positions
# (5,6,7,8) that were an artifact of the very trigger being tuned, not the raw ranking.
# Recomputing directly from _semantic_candidates/_keyword_candidates before fusion-
# ordering was ever touched, the real positions for salary-stated questions are much
# deeper and more scattered, e.g. [14, 35] for the reported failure, [13, 34, 49] for a
# near-paraphrase, [6, 33, 35, 47] for another. Depth 20 (the first, contaminated pick)
# fires on 0 of these; depth 35 fires on 6 of 7 -- the seventh is a bare "راتبي 1000
# دينار" fragment with no verb or question, where only ONE family member appears
# anywhere in the top 50 candidates, so no depth-based trigger can reach it.
#
# At depth 35 / count 2: fires on the eval/eval_rates.jsonl probe set 10/10, and on
# eval_v3 (60 general questions) only 2/60 (q10, q16) -- both confirmed to cost
# nothing on retrieval metrics (recall@1/@5, MRR, full coverage all unchanged; see
# scripts/24. deterministic_calculation.ipynb).
#
# Phase 25 widened 35 -> 37. Two new probe questions ("راتبي 1000 دينار كم يُقتطع
# مني" and a رواتب+"وضح كل نسبة" phrasing) each had only ONE family member within
# depth 35 (Article 42 at true raw rank 36 and 37 respectively) -- one member is
# exactly the "incidental" case COUNT=2 exists to reject, so neither fired and both
# fell back to an incomplete family. Swept depth 35-40 against eval_v3 + eval_rates
# + the two new probes with FAMILY_TOP_K's own slicing (8 when fired, not 5) applied
# correctly this time -- an earlier pass of this same sweep truncated to 5
# unconditionally and wrongly read eval_rates as 9/10 instead of the true 10/10.
# 37 is the exact minimum that catches both new probes (36 catches only one) with
# recall@1/@5, MRR and full coverage on eval_v3 bit-identical to depth 35 (same two
# questions fire, nothing new). 39+ starts firing on q05/q12/q43, which stays
# "covered" within the widened 8 slots but costs recall@1/MRR by knocking the
# correct article off rank 1 -- so wider is not free past this point.
FAMILY_TRIGGER_DEPTH = 37
FAMILY_TRIGGER_COUNT = 2
#
# Phase 25b: FAMILY_TOP_K was raised 8 -> 12 (see above) specifically to fix q10
# ("am I eligible for a maternity payment"), the one eval_v3 question this whole
# mechanism was still costing. Root cause: q10's family fires correctly (Article 42
# clusters with one other member within FAMILY_TRIGGER_DEPTH) but the target article,
# SSL 44, sits at combined position 12 after expansion -- one slot past the old
# FAMILY_TOP_K=8 cutoff. Tested three candidates against eval_v3 + eval_rates before
# picking this one:
#   A. FAMILY_TOP_K 8->12 (adopted): fixes q10, zero regressions elsewhere, full
#      coverage improves 0.950 -> 0.967.
#   B. exclude Article 42 from the >=COUNT trigger members (it is topically noisy --
#      shared vocabulary with ordinary maternity-eligibility questions unrelated to
#      contributions): does NOT fix q10 -- the family simply never fires for that
#      question, so the extra slots from A never get used -- but genuinely improves
#      the rest of eval_v3 (recall@1 .883->.900, MRR .926->.939, worst_rank 5->4),
#      because Article 42 alone was over-triggering on other, unrelated questions too.
#   A+B together: numerically identical to B alone here, since B pre-empts A's
#      mechanism for q10 specifically (family never fires, so widening the slice is
#      moot). Not useful as a combination for this goal.
# B's finding is real and worth revisiting on its own terms, but it does not solve
# the problem it was tested against, so only A shipped.

_ARABIC_RE = re.compile(r"[؀-ۿ]")


def detect_language(text):
    """'ar' if the text contains any Arabic script, else 'en'.

    Same rule as generation.detect_language, duplicated rather than imported so
    retrieval keeps no dependency on generation. If the two ever diverge, retrieval
    would filter to one language while the answer is written in the other.
    """
    return "ar" if _ARABIC_RE.search(str(text)) else "en"

AR_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰـ]")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize_ar(text):
    text = AR_DIACRITICS.sub("", text)
    text = re.sub(r"[آأإ]", "ا", text)  # alef variants -> bare alef
    text = text.replace("ى", "ي")  # alef maksura -> yaa
    text = text.replace("ة", "ه")  # ta marbuta -> ha
    return text


def tokenize(text):
    text = normalize_ar(text)
    return [t.lower() for t in WORD_RE.findall(text)]


def _make_embedder(provider):
    """Return a callable str -> list[float] for query embedding.

    Kept as a small factory rather than an inline branch so that switching providers is
    one constant, and so the client is built once at Retriever construction instead of
    per request. Each provider is imported lazily: the deployed app only needs the SDK
    for the provider it actually uses.
    """
    if provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        cfg = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")

        def embed(query):
            # RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT is not cosmetic -- the corpus was
            # indexed as RETRIEVAL_DOCUMENT, and mixing them degrades similarity.
            resp = client.models.embed_content(model=GEMINI_MODEL, contents=[query], config=cfg)
            return list(resp.embeddings[0].values)

        return embed

    if provider == "cohere":
        import cohere

        client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

        def embed(query):
            resp = client.embed(texts=[query], model=COHERE_MODEL,
                                input_type="search_query", embedding_types=["float"])
            return resp.embeddings.float_[0]

        return embed

    raise ValueError(f"unknown EMBED_PROVIDER: {provider!r}")


class Retriever:
    def __init__(self):
        load_dotenv(PROJECT / ".env")
        self.chunks = pd.read_parquet(CHUNKS_PATH)
        self.chunks["article_number"] = self.chunks["article_number"].astype(str)
        self.chunks["clause"] = self.chunks["clause"].fillna("").astype(str)
        self.chunk_ids = self.chunks["chunk_id"].tolist()
        self.id_to_row = {cid: i for i, cid in enumerate(self.chunk_ids)}

        self.embedder = _make_embedder(EMBED_PROVIDER)
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

        corpus_tokens = [tokenize(t) for t in self.chunks["text"].tolist()]
        self.bm25 = BM25Okapi(corpus_tokens, k1=BM25_K1, b=BM25_B)
        self._masks = {}  # language -> boolean mask over chunk rows, built once
        self._families = self._build_families()

    def _build_families(self):
        """{(law, language): [{article numbers}, ...]} -- sibling groups, split by scope.

        Derived from the text, not declared by hand, so it stays correct if the corpus
        is rebuilt. Both markers independently return the same five SSL articles.

        The scope split is not cosmetic. The marker groups {24, 42, 48, 59, 72}, but
        **Article 72 is the public-sector and military funding regime** -- an alternative
        to the other four, not a fifth component of the same total. It carries its own
        11% and 6.5%, numerically identical to the private-sector ones, plus 17%, 21.5%
        and 26%. Promoting all five together put 16 percentages in front of the model
        when only 6 answered the question, and measured 3/6 on the resulting arithmetic.

        Members are therefore grouped by scope, and expansion promotes only the group
        the retrieved article belongs to: a private-sector question gets {24, 42, 48, 59}
        and a military one gets 72 alone.
        """
        fams = {}
        if not EXPAND_FAMILIES:
            return fams
        from .calc import article_components  # local: keeps the import graph one-way

        for (law, lang), grp in self.chunks.groupby(["law", "language"]):
            marker = FAMILY_MARKERS.get(lang)
            if not marker:
                continue
            by_scope = {}
            for r in grp.itertuples():
                text = str(r.text)
                if marker not in text:
                    continue
                comps = article_components(text)
                # "private" if the article states any ordinary private-sector rate;
                # otherwise the scope its components actually describe.
                if any(c["scope"] == "private" and c["bearer"] for c in comps):
                    scope = "private"
                else:
                    scopes = [c["scope"] for c in comps] or ["private"]
                    scope = max(set(scopes), key=scopes.count)
                by_scope.setdefault(scope, set()).add(str(r.article_number))
            groups = [m for m in by_scope.values() if len(m) > 1]
            if groups:
                fams[(law, lang)] = groups
        return fams

    def _embed_query(self, query):
        return self.embedder(query)

    def _semantic_candidates(self, query, top_k, language=None):
        emb = self._embed_query(query)
        kwargs = {"vector": emb, "top_k": top_k, "include_metadata": False}
        if language:
            kwargs["filter"] = {"language": {"$eq": language}}
        res = self.index.query(**kwargs)
        return [m["id"] for m in res["matches"]]  # already rank-ordered, best first

    def _keyword_candidates(self, query, top_k, language=None):
        scores = self.bm25.get_scores(tokenize(query))
        if language:
            # -inf rather than dropping rows, so positions still line up with chunk_ids
            scores = np.where(self._lang_mask(language), scores, -np.inf)
        top_idx = np.argsort(-scores)[:top_k]
        return [self.chunk_ids[i] for i in top_idx]

    def _expand_families(self, chunk_ids, language):
        """Bring every member of a triggered family to the front, in relative rank order.

        Returns (ids, fired). Fires when >= FAMILY_TRIGGER_COUNT members of the SAME
        family are clustered within the first FAMILY_TRIGGER_DEPTH candidates -- one
        member showing up is incidental, several clustered together is the topic. See
        FAMILY_TRIGGER_DEPTH/COUNT above for why a single-member "top_k" trigger missed
        the case this feature exists for.

        Members are moved to the FRONT rather than inserted beside the trigger. That is
        not cosmetic: the caller slices the returned list to FAMILY_TOP_K afterward, and
        a trigger at rank 14-17 (routine once a salary figure dilutes the query) means
        `chunk_ids[:i+1]` alone already exceeds FAMILY_TOP_K -- prepending it, as an
        earlier version did, pushed every sibling past the slice before the caller ever
        saw them. Confirmed with a scratch repro before this rewrite: the old insertion
        delivered 0 of 3 siblings into an 8-slot answer when the trigger sat at rank 14.
        Members already in chunk_ids keep their relative rank order; a sibling that was
        never independently retrieved (a real case -- SSL 48 does not surface in the top
        30 for some phrasings) is appended after them, since there is no rank to sort by.
        """
        if not self._families:
            return chunk_ids, False
        pos = {cid: i for i, cid in enumerate(chunk_ids)}
        window = chunk_ids[:FAMILY_TRIGGER_DEPTH]
        seen = set()
        for cid in window:
            row = self.chunks.iloc[self.id_to_row[cid]]
            groups = self._families.get((row["law"], language)) or []
            # only the scope group this article belongs to -- see _build_families
            fam = next((g for g in groups if str(row["article_number"]) in g), None)
            if fam is None:
                continue
            key = (row["law"], frozenset(fam))
            if key in seen:
                continue
            seen.add(key)

            present = sum(
                1 for c2 in window
                if self.chunks.iloc[self.id_to_row[c2]]["law"] == row["law"]
                and str(self.chunks.iloc[self.id_to_row[c2]]["article_number"]) in fam
            )
            if present < FAMILY_TRIGGER_COUNT:
                continue

            member_ids = []
            for art in sorted(fam, key=lambda a: int(re.sub(r"\D", "", a) or 0)):
                sid = self._article_chunk_id(row["law"], language, art)
                if sid:
                    member_ids.append(sid)
            if not member_ids:
                continue
            member_ids.sort(key=lambda c: pos.get(c, 10**9))
            rest = [c for c in chunk_ids if c not in member_ids]
            return member_ids + rest, True
        return chunk_ids, False

    def _article_chunk_id(self, law, language, article_number):
        if not hasattr(self, "_by_article"):
            self._by_article = {
                (r.law, r.language, str(r.article_number)): r.chunk_id
                for r in self.chunks.itertuples()
            }
        return self._by_article.get((law, language, str(article_number)))

    def _lang_mask(self, language):
        if language not in self._masks:
            self._masks[language] = (self.chunks["language"] == language).to_numpy()
        return self._masks[language]

    def retrieve(self, query, top_k=5, language=None):
        """Hybrid retrieval: RRF-combine Pinecone semantic + BM25 keyword rankings.

        top_k was 10 from Phase 8b until Phase 16. That default was never about recall:
        scripts/generation_config_tuning.ipynb found top_k=10 and top_k=5 tied on hit rate
        and fabrication rate, and picked 10 only because it lifted the LLM-judge worst
        cases (min faithfulness 4 vs 2, min relevancy 3 vs 2). That in turn traced to a
        handful of questions -- q29 especially -- whose correct article sat at rank 6-7,
        inside recall@10 but outside recall@5. The extra slots existed to catch those.

        Measured again in scripts/embeddings_hybrid_v3.ipynb on article-level chunks,
        searching 30 deep across both eval sets (120 question-runs): with the current
        embedding model, *no* question puts its correct article between rank 6 and 10.
        Worst rank is 4 on eval_v3, 2-3 on eval_v2. Slots 6-10 are padding, and they
        are not free -- they double the context handed to the generator (mean 4172 ->
        2218 tokens, worst case 21997 -> 11058), which is both cost and, per Phase 12a,
        crowding risk.

        Two caveats worth keeping in view:
          - This is a retrieval measurement. Phase 8b's original argument was about
            answer quality, and that has not been re-tested at top_k=5 under article
            chunking; generation work is parked until the retriever is settled.
          - Worst rank drifted 2-3 -> 4 as the eval questions got harder (statute-worded
            v2 -> naturally-worded v3). Both stay inside 5, but the trend runs toward
            deeper ranks, so a harder question set could change this.

        Note this is model-dependent, not a universal property: on the same test,
        qwen3-embedding-0.6b puts 9 of 60 correct articles in slots 6-10. If the
        embedding model changes, re-run that check before trusting top_k=5.
        """
        lang = (language or detect_language(query)) if FILTER_BY_LANGUAGE else None
        semantic_ids = self._semantic_candidates(query, CANDIDATE_TOP_K, lang)
        keyword_ids = self._keyword_candidates(query, CANDIDATE_TOP_K, lang)

        # Fall back to an unfiltered search rather than returning nothing. This should
        # not fire -- both languages cover the whole corpus -- but a wrong language
        # guess on a code-switched question would otherwise search an empty pool, and
        # degrading to today's behaviour beats degrading to no answer.
        if lang and not semantic_ids:
            semantic_ids = self._semantic_candidates(query, CANDIDATE_TOP_K)
            keyword_ids = self._keyword_candidates(query, CANDIDATE_TOP_K)

        rrf_scores = {}
        for rank, cid in enumerate(semantic_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + RRF_W / (RRF_K + rank)
        for rank, cid in enumerate(keyword_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1 - RRF_W) / (RRF_K + rank)

        ordered = [cid for cid, _ in
                   sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)]
        ordered, fired = self._expand_families(ordered, lang or detect_language(query))
        limit = FAMILY_TOP_K if (fired and top_k <= FAMILY_TOP_K) else top_k
        ranked = [(cid, rrf_scores.get(cid, 0.0)) for cid in ordered[:limit]]

        results = []
        for cid, score in ranked:
            row = self.chunks.iloc[self.id_to_row[cid]]
            results.append({
                "chunk_id": cid,
                "score": score,
                "law": row["law"],
                "language": row["language"],
                "article_number": row["article_number"],
                "clause": row["clause"],
                "text": row["text"],
                "source_file": row["source_file"],
            })
        return results
