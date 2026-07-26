"""Hybrid retrieval: Pinecone semantic search + BM25 keyword search, combined via RRF.

Config (k=5, w=0.85 for RRF; k1=1.5, b=0.75 for BM25) was chosen empirically in
scripts/search_method_evaluation.ipynb -- see that notebook for the full comparison
against pure semantic, pure keyword, weighted-sum hybrid, and the RRF/BM25 parameter
sweeps. Do not change these constants without re-running that eval.
"""
import os
import re
from pathlib import Path

import cohere
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

PROJECT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT / "0. cleaned_data" / "chunks.parquet"
PINECONE_INDEX_NAME = "jordan-labor-law"
COHERE_MODEL = "embed-v4.0"

RRF_K = 5
RRF_W = 0.85  # weight on the semantic ranker; keyword gets (1 - RRF_W)
BM25_K1 = 1.5
BM25_B = 0.75
CANDIDATE_TOP_K = 50  # how many candidates each ranker contributes before fusing

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


class Retriever:
    def __init__(self):
        load_dotenv(PROJECT / ".env")
        self.chunks = pd.read_parquet(CHUNKS_PATH)
        self.chunks["article_number"] = self.chunks["article_number"].astype(str)
        self.chunks["clause"] = self.chunks["clause"].fillna("").astype(str)
        self.chunk_ids = self.chunks["chunk_id"].tolist()
        self.id_to_row = {cid: i for i, cid in enumerate(self.chunk_ids)}

        self.cohere_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

        corpus_tokens = [tokenize(t) for t in self.chunks["text"].tolist()]
        self.bm25 = BM25Okapi(corpus_tokens, k1=BM25_K1, b=BM25_B)

    def _embed_query(self, query):
        resp = self.cohere_client.embed(
            texts=[query], model=COHERE_MODEL, input_type="search_query", embedding_types=["float"]
        )
        return resp.embeddings.float_[0]

    def _semantic_candidates(self, query, top_k):
        emb = self._embed_query(query)
        res = self.index.query(vector=emb, top_k=top_k, include_metadata=False)
        return [m["id"] for m in res["matches"]]  # already rank-ordered, best first

    def _keyword_candidates(self, query, top_k):
        scores = self.bm25.get_scores(tokenize(query))
        top_idx = np.argsort(-scores)[:top_k]
        return [self.chunk_ids[i] for i in top_idx]

    def retrieve(self, query, top_k=10):
        """Hybrid retrieval: RRF-combine Pinecone semantic + BM25 keyword rankings.

        top_k=10 (not 5) is the production default -- tested head-to-head against top_k=5
        in scripts/generation_config_tuning.ipynb: same hit-rate/fabrication-rate on the
        regex proxy, but noticeably better LLM-judge worst-case scores (min faithfulness
        4 vs 2, min relevancy 3 vs 2), since a few eval questions have their correct
        article sitting at rank 6-7 -- inside recall@10 but outside recall@5.
        """
        semantic_ids = self._semantic_candidates(query, CANDIDATE_TOP_K)
        keyword_ids = self._keyword_candidates(query, CANDIDATE_TOP_K)

        rrf_scores = {}
        for rank, cid in enumerate(semantic_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + RRF_W / (RRF_K + rank)
        for rank, cid in enumerate(keyword_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1 - RRF_W) / (RRF_K + rank)

        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

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
