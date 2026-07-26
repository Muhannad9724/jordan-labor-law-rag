"""Answer generation: grounded chat over retrieved chunks.

Two mechanisms, depending on the model:
- Cohere (`Generator`): native `documents` + `citation_options` params -- the model
  returns structured citations mapping spans of its answer back to specific document
  ids, which are mapped back to (law, article_number) here.
- Any OpenAI-compatible endpoint (`OpenAICompatibleGenerator`, used for OpenAI direct
  and for OpenRouter-hosted open models like Qwen): no native grounding API, so the
  documents are formatted into the prompt and the model is instructed to name the law
  and article number inline; citations are then extracted from the answer text via
  regex and checked against what was actually retrieved, which doubles as a fabricated-
  citation check (did it cite an article that wasn't even in the context it was given).

Production pick: `get_production_generator()` -- GPT-4o-mini via OpenRouter. Chosen in
scripts/generation_llm_comparison.ipynb over Cohere command-a-03-2025 and Qwen2.5-72B on
a 10-question correctness/fabrication check (10/10 correct, 0 fabricated citations, and
the cheapest of the three at ~$0.0004/question). `Generator` (Cohere) is kept for that
comparison, not used in production.
"""
import os
import re
import time
from pathlib import Path

import cohere
from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parent.parent
CHAT_MODEL = "command-a-03-2025"

SYSTEM_PROMPT = """You are a legal information assistant for Jordanian labor law and social security law (the 1996 Labor Law, its 2019 Professional Work Law amendment, and the Social Security Law). You are given law article excerpts as documents. Answer the user's question using ONLY the information in those documents.

Rules:
1. Base your answer strictly on the provided documents. Do not use outside knowledge or assumptions about the law.
2. If the documents do not contain enough information to answer the question, say so clearly and honestly instead of guessing.
3. Always mention which law and article number(s) support your answer (e.g., "Labor Law 1996, Article 61").
4. Answer in the same language the question was asked in: an Arabic question gets an Arabic answer, an English question gets an English answer.
5. End every answer with a brief disclaimer that this is general information, not legal advice, and the user should consult a licensed lawyer for their specific situation."""

LAW_DISPLAY_NAMES = {
    "ssl": "Social Security Law",
    "labor96": "Labor Law 1996",
    "prof19": "Professional Work Law 2019",
}


class Generator:
    def __init__(self):
        load_dotenv(PROJECT / ".env")
        self.client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

    def _to_documents(self, chunks):
        return [
            {
                "id": c["chunk_id"],
                "data": {
                    "law": LAW_DISPLAY_NAMES.get(c["law"], c["law"]),
                    "article_number": c["article_number"],
                    "language": c["language"],
                    "text": c["text"],
                },
            }
            for c in chunks
        ]

    def generate(self, query, chunks):
        """Generate a grounded answer from already-retrieved chunks (see src/retrieval.py)."""
        resp = self.client.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            documents=self._to_documents(chunks),
            citation_options={"mode": "ENABLED"},
        )

        answer = "".join(item.text for item in (resp.message.content or []) if item.type == "text")

        citations = []
        for cit in (resp.message.citations or []):
            for src in (cit.sources or []):
                doc = getattr(src, "document", None) or {}
                citations.append({
                    "cited_text": cit.text,
                    "law": doc.get("law"),
                    "article_number": doc.get("article_number"),
                    "language": doc.get("language"),
                })

        return {"answer": answer, "citations": citations, "response": resp}


CONTEXT_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You will be given a list of source documents, each labeled with its law and article number. When you cite a claim, name the actual law and article number directly in your answer text (e.g., "Labor Law 1996, Article 61") -- do not refer to documents by number (e.g. do not write "Document 1")."""

ARTICLE_NUM_RE = re.compile(r"[Aa]rticle\D{0,15}?(\d+)|المادة\D{0,5}?(\d+)")


def _format_context(chunks):
    parts = []
    for i, c in enumerate(chunks, start=1):
        law_name = LAW_DISPLAY_NAMES.get(c["law"], c["law"])
        parts.append(
            f"[Document {i}]\nLaw: {law_name}\nArticle: {c['article_number']}\n"
            f"Language: {c['language']}\nText: {c['text']}"
        )
    return "\n\n".join(parts)


class OpenAICompatibleGenerator:
    """Generator for any OpenAI-compatible chat endpoint (OpenAI direct, or OpenRouter for open models)."""

    def __init__(self, model_id, api_key_env, base_url=None, extra_body=None, temperature=None):
        load_dotenv(PROJECT / ".env")
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url)
        self.model_id = model_id
        self.extra_body = extra_body
        self.temperature = temperature  # None = provider default (unset, untested -- see scripts/generation_config_tuning.ipynb)

    def generate(self, query, chunks, max_retries=5):
        context = _format_context(chunks)
        available_articles = {c["article_number"] for c in chunks}

        resp = None
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {query}"},
                    ],
                    extra_body=self.extra_body,
                    **({"temperature": self.temperature} if self.temperature is not None else {}),
                )
                if resp.choices:
                    break
            except Exception as e:
                last_error = e
            time.sleep(min(30, 2 ** attempt))
        else:
            raise RuntimeError(
                f"No usable response from {self.model_id} after {max_retries} attempts"
                f"{f' (last error: {last_error})' if last_error else ''}"
            )

        answer = resp.choices[0].message.content or ""

        cited_articles = {g1 or g2 for g1, g2 in ARTICLE_NUM_RE.findall(answer)}
        citations = [
            {"article_number": a, "in_retrieved_context": a in available_articles}
            for a in cited_articles
        ]

        return {"answer": answer, "citations": citations, "response": resp}


PRODUCTION_MODEL_ID = "openai/gpt-4o-mini"
PRODUCTION_API_KEY_ENV = "OPENROUTER_API_KEY"
PRODUCTION_BASE_URL = "https://openrouter.ai/api/v1"


def get_production_generator(temperature=None):
    """The chosen generation backend for this project -- see module docstring."""
    return OpenAICompatibleGenerator(
        model_id=PRODUCTION_MODEL_ID,
        api_key_env=PRODUCTION_API_KEY_ENV,
        base_url=PRODUCTION_BASE_URL,
        temperature=temperature,
    )
