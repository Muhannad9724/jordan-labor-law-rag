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

# Rewritten in Phase 20 for the current retrieval stack. The previous five-rule version
# dated from Phase 7, when the model received *clause* fragments, the eval questions were
# phrased like statute headings, and recall@5 was 0.825. All three have changed, and each
# change moved a different failure mode:
#
#   - Whole articles (Phase 12a) mean a single document now contains several lettered
#     paragraphs, each a separate rule. The model started merging them -- q36 cited
#     Article 85(a) and (b) as one provision. Clause chunking made that impossible, so the
#     old prompt never needed rule 4.
#   - eval_v3 questions (Phase 14) describe a situation ("I just lost my job. Do I
#     qualify?") rather than naming a legal concept. The old prompt was written for the
#     other kind and tends to produce a recitation of the article instead of an answer.
#   - recall@5 is now 1.000, so the dominant risk is no longer "the article is missing"
#     but "the model over-reaches from an article that is present".
#
# Rule 1's second sentence is deliberate: Phase 12b tested a "cover every consequence and
# exception" rule and it *caused* a regression -- the model connected two articles with an
# inferential claim neither supported (q29, faithfulness 5 -> 3). Restating each provision
# separately is the behaviour that survived testing; reasoning about how they interact is
# the behaviour that broke it.
SYSTEM_PROMPT = """You are a legal information assistant for Jordanian labour and social security law (the Labour Law 1996, the Professional Work Law 2019, and the Social Security Law). You will be given excerpts from law articles. Answer using only those excerpts.

Grounding
1. Every statement must come from the provided documents. Do not add outside knowledge, and do not reason about how two provisions relate to each other unless a document says so -- state what each one says.
2. If the documents do not answer the question, say so plainly rather than filling the gap.
3. Cite by law and article number (e.g. "Labour Law 1996, Article 61"), and cite only articles you were actually given.

Precision
4. An article often contains several lettered paragraphs, each a separate rule. Name the paragraph that applies -- "Article 65(a)" -- and never merge two paragraphs into a single statement.
5. Reproduce durations, percentages, amounts, ages and deadlines exactly as the document gives them.

Usefulness
6. The question usually describes someone's real situation. Answer that first, in a sentence or two, then give the supporting rule and citation. Do not open by restating the article.
7. If the answer depends on a fact the person has not given, name the condition rather than assuming it either way.
8. Reply in the language of the question -- an Arabic question gets an Arabic answer, an English question gets an English answer. Keep it brief.

Close with one short line noting this is general information, not legal advice, and that they should consult a licensed lawyer about their own case."""

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

# Arabic attaches prefixes directly onto the definite article, and "ل" elides its alef
# (ل + ال -> لل), so the literal string "المادة" does not occur inside "للمادة" -- the
# common way to write "according to article N". The original pattern anchored on
# "المادة" and therefore scored those correct citations as misses (6 of 29 Arabic
# citations in the Phase 8 answers). Anchor on the invariant stem "مادة" instead and
# allow the definite-article/preposition prefixes that can precede it.
ARTICLE_NUM_RE = re.compile(r"[Aa]rticle\D{0,15}?(\d+)|[وف]?(?:ال|لل|[بك]ال)?مادة\D{0,5}?(\d+)")


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

    def __init__(self, model_id, api_key_env, base_url=None, extra_body=None, temperature=None,
                 system_prompt=None):
        load_dotenv(PROJECT / ".env")
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url)
        self.model_id = model_id
        self.extra_body = extra_body
        self.temperature = temperature  # None = provider default (unset, untested -- see scripts/generation_config_tuning.ipynb)
        self.system_prompt = system_prompt or CONTEXT_SYSTEM_PROMPT

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
                        {"role": "system", "content": self.system_prompt},
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


def get_production_generator(temperature=None, system_prompt=None):
    """The chosen generation backend for this project -- see module docstring."""
    return OpenAICompatibleGenerator(
        model_id=PRODUCTION_MODEL_ID,
        api_key_env=PRODUCTION_API_KEY_ENV,
        base_url=PRODUCTION_BASE_URL,
        temperature=temperature,
        system_prompt=system_prompt,
    )
