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
import json
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
# Rule 6's second sentence is deliberate: Phase 12b tested a "cover every consequence and
# exception" rule and it *caused* a regression -- the model connected two articles with an
# inferential claim neither supported (q29, faithfulness 5 -> 3). Restating each provision
# separately is the behaviour that survived testing; reasoning about how they interact is
# the behaviour that broke it.
#
# Phase 21 adds the Coverage section, in response to the Phase 20 measurement. The dominant
# failure was omission, not hallucination -- 18 of 19 relevancy penalties were "omits X" --
# and it was NOT a retrieval failure: 9 of the 10 articles the judge said were missing were
# already sitting in the context we handed the model, and the gold article was at rank 1 in
# 16 of the 19 penalised questions. Measured directly, answers named a mean of 1.22 of the
# 3.87 distinct articles offered (32%), and 48 of 60 cited exactly one.
#
# That behaviour was instructed. The Phase 20 prompt said "Keep it brief", "answer in a
# sentence or two", and "do not reason about how two provisions relate" -- three pushes
# toward short single-source answers, and no instruction anywhere to check every document.
# So the model was obeying and the judge was penalising the obedience.
#
# The distinction that makes this safe to fix: *listing* two provisions separately is not
# the *inference across* provisions that broke Phase 12b's rule 7. Rule 6 states that
# guardrail immediately after the coverage rules, where the failure would occur.
#
# Rules 8, 9 and the second sentence of rule 3 target the other measured defects: 6 of the
# 14 faithfulness penalties were qualified rules stated flat (q42 ten weeks post-birth vs
# ten total, q15 a gender-neutral rule stated as female-specific, q03 a deadline attached
# to the wrong obligation, q28 last wage read as number of contributions, q02 conditional
# coverage stated as unconditional), and 3 answers named the wrong law entirely.
#
# Cost of the added length: 12 rules instead of 8. Every one is anchored to a counted
# failure, but rule dilution is a real risk and is part of what the next run measures.
#
# Phase 22 adds two more, from direct user feedback rather than a measured judge failure.
#
# Rule 11 rewritten: was "name the condition rather than assuming it either way" -- correct
# per Phase 20/21 (it's why q15's gender-neutral-vs-female-specific error doesn't recur), but
# only for conditions the documents themselves lay out as a short closed set. It said nothing
# about facts that only the person asking could supply -- their age, years of service, a
# date -- and the model would either guess one case or dump every branch. Now it asks.
#
# IMPORTANT LIMITATION, not solved by this prompt change: the API is single-turn. /ask takes
# only `question`, no prior conversation, and retrieval runs fresh on whatever text comes in.
# If the model asks "how many years have you contributed?" and the person replies with just
# "12", that reply becomes an entirely new, context-free query -- retrieval has no idea it's
# the answer to the model's own question, and the reply likely returns poor or unrelated
# results. This rule makes the model ask a better question; it does not give this app a way
# to hear the answer. Closing that gap needs actual conversation memory (thread history
# passed into /ask and folded into retrieval), which is a real feature, not a prompt tweak,
# and is not implemented here.
#
# Rule 13 (new "Judgment questions" section) is a different failure entirely: a question like
# "should I join the optional social security?" is not a fact lookup, it is a request for a
# personal decision, and the previous prompt had no instruction for that shape of question at
# all. Rule 2 ("say so plainly if the documents don't answer") would fire wrong here -- the
# documents DO describe the optional-insurance provisions, they just don't say what any given
# person should do, and refusing outright would be unhelpful for a question people genuinely
# ask. The other failure mode is the opposite one: giving a directive personal recommendation
# ("yes, join") would be actual advice, not legal information, and is exactly the boundary the
# closing disclaimer line already exists to keep. The rule threads between the two: state the
# benefits and the obligations/costs the documents describe, neutrally, and stop there.
SYSTEM_PROMPT = """You are a legal information assistant for Jordanian labour and social security law (the Labour Law 1996, the Professional Work Law 2019, and the Social Security Law). You will be given excerpts from law articles. Answer using only those excerpts.

Grounding
1. Every statement must come from the provided documents. Do not add outside knowledge.
2. If the documents do not answer the question, say so plainly rather than filling the gap.
3. Cite by law and article number (e.g. "Labour Law 1996, Article 61"), and cite only articles you were actually given. Name the law the document itself is labelled with -- never attribute a provision to a different law.

Coverage
4. Read every document you were given before you answer. More than one article often bears on the same question. Address each article that does, with its own citation -- do not stop at the first one that answers it.
5. Within an article you cite, cover every lettered paragraph that bears on the question. A paragraph bears on the question if it adds a condition, an exception, a limit, a minimum, or a further step the person would need to take.
6. State what each provision says on its own terms. Do not explain how two provisions combine, qualify or override one another unless a document says so -- where two articles both apply, give each separately and let them stand side by side.

Precision
7. An article often contains several lettered paragraphs, each a separate rule. Name the paragraph that applies -- "Article 65(a)" -- and never merge two paragraphs into a single statement.
8. Reproduce durations, percentages, amounts, ages and deadlines exactly as the document gives them, and attach each one to the specific obligation the text attaches it to.
9. Where the text qualifies a rule, state the qualification together with the rule. Never present a conditional rule as unconditional, and never widen a rule beyond the persons or cases the text names.

Usefulness
10. The question usually describes someone's real situation. Answer that first, in a sentence or two, then give the supporting rules and citations. Do not open by restating the article.
11. If the answer depends on a fact about the person's own situation that they have not given -- their age, years of service, marital status, a date, an amount -- ask them for it in one short question instead of guessing or covering every possible case. If the documents instead give a short, complete set of conditions you can state directly, state them.
12. Reply in the language of the question -- an Arabic question gets an Arabic answer, an English question gets an English answer. Write plainly and without padding, but do not leave a relevant provision out in order to keep the answer short.

Judgment questions
13. Some questions ask for a personal decision or opinion ("should I...", "is it worth it...", "is it better to...") rather than a fact the documents state directly. If the documents describe the relevant provisions, lay out the concrete benefits and the obligations or costs they describe, neutrally and side by side, so the person can weigh it themselves. Do not tell them what to decide, and do not reason beyond what the documents state.

Close with one short line noting this is general information, not legal advice, and that they should consult a licensed lawyer about their own case."""

LAW_DISPLAY_NAMES = {
    "ssl": "Social Security Law",
    "labor96": "Labor Law 1996",
    "prof19": "Professional Work Law 2019",
}

# Phase 21 fix. `_format_context` labelled every document with the English names above, so
# an Arabic question got Arabic article text under an English heading -- and the model cited
# what it was shown: 8 of 30 Arabic answers in the Phase 20 run wrote "Article 65" instead of
# "المادة (65)" (q16, q27, q44, q46, q47, q48, q53, q56). The answer bodies were correctly
# Arabic; only the citations were not, which is why no answer was flagged as wrong-language.
# The context scaffolding is localised too, not just the law name -- an Arabic heading reading
# "Article: 65" would reintroduce the same pull toward an English citation.
LAW_DISPLAY_NAMES_AR = {
    "ssl": "قانون الضمان الاجتماعي",
    "labor96": "قانون العمل لسنة 1996",
    "prof19": "قانون العمل المهني لسنة 2019",
}

CONTEXT_LABELS = {
    "en": {"doc": "Document", "law": "Law", "article": "Article",
           "language": "Language", "text": "Text"},
    "ar": {"doc": "وثيقة", "law": "القانون", "article": "المادة",
           "language": "اللغة", "text": "النص"},
}

_ARABIC_RE = re.compile(r"[؀-ۿ]")


def detect_language(text):
    """'ar' if the text contains Arabic script, else 'en'.

    Deliberately simple: the eval set and the live app both send a question in one language,
    and rule 12 already tells the model to answer in the question's language. This only picks
    which labels the context carries.
    """
    return "ar" if _ARABIC_RE.search(str(text)) else "en"


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

You will be given a list of source documents, each labeled with its law and article number. When you cite a claim, name the actual law and article number directly in your answer text -- in English "Labor Law 1996, Article 61", in Arabic "قانون العمل لسنة 1996، المادة (61)". Write the citation in the same language as the rest of your answer. Do not refer to documents by number (e.g. do not write "Document 1")."""

# Arabic attaches prefixes directly onto the definite article, and "ل" elides its alef
# (ل + ال -> لل), so the literal string "المادة" does not occur inside "للمادة" -- the
# common way to write "according to article N". The original pattern anchored on
# "المادة" and therefore scored those correct citations as misses (6 of 29 Arabic
# citations in the Phase 8 answers). Anchor on the invariant stem "مادة" instead and
# allow the definite-article/preposition prefixes that can precede it.
ARTICLE_NUM_RE = re.compile(r"[Aa]rticle\D{0,15}?(\d+)|[وف]?(?:ال|لل|[بك]ال)?مادة\D{0,5}?(\d+)")

# Arabic-Indic digits. Latent until Phase 21 made Arabic citations actually happen: an answer
# writing "المادة (٦٥)" yields the citation "٦٥", which never equals the ground truth "65".
# Python's \d matches these, so the regex finds them and the *comparison* silently fails --
# the same shape of bug as the المادة/للمادة one above. Measured at 3 of 60 answers in the
# Phase 21 run, none of which flipped a score, but normalise before comparing regardless.
_ARABIC_INDIC_DIGITS = {ord(c): str(i % 10) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")}


def normalize_digits(text):
    """Map Arabic-Indic and extended Arabic-Indic digits onto ASCII 0-9."""
    return str(text).translate(_ARABIC_INDIC_DIGITS)


def extract_cited_articles(answer):
    """Article numbers an answer cites, digit-normalised. Use this, not ARTICLE_NUM_RE directly."""
    return {a or b for a, b in ARTICLE_NUM_RE.findall(normalize_digits(answer))}


# Matches how the *statutes themselves* name other articles, including plurals and lists:
# "articles (32) and (33)", "المادتين (32) و (33)", "المواد ...". Needed for the fabrication
# check below -- Article 25's own text says "entitlements specified in articles (32) and (33)
# of this Law", so an answer quoting it verbatim was being scored as inventing Article 32.
_XREF_RE = re.compile(r"(?:[Aa]rticles?|[وف]?(?:ال|لل|[بك]ال)?ماد(?:ة|تين|تي)|المواد)"
                      r"((?:\s*\(?\d+\)?(?:\s*(?:and|و|,|،)\s*)?)+)")


def grounded_article_numbers(chunks):
    """Article numbers an answer may legitimately name: the retrieved articles, plus any
    article those articles cross-reference in their own text.

    51 of the 531 corpus units cross-reference another article, so treating "cited but not
    retrieved" as fabrication produces false positives at a rate that scales with how much
    of the source the model quotes -- i.e. it penalises faithfulness. Phase 20 flagged 1 and
    Phase 21 flagged 4; by this definition both arms have 0 real fabrications.
    """
    nums = {c["article_number"] for c in chunks}
    for c in chunks:
        for grp in _XREF_RE.findall(normalize_digits(c["text"])):
            nums |= set(re.findall(r"\d+", grp))
    return nums


def _format_context(chunks, lang="en"):
    """Format retrieved chunks as labelled documents, in `lang` ('en' or 'ar').

    The label language follows the *question*, not the chunk -- a chunk's own language is
    still stated in its `language` field. See LAW_DISPLAY_NAMES_AR for why this matters.
    """
    names = LAW_DISPLAY_NAMES_AR if lang == "ar" else LAW_DISPLAY_NAMES
    lb = CONTEXT_LABELS.get(lang, CONTEXT_LABELS["en"])
    parts = []
    for i, c in enumerate(chunks, start=1):
        law_name = names.get(c["law"], c["law"])
        parts.append(
            f"[{lb['doc']} {i}]\n{lb['law']}: {law_name}\n{lb['article']}: {c['article_number']}\n"
            f"{lb['language']}: {c['language']}\n{lb['text']}: {c['text']}"
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
        lang = detect_language(query)
        context = _format_context(chunks, lang)
        user_msg = (f"الوثائق:\n{context}\n\nالسؤال: {query}" if lang == "ar"
                    else f"Documents:\n{context}\n\nQuestion: {query}")
        available_articles = {c["article_number"] for c in chunks}

        resp = None
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_msg},
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

    def suggest_followups(self, query, chunks, answer, n=3):
        """Follow-up questions grounded in `chunks`. Returns [] on any failure.

        Never raises. This is a garnish on an answer the user already has, so a
        malformed model response, a rate limit or a billing error must degrade to
        "no chips" rather than break the response they actually asked for.

        Article text is truncated hard -- proposing a question *about* an article only
        needs to know what it covers, not to quote it.
        """
        lang = detect_language(query)
        names = LAW_DISPLAY_NAMES_AR if lang == "ar" else LAW_DISPLAY_NAMES
        listing = "\n".join(
            f"- {names.get(c['law'], c['law'])} {c['article_number']}: "
            f"{' '.join(str(c['text']).split())[:280]}"
            for c in chunks
        )
        user_msg = (f"Articles available:\n{listing}\n\n"
                    f"User's question: {query}\n\nAnswer they received: {str(answer)[:1200]}")
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=FOLLOWUP_MAX_TOKENS,
                extra_body=self.extra_body,
                messages=[{"role": "system", "content": FOLLOWUP_PROMPT},
                          {"role": "user", "content": user_msg}],
            )
            content = (resp.choices[0].message.content or "") if resp.choices else ""
            m = re.search(r"\[.*\]", content, re.DOTALL)
            if not m:
                return []
            items = json.loads(m.group(0))
        except Exception:
            return []

        out, seen = [], {query.strip()}
        for q in items:
            q = " ".join(str(q).split())
            # Cheaper to drop a bad suggestion than to trust the model followed the rules.
            if q and q not in seen and len(q) <= 120:
                seen.add(q)
                out.append(q)
        return out[:n]


# Follow-up suggestions shown under an answer. Grounded in the articles that were
# actually retrieved, for one specific reason: an ungrounded model will happily suggest
# "what about maternity leave for self-employed workers?" when nothing in the corpus
# answers it, and a suggestion chip that leads to "the documents do not cover this" is
# worse than no chip. Constraining it to the retrieved articles makes every suggestion
# answerable by construction.
#
# This also does something the prompt could not. Phase 21 closed most omissions, but 7
# questions still lose relevancy for skipping a second article that *was* in the context
# (q01, q05, q09, q10, q19, q29, q35). Those articles are exactly what this proposes as
# the next question -- the interface surfaces what the answer left out.
FOLLOWUP_PROMPT = """You suggest follow-up questions for a Jordanian labour and social security law assistant.

Below are the law articles that were retrieved for the user's question, and the answer they received. Propose exactly 3 short follow-up questions.

Rules:
- Each question must be answerable from the articles listed below. Do not propose anything they do not cover.
- Prefer articles the answer did not already cover in full.
- Write the way a person describes their situation, not the way a statute is titled. Not "What are the conditions for entitlement to the unemployment allowance?" but "Do I have to have worked a certain time to qualify?"
- Keep each under 12 words.
- Do not repeat the user's original question.
- Write them in the SAME LANGUAGE as the user's question.

Respond with ONLY a JSON array of 3 strings, no other text."""

FOLLOWUP_MAX_TOKENS = 300


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
