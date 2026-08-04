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

Production pick: `get_production_generator()` -- Claude Haiku 4.5 via OpenRouter, switched
from `gpt-4o-mini` in Phase 31 (scripts/29-30...ipynb). GPT-4o-mini was the Phase 7 pick
(against Cohere command-a-03-2025 and Qwen2.5-72B, clause chunks, the old prompt, the v1
eval set -- all since replaced). The Phase 31 re-review tested `gemini-2.5-flash`, `gpt-4o`
and `claude-haiku-4.5` against gpt-4o-mini on all 60 eval_v3 questions, paired identical
context: none beat production on both judge metrics by a pre-committed margin (gemini lost
on relevancy from padding plus a real truncation bug; gpt-4o was directionally better on
faithfulness but not significantly so). Haiku's LLM-judge comparison (necessarily graded by
`gpt-4o`, not this project's usual Sonnet-5 judge, to avoid vendor-family bias -- Haiku is
Anthropic, same as Sonnet 5) came back almost entirely tied, but that traces to `gpt-4o`
being a substantially more lenient grader than Sonnet 5 (the same 60 production answers
scored 0.17 higher under `gpt-4o` than under Sonnet 5 -- a ceiling effect, not evidence of
no difference), not to Haiku actually matching production. The regex proxy, unaffected by
judge choice, was unambiguous: 60/60 correct-article-cited, 58/60 fully complete, 0/60
fabrications -- the best and only zero-fabrication result across every arm tested in this
project's history. Adopted on that basis plus a manual read of a sample of its 60 answers
(clean; the two real defects the judge did catch -- a wrong pension age in one answer, an
ungrounded "gross fault" inference in another -- were isolated, not a pattern), not on a
statistically significant judge win, which this particular judge setup could not reliably
produce either way. `SYSTEM_PROMPT` was compacted in the same phase (22 rules, ~34% fewer
tokens, no rule dropped) and shipped together with the model switch -- the two were tested
as one bundled configuration, so this switch cannot be attributed to the model alone.
`Generator` (Cohere) is kept for the original Phase 7 comparison, not used in production.
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
# That rule originally shipped with a real gap: the API was single-turn, so if the model
# asked "how many years have you contributed?" and the person replied "12", that reply
# became a context-free query and retrieval had no idea what it referred to. The model
# asked a better question but the app could not hear the answer.
#
# CLOSED in Phase 23. /ask now accepts `history`, and -- the part that actually matters --
# the follow-up is rewritten into a standalone question *before* retrieval runs, since
# retrieval is semantic and "12" embeds to nothing useful. See rewrite_for_retrieval().
#
# Multi-turn then introduced its own regression, caught on the first real test: with the
# same question and BYTE-IDENTICAL retrieved articles, the model asked for the missing
# fact when history was empty and skipped the question when an earlier turn had already
# answered something similar. Conversational context was overriding the rule -- a model
# will not re-ask what the transcript shows as already settled. Hence rule 11's closing
# sentence, which scopes the rule to the current question rather than the thread.
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
#
# Rules 21-22 are the generation-side half of a retrieval fix (src/retrieval.py's
# PRECONDITION_ARTICLES). Tracing eval_v3 q05 and q43 found the model answering a
# work-injury question from labor96 Articles 90/91 alone, or blending them with ssl
# Article 30, without ever knowing that Article 86 makes the whole labor96 chapter
# conditional on NOT being covered by Social Security -- because Article 86 was not
# retrieved. Fixing retrieval alone was measured NOT to fix this: with Article 86
# added to context but no instruction, none of four test answers stated the
# condition, and q05 got WORSE (a new cross-contamination error, applying Article
# 91's wage formula to an SSL pension) -- proof that more context without guidance
# can actively hurt. Rule 21 is the fix: an explicit carve-out of rule 6, the same
# pattern rule 18 already uses for arithmetic, because stating a condition a document
# actually writes is reading the text, not the inference rule 6 forbids.
#
# Rule 22 covers a different, adjacent finding from the same investigation: labor96
# Article 3 excludes public servants, municipal workers, and (separately) domestic/
# agricultural workers from the entire Labour Law, and prof19 Article 3 gates the
# whole Professional Work Law on a ministerial inclusion decision -- neither answer
# the question asked, but both can silently change whether the rest of the answer
# even applies to the person. Deliberately NOT modelled as retrieval facts the way
# Article 86 is: user's call, after the tradeoff was stated -- a fact stated directly
# in the prompt can go silently stale if the article is ever amended, unlike
# everything else here which is re-read from retrieved text every time. Accepted for
# these two specifically because they are short, simple, and not the kind of thing
# (a rate, a threshold) this project's history shows gets amended often. ssl Article
# 4's own coverage conditions stayed a retrieval rule rather than being summarised
# here, since they are longer and more detailed (age, category, wage-minimum tests).
# Compacted in Phase 31 (scripts/29...ipynb) as part of the model switch above: same 22
# rules, no rule dropped, phrasing tightened -- 34% fewer tokens (7,913 -> 5,189 chars),
# sent on every single call. The history in the comment block above this one documents WHY
# each rule exists and stays accurate; it describes the reasoning, not the exact wording,
# which changed here. Rules 14-20 (the calculator-block rules) were kept close to their
# original wording deliberately -- they gate deterministic output from src/calc.py and are
# tied to real measured bugs (Phase 24-26), not worth risking for a marginal token saving.
# Rule 21 was tightened to say "lead with that condition" rather than "state that
# condition", matching what a same-phase position experiment (scripts/28...ipynb, section 9)
# showed actually moves behaviour on a model that needed it -- untested here on Haiku
# specifically, since Haiku handled the underlying case correctly without needing the push.
# Validated before shipping: a 9-answer smoke test (3 models x 3 known-tricky questions)
# confirmed the calculator blocks still fire and narrate correctly, then the full Phase 31
# re-review (60 questions, paired context) found no regression versus the original wording.
SYSTEM_PROMPT = """You are a legal information assistant for Jordanian labour and social security law (the Labour Law 1996, the Professional Work Law 2019, and the Social Security Law). Answer using only the provided excerpts.

Grounding
1. Every statement must come from the documents -- no outside knowledge.
2. If the documents don't answer the question, say so plainly rather than filling the gap.
3. Cite by law and article number (e.g. "Labour Law 1996, Article 61"), only articles you were given, under the law each document is actually labelled with.

Coverage
4. Read every document before answering. More than one article often bears on the question -- address each one that does, with its own citation, not just the first.
5. Within a cited article, cover every lettered paragraph that adds a condition, exception, limit, minimum, or further step -- and every numbered point nested inside one of those paragraphs (e.g. different categories of beneficiary each with their own condition), not just the paragraph as a whole.
6. State each provision on its own terms. Don't explain how two provisions combine or override each other unless a document says so -- give each separately, side by side.

Precision
7. Name the specific paragraph that applies ("Article 65(a)"), never merging two paragraphs into one statement.
8. Reproduce durations, percentages, amounts, ages and deadlines exactly as given, attached to the specific obligation they belong to.
9. State any qualification together with the rule it qualifies. Never present a conditional rule as unconditional or widen it beyond the persons/cases named.

Usefulness
10. Answer the person's actual situation first, in a sentence or two, then give supporting rules and citations. Don't open by restating the article.
11. If the answer depends on a fact the person hasn't given (age, years of service, marital status, a date, an amount, which law covers them, or sex where a provision sets separate figures) -- ask for it in one short question rather than guessing or covering every case. Never silently default to one case (e.g. male figures) when a provision states both. If the documents give a short, complete set of conditions instead, state them directly. Judge the current question on its own terms even if an earlier turn answered without the fact. Exception: if a computed block already gives both cases (rule 20), presenting both IS the answer -- don't also ask.
12. Reply in the question's language. Be plain and unpadded, but never drop a relevant provision to stay short.

Judgment questions
13. For a personal-decision question ("should I...", "is it worth it..."), lay out the documents' benefits and obligations/costs neutrally, side by side, so the person can weigh it. Don't recommend, and don't reason beyond what the documents state.

Totals and calculations
14. For a total, combined rate, or amount worked out from a stated figure, account for EVERY contributing provision, not just the most prominent one -- contributions especially span several articles, one per insurance type.
15. Convert any rate the documents write in words to a numeral before calculating (using the conversion aid) -- it counts the same as a digit rate.
16. Show each component on its own line with its article and insurance type (never drop the name). Then the total. Keep employer and employee amounts separate; don't combine them unless asked for the combined total.
17. Apply any stated ceiling or floor, and say plainly when it doesn't affect the figure.
18. Adding up rates the documents state outright is arithmetic, not the inference rule 6 forbids -- rule 6 still governs everything else. If the total depends on which category the person falls into, say which case you calculated and name the others.
19. A computed block's arithmetic is already done and checked -- use its rates, amounts and totals exactly, without re-deriving, re-rounding, or contradicting them. It already satisfies rules 14-17; you still cite each rate's article and add anything the documents say that the block doesn't cover.
20. Follow a block's sex-differentiated condition exactly: one figure means sex is known (state only that one); both figures mean it isn't (present both, labelled) -- this has already been determined, not something to re-decide.

Multiple laws, and who a law covers
21. If a document states a condition that decides which of two provisions applies (most often, coverage by a different law), lead with that condition -- this is reading the text, not the inference rule 6 forbids. When two laws address the same event, never merge them into one undifferentiated answer: attribute each obligation to the party the text names, and never describe one law's institution's payment (e.g. Social Security) as required from the other's party (e.g. the employer).
22. Labour Law 1996 excludes government/municipal employees and domestic/agricultural workers (separate rules apply). Professional Work Law 2019 only covers professions the Council of Ministers has included. Treat a retrieved document's own coverage conditions the same way. Mention whichever is relevant briefly, once -- not every turn, and not as a second full answer; if these documents don't cover the person's situation, say so per rule 2.

Close with one short line: general information, not legal advice, consult a licensed lawyer about their own case."""

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
           "language": "Language", "text": "Text",
           "rates": "Rates written in words, as numerals (conversion aid, not statute text)"},
    "ar": {"doc": "وثيقة", "law": "القانون", "article": "المادة",
           "language": "اللغة", "text": "النص",
           "rates": "نسب مكتوبة بالحروف، مقابلها بالأرقام (للحساب فقط، وليست من نص المادة)"},
}

_ARABIC_RE = re.compile(r"[؀-ۿ]")

# Rates spelled out in words instead of digits. Most articles write "(١١%)", but a few
# write "(اثنين بالمائة)" or "ثلاثة أرباع الواحد بالمائة". That difference is not
# cosmetic -- it was measured breaking the system twice over:
#   retrieval  -- asked "how much does the company pay", the spelled-out articles ranked
#                 11, 25 and worse-than-30 while the digit one ranked 3
#   generation -- even with all four articles in context and an explicit instruction to
#                 convert words to numerals, the model summed 11% + 2% + 0.5% and
#                 silently dropped the 0.75% maternity rate, under-reporting the
#                 employer total as 135 JD instead of 142.5 JD
#
# `_format_context` therefore appends a clearly-labelled conversion note per document.
# The statute text itself is never rewritten: the model must still quote and cite the
# article as written, and a numeral the law does not contain must not appear inside a
# quotation. The note is an aid beside the text, not a substitute for it.
#
# The vocabulary is closed and small -- 25 occurrences across the whole corpus, seven
# distinct number words -- so this is a lookup, not a general Arabic number parser.
AR_NUMBER_WORDS = {
    "نصف": 0.5, "ربع": 0.25, "ثلث": 1 / 3,
    "واحد": 1, "واحدة": 1, "اثنين": 2, "اثنان": 2, "ثلاثة": 3, "ثلاث": 3,
    "أربعة": 4, "اربعة": 4, "خمسة": 5, "ستة": 6, "سبعة": 7, "ثمانية": 8,
    "تسعة": 9, "عشرة": 10, "عشرين": 20, "ثلاثين": 30, "ثلاثون": 30,
    "أربعين": 40, "اربعين": 40, "خمسين": 50, "ستين": 60, "سبعين": 70,
    "ثمانين": 80, "تسعين": 90, "مئة": 100, "مائة": 100,
}
# Multi-word fractions, checked before single words so "ثلاثة أرباع الواحد" is not read as 3.
AR_NUMBER_PHRASES = {
    "ثلاثة أرباع الواحد": 0.75, "ثلاثة ارباع الواحد": 0.75,
    "ثلاثة أرباع": 0.75, "ثلاثة ارباع": 0.75,
    "واحد ونصف": 1.5, "اثنين ونصف": 2.5,
}
_PCT_WORDS_RE = re.compile(r"((?:[^\s()]+\s+){0,3}[^\s()]+)\s*بالما?ئة")


def _strip_ar(tok):
    return tok.strip("()[]،.,:؛").replace("ـ", "")


def spelled_out_rates(text):
    """[(phrase, numeric percent)] for rates written in words rather than digits."""
    out, seen = [], set()
    for m in _PCT_WORDS_RE.finditer(str(text)):
        toks = [_strip_ar(t) for t in m.group(1).split() if _strip_ar(t)]
        val, phrase = None, None
        for n in (3, 2):                       # longest phrase first
            if len(toks) >= n:
                cand = " ".join(toks[-n:])
                if cand in AR_NUMBER_PHRASES:
                    val, phrase = AR_NUMBER_PHRASES[cand], cand
                    break
        if val is None and toks and toks[-1] in AR_NUMBER_WORDS:
            val, phrase = AR_NUMBER_WORDS[toks[-1]], toks[-1]
        if val is None:
            continue
        label = f"{phrase} بالمائة"
        if label not in seen:
            seen.add(label)
            pretty = f"{val:g}"
            out.append((label, f"{pretty}%"))
    return out


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
        block = (
            f"[{lb['doc']} {i}]\n{lb['law']}: {law_name}\n{lb['article']}: {c['article_number']}\n"
            f"{lb['language']}: {c['language']}\n{lb['text']}: {c['text']}"
        )
        rates = spelled_out_rates(c["text"])
        if rates:
            pairs = "، ".join(f"{w} = {n}" for w, n in rates)
            block += f"\n{lb['rates']}: {pairs}"
        parts.append(block)
    return "\n\n".join(parts)


class OpenAICompatibleGenerator:
    """Generator for any OpenAI-compatible chat endpoint (OpenAI direct, or OpenRouter for open models)."""

    def __init__(self, model_id, api_key_env, base_url=None, extra_body=None, temperature=None,
                 system_prompt=None, max_tokens=None):
        load_dotenv(PROJECT / ".env")
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url)
        self.model_id = model_id
        self.extra_body = extra_body
        self.temperature = temperature  # None = provider default (unset, untested -- see scripts/generation_config_tuning.ipynb)
        self.system_prompt = system_prompt or CONTEXT_SYSTEM_PROMPT
        # None = provider default, same as `temperature` -- but per the Phase 20 judge lesson
        # (LESSONS.md 3.3), OpenRouter reserves credit against a request's max_tokens *ceiling*,
        # not actual usage. Left unset here for `gpt-4o`/`gpt-4o-mini` (production, and every
        # generation number this project has published, is measured with it unset and has never
        # hit this). Repeated the failure on `claude-haiku-4.5` during the Phase 31 re-review
        # (scripts/30...ipynb): its unset ceiling defaults to 64,000 tokens/call, and after
        # enough calls the account's available balance couldn't cover the reservation --
        # a 402 with a healthy balance still showing, same shape as the original judge bug.
        self.max_tokens = max_tokens

    def rewrite_for_retrieval(self, query, history):
        """Rewrite a follow-up into a standalone question. Returns `query` on any failure.

        This is what makes multi-turn actually work, and it is a *retrieval* fix, not a
        generation one. Retrieval is stateless and semantic: it embeds the query text and
        nothing else. A reply like "yes, I'm covered" or "12 years" embeds to almost
        nothing useful -- there is no signal in it about pensions or end-of-service
        compensation -- so the retriever returns unrelated articles and the answer is
        built on the wrong documents. Passing history to the *generator* alone would not
        fix that; by then the wrong context has already been fetched.

        So the history is folded into the query before it reaches the retriever:
        "yes, I'm covered" + the preceding exchange becomes "How much end-of-service
        compensation am I owed if I am subject to the Social Security Law?", which embeds
        to the right neighbourhood.

        Falls back to the raw query on any failure -- a bad rewrite should degrade to
        today's single-turn behaviour, never to an error.
        """
        if not history:
            return query
        lang = detect_language(query)
        transcript = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {str(m['content'])[:600]}"
            for m in history[-REWRITE_HISTORY_TURNS:]
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=REWRITE_MAX_TOKENS,
                extra_body=self.extra_body,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user",
                     "content": f"Conversation so far:\n{transcript}\n\n"
                                f"Latest user message: {query}\n\n"
                                f"Standalone question (in {'Arabic' if lang == 'ar' else 'English'}):"},
                ],
            )
            out = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        except Exception:
            return query

        out = " ".join(out.split())
        # Guard against the model returning commentary, an empty string, or a runaway
        # paraphrase instead of a question. Anything suspect falls back to the raw query.
        if not out or len(out) > 400:
            return query
        return out

    def _reparse_for_completeness(self, query):
        """A more explicit restatement of `query`, or None on any failure/no-op.

        Phase 35's deep dive (q29) found this works where a static prompt rule
        (SYSTEM_PROMPT rule 5) only partially did: q29 measured 0/9 with no
        intervention, ~30% with the broadened rule, 12/12 across two independent
        6-trial runs with this reparse step. Best-guess mechanism: noticing that a
        dense article covers several distinct cases and choosing to enumerate all
        of them is a harder task than answering an ALREADY-explicit "list every
        category and its own condition" question -- splitting a hard task into an
        easier decomposition step plus an easier execution step raised reliability
        here in a way another generic prompt rule did not.

        Deliberately narrow: only makes the ORIGINAL question's own scope
        explicit (asks for every relevant category if the question could have
        more than one), never introduces a new comparison axis (sector, pension
        type, etc.) the person did not ask about -- an earlier, looser version of
        this prompt invented such comparisons and roughly tripled answer length
        pulling in tangential articles. Still measured ~2-3x longer answers than
        without it even after tightening, because a genuinely more complete
        question legitimately triggers more of SYSTEM_PROMPT's coverage rules
        (rule 4) -- disclosed, not hidden; this is not free.

        Runs on `COMPLETENESS_REPARSE_MODEL_ID`, not `self.model_id` -- this call
        never reads legal text, just rewrites one question into a more explicit
        version of itself, a narrower task than the main generation call it feeds.
        Deliberately the same choice of model this project already trusted for
        full legal-answer generation for most of its history (`gpt-4o-mini` was
        production through Phase 30), so it is not an unproven cost-cutting
        gamble -- it is being asked to do less than what it used to do reliably.
        Falls back to `self.model_id` if the cheap call errors, so a bad key/quota
        on the cheaper model degrades to the old cost rather than silently losing
        the completeness gain.
        """
        for model_id in (COMPLETENESS_REPARSE_MODEL_ID, self.model_id):
            try:
                resp = self.client.chat.completions.create(
                    model=model_id,
                    max_tokens=COMPLETENESS_REPARSE_MAX_TOKENS,
                    extra_body=self.extra_body,
                    messages=[
                        {"role": "system", "content": COMPLETENESS_REPARSE_PROMPT},
                        {"role": "user", "content": f"Original question: {query}\n\nRewritten question:"},
                    ],
                )
                out = (resp.choices[0].message.content or "").strip() if resp.choices else ""
                break
            except Exception:
                out = ""
        else:
            return None

        out = " ".join(out.split())
        if not out or len(out) > 600 or out == query:
            return None
        return out

    def generate(self, query, chunks, history=None, max_retries=5):
        lang = detect_language(query)
        context = _format_context(chunks, lang)

        # Contribution arithmetic is done in code, not by the model -- see src/calc.py.
        # Measured at 3/6 and 2/6 correct when the model summed the six rates itself,
        # with every article already in context. `compute_contributions` returns None
        # unless the full private funding family was retrieved, so an incomplete
        # context leaves the model to answer unaided rather than handing it a
        # confidently wrong total.
        calc = None
        try:
            from .calc import (compute_contributions, compute_end_of_service,
                               compute_gender_conditions, compute_overtime, compute_leave_accrual,
                               extract_salary, extract_service, extract_sex, extract_hourly_wage,
                               format_calculation_block, format_end_of_service_block,
                               format_gender_block, format_overtime_block, format_leave_block,
                               disability_pension_scope_note)
            salary = extract_salary(query)
            calc = compute_contributions(chunks, salary=salary)
            if calc:
                context += "\n\n" + format_calculation_block(calc, lang)
            eos = compute_end_of_service(chunks, salary=salary, years=extract_service(query))
            if eos:
                context += "\n\n" + format_end_of_service_block(eos, lang)
                calc = {"contributions": calc, "end_of_service": eos} if calc else \
                       {"end_of_service": eos}
            # Sex-differentiated pension conditions (SSL Art 62/64) -- see src/calc.py for
            # why a prompt rule alone left this wrong 2 times in 6. `extract_sex` reads only
            # the PERSON's own words (this question + prior user turns), never the
            # assistant's, since the assistant's own explanation routinely says "if
            # male... if female..." once it's retrieved and would falsely look like the
            # person had already told us.
            sex = extract_sex(query, history)
            gender = compute_gender_conditions(chunks, sex=sex)
            if gender:
                context += "\n\n" + format_gender_block(gender, lang, sex=sex)
                calc = {**(calc or {}), "gender_conditions": gender}
            # Overtime (Art 59) and leave accrual (Art 61) -- same "read the rate from the
            # text, decline rather than invent" discipline as everything above. Neither
            # needs a precondition question the way end-of-service does; see src/calc.py
            # for why each stops short of a JD amount / a single blended total.
            ot = compute_overtime(chunks, hourly_wage=extract_hourly_wage(query))
            if ot:
                context += "\n\n" + format_overtime_block(ot, lang)
                calc = {**(calc or {}), "overtime": ot}
            leave = compute_leave_accrual(chunks, years=extract_service(query))
            if leave:
                context += "\n\n" + format_leave_block(leave, lang)
                calc = {**(calc or {}), "leave_accrual": leave}
            # Article 40 vs Article 85 (Phase 34) -- pure disambiguation, no
            # computed figures, see src/calc.py for why the model keeps
            # misapplying Article 85's disability-pension paragraphs here.
            scope_note = disability_pension_scope_note(chunks, lang)
            if scope_note:
                context += "\n\n" + scope_note
                calc = {**(calc or {}), "disability_pension_scope_note": True}
        except Exception:
            calc = None  # never let the aid break an answer the model could still give

        user_msg = (f"الوثائق:\n{context}\n\nالسؤال: {query}" if lang == "ar"
                    else f"Documents:\n{context}\n\nQuestion: {query}")

        # Completeness reparse (Phase 35) -- appended AFTER the original question,
        # never replacing it, so the person's own words still drive the answer's
        # tone (rule 10) and the calc extractors above still read the ORIGINAL
        # query, not this rewrite. See _reparse_for_completeness for why this
        # exists and its measured cost (longer answers).
        reparsed = self._reparse_for_completeness(query)
        if reparsed:
            user_msg += (f"\n\n[صياغة أكثر تفصيلاً لنفس السؤال -- تأكد من تغطية كل ما تشمله]: {reparsed}"
                         if lang == "ar" else
                         f"\n\n[A more explicit restatement of the same question -- make sure your answer covers everything it asks]: {reparsed}")

        available_articles = {c["article_number"] for c in chunks}

        # Prior turns go in as real chat messages, before the documents+question turn.
        # Only the text is carried, never the old documents: each turn is grounded in its
        # own freshly retrieved context, so replaying stale articles would invite citing
        # something that is not in *this* turn's context -- the fabrication failure mode
        # the citation check exists to catch.
        messages = [{"role": "system", "content": self.system_prompt}]
        trimmed = (history or [])[-GENERATION_HISTORY_TURNS:]
        for m in trimmed:
            role = "user" if m.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": str(m.get("content", ""))[:2000]})
        # Recency beats the system prompt when the transcript demonstrates the opposite
        # behaviour -- see MULTITURN_REMINDER. Only when there is a transcript to compete
        # with; a first turn follows rule 11 on its own.
        if trimmed:
            messages.append({"role": "system", "content": MULTITURN_REMINDER})
        messages.append({"role": "user", "content": user_msg})

        resp = None
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    extra_body=self.extra_body,
                    **({"temperature": self.temperature} if self.temperature is not None else {}),
                    **({"max_tokens": self.max_tokens} if self.max_tokens is not None else {}),
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

        return {"answer": answer, "citations": citations, "response": resp,
                "calculation": calc}

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

    def review_and_revise(self, query, chunks, draft_answer):
        """Check `draft_answer` against every retrieved document and revise if
        something relevant was skipped. Returns (answer, revised: bool).

        Never raises -- any failure returns the draft unchanged, same "degrade to
        today's behaviour" discipline as rewrite_for_retrieval/suggest_followups. See
        REVIEW_PROMPT above for why this exists (the 7-of-9 generation-gap finding).

        Full article TEXT is given here, unlike suggest_followups' truncated topic
        listing -- checking completeness requires actually reading the paragraphs, not
        just knowing a document's subject.
        """
        listing = "\n\n".join(
            f"{LAW_DISPLAY_NAMES.get(c['law'], c['law'])}, Article {c['article_number']} "
            f"({c['language']}):\n{c['text']}"
            for c in chunks
        )
        user_msg = f"Documents given:\n{listing}\n\nQuestion: {query}\n\nDraft answer:\n{draft_answer}"
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=REVIEW_MAX_TOKENS,
                extra_body=self.extra_body,
                messages=[{"role": "system", "content": REVIEW_PROMPT},
                          {"role": "user", "content": user_msg}],
            )
            content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        except Exception:
            return draft_answer, False
        # Guard against an empty response or a runaway generation -- either falls back
        # to the draft rather than risk showing something worse than what already existed.
        if not content or len(content) > len(draft_answer) * 4 + 1000:
            return draft_answer, False
        return content, content != draft_answer.strip()


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

# Second-pass completeness review. See OpenAICompatibleGenerator.review_and_revise.
#
# A subagent diagnosed the 9 eval_v3 questions that survived Phase 21's coverage rule
# in both the before and after arms: 7 of 9 (q01, q05, q09-Phase20, q19, q29, q35, q49)
# were GENERATION gaps, not retrieval ones -- the missing article, or a specific
# lettered paragraph within an article already cited, was fully present in the SAME
# context the model already had. More coverage-rule wording was not going to close
# this: Phase 21 already told the model to "read every document" and "cover every
# lettered paragraph", and these 7 are exactly the cases where that instruction did
# not fire, in a single generative pass where the model stops once it has found one
# article that satisfies the question.
#
# The FOLLOWUP_PROMPT comment above already names this same pattern -- the missed
# article is literally what gets proposed as a follow-up chip. That is a UI
# workaround (surface what was skipped as the NEXT question), not a fix to the
# answer that was actually given. This is the fix attempt: a second, separate model
# call that re-reads the full document list against the draft AFTER a "good enough"
# answer already exists -- the exact point the first pass stops checking -- rather
# than hoping the first pass keeps reading before it commits to an answer.
#
# Cost: one extra LLM call per question, roughly doubling generation cost and
# latency. Not wired into generate() or /ask by default -- see the eval notebook
# (scripts/28. completeness_review.ipynb) for the measured decision before this
# is adopted in production.
REVIEW_PROMPT = """You review a draft answer from a Jordanian labour and social security law assistant for completeness against the documents it was given.

You will be shown the documents that were retrieved for a question, the question itself, and a draft answer already written from those documents.

Task: check whether every article, and every lettered paragraph within an article, that bears on the question is reflected in the draft. "Bears on the question" means it adds a condition, an exception, a limit, an amount, or a further step relevant to what was asked -- not every provision that happens to sit in the same article.

If the draft already covers everything relevant: return it completely unchanged, word for word.

If something relevant was missed: rewrite the answer to include it, keeping everything the draft already got right. Follow the same rules the draft was written under:
- Every statement must come from the documents given. Do not add outside knowledge, and cite only articles you were actually given.
- State what each provision says on its own terms. Do not explain how two provisions combine, qualify or override one another unless a document says so -- add the missing provision beside the existing ones, do not reason about how they relate.
- Reproduce durations, percentages, amounts, ages and deadlines exactly as the documents give them.
- Reply in the same language as the draft.
- Answer the person's situation first, then the supporting rules and citations -- do not open by restating an article.

Respond with ONLY the final answer text -- the unchanged draft, or your revised version. No preamble, no explanation of what you changed, no markdown fencing."""

REVIEW_MAX_TOKENS = 1500

# Query rewriting for multi-turn. See OpenAICompatibleGenerator.rewrite_for_retrieval.
#
# "Resolve references" is the whole job: pronouns ("it", "that"), ellipsis ("what about
# after 5 years?"), and bare answers to the assistant's own question ("yes", "12 years").
# The last of those is the case rule 11 creates and is the reason this exists at all --
# the model now asks for a missing personal fact, so the very next message is frequently
# a fragment that means nothing on its own.
#
# Explicitly NOT a place to add information: the rewrite feeds a semantic retriever, and
# inventing legal terms the user never said would pull in articles they never asked about.
REWRITE_PROMPT = """You rewrite a follow-up message into a standalone search query for a Jordanian labour and social security law database.

The user's latest message may be a fragment that only makes sense in context -- a pronoun ("what about it?"), an ellipsis ("and after five years?"), or a bare answer to a question the assistant just asked ("yes", "12 years", "I'm covered").

Rewrite it into ONE self-contained question that carries over the necessary context from the conversation.

Rules:
- Output only the rewritten question. No preamble, no explanation, no quotes.
- Keep it in the same language as the user's latest message.
- Use only information present in the conversation. Do not add legal terms, article numbers, or facts nobody mentioned.
- If the latest message is already self-contained, return it unchanged."""

REWRITE_MAX_TOKENS = 200

# Query reparser for completeness (Phase 35) -- a second, separate LLM call from
# REWRITE_PROMPT above. That one rewrites a follow-up into a standalone RETRIEVAL
# query; this one takes the question that will actually be shown to the GENERATOR
# and makes its own scope explicit, so a dense article's several distinct cases
# get asked for by name instead of the model having to notice them unprompted.
#
# Tightened once already after a first version invented new comparison axes the
# person never asked about (e.g. adding "does this differ by employer sector?" to
# a question that never raised sector at all) -- roughly tripled answer length
# pulling in tangential articles that axis triggered. This version is restricted
# to making the ORIGINAL question's own scope explicit, nothing beyond it.
COMPLETENESS_REPARSE_PROMPT = """You rewrite a legal question into a more explicit version of the SAME question, for a Jordanian labour and social security law assistant. Output must be a single question, not a numbered list of separate questions.

Do NOT add any new topic, angle, or comparison the person did not ask about (e.g. do not ask whether the answer differs by sector, employer type, or pension type, unless the original question already asked that). Only make the ORIGINAL question's own scope explicit: if it asks who is entitled to something and there could be more than one category of eligible person, make the question explicitly ask for EVERY such category and that category's own specific condition -- nothing beyond that. If the question does not have this shape (no plausible multiple categories/cases), return it unchanged.

Return only the rewritten question, in the same language as the original, as ONE sentence. Nothing else."""

COMPLETENESS_REPARSE_MAX_TOKENS = 150

# Deliberately NOT PRODUCTION_MODEL_ID (Phase 36 follow-up). Rewriting one question
# into a more explicit version of itself never reads legal text and needs no legal
# reasoning -- a narrower task than the generation call it feeds, so it does not
# need the same model. Same OpenRouter account/base_url as the main generator, so
# no separate client or API key is needed -- just a different `model=` string.
#
# `gpt-4o-mini` was tried first (it was this project's own production model for
# full legal-answer generation through Phase 30) and measured too weak for even
# this narrower task: on the q29 test case its rewrites collapsed into plain
# paraphrases that dropped the explicit "every category, its own condition"
# framing, and completeness fell from 18/18 (reparsing on the production model) to
# roughly 1/4. `google/gemini-2.5-flash` measured 6/6 on the same test -- its
# rewrites kept the explicit per-category framing -- at $0.30/$2.50 per 1M
# tokens vs Haiku 4.5's $1/$5, roughly a third of the cost. Chosen over the
# production model specifically because this task doesn't need it, not because
# it's assumed to be "good enough" generically -- see the model comparison in
# scripts/33 and its Phase 36 follow-up in CLAUDE.md.
COMPLETENESS_REPARSE_MODEL_ID = "google/gemini-2.5-flash"

# Injected as a system message AFTER the history and immediately BEFORE the current
# question -- position is the entire point.
#
# Rule 11 alone does not survive multi-turn, and this was measured rather than assumed:
# with a prior turn in history, the model asked for a missing fact in 0 of 6 trials both
# locally and in production, with byte-identical retrieved articles and the same question
# that asks reliably with no history. Strengthening rule 11's wording in the system prompt
# changed nothing (still 0/6).
#
# The reason is that the assistant's own earlier answer sits in the transcript as a
# concrete demonstration of answering without asking. An in-context example outweighs an
# instruction in a 13-rule system prompt, so the model imitates its own prior turn. The
# counter is recency: restate the rule in the last position before the question, where it
# competes on equal footing. Only sent when history exists -- a first turn already obeys
# rule 11 without it.
MULTITURN_REMINDER = """Before answering the question below, judge it on its own terms.

If answering it correctly depends on a fact about this person that they have not stated -- their age, years of service, marital status, a date, an amount, which law covers them, or their sex where a provision sets separate figures for men and women -- ask them for that one fact in a short question instead of answering as though it were settled. Never silently pick one case (e.g. the male figures) when the documents give both. Exception: if a computed block below already gives you both cases for that fact, state both instead of asking -- that block is the answer, not a reason to ask a further question.

How earlier turns in this conversation were answered does not decide this. If the fact is still missing now, ask for it now."""
# How much history each step sees. Rewriting only needs the immediately preceding
# exchange to resolve a reference, so it stays tight and cheap; generation gets a longer
# window so the thread reads coherently. Both are capped -- unbounded history would grow
# cost per turn without bound on a public endpoint.
REWRITE_HISTORY_TURNS = 4
GENERATION_HISTORY_TURNS = 6


PRODUCTION_MODEL_ID = "anthropic/claude-haiku-4.5"
PRODUCTION_API_KEY_ENV = "OPENROUTER_API_KEY"
PRODUCTION_BASE_URL = "https://openrouter.ai/api/v1"

# Required as of the Phase 31 switch to Haiku -- see OpenAICompatibleGenerator.__init__'s
# max_tokens note. gpt-4o-mini never needed this (unset ceiling never caused a problem in
# this project's whole history with it); Haiku's unset ceiling is 64,000 tokens/call, and
# that reservation 402'd a live OpenRouter key mid-testing despite real usage per answer
# staying under 600 tokens. 4000 is generous headroom above every answer length measured so
# far (worst case ~1,100 tokens) while nowhere near large enough to trigger the same failure.
PRODUCTION_MAX_TOKENS = 4000


def get_production_generator(temperature=None, system_prompt=None, max_tokens=PRODUCTION_MAX_TOKENS):
    """The chosen generation backend for this project -- see module docstring."""
    return OpenAICompatibleGenerator(
        model_id=PRODUCTION_MODEL_ID,
        api_key_env=PRODUCTION_API_KEY_ENV,
        base_url=PRODUCTION_BASE_URL,
        temperature=temperature,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )
