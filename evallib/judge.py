"""LLM-as-judge scaffolding -- one definition of the grader for the whole project.

The judge scores an answer against the documents it was given, 1-5 on Faithfulness
and Relevancy. Claude Sonnet is used deliberately: it is from a different vendor
than any model being graded, so it cannot prefer its own style.

Verified before extracting: `JUDGE_PROMPT` and `JUDGE_MODEL` were **byte-identical**
in all seven notebooks that defined them, and the two inline context builders
produced the same string as `format_context_for_judge`. Judge scores really are
comparable across phases -- that was an assumption the project had been relying on
without ever checking it.

## One deliberate behaviour change, stated rather than slipped in

Two generations of this function existed:

  * the original (notebooks 08/09/10/14/17) sent **no** `max_tokens`
  * the Phase 20 version sent `max_tokens=2000` and reported failures in detail

Leaving `max_tokens` unset is what killed a run mid-way: the provider reserves
credit against the *ceiling*, not actual usage, and an unset ceiling defaults to
the model's maximum. See `LESSONS.md` section 3.3.

This module keeps `max_tokens` as an explicit argument, defaulting to `None` so the
older notebooks reproduce their original behaviour exactly. The improved failure
reporting is applied to both, since it only affects what is printed on a retry and
cannot change a score that succeeds.

**New work should pass `max_tokens`.**
"""

from __future__ import annotations

import json
import re
import time

__all__ = ["JUDGE_MODEL", "JUDGE_PROMPT", "JUDGE_MAX_TOKENS",
           "format_context_for_judge", "make_judge",
           "BEHAVIOR_JUDGE_PROMPT", "make_behavior_judge"]

JUDGE_MODEL = "anthropic/claude-sonnet-5"

# The value adopted in Phase 20. Unset -> provider reserves the model maximum and
# can fail a request on a healthy balance; 300 -> empty responses. Both observed.
JUDGE_MAX_TOKENS = 2000

# VERBATIM from the notebooks -- do not reword, reformat or "improve" this.
# Every judge score this project has published was produced by this exact string.
# Changing a single word changes the grader, which would make new scores
# incomparable with every number in ROADMAP.md without anything visibly breaking.
# `test_equivalence.py` asserts this matches the notebooks byte for byte.
JUDGE_PROMPT = """You are grading an AI assistant's answer to a legal question about Jordanian labor/social security law. Score it on two dimensions, each 1-5.

FAITHFULNESS (1-5): Is every factual claim in the answer actually supported by the source documents below? 5 = every claim is directly backed by the documents, nothing added. 3 = mostly supported but with some unsupported/embellished detail. 1 = contains claims that contradict or aren't found anywhere in the documents.

RELEVANCY (1-5): Does the answer directly and completely address the question asked? 5 = fully addresses it, on-topic, complete. 3 = addresses it but incomplete or padded with tangential info. 1 = doesn't actually answer the question.

Question: {question}

Source documents given to the assistant:
{context}

Assistant's answer:
{answer}

Respond with ONLY this JSON, no other text:
{{"faithfulness_score": <1-5>, "faithfulness_reason": "<one sentence>", "relevancy_score": <1-5>, "relevancy_reason": "<one sentence>"}}"""


def format_context_for_judge(chunks):
    """Render retrieved chunks the way the judge prompt expects."""
    return "\n\n".join(
        f"[{c['law']} Article {c['article_number']} ({c['language']})]\n{c['text']}"
        for c in chunks
    )


def make_judge(client, model=JUDGE_MODEL, max_tokens=None, verbose=True):
    """Build a `judge(question, chunks, answer, max_retries=5)` bound to a client.

    A factory rather than a plain function so notebooks keep their existing call
    sites unchanged while the client stays an explicit dependency instead of a
    module-level global.
    """

    def judge(question, chunks, answer, max_retries=5):
        prompt = JUDGE_PROMPT.format(
            question=question, context=format_context_for_judge(chunks), answer=answer
        )
        last = None
        for attempt in range(max_retries):
            try:
                kwargs = {"model": model,
                          "messages": [{"role": "user", "content": prompt}]}
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                r = client.chat.completions.create(**kwargs)
                if r.choices:
                    content = r.choices[0].message.content
                    if not content:
                        last = f"empty content (finish_reason={r.choices[0].finish_reason})"
                    else:
                        m = re.search(r"\{.*\}", content, re.DOTALL)
                        if m:
                            return json.loads(m.group(0))
                        last = f"no JSON in: {content[:200]!r}"
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:200]}"
            wait = min(30, 2 ** attempt)
            if verbose:
                print(f"    judge retry {attempt + 1}/{max_retries}: {last} -- waiting {wait}s")
            time.sleep(wait)
        raise RuntimeError(f"judge failed after {max_retries} attempts: {last}")

    return judge


# ---------------------------------------------------------------------------
# Behavior judge -- a SEPARATE grader from JUDGE_PROMPT above, deliberately.
#
# JUDGE_PROMPT scores general faithfulness/relevancy and must stay byte-identical
# for historical comparability (see the module docstring). It doesn't fit the
# eval_mechanisms_v1 questions that test a specific instructed BEHAVIOR rather than
# citation accuracy -- e.g. "did the model correctly decline an out-of-scope
# question" or "did it stay neutral on a judgment question" -- where a fully
# faithful, on-topic-sounding answer can still fail the actual thing being tested
# (a faithful, relevant recommendation is still a rule-13 violation on a judgment
# question). This judge takes the `expected_behavior` field those questions carry
# (same field eval_multiturn_v1 already uses, there for manual reading) and asks a
# single pass/fail question instead of a 1-5 score, since these are binary
# instructed behaviors, not a quality gradient.
BEHAVIOR_JUDGE_PROMPT = """You are checking whether an AI legal assistant's answer displays ONE SPECIFIC required behavior, for a Jordanian labor/social security law assistant. Do not grade general quality -- score ONLY whether the described behavior happened.

Required behavior: {expected_behavior}

Question: {question}

Source documents given to the assistant:
{context}

Assistant's answer:
{answer}

Respond with ONLY this JSON, no other text:
{{"pass": <true or false>, "reason": "<one sentence explaining why>"}}"""


def make_behavior_judge(client, model=JUDGE_MODEL, max_tokens=None, verbose=True):
    """Build a `judge(question, chunks, answer, expected_behavior, max_retries=5)`
    bound to a client -- same shape and retry/parsing behavior as make_judge()."""

    def judge(question, chunks, answer, expected_behavior, max_retries=5):
        prompt = BEHAVIOR_JUDGE_PROMPT.format(
            expected_behavior=expected_behavior, question=question,
            context=format_context_for_judge(chunks), answer=answer,
        )
        last = None
        for attempt in range(max_retries):
            try:
                kwargs = {"model": model,
                          "messages": [{"role": "user", "content": prompt}]}
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                r = client.chat.completions.create(**kwargs)
                if r.choices:
                    content = r.choices[0].message.content
                    if not content:
                        last = f"empty content (finish_reason={r.choices[0].finish_reason})"
                    else:
                        m = re.search(r"\{.*\}", content, re.DOTALL)
                        if m:
                            return json.loads(m.group(0))
                        last = f"no JSON in: {content[:200]!r}"
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:200]}"
            wait = min(30, 2 ** attempt)
            if verbose:
                print(f"    behavior judge retry {attempt + 1}/{max_retries}: {last} -- waiting {wait}s")
            time.sleep(wait)
        raise RuntimeError(f"behavior judge failed after {max_retries} attempts: {last}")

    return judge
