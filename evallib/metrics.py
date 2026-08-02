"""Retrieval metrics -- the single definition of recall@k and MRR for this project.

Before this module existed the same arithmetic was written out separately in seven
notebooks. All seven agreed, but that was luck rather than design: `LESSONS.md`
records a scoring bug (the Arabic `للمادة` regex) that was fixed in one place while
other runs stayed wrong forever, because nothing tied the copies together. Seven
copies of a metric is seven places a future bug can hide.

`test_equivalence.py` proves this module reproduces every one of those seven
variants exactly, so replacing them changed no published number.

## The rank convention

`rank` is the 1-based position of the correct article in the result list, or NaN if
it never appeared. Everything here follows from that.

## The recall@10 assumption -- read this before reusing the module

`recall@10` is computed as `rank.notna()`, i.e. "was it found at all". That is only
the same thing as "found within 10" **when the search was run exactly 10 deep**,
which is what every notebook that used these metrics did.

Search 30 deep and this column silently becomes recall@30 under a recall@10 label.
The original inline copies all had this assumption baked in with no comment; it is
kept for exact compatibility, and `recall_at()` below is the honest version to use
for new work.
"""

from __future__ import annotations

__all__ = ["metrics", "rank_of_correct", "recall_at", "mrr"]


def _ranks(data, col="rank"):
    """Accept either a DataFrame (take `col`) or an already-extracted Series."""
    return data[col] if hasattr(data, "columns") else data


def recall_at(ranks, k):
    """Share of questions whose correct article landed at position <= k."""
    n = len(ranks)
    return (ranks.notna() & (ranks <= k)).sum() / n if n else 0.0


def mrr(ranks):
    """Mean reciprocal rank. A miss (NaN) contributes 0, not a dropped row --
    hence dividing by the full n rather than by len(dropna())."""
    n = len(ranks)
    return (1 / ranks.dropna()).sum() / n if n else 0.0


def metrics(data, col="rank", *, ndigits=None, include_n=True, full_coverage=False):
    """recall@1 / @5 / @10 / MRR over a rank column.

    Args:
        data: DataFrame with a rank column, or the rank Series itself.
        col: rank column name when `data` is a DataFrame.
        ndigits: round every value to this many decimals (display only).
        include_n: emit the row count under key "n".
        full_coverage: emit "full_coverage@10" from a `full_coverage` column.
            Off by default and must be asked for. An earlier draft inferred it from
            the column merely being present -- `test_equivalence.py` caught that
            immediately, because a DataFrame that happens to carry the column would
            silently gain an extra key and change the shape of a saved CSV. Output
            schema should never depend on what a caller's frame happens to contain.
    """
    r = _ranks(data, col)
    n = len(r)
    if n == 0:
        return {"n": 0} if include_n else {}

    out = {}
    if include_n:
        out["n"] = n
    out["recall@1"] = (r == 1).sum() / n
    out["recall@5"] = recall_at(r, 5)
    out["recall@10"] = r.notna().sum() / n  # see module docstring
    out["mrr"] = mrr(r)

    if full_coverage:
        out["full_coverage@10"] = data["full_coverage"].sum() / n

    if ndigits is not None:
        out = {k: (round(v, ndigits) if k != "n" else v) for k, v in out.items()}
    return out


def rank_of_correct(results, question):
    """1-based position of the correct article in `results`, or None if absent.

    A hit requires all three of law, language and article number to match. Language
    is part of the identity on purpose: the corpus holds each article in both
    Arabic and English, so ignoring it would count the wrong-language copy as a hit
    and quietly inflate every score.
    """
    correct = set(question.answer_article_numbers)
    for pos, r in enumerate(results, start=1):
        if (r["law"] == question.answer_law_code
                and r["language"] == question.answer_language
                and r["article_number"] in correct):
            return pos
    return None
