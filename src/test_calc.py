"""Offline tests for src/calc.py. No API calls -- reads the corpus and asserts.

Ground truth is external: the Social Security Corporation's published split is
employee 7.5%, employer 14.25%, total 21.75%. The whole point of the module is
that it reaches that from the article text alone, in both languages, every time.

Run:  python -m src.test_calc
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.calc import (  # noqa: E402
    article_components, compute_contributions, extract_salary,
    format_calculation_block, _private_funding_articles,
)

EMPLOYEE, EMPLOYER = 7.5, 14.25
corpus = pd.read_parquet(PROJECT / "0. cleaned_data" / "chunks_articles.parquet")


def chunk(law, lang, art):
    r = corpus[(corpus.law == law) & (corpus.language == lang)
               & (corpus.article_number.astype(str) == art)]
    assert not r.empty, f"{law}:{lang}:{art} missing from corpus"
    row = r.iloc[0]
    return {"law": law, "language": lang, "article_number": art,
            "text": str(row["text"]), "chunk_id": row["chunk_id"]}


def family(lang, extra=()):
    arts = sorted(_private_funding_articles(lang) | set(extra), key=int)
    return [chunk("ssl", lang, a) for a in arts]


# ---------------------------------------------------------------- derived set
for lang in ("ar", "en"):
    got = _private_funding_articles(lang)
    assert got == {"24", "42", "48", "59"}, f"{lang}: private funding family = {got}"
    assert "72" not in got, f"{lang}: Article 72 (public sector) leaked into the private family"
print("private funding family derived as {24, 42, 48, 59} in both languages; 72 excluded")

# ------------------------------------------------------- the split, both langs
for lang in ("ar", "en"):
    res = compute_contributions(family(lang))
    assert res is not None, f"{lang}: returned None on a complete family"
    assert res["employee_rate"] == EMPLOYEE, f"{lang}: employee {res['employee_rate']} != {EMPLOYEE}"
    assert res["employer_rate"] == EMPLOYER, f"{lang}: employer {res['employer_rate']} != {EMPLOYER}"
    assert res["total_rate"] == 21.75
    arts = {c["article"] for c in res["components"]}
    assert arts == {"24", "42", "48", "59"}, f"{lang}: components drawn from {arts}"
print(f"rates read from article text: employee {EMPLOYEE}% / employer {EMPLOYER}% / total 21.75%")

# ------------------------------------------- decoys must not enter the total
for lang in ("ar", "en"):
    res = compute_contributions(family(lang, extra=["72"]))
    assert res["employee_rate"] == EMPLOYEE and res["employer_rate"] == EMPLOYER, \
        f"{lang}: Article 72 changed the total -- scope filter failed"
    rates = [c["rate"] for c in res["components"]]
    for decoy in (17.5, 50, 17, 21.5, 26):
        assert decoy not in rates, f"{lang}: decoy {decoy}% was counted"
print("Article 72 (public/military), 17.5% (voluntary) and 50% (reduction cap) all excluded")

# ------------------------------------------------------- completeness guard
for lang in ("ar", "en"):
    assert compute_contributions([chunk("ssl", lang, "59")]) is None, \
        f"{lang}: computed a total from Article 59 alone"
    partial = [chunk("ssl", lang, a) for a in ("24", "48", "59")]
    assert compute_contributions(partial) is None, f"{lang}: computed without Article 42"
    assert compute_contributions([]) is None
print("partial families return None rather than a confident wrong total")

# ------------------------------------------------------------------- amounts
res = compute_contributions(family("ar"), salary=1000)
assert res["employee_amount"] == 75.0, res["employee_amount"]
assert res["employer_amount"] == 142.5, res["employer_amount"]
assert res["total_amount"] == 217.5, res["total_amount"]
res = compute_contributions(family("en"), salary=400)
assert res["employee_amount"] == 30.0 and res["employer_amount"] == 57.0
print("amounts: 1000 JD -> 75 / 142.5 / 217.5   400 JD -> 30 / 57")

# --------------------------------------------------- rounding, half-up not half-even
# 850 * 14.25% = 121.125 exactly. Python's round() gives 121.12 (half to even); every
# payslip convention gives 121.13. Measured: the model reproduced the wrong figure in
# 3 of 3 trials, faithfully -- the calculator was wrong, not the model.
res = compute_contributions(family("en"), salary=850)
assert res["employer_amount"] == 121.13, f"850 JD employer = {res['employer_amount']}"
assert res["employee_amount"] == 63.75
assert round(850 * 14.25 / 100, 2) == 121.12, "round() no longer banker's -- revisit this note"
# per-component amounts, so the model never multiplies either
by_art = {(c["article"], c["bearer"]): c["amount"] for c in res["components"]}
assert by_art[("42", "employer")] == 6.38, by_art[("42", "employer")]   # 0.75% of 850 = 6.375
assert by_art[("59", "employer")] == 93.5 and by_art[("48", "employee")] == 8.5
assert all(c["amount"] is not None for c in res["components"])
assert all(c["amount"] is None for c in compute_contributions(family("en"))["components"])
print("rounding: half-up, so 850 JD -> 121.13 not 121.12; every component carries its own amount")

# ------------------------------------------------------------------- ceiling
for lang in ("ar", "en"):
    with_20 = family(lang) + [chunk("ssl", lang, "20")]
    res = compute_contributions(with_20, salary=5000)
    assert res["ceiling"] == 3000, f"{lang}: ceiling parsed as {res['ceiling']}"
    assert res["capped"] is True
    assert res["employee_amount"] == 225.0, res["employee_amount"]     # 3000 * 7.5%
    assert res["employer_amount"] == 427.5, res["employer_amount"]     # 3000 * 14.25%
    res = compute_contributions(with_20, salary=1000)
    assert res["capped"] is False and res["employee_amount"] == 75.0
    # not retrieved means not stated -- never invent it
    assert compute_contributions(family(lang), salary=5000)["ceiling"] is None
print("ceiling: 3000 JD parsed from Article 20(a) in both languages, applied only above it,")
print("         and never invented when Article 20 was not retrieved")

# ------------------------------------------------------------ salary parsing
cases = [
    ("راتبي 1000 دينار، كم أدفع؟", 1000),
    ("راتبي ١٢٥٠ ديناراً", 1250),
    ("My salary is 850 JD", 850),
    ("I earn 1,200 dinars a month", 1200),
    ("كم النسبة التي تتحملها الشركة عن الموظف في الضمان الاجتماعي", None),
    ("What percentage does the company pay?", None),
]
for q, want in cases:
    got = extract_salary(q)
    assert got == want, f"extract_salary({q!r}) = {got}, expected {want}"
print("salary parsing: digits, Arabic-Indic digits and thousands separators; None when absent")

# ----------------------------------------------------------- rendered block
for lang in ("ar", "en"):
    block = format_calculation_block(
        compute_contributions(family(lang) + [chunk("ssl", lang, "20")], salary=1000), lang)
    for token in ("7.5", "14.25", "75", "142.5", "21.75"):
        assert token in block, f"{lang}: {token} missing from rendered block"
    for art in ("24", "42", "48", "59"):
        assert art in block, f"{lang}: article {art} missing from rendered block"
    # as a RATE, not a substring -- the combined amount 217.5 JD legitimately contains "17.5"
    for decoy in ("17.5%", "50%", "26%", "21.5%"):
        assert decoy not in block, f"{lang}: decoy {decoy} reached the block"
print("rendered block carries every component, both totals and no decoy\n")

print(format_calculation_block(
    compute_contributions(family("ar") + [chunk("ssl", "ar", "20")], salary=1000), "ar"))
print()
print(format_calculation_block(
    compute_contributions(family("en") + [chunk("ssl", "en", "20")], salary=5000), "en"))
print("\nALL CALC TESTS PASSED")
