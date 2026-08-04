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
    article_components, compute_contributions, compute_end_of_service,
    compute_gender_conditions, compute_overtime, compute_leave_accrual,
    extract_salary, extract_service, extract_sex, extract_hourly_wage,
    format_calculation_block, format_end_of_service_block, format_gender_block,
    format_overtime_block, format_leave_block, disability_pension_scope_note,
    _private_funding_articles,
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
print("rendered block carries every component, both totals and no decoy")

# --------------------------------------------------- insurance name per component
# Live report: the model cited article+percent correctly but sometimes dropped what
# the rate was FOR, inconsistently across trials, because nothing forced it to.
# Fixed by extracting the insurance name from the same funding-marker sentence
# already used elsewhere and putting it in the block itself.
EXPECTED_NAMES = {
    ("24", "ar"): "تأمين إصابات العمل", ("24", "en"): "Work Injury Insurance",
    ("42", "ar"): "تأمين الأمومة", ("48", "ar"): "تأمين التعطل عن العمل",
    ("59", "ar"): "تأمين الشيخوخة والعجز والوفاة",
}
for lang in ("ar", "en"):
    res = compute_contributions(family(lang), salary=1000)
    by_art = {c["article"]: c.get("name") for c in res["components"]}
    for art in ("24", "42", "48", "59"):
        assert by_art[art], f"{lang}: article {art} has no insurance name attached"
        want = EXPECTED_NAMES.get((art, lang))
        if want:
            assert by_art[art] == want, f"{lang} {art}: got {by_art[art]!r}, expected {want!r}"
    block = format_calculation_block(res, lang)
    assert "Insurance" in block or "تأمين" in block, f"{lang}: no insurance name reached the block"
    # 42/48/59's English text opens with a stray list marker ("a.\t The ...") -- must be
    # stripped, not leaked into the block
    assert "a." not in block.split("\n")[0] and "\t" not in block
print("insurance name (Work Injury / Maternity / Unemployment / Old-age) attached per component")

# ------------------------------------------- ceiling exception, Article 20(c)(1)
# Anyone insured before 15/10/2009 has a 5,000 ceiling, not 3,000. Which one applies
# is a fact about the person that the question never gives, so the calculator must
# report both and make the model ask -- the first version stated 3,000 flatly, which
# is simply wrong for everyone Article 20(c) exempts.
for lang in ("ar", "en"):
    with_20 = family(lang) + [chunk("ssl", lang, "20")]
    r = compute_contributions(with_20, salary=4000)
    assert r["ceiling"] == 3000 and r["ceiling_exception"] == 5000, \
        f"{lang}: ceilings {r['ceiling']} / {r['ceiling_exception']}"
    assert r["ceiling_exception_date"] is not None, f"{lang}: cutoff date not parsed"
    assert r["ceiling_exception_applies"] is True
    block = format_calculation_block(r, lang)
    assert "5000" in block and ("ASK" in block or "اسأل" in block), \
        f"{lang}: the exemption is not raised as a question"
    # below the standard ceiling neither figure bites -- asking would be noise
    low = compute_contributions(with_20, salary=1000)
    assert low["ceiling_exception_applies"] is False
    assert "5000" not in format_calculation_block(low, lang), f"{lang}: exemption raised needlessly"
print("ceiling exception: 5,000 for pre-2009 insured surfaced as a question above 3,000 only")

# ------------------------------------------------- end of service, Art 32
for lang, salary, years, months, amount in [("ar", 1000, 10, 10, 10000),
                                            ("en", 850, 4.5, 4.5, 3825)]:
    ch = [chunk("labor96", lang, "32")]
    r = compute_end_of_service(ch, salary=salary, years=years)
    assert r is not None, f"{lang}: Article 32 in context but returned None"
    assert r["months_per_year"] == 1.0 and r["months_owed"] == months
    assert r["amount"] == amount, f"{lang}: amount {r['amount']} != {amount}"
    # the precondition most workers fail -- Art 32 applies only OUTSIDE social security
    assert r["requires_not_ssl"] is True, f"{lang}: the not-subject-to-SSL condition was not found"
    block = format_end_of_service_block(r, lang)
    assert ("ASK" in block or "اسأل" in block), f"{lang}: coverage question not raised"
    # no service stated -> nothing to compute; the model must ask instead
    assert compute_end_of_service(ch, salary=salary, years=None) is None
    # wage not given -> still useful, expressed in months of wage
    nw = compute_end_of_service(ch, salary=None, years=years)
    assert nw["amount"] is None and nw["months_owed"] == months
    # wrong article in context -> decline
    assert compute_end_of_service([chunk("labor96", lang, "28")], salary=salary, years=years) is None
print("end of service: rate matched as a phrase, pro-rated, and gated on the not-in-SSL condition")

cases = [("عملت 10 سنوات", 10.0), ("خدمتي 3 سنوات و 6 أشهر", 3.5),
         ("I worked for 7 years", 7.0), ("I have 2 years and 3 months of service", 2.25),
         ("راتبي 1000 دينار", None), ("كم مكافأة نهاية الخدمة؟", None)]
for q, want in cases:
    got = extract_service(q)
    assert got == want, f"extract_service({q!r}) = {got}, expected {want}"
print("service parsing: years, years+months, Arabic and English; None when absent\n")

# Phase 25's eval found the digit-only regex above missed a natural "عملت عشر سنوات"
# ("I worked ten years"): extract_service returned None, the calculator declined, and
# all 4 trials fell back to the exact pre-Phase-24a bug (a confident wage x years
# figure stated as owed, the not-subject-to-SSL precondition never actually raised).
# سنة (year) is feminine and شهر (month) is masculine, and Arabic 3-10 reverses gender
# agreement with the counted noun, so these need separate word forms for the same
# number -- "عشر سنوات" (ten years) vs "عشرة أشهر" (ten months).
word_cases = [
    ("عملت عشر سنوات براتب 800 دينار", 10.0),      # the exact reported phrasing
    ("عملت ثلاث سنوات", 3.0), ("عملت أربع سنوات", 4.0), ("عملت اربع سنوات", 4.0),
    ("عملت خمس سنوات", 5.0), ("عملت ست سنوات", 6.0), ("عملت سبع سنوات", 7.0),
    ("عملت ثمان سنوات", 8.0), ("عملت ثماني سنوات", 8.0), ("عملت تسع سنوات", 9.0),
    ("عملت سنتين", 2.0), ("عملت سنتان", 2.0), ("عملت سنة واحدة", 1.0),
    ("عملت عشرين سنة", 20.0), ("عملت ثلاثين عاماً", 30.0),
    ("عملت خمس وعشرين سنة", 25.0), ("عملت ثلاث وثلاثين سنة", 33.0),
    ("عملت سنتين وثلاثة أشهر", 2.25), ("عملت سنة واحدة وشهرين", 1 + 2 / 12),
    ("انتهت السنة الدراسية", None),  # bare سنة without واحدة must NOT match -- decline
]
for q, want in word_cases:
    got = extract_service(q)
    ok = got == want or (want is not None and got is not None and abs(got - want) < 1e-3)
    assert ok, f"extract_service({q!r}) = {got}, expected {want}"
print("service parsing: word-form Arabic numbers (feminine years, masculine months, "
      "dual, compounds); bare سنة alone still declines\n")

# --------------------------------------- sex-differentiated pension conditions
# Live report: a pension-eligibility answer used the MALE-only thresholds for
# Article 62 (retirement age) and Article 64 (early-retirement subscriptions)
# without ever surfacing that the same sentence gives lower figures for women.
# Ground truth from the article text itself, both languages, both known
# provisions: 62 -> age 60 male / 55 female. 64 -> 252 male / 228 female.
for lang in ("ar", "en"):
    ch = [chunk("ssl", lang, "62"), chunk("ssl", lang, "64")]
    cond = compute_gender_conditions(ch)
    by_art = {c["article"]: c for c in cond}
    assert by_art["62"]["male"] == 60 and by_art["62"]["female"] == 55, by_art["62"]
    assert by_art["64"]["male"] == 252 and by_art["64"]["female"] == 228, by_art["64"]
print("gender conditions: Art 62 age 60/55, Art 64 subscriptions 252/228, both languages")

# no gender-differentiated article retrieved -> nothing invented
assert compute_gender_conditions([chunk("ssl", "ar", "59")]) == []
print("no gender-differentiated article in context -> [] (nothing invented)")

# COMPLETENESS GUARD (Phase 25a): eval_v3 showed this firing on ordinary questions
# whenever Article 62 OR 64 merely appeared in context, e.g. a question about buying
# a missing subscription period pulling in Article 64 as an incidental sibling, not
# the article the question is actually about. A single article present but NOT the
# top-ranked chunk (i.e. something else outranks it) is not a strong enough signal
# -- declines.
assert compute_gender_conditions([chunk("ssl", "ar", "59"), chunk("ssl", "ar", "62")]) == []
assert compute_gender_conditions([chunk("ssl", "en", "59"), chunk("ssl", "en", "64")]) == []
print("gender block: a lower-ranked lone match (something else outranks it) declines")

# RANK EXCEPTION (Phase 32, mt02): a question genuinely and only about early
# retirement retrieves just Article 64, as the TOP-ranked chunk -- Article 62
# (retirement age, a different topic) has no reason to also be retrieved. That
# should NOT decline just because its sibling article is absent.
top64 = compute_gender_conditions([chunk("ssl", "en", "64"), chunk("ssl", "en", "59")])
assert len(top64) == 1 and top64[0]["article"] == "64" and top64[0]["male"] == 252, top64
top62 = compute_gender_conditions([chunk("ssl", "ar", "62"), chunk("ssl", "ar", "59")])
assert len(top62) == 1 and top62[0]["article"] == "62" and top62[0]["male"] == 60, top62
print("gender block: a lone match that IS the top-ranked chunk fires on its own")

BOTH_AR = [chunk("ssl", "ar", "62"), chunk("ssl", "ar", "64")]
BOTH_EN = [chunk("ssl", "en", "62"), chunk("ssl", "en", "64")]

# sex unknown -> both figures, clearly labelled, in the block
both = format_gender_block(compute_gender_conditions(BOTH_AR), "ar", sex=None)
assert "60" in both and "55" in both and ("للذكر" in both and "للأنثى" in both)
both_en = format_gender_block(compute_gender_conditions(BOTH_EN), "en", sex=None)
assert "60" in both_en and "55" in both_en and "male" in both_en and "female" in both_en
print("sex unknown: block states both figures, labelled, in both languages")

# sex known -> ONLY that figure appears for EVERY row; the other never leaks in
male_only = format_gender_block(compute_gender_conditions(BOTH_EN), "en", sex="male")
assert "60" in male_only and "252" in male_only, male_only
assert "55" not in male_only and "228" not in male_only, male_only
female_only = format_gender_block(compute_gender_conditions(BOTH_AR), "ar", sex="female")
assert "55" in female_only and "228" in female_only, female_only
assert "60" not in female_only and "252" not in female_only, female_only
print("sex known: block states only that figure, the other never leaks in")

# extract_sex reads the PERSON's own words, never the assistant's prior recitation --
# the whole point, since the assistant routinely says "if male... if female..." once
# rule 11 fires, which would otherwise look like the person had already told us
sex_cases = [
    ("أنا ذكر وعمري 30", None, "male"), ("I am a woman, 40 years old", None, "female"),
    ("عمري 50", None, None),   # no sex stated anywhere
    ("عمري 50", [{"role": "user", "content": "أنا رجل"}], "male"),
    # assistant's own recitation must NOT be read as the person's answer
    ("عمري 50", [{"role": "assistant",
                  "content": "إذا كنت ذكراً..سن الستين. إذا كنت أنثى..سن الخامسة والخمسين."}], None),
]
for q, hist, want in sex_cases:
    got = extract_sex(q, hist)
    assert got == want, f"extract_sex({q!r}, {hist!r}) = {got}, expected {want}"
print("extract_sex: reads the person's own words only, ignores the assistant's recitation\n")

# --------------------------------------------------------- overtime, Art 59
# Ground truth read directly from the article text: 125% ordinary overtime,
# 150% for the weekly day off / a holiday, both languages.
for lang in ("ar", "en"):
    r = compute_overtime([chunk("labor96", lang, "59")])
    assert r is not None, f"{lang}: Article 59 in context but returned None"
    assert r["ordinary_rate"] == 125.0 and r["holiday_rate"] == 150.0, r
    assert r["ordinary_amount"] is None and r["holiday_amount"] is None
    block = format_overtime_block(r, lang)
    assert "125" in block and "150" in block
    # no hourly wage stated -> the rates alone, never a JD figure invented from a
    # monthly salary
    assert "ASK" not in block and "اسأل" not in block  # not a fact to ask for, just absent
print("overtime: 125%/150% read from Article 59 text, both languages")

# hourly wage stated -> JD amounts computed; a bare monthly salary must NOT trigger this
r = compute_overtime([chunk("labor96", "en", "59")], hourly_wage=4.0)
assert r["ordinary_amount"] == 5.0 and r["holiday_amount"] == 6.0, r  # 4 * 125% / 4 * 150%
assert extract_hourly_wage("my salary is 800 JD") is None
assert extract_hourly_wage("I earn 4 JD per hour") == 4.0
assert extract_hourly_wage("أجري بالساعة 4 دينار") == 4.0
print("overtime: JD amounts only from an explicit hourly wage, never a monthly one")

# wrong article in context -> decline
assert compute_overtime([chunk("labor96", "en", "58")]) is None
print("overtime: declines when Article 59 was not retrieved\n")

# Article 58 also retrieved -> flag the rates as conditional and ask, same pattern
# as compute_end_of_service's SSL-coverage condition
for lang in ("ar", "en"):
    r = compute_overtime([chunk("labor96", lang, "59"), chunk("labor96", lang, "58")])
    assert r["requires_not_excluded_role"] is True, r
    block = format_overtime_block(r, lang)
    ask_word = "اسأل" if lang == "ar" else "ASK"
    assert ask_word in block, f"{lang}: condition line missing from block:\n{block}"
    # the rates themselves are unaffected -- only the framing changes
    assert "125" in block and "150" in block
print("overtime: Article 58 present -> rates flagged conditional, model told to ask")

# ---------------------------------------------------- annual leave, Art 61
# Ground truth read directly from the article text: 14 days/year base, 21
# after 5 consecutive years with the same employer, both languages.
for lang in ("ar", "en"):
    r = compute_leave_accrual([chunk("labor96", lang, "61")], years=3.0)
    assert r is not None, f"{lang}: Article 61 in context but returned None"
    assert r["base_rate_days"] == 14 and r["extended_rate_days"] == 21 \
        and r["extended_threshold_years"] == 5, r
    assert r["threshold_crossed"] is False and r["accrued_days"] == 42.0, r  # 14 * 3
    block = format_leave_block(r, lang)
    assert "42" in block
print("leave accrual: 14/21 days and the 5-year threshold read from Article 61, both languages")

# below the threshold -> exact total; at/past it -> no blended total, both rates stated
below = compute_leave_accrual([chunk("labor96", "en", "61")], years=4.5)
assert below["accrued_days"] == 63.0 and below["threshold_crossed"] is False  # 14 * 4.5
crossed = compute_leave_accrual([chunk("labor96", "en", "61")], years=6.0)
assert crossed["threshold_crossed"] is True and crossed["accrued_days"] is None, crossed
block = format_leave_block(crossed, "en")
assert "21" in block and "14" in block and "DO NOT calculate" in block
print("leave accrual: exact total below 5 years; no invented blended total once past it")

# no years stated -> rate only, no total
no_years = compute_leave_accrual([chunk("labor96", "en", "61")], years=None)
assert no_years["accrued_days"] is None and no_years["threshold_crossed"] is None
assert "not stated" in format_leave_block(no_years, "en")
# Article 63 cash-out is deliberately never computed -- no daily-wage conversion in the law
assert "not computed" in format_leave_block(below, "en")
print("leave accrual: no years stated -> rate only; Art 63 cash value never invented")

# wrong article in context -> decline
assert compute_leave_accrual([chunk("labor96", "en", "63")], years=3.0) is None
print("leave accrual: declines when Article 61 was not retrieved\n")

# ------------------------------- Article 40 vs 85 disambiguation (Phase 34)
# Phase 21/25/31/34 repeatedly found the model misapplying one of Article 85's
# disability-pension paragraphs to a Work Injury Disability Pension question,
# instead of the directly-applicable Article 40. Fires only when BOTH are
# retrieved together -- nothing to compute, no precondition.
note_en = disability_pension_scope_note([chunk("ssl", "en", "40"), chunk("ssl", "en", "85")], "en")
assert note_en is not None and "Article (40)" in note_en and "Article (85)" in note_en
note_ar = disability_pension_scope_note([chunk("ssl", "ar", "40"), chunk("ssl", "ar", "85")], "ar")
assert note_ar is not None and "40" in note_ar and "85" in note_ar
assert disability_pension_scope_note([chunk("ssl", "en", "40")], "en") is None  # 85 absent
assert disability_pension_scope_note([chunk("ssl", "en", "85")], "en") is None  # 40 absent
print("disability pension scope note: fires only when both Article 40 and 85 are retrieved\n")

print(format_calculation_block(
    compute_contributions(family("ar") + [chunk("ssl", "ar", "20")], salary=1000), "ar"))
print()
print(format_calculation_block(
    compute_contributions(family("en") + [chunk("ssl", "en", "20")], salary=5000), "en"))
print()
print(format_gender_block(compute_gender_conditions([chunk("ssl", "ar", "62"), chunk("ssl", "ar", "64")]), "ar"))
print("\nALL CALC TESTS PASSED")
