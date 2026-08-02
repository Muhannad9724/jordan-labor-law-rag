"""Deterministic social-security contribution arithmetic.

WHY THIS EXISTS
---------------
Phase 24 measured the generator answering "my salary is 1000 JD, what do I pay and
what does the company pay" at **3/6 correct** with all four funding articles in
context, and 2/6 at another salary. Baseline before the retrieval work was 0/6
(it answered from Article 59 alone). So context was no longer the problem and the
prompt was not going to close the rest: the model was being asked to find six
rates spread over four articles, attribute each to the right side, sum them, and
multiply -- four chained steps, no calculator, at non-zero temperature.

This module does the arithmetic in code. The model is handed finished figures and
writes the answer around them. It never adds anything up itself.

The rates are still READ FROM THE RETRIEVED ARTICLE TEXT, never hardcoded -- if
the corpus is re-extracted or the law is amended, the numbers here follow. What is
hardcoded is only *how to read* the text, and that was derived by measurement, not
assumption:

TRAP 1 -- the bearer is named by the verb, not the actor.
    48(a)  الاشتراكات التي تؤديها المنشأة بنسبة (نصف بالمائة)     employer's own money
    48(b)  الاشتراكات التي تقتطعها المنشأة بنسبة (واحد بالمائة)   deducted from the worker
    Same subject, "the firm", in both. Only تؤدي (pays) vs تقتطع (deducts)
    distinguishes them; English is likewise "paid by" vs "deducted by". A first
    implementation keyed on the actor and tagged all six rates as employer.

TRAP 2 -- both verbs sit in the same window.
    An article states its employer clause then its employee clause, so a window
    wide enough to catch the governing verb also catches the other one. The clause
    that owns a rate is the NEAREST verb to its left, not the first one found.

TRAP 3 -- the funding family is not homogeneous.
    Sibling expansion (src/retrieval.py) promotes {24, 42, 48, 59, 72} together
    because all five share the "تتكون مصادر تمويل" marker. But **Article 72 is the
    public-sector and military regime** -- an ALTERNATIVE to the other four, not a
    component of the same total. It carries its own 11% and 6.5%, numerically
    identical to the private ones. Article 59 additionally carries 17.5%
    (voluntary members) and 50% (a reduction cap). Counting every percentage in
    the retrieved family gives 16 figures of which 6 are the answer, so scope
    filtering is not a refinement -- without it the total is simply wrong.

COMPLETENESS GUARD
------------------
Computing from a partial family would reproduce the original bug in a more
confident voice: Article 59 alone yields 11%/6.5%, which is wrong but looks
authoritative. `compute_contributions` therefore returns None unless every
private-scope funding article is present. The expected set is derived from the
corpus, not written down here.
"""
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CORPUS = PROJECT / "0. cleaned_data" / "chunks_articles.parquet"

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# The marker that defines a funding article, shared with src/retrieval.py.
FUNDING_MARKERS = {"ar": "تتكون مصادر تمويل", "en": "shall be financed"}

# The insurance TYPE each article's rate belongs to (Work Injury, Maternity, ...) is
# not in `format_calculation_block` -- it's read out of the same marker sentence.
# Measured live: without a name attached to each line, the model cites the article and
# percent correctly but omits what the rate is FOR in some trials and states it in
# others (inconsistent, not wrong), because nothing forces it to. Extracted the same
# way as everywhere else in this module -- from the text, not hardcoded -- so a
# re-extracted corpus or a renamed insurance type carries through automatically.
_INSURANCE_NAME_AR = re.compile(r"تتكون مصادر تمويل\s+(.+?)\s+مما يلي")
_INSURANCE_NAME_EN = re.compile(r"(.+?)\s+shall be financed", re.I)
_LEADING_MARKER_RE = re.compile(r"^[a-zA-Z]\.\s*")


def _insurance_name(text, language):
    """The insurance type an article's funding clause names, or None if unmatched."""
    pat = _INSURANCE_NAME_AR if language == "ar" else _INSURANCE_NAME_EN
    m = pat.search(str(text))
    if not m:
        return None
    name = re.sub(r"\s+", " ", m.group(1)).strip()
    name = _LEADING_MARKER_RE.sub("", name).strip()
    return name or None

# Who bears the cost -- keyed on the VERB (trap 1). Order within a list is
# irrelevant; the nearest match wins (trap 2).
BEARER_CUES = {
    "employer": ["تؤديها", "يؤديها", "تؤدي ", "يؤدي ", "paid by"],
    "employee": ["تقتطعها", "يقتطعها", "تقتطع", "يقتطع", "deducted by", "deducted from"],
}

# Rates that are not part of an ordinary private-sector payslip (trap 3).
SCOPE_CUES = {
    "voluntary": ["اختيارية", "المنتسبون", "voluntary"],
    "public": ["القطاع العام", "القوات المسلحة", "الأجهزة الأمنية", "العسكريين",
               "public sector", "Armed", "Security Corps", "military"],
}

_PCT_RE = re.compile(r"[\(\s]([٠-٩\d]+(?:[.,،][٠-٩\d]+)?)\s*%")
_WINDOW = 200

# Wage ceiling, Article 20(a). Arabic writes it in words, English in digits.
_AR_THOUSANDS = {"ثلاثة آلاف": 3000, "ثلاثة الاف": 3000,
                 "خمسة آلاف": 5000, "خمسة الاف": 5000}
_CEILING_AR = re.compile(r"الحد الأعلى للأجر[^.]{0,80}?(" + "|".join(_AR_THOUSANDS) + r")\s*دينار")
_CEILING_EN = re.compile(r"maximum wage[^.]{0,120}?JOD\s*\((\d[\d,]*)\)", re.I)

# Article 20(c)(1) exempts anyone insured before the 2009 amendment took effect, for whom
# the ceiling is 5,000 rather than 3,000. Whether it applies turns on a fact about the
# person that the question never states, so the calculator must NOT pick a side -- it
# reports the ordinary figures and tells the model to ask (SYSTEM_PROMPT rule 11).
# Stating 3,000 flatly, as the first version did, is wrong for everyone it exempts.
_EXEMPT_AR = re.compile(r"يستثنى[^:]{0,120}?:?-?\s*(?:١|1)?\.?\s*المؤمن عليه[^.]{0,200}?"
                        r"(" + "|".join(_AR_THOUSANDS) + r")\s*دينار")
_EXEMPT_EN = re.compile(r"exempted[^§]{0,400}?JOD\s*\((\d[\d,]*)\)", re.I)
_DATE_RE = re.compile(r"\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4}|October\s+\d{1,2},\s*\d{4}", re.I)

# A salary the person stated in their own question.
_SALARY_RE = re.compile(
    r"([٠-٩\d][٠-٩\d,\.]*)\s*(?:دينار|دنانير|jods?|jd|dinars?)"
    r"|(?:دينار|دنانير|jods?|jd|dinars?)\s*([٠-٩\d][٠-٩\d,\.]*)", re.I)

_corpus_cache = {}


def _to_num(s):
    return float(str(s).translate(AR_DIGITS).replace(",", ".").replace("،", "."))


def _amount(base, rate):
    """base * rate%, rounded half-up to the fils.

    Not `round()`: Python rounds half to even, so a wage of 850 JD gave an employer
    share of 121.125 -> 121.12 where every payslip convention says 121.13. The model
    reproduced 121.12 faithfully in 3 of 3 trials, which is the calculator working --
    it was the calculator that was wrong.
    """
    return _money(Decimal(str(base)) * Decimal(str(rate)) / Decimal(100))


def _money(value):
    """Round a Decimal to the fils, half up."""
    return float(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _times(base, factor):
    """base * factor, rounded half-up to the fils. Used for months-of-wage."""
    return _money(Decimal(str(base)) * Decimal(str(factor)))


def _spelled_out(text):
    """Rates written in words. Delegates to the Phase 24 converter already in generation."""
    from .generation import spelled_out_rates
    return [(phrase, float(str(v).rstrip("%"))) for phrase, v in spelled_out_rates(text)]


def _nearest(window, cues_by_key, default=None):
    """The key whose cue appears closest to the END of `window` (trap 2)."""
    best_at, best_key = -1, default
    for key, cues in cues_by_key.items():
        for c in cues:
            at = window.rfind(c)
            if at > best_at:
                best_at, best_key = at, key
    return best_key


def article_components(text):
    """[{rate, bearer, scope}] for every percentage stated in one article's text."""
    hits = [(m.start(), _to_num(m.group(1))) for m in _PCT_RE.finditer(text)]
    for phrase, val in _spelled_out(text):
        at = text.find(phrase)
        if at >= 0:
            hits.append((at, val))

    out = []
    for pos, rate in sorted(hits):
        win = text[max(0, pos - _WINDOW): pos + 40]
        out.append({
            "rate": rate,
            "bearer": _nearest(win, BEARER_CUES),
            "scope": _nearest(win, SCOPE_CUES, default="private"),
        })
    return out


def _private_funding_articles(language):
    """Article numbers whose funding components apply to an ordinary private worker.

    Derived from the corpus so it follows a re-extraction or an amendment; this is
    the completeness guard's expected set, never a source of rates.
    """
    if language in _corpus_cache:
        return _corpus_cache[language]
    import pandas as pd
    corpus = pd.read_parquet(CORPUS)
    marker = FUNDING_MARKERS.get(language, FUNDING_MARKERS["en"])
    wanted = set()
    for r in corpus.itertuples():
        if r.law != "ssl" or r.language != language or marker not in str(r.text):
            continue
        comps = article_components(str(r.text))
        if any(c["scope"] == "private" and c["bearer"] for c in comps):
            wanted.add(str(r.article_number))
    _corpus_cache[language] = wanted
    return wanted


def extract_salary(question):
    """The monthly wage the person stated, or None. Only fires on an explicit currency."""
    m = _SALARY_RE.search(str(question))
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    try:
        val = float(str(raw).translate(AR_DIGITS).replace(",", ""))
    except ValueError:
        return None
    return val if val > 0 else None


def _ceiling(chunks):
    """(standard, exception, cutoff_date, article) from Article 20 if retrieved.

    All-None when Article 20 is not in context: not retrieved means not stated, and
    inventing 3000 would be exactly the kind of ungrounded number this system exists
    to avoid. `exception` is the higher Article 20(c)(1) ceiling, reported alongside
    the standard one rather than chosen between -- see _EXEMPT_AR.
    """
    for c in chunks:
        if c.get("law") != "ssl" or str(c.get("article_number")) != "20":
            continue
        text = str(c["text"])
        std = exc = None
        m = _CEILING_EN.search(text)
        if m:
            std = float(m.group(1).replace(",", ""))
            m2 = _EXEMPT_EN.search(text)
            if m2:
                exc = float(m2.group(1).replace(",", ""))
        else:
            m = _CEILING_AR.search(text)
            if m:
                std = float(_AR_THOUSANDS[m.group(1)])
                m2 = _EXEMPT_AR.search(text)
                if m2:
                    exc = float(_AR_THOUSANDS[m2.group(1)])
        if std is None:
            continue
        if exc is not None and exc <= std:
            exc = None                       # only a *higher* ceiling is an exemption
        d = _DATE_RE.search(text.translate(AR_DIGITS))
        return std, exc, (d.group(0) if d else None), "20"
    return None, None, None, None


def compute_contributions(chunks, salary=None):
    """Employee/employer contribution rates and amounts, or None if not computable.

    Returns None -- deliberately leaving the model to answer unaided -- when the
    retrieved documents do not contain every private-scope funding article. A
    partial total is worse than no total: it is wrong and confident.
    """
    by_lang = {}
    for c in chunks:
        if c.get("law") != "ssl":
            continue
        by_lang.setdefault(c.get("language", "ar"), {})[str(c.get("article_number"))] = str(c["text"])

    for language, arts in by_lang.items():
        expected = _private_funding_articles(language)
        if not expected or not expected <= set(arts):
            continue

        comps = []
        for art in sorted(expected, key=int):
            name = _insurance_name(arts[art], language)
            for c in article_components(arts[art]):
                if c["scope"] == "private" and c["bearer"]:
                    comps.append({"article": art, "name": name, **c})
        if not comps:
            continue

        emp = round(sum(c["rate"] for c in comps if c["bearer"] == "employee"), 4)
        er = round(sum(c["rate"] for c in comps if c["bearer"] == "employer"), 4)

        cap, cap_exc, cap_date, cap_art = _ceiling(chunks)
        base = salary
        capped = bool(salary and cap and salary > cap)
        if capped:
            base = cap
        # Only raise the exemption where it could change the answer: below the standard
        # ceiling neither figure bites, and asking would be noise.
        exc_applies = bool(capped and cap_exc)

        # Per-line amounts too, not just the totals. The model was multiplying each
        # rate by the wage itself to show its working -- correct in the trial run, but
        # it is six more chances to drift, and there is no reason to leave any of them.
        for c in comps:
            c["amount"] = _amount(base, c["rate"]) if base else None

        return {
            "language": language,
            "components": comps,
            "employee_rate": emp,
            "employer_rate": er,
            "total_rate": round(emp + er, 4),
            "salary": salary,
            "base": base,
            "ceiling": cap,
            "ceiling_article": cap_art,
            "ceiling_exception": cap_exc,
            "ceiling_exception_date": cap_date,
            "ceiling_exception_applies": exc_applies,
            "capped": capped,
            "employee_amount": _amount(base, emp) if base else None,
            "employer_amount": _amount(base, er) if base else None,
            "total_amount": _amount(base, emp + er) if base else None,
        }
    return None


# ---------------------------------------------------------------------------
# End-of-service remuneration, Labour Law 1996 Article 32.
#
# Structurally different from contributions, and the difference is the whole point:
# Article 32 opens "a worker WHO IS NOT SUBJECT TO the provisions of the Social
# Security Law". Most Jordanian workers are subject to it, so the formula does not
# apply to them -- an answer that just multiplies wage by years is wrong for the
# majority of the people who would ask. The precondition is surfaced as a question
# the model must put to the person (SYSTEM_PROMPT rule 11), never assumed either way.
#
# The multiplier is matched as a PHRASE rather than hardcoded. If the law is amended
# to half a month per year, the phrase stops matching and the calculator declines
# instead of quietly computing the old rate.
_EOS_RATE = {
    "en": (re.compile(r"one\s+month.{0,3}s?\s+wage\s+for\s+each\s+year", re.I), 1.0),
    "ar": (re.compile(r"أجر\s+شهر\s+عن\s+كل\s+سنة"), 1.0),
}
_EOS_NOT_SSL = {
    "en": re.compile(r"not\s+subject\s+to\s+the\s+provisions\s+of\s+the\s+Social\s+Security", re.I),
    "ar": re.compile(r"غير\s+الخاضع\s+لأحكام\s+قانون\s+الضمان"),
}
_YEARS_RE = re.compile(r"([٠-٩\d]+(?:[.,،][٠-٩\d]+)?)\s*(?:سنوات|سنة|سنين|أعوام|اعوام|عام|years?)")
_MONTHS_RE = re.compile(r"([٠-٩\d]+)\s*(?:شهور|أشهر|اشهر|شهرا|شهراً|شهر|months?)")

# Word-form years/months of service. Phase 25's eval found a natural "عملت عشر سنوات"
# ("I worked ten years") returned None from _YEARS_RE above -- digits only -- so the
# calculator declined and the model fell back to stating a wage-times-years figure as
# owed unconditionally, exactly the failure Phase 24a's precondition guard exists to
# prevent. Unlike AR_NUMBER_WORDS in generation.py (built for rate percentages, an
# effectively ungendered count), سنة (year) is FEMININE and شهر (month) is MASCULINE,
# and Arabic 3-10 REVERSES gender agreement with the counted noun -- "عشر سنوات" (ten
# years) and "عشرة أشهر" (ten months) use different words for the same number 10. Two
# separate unit dicts, not one, is not an oversight.
_AR_UNITS_FEM = {"ثلاث": 3, "أربع": 4, "اربع": 4, "خمس": 5, "ست": 6, "سبع": 7,
                 "ثمان": 8, "ثماني": 8, "تسع": 9, "عشر": 10}
_AR_UNITS_MASC = {"ثلاثة": 3, "أربعة": 4, "اربعة": 4, "خمسة": 5, "ستة": 6, "سبعة": 7,
                  "ثمانية": 8, "تسعة": 9, "عشرة": 10}
_AR_TENS = {"عشرين": 20, "ثلاثين": 30, "أربعين": 40, "اربعين": 40, "خمسين": 50}


def _ar_special_re(one_word, two_word):
    """"سنة واحدة" (one year) / "سنتين"|"سنتان" (two years, a distinct dual WORD, not a
    count of "سنة") -- these already stand alone, unlike "خمس سنوات" where the noun
    follows the number. A first version required a trailing noun after these too and
    silently matched nothing, since there is no second noun word to find.
    """
    return re.compile(r"(?P<one>" + one_word + r")|(?P<two>" + two_word + r")")


def _ar_numeric_re(units, nouns):
    """Bare unit ("خمس"), bare tens ("عشرين") or a compound ("خمس وعشرين"), each
    followed by `nouns`. Compound checked first (longest match) so it isn't swallowed
    by the bare-unit alternative matching just the first word.
    """
    units_alt = "|".join(sorted(units, key=len, reverse=True))
    tens_alt = "|".join(sorted(_AR_TENS, key=len, reverse=True))
    return re.compile(
        r"(?:(?P<unit>" + units_alt + r")\s*و(?P<tens_c>" + tens_alt + r")"
        r"|(?P<tens>" + tens_alt + r")"
        r"|(?P<bare>" + units_alt + r"))"
        r"\s*(?:" + nouns + r")"
    )


# Requiring "واحدة"/"واحد" for 1 rather than matching bare "سنة"/"شهر" avoids a false
# hit on "نهاية السنة" and similar -- decline rather than guess, same rule as
# everywhere else in this module.
_AR_YEARS_SPECIAL_RE = _ar_special_re(one_word=r"سنة\s*واحدة", two_word=r"سنتين|سنتان")
_AR_YEARS_NUMERIC_RE = _ar_numeric_re(_AR_UNITS_FEM, r"سنوات|سنة|سنين|أعوام|اعوام|عام")
_AR_MONTHS_SPECIAL_RE = _ar_special_re(one_word=r"شهر\s*واحد", two_word=r"شهرين|شهران")
_AR_MONTHS_NUMERIC_RE = _ar_numeric_re(_AR_UNITS_MASC, r"شهور|أشهر|اشهر|شهرا|شهراً|شهر")


def _ar_word_value(text, special_re, numeric_re):
    m = special_re.search(text)
    if m:
        return 1 if m.group("one") else 2
    m = numeric_re.search(text)
    if not m:
        return None
    if m.group("unit"):
        unit_val = _AR_UNITS_FEM.get(m.group("unit")) or _AR_UNITS_MASC.get(m.group("unit")) or 0
        return unit_val + _AR_TENS[m.group("tens_c")]
    if m.group("tens"):
        return _AR_TENS[m.group("tens")]
    if m.group("bare"):
        return _AR_UNITS_FEM.get(m.group("bare")) or _AR_UNITS_MASC.get(m.group("bare"))
    return None


def extract_service(question):
    """Years of service the person stated, months folded in, or None.

    "3 years and 6 months" -> 3.5. Months alone -> a fraction of a year, since
    Article 32 gives "a proportional remuneration" for parts of a year. Tries
    digits first, then word-form Arabic numbers (see _ar_word_value above).
    """
    q = str(question)
    y = _YEARS_RE.search(q)
    m = _MONTHS_RE.search(q)
    total = 0.0
    y_val = m_val = None
    if y:
        y_val = float(y.group(1).translate(AR_DIGITS).replace(",", ".").replace("،", "."))
    else:
        wy = _ar_word_value(q, _AR_YEARS_SPECIAL_RE, _AR_YEARS_NUMERIC_RE)
        if wy is not None:
            y_val = float(wy)
    if m:
        m_val = float(m.group(1).translate(AR_DIGITS))
    else:
        wm = _ar_word_value(q, _AR_MONTHS_SPECIAL_RE, _AR_MONTHS_NUMERIC_RE)
        if wm is not None:
            m_val = float(wm)
    if y_val:
        total += y_val
    if m_val:
        total += m_val / 12.0
    return round(total, 4) if (y_val or m_val) and total > 0 else None


def compute_end_of_service(chunks, salary=None, years=None):
    """End-of-service remuneration under Labour Law Art 32, or None if not computable.

    None when Article 32 was not retrieved, when its rate phrase does not match, or
    when the person has not said how long they worked -- the last is a fact only they
    have, so the model is left to ask for it rather than handed a fabricated figure.
    """
    if years is None or years <= 0:
        return None
    for c in chunks:
        if c.get("law") != "labor96" or str(c.get("article_number")) != "32":
            continue
        lang = c.get("language", "en")
        pat = _EOS_RATE.get(lang)
        if not pat:
            continue
        text = str(c["text"])
        if not pat[0].search(text):
            continue                       # rate reworded or amended -- decline, don't guess
        months = round(years * pat[1], 4)
        return {
            "law": "labor96", "article": "32", "language": lang,
            "years": years, "months_per_year": pat[1], "months_owed": months,
            "salary": salary,
            "amount": _times(salary, months) if salary else None,
            "requires_not_ssl": bool(_EOS_NOT_SSL.get(lang, re.compile(r"$^")).search(text)),
        }
    return None


_EOS_LABELS = {
    "en": {
        "head": "COMPUTED FIGURES -- end of service (arithmetic already done, do not recalculate)",
        "rate": "Rate ({art}): {r:g} month's wage per year of actual service",
        "service": "Service stated: {y:g} years",
        "months": "Remuneration: {m:g} months' wage",
        "amount": "On the wage of {s:g} JD given: {a:g} JD",
        "nowage": "The person did not give their wage, so state the result in months' wage.",
        "cond": ("CONDITION -- Article {art} applies ONLY to a worker who is NOT subject to the "
                 "Social Security Law. Most workers are subject to it, and for them this "
                 "remuneration does not arise. The person has not said which they are, so give "
                 "the figure above as conditional on that and ASK whether they are covered by "
                 "social security. Do not present it as what they are owed."),
    },
    "ar": {
        "head": "أرقام محسوبة مسبقاً — مكافأة نهاية الخدمة (الحساب تم، لا تعِد حسابه)",
        "rate": "المعدل (المادة {art}): أجر {r:g} شهر عن كل سنة من الخدمة الفعلية",
        "service": "مدة الخدمة المذكورة: {y:g} سنة",
        "months": "المكافأة: أجر {m:g} شهر",
        "amount": "على الأجر المذكور {s:g} دينار: {a:g} دينار",
        "nowage": "لم يذكر الشخص أجره، فاذكر النتيجة بعدد أشهر الأجر.",
        "cond": ("شرط — المادة ({art}) تنطبق فقط على العامل غير الخاضع لأحكام قانون الضمان "
                 "الاجتماعي. ومعظم العمال خاضعون له ولا تستحق لهم هذه المكافأة. لم يذكر الشخص "
                 "أي الحالتين ينطبق عليه، فاذكر الرقم أعلاه مشروطاً بذلك واسأله هل هو مشمول "
                 "بالضمان الاجتماعي. لا تقدّمه على أنه ما يستحقه."),
    },
}


def format_end_of_service_block(result, lang="en"):
    lb = _EOS_LABELS.get(lang, _EOS_LABELS["en"])
    cur = "دينار" if lang == "ar" else "JD"
    art = result["article"]
    lines = [f"[{lb['head']}]",
             lb["rate"].format(art=art, r=result["months_per_year"]),
             lb["service"].format(y=result["years"]),
             lb["months"].format(m=result["months_owed"])]
    if result["amount"] is not None:
        lines.append(lb["amount"].format(s=result["salary"], a=result["amount"]))
    else:
        lines.append(lb["nowage"])
    if result["requires_not_ssl"]:
        lines.append(lb["cond"].format(art=art))
    return "\n".join(lines)


_BLOCK_LABELS = {
    "en": {
        "head": "COMPUTED FIGURES (arithmetic already done for you -- do not recalculate)",
        "note": ("Each rate below was read from the article text shown above. Use these "
                 "figures as given, cite the article beside each one, and do not add, "
                 "re-derive or adjust any of them."),
        "deducted": "Deducted from the employee",
        "paid": "Paid by the employer",
        "emp_total": "Employee total",
        "er_total": "Employer total",
        "combined": "Combined total",
        "on": "on a monthly wage of",
        "cap_hit": ("The wage exceeds the maximum contributory wage in Article {a}, so the "
                    "amounts above are calculated on {c:g} JD, not on the stated wage."),
        "cap_ok": ("The maximum contributory wage in Article {a} is {c:g} JD, which does not "
                   "affect this calculation."),
        "cap_exc": ("The figures above assume the ordinary {c:g} JD ceiling. Article {a}(c) sets "
                    "a higher ceiling of {e:g} JD for anyone insured before {d}, and the amounts "
                    "would be larger for them. You have not been told which applies, so give the "
                    "figures above and then ASK whether they were insured before {d} -- do not "
                    "assume either case, and do not calculate the {e:g} JD version unasked."),
    },
    "ar": {
        "head": "أرقام محسوبة مسبقاً (الحساب تم، لا تعِد حسابه)",
        "note": ("كل نسبة أدناه مقروءة من نص المواد المعروضة أعلاه. استخدم هذه الأرقام كما هي، "
                 "واذكر المادة بجانب كل نسبة، ولا تجمع أو تعدّل أياً منها."),
        "deducted": "يُقتطع من المؤمن عليه",
        "paid": "تؤديه المنشأة",
        "emp_total": "مجموع ما يُقتطع من المؤمن عليه",
        "er_total": "مجموع ما تؤديه المنشأة",
        "combined": "المجموع الكلي",
        "on": "على أجر شهري مقداره",
        "cap_hit": ("الأجر يتجاوز الحد الأعلى للأجر الخاضع للاقتطاع في المادة ({a})، لذلك "
                    "حُسبت المبالغ أعلاه على {c:g} دينار وليس على الأجر المذكور."),
        "cap_ok": ("الحد الأعلى للأجر الخاضع للاقتطاع في المادة ({a}) هو {c:g} دينار، وهو لا "
                   "يؤثر على هذا الحساب."),
        "cap_exc": ("الأرقام أعلاه محسوبة على الحد الأعلى العادي وهو {c:g} دينار. والفقرة (ج) من "
                    "المادة ({a}) تضع حداً أعلى أكبر مقداره {e:g} دينار لمن شُمل بالتأمين قبل "
                    "{d}، وتكون المبالغ أكبر في حالته. لم يُذكر أي الحالتين تنطبق، فاذكر الأرقام "
                    "أعلاه ثم اسأل هل شُمل بالتأمين قبل {d} — لا تفترض إحدى الحالتين، ولا تحسب "
                    "حالة الـ {e:g} دينار دون أن يُطلب منك."),
    },
}


def format_calculation_block(result, lang="en"):
    """Render `compute_contributions` output for injection into the model's context."""
    lb = _BLOCK_LABELS.get(lang, _BLOCK_LABELS["en"])
    art = "المادة" if lang == "ar" else "Article"
    cur = "دينار" if lang == "ar" else "JD"

    lines = [f"[{lb['head']}]", lb["note"], ""]
    for side, key in (("employee", "deducted"), ("employer", "paid")):
        rows = [c for c in result["components"] if c["bearer"] == side]
        if not rows:
            continue
        lines.append(f"{lb[key]}:")
        for c in rows:
            label = f" ({c['name']})" if c.get("name") else ""
            line = f"  {art} ({c['article']}){label}: {c['rate']:g}%"
            if c.get("amount") is not None:
                line += f" = {c['amount']:g} {cur}"
            lines.append(line)
        total = result[f"{side}_rate"]
        amount = result[f"{side}_amount"]
        label = lb["emp_total"] if side == "employee" else lb["er_total"]
        line = f"  {label}: {total:g}%"
        if amount is not None:
            line += f" = {amount:g} {cur}"
        lines.append(line)
        lines.append("")

    line = f"{lb['combined']}: {result['total_rate']:g}%"
    if result["total_amount"] is not None:
        line += f" = {result['total_amount']:g} {cur}"
    lines.append(line)

    if result["salary"]:
        lines.append(f"({lb['on']} {result['salary']:g} {cur})")
    if result["ceiling"]:
        key = "cap_hit" if result["capped"] else "cap_ok"
        lines.append(lb[key].format(a=result["ceiling_article"], c=result["ceiling"]))
    if result.get("ceiling_exception_applies"):
        lines.append(lb["cap_exc"].format(
            a=result["ceiling_article"], c=result["ceiling"],
            e=result["ceiling_exception"],
            d=result["ceiling_exception_date"] or ("the 2009 amendment" if lang == "en"
                                                   else "تعديل عام 2009")))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sex-differentiated pension conditions, SSL Articles 62 and 64.
#
# WHY THIS EXISTS
# A live conversation asked about pension eligibility (200 subscriptions, then
# "my age is 50") and got an answer built entirely on the MALE thresholds --
# Article 62's retirement age (60) and Article 64's early-retirement subscription
# count (252) -- without ever surfacing that both articles set separate, lower
# figures for women (55 / 228) in the very same sentence. A prompt rule fix
# (SYSTEM_PROMPT rule 11, MULTITURN_REMINDER) narrowed this from "always wrong"
# to "wrong 2 times in 6" -- real progress, but exactly the ceiling LESSONS.md
# 5.4 already named: a prompt cannot make an unreliable capability reliable.
#
# So, same as contributions and end-of-service: extract both figures from the
# text in code, and stop leaving it to the model to remember there are two.
#
# WHITESPACE BUG, FOUND BUILDING THIS
# The first version of the English regexes used a literal space between words
# and silently matched the WRONG clause -- Article 64 has several sex-split
# sub-clauses (historical transition provisions for people who were 42-48
# years old before an earlier amendment), and a fixed single space doesn't
# cross the "\n" the PDF extraction left between "and" and the next "(number)".
# `_norm()` collapses all whitespace before any of these patterns run.
def _norm(text):
    return re.sub(r"\s+", " ", str(text))


# Ages are spelled out in Arabic ("سن الستين", "سن الخامسة والخمسين"); the
# vocabulary is closed to what Article 62 actually uses, same approach as
# AR_NUMBER_WORDS above -- a lookup, not a general ordinal-number parser.
_AGE_WORDS_AR = {"الستين": 60, "الخامسة والخمسين": 55, "الخمسين": 50}
_AGE_ALT_AR = "|".join(sorted(_AGE_WORDS_AR, key=len, reverse=True))

# decode(group, language) -- Arabic Art 62 needs the word lookup; every other case
# (Arabic Art 64's digits, and English throughout) is a plain number, via _to_num.
# A first version shared one `decode` across both languages and silently dropped
# English Article 62 entirely: _AGE_WORDS_AR.get("60") is None, since the English
# capture group is the digit string "60", not the Arabic word "الستين".
def _decode_age(g, language):
    return _AGE_WORDS_AR.get(g) if language == "ar" else _to_num(g)


_GENDER_PATTERNS = {
    # article -> (kind, compiled AR pattern, compiled EN pattern, decoder(group, lang))
    "62": ("age",
           re.compile(r"الذكر\s+سن\s+(" + _AGE_ALT_AR + r")\s+والمؤمن عليها الأنثى\s+سن\s+(" + _AGE_ALT_AR + r")"),
           re.compile(r"age of \((\d+)\) for the male and \((\d+)\) for the female", re.I),
           _decode_age),
    "64": ("subscriptions",
           re.compile(r"\(([٠-٩]+)\)\s*اشتراكاً\s*فعلياً\s*بالنسبة\s*للذكر\s*و\s*\(([٠-٩]+)\)\s*اشتراكاً\s*فعلياً\s*بالنسبة\s*للأنثى"),
           re.compile(r"is not less than \((\d+)\) contributions for the male and \((\d+)\) actual contributions for the female", re.I),
           lambda g, language: _to_num(g)),
}

_GENDER_LABEL = {
    ("62", "ar"): "سن التقاعد (تقاعد الشيخوخة)", ("62", "en"): "retirement age (old-age pension)",
    ("64", "ar"): "عدد الاشتراكات المطلوب (التقاعد المبكر)",
    ("64", "en"): "required subscriptions (early retirement)",
}

_SEX_MALE_RE = re.compile(r"\b(ذكر|رجل)\b|\b(male|man)\b", re.I)
_SEX_FEMALE_RE = re.compile(r"\b(أنثى|امرأة)\b|\b(female|woman)\b", re.I)


def extract_sex(query, history=None):
    """'male' / 'female' / None, read from the PERSON's own words only.

    Deliberately scans only the current question plus prior USER turns, never
    the assistant's. Once rule 11 fires, the assistant's own explanation
    routinely says "if male... if female..." -- scanning that text would make
    every follow-up turn look like the person had already stated their sex,
    defeating the whole point.
    """
    texts = [query]
    for turn in (history or []):
        if turn.get("role") == "user":
            texts.append(str(turn.get("content", "")))
    blob = " ".join(texts)
    m, f = _SEX_MALE_RE.search(blob), _SEX_FEMALE_RE.search(blob)
    if m and not f:
        return "male"
    if f and not m:
        return "female"
    return None  # absent, or both mentioned (e.g. quoting the law back) -- don't guess


def compute_gender_conditions(chunks, sex=None):
    """[{article, kind, label, male, female}] for every sex-differentiated SSL
    condition among the retrieved articles. [] if none apply -- never invents a
    split that isn't actually in front of the model.

    COMPLETENESS GUARD -- Phase 25's eval found this had none, unlike contributions
    and end-of-service: it fired whenever Article 62 OR 64 merely showed up in
    context, whether or not the question was actually about retirement-age or
    early-retirement eligibility. Checked what was actually retrieved on the 4
    eval_v3 questions that triggered it: the two genuinely about retirement
    eligibility (a normal-retirement-age question, an early-retirement-continuation
    question) had BOTH articles clustered together; the two that look like ordinary
    crowding (one about buying a missing subscription period, unrelated to the sex
    split) had only ONE. One article present is incidental; both together is a real
    signal the question is about eligibility broadly -- so this now requires every
    pattern in _GENDER_PATTERNS to have matched, not just one. Confirmed this does
    not affect the actual reported bug: every real gender-eligibility question in
    Phase 24e/25's own tests retrieved both articles together, never one alone.
    """
    out = []
    for c in chunks:
        if c.get("law") != "ssl":
            continue
        art = str(c.get("article_number"))
        spec = _GENDER_PATTERNS.get(art)
        if not spec:
            continue
        kind, pat_ar, pat_en, decode = spec
        lang = c.get("language", "ar")
        text = _norm(c["text"])
        pat = pat_ar if lang == "ar" else pat_en
        m = pat.search(text)
        if not m:
            continue
        g1, g2 = m.group(1), m.group(2)
        male = decode(g1, lang)
        female = decode(g2, lang)
        if male is None or female is None:
            continue  # a word this closed lookup doesn't know -- decline, don't guess
        out.append({"article": art, "kind": kind, "language": lang,
                     "label": _GENDER_LABEL.get((art, lang)),
                     "male": male, "female": female})
    if len({o["article"] for o in out}) < len(_GENDER_PATTERNS):
        return []
    return out


_GENDER_LABELS = {
    "en": {
        "head": "COMPUTED FIGURES -- this law sets different numbers for men and women here",
        "both": ("The person's sex is not known. State BOTH figures below, clearly labelled "
                 "\"if male\" / \"if female\", for every row -- do not pick one silently and "
                 "do not answer as though only one case exists."),
        "known": "The person has told you they are {sex}. Use only the {sex} figure below.",
        "row": "{label} (Article {art}): male {m:g} -- female {f:g}",
        "row_known": "{label} (Article {art}): {v:g}",
    },
    "ar": {
        "head": "أرقام محسوبة مسبقاً -- هذا القانون يضع أرقاماً مختلفة للرجال والنساء هنا",
        "both": ("جنس الشخص غير معروف. اذكر كلا الرقمين أدناه بوضوح، \"إذا كان ذكراً\" / "
                 "\"إذا كانت أنثى\"، لكل بند -- لا تختر أحدهما بصمت ولا تجب وكأن حالة واحدة فقط موجودة."),
        "known": "أخبرك الشخص أنه {sex}. استخدم رقم {sex} فقط أدناه.",
        "row": "{label} (المادة {art}): للذكر {m:g} -- للأنثى {f:g}",
        "row_known": "{label} (المادة {art}): {v:g}",
    },
}


def format_gender_block(conditions, lang="en", sex=None):
    """Render `compute_gender_conditions` output. `conditions` must be non-empty."""
    lb = _GENDER_LABELS.get(lang, _GENDER_LABELS["en"])
    sex_word = ({"male": "ذكر", "female": "أنثى"} if lang == "ar"
                else {"male": "male", "female": "female"}).get(sex)
    lines = [f"[{lb['head']}]", lb["known"].format(sex=sex_word) if sex else lb["both"], ""]
    for c in conditions:
        label = c["label"] or c["article"]
        if sex:
            v = c[sex]
            lines.append(lb["row_known"].format(label=label, art=c["article"], v=v))
        else:
            lines.append(lb["row"].format(label=label, art=c["article"], m=c["male"], f=c["female"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Overtime pay, Labour Law 1996 Article 59.
#
# The rate itself is clean: 125% of the regular hourly wage per overtime hour,
# 150% for work on the weekly day off or a holiday. What is NOT clean is
# turning a stated MONTHLY salary into an hourly figure -- no article anywhere
# in the corpus gives that conversion (Art 56 sets 8 hours/day, 48 hours/week,
# but never states a divisor for wage-per-hour). Inventing one (/26 working
# days, /8 hours, whatever) would be exactly the ungrounded-number failure this
# module exists to prevent, so this calculator ALWAYS reports the two rates
# (safe -- pure text extraction, nothing about the person needed) and only
# reports a JD figure if the person states an HOURLY wage directly. A bare
# "my salary is 800 JD" is left as a percentage, never silently divided.
_OT_HOURLY_RE = re.compile(
    r"([٠-٩\d][٠-٩\d,\.]*)\s*(?:دينار|دنانير|jods?|jd|dinars?)\s*(?:في|بال|/|per)\s*(?:ال)?(?:ساعة|hour)"
    r"|(?:ساعة|hour)\s*(?:بـ|بمبلغ|:)?\s*([٠-٩\d][٠-٩\d,\.]*)\s*(?:دينار|دنانير|jods?|jd|dinars?)",
    re.I)


def extract_hourly_wage(question):
    """An hourly wage the person stated explicitly (JD/hour), or None.

    Deliberately narrower than `extract_salary` -- a bare currency figure is a
    MONTHLY wage everywhere else in this module, so this only fires when an
    hour qualifier sits right next to the figure, never assumed.
    """
    m = _OT_HOURLY_RE.search(str(question))
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    try:
        val = float(str(raw).translate(AR_DIGITS).replace(",", ""))
    except ValueError:
        return None
    return val if val > 0 else None


def compute_overtime(chunks, hourly_wage=None):
    """Overtime pay rates under Art 59, or None if the article was not
    retrieved or its two rates could not be found (reworded, amended --
    decline rather than guess which figure is which).
    """
    for c in chunks:
        if c.get("law") != "labor96" or str(c.get("article_number")) != "59":
            continue
        lang = c.get("language", "en")
        text = str(c["text"])
        pcts = [_to_num(m.group(1)) for m in _PCT_RE.finditer(text)]
        if len(pcts) != 2:
            continue
        ordinary_rate, holiday_rate = pcts
        return {
            "law": "labor96", "article": "59", "language": lang,
            "ordinary_rate": ordinary_rate, "holiday_rate": holiday_rate,
            "hourly_wage": hourly_wage,
            "ordinary_amount": _amount(hourly_wage, ordinary_rate) if hourly_wage else None,
            "holiday_amount": _amount(hourly_wage, holiday_rate) if hourly_wage else None,
        }
    return None


_OT_LABELS = {
    "en": {
        "head": "COMPUTED FIGURES -- overtime pay (Article 59, rates already read from the text)",
        "rates": ("Ordinary overtime: {o:g}% of the regular hourly wage, per overtime hour. Work "
                  "on the weekly day off or a holiday: {h:g}% of the regular hourly wage."),
        "amounts": ("On the stated hourly wage of {w:g} JD: {oa:g} JD per ordinary overtime hour, "
                    "{ha:g} JD per hour on a day off or holiday."),
        "nowage": ("The person did not state an HOURLY wage, so give the rates as percentages of "
                   "their regular hourly wage -- do not invent a JD figure from a monthly salary; "
                   "the documents do not state how to convert one into the other."),
    },
    "ar": {
        "head": "أرقام محسوبة مسبقاً -- العمل الإضافي (المادة 59، النسب مقروءة من النص)",
        "rates": ("العمل الإضافي العادي: {o:g}% من أجر الساعة المعتاد عن كل ساعة إضافية. العمل في "
                  "يوم العطلة الأسبوعية أو عطلة رسمية: {h:g}% من أجر الساعة المعتاد."),
        "amounts": ("على أجر الساعة المذكور {w:g} دينار: {oa:g} دينار لكل ساعة عمل إضافي عادية، "
                    "{ha:g} دينار لكل ساعة في يوم العطلة أو العطلة الرسمية."),
        "nowage": ("لم يذكر الشخص أجر الساعة، فاذكر النسب كنسبة من أجر الساعة المعتاد -- لا تخترع "
                   "رقماً بالدينار من راتب شهري، فالوثائق لا تنص على طريقة التحويل بينهما."),
    },
}


def format_overtime_block(result, lang="en"):
    """Render `compute_overtime` output for injection into the model's context."""
    lb = _OT_LABELS.get(lang, _OT_LABELS["en"])
    lines = [f"[{lb['head']}]",
             lb["rates"].format(o=result["ordinary_rate"], h=result["holiday_rate"])]
    if result["hourly_wage"] and result["ordinary_amount"] is not None:
        lines.append(lb["amounts"].format(w=result["hourly_wage"], oa=result["ordinary_amount"],
                                           ha=result["holiday_amount"]))
    else:
        lines.append(lb["nowage"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Annual leave accrual, Labour Law 1996 Article 61 (and Article 63's cash-out
# on unused leave, mentioned only -- see below).
#
# The rate: 14 days per year of service, rising to 21 once the worker has
# completed 5 consecutive years WITH THE SAME EMPLOYER. `years` from
# `extract_service` is treated as continuous service with the current
# employer -- the same simplifying read end-of-service already makes.
#
# What this deliberately does NOT do: collapse a span that crosses the
# threshold into one blended total. "6 years of service" is not "6 x 21 days"
# -- the first 5 years accrued at 14 days/year, and this module has no hire
# date, only a total years figure, so it cannot know exactly when the 21-day
# rate started applying. Multiplying the CURRENT rate by the WHOLE span would
# be precisely the confident-but-wrong pattern the rest of this module exists
# to avoid (see compute_contributions' completeness guard, compute_end_of_service's
# precondition). Past the threshold this returns the two rates and the
# threshold, not a single number, and tells the model to state it that way.
#
# The cash value of unused leave (Art 63) is NOT computed here at all: same
# problem as overtime, no article states how to derive a daily wage from a
# monthly one, so a JD figure here would be invented, not read.
_LEAVE_BASE_RE = {"en": re.compile(r"fourteen\s+days", re.I),
                  "ar": re.compile(r"أربعة\s*عشر\s*يوماً")}
_LEAVE_EXT_RE = {"en": re.compile(r"twenty-one\s+days", re.I),
                 "ar": re.compile(r"واحداً?\s*وعشرين\s*يوماً")}
_LEAVE_THRESHOLD_RE = {"en": re.compile(r"five\s+consecutive\s+years", re.I),
                       "ar": re.compile(r"خمس\s*سنوات\s*متصلة")}
_LEAVE_BASE_RATE, _LEAVE_EXT_RATE, _LEAVE_THRESHOLD_YEARS = 14, 21, 5


def compute_leave_accrual(chunks, years=None):
    """Annual leave entitlement under Art 61, or None if the article was not
    retrieved or its wording no longer matches the rates above (decline, don't
    guess -- same discipline as `compute_end_of_service`'s rate phrase match).
    """
    for c in chunks:
        if c.get("law") != "labor96" or str(c.get("article_number")) != "61":
            continue
        lang = c.get("language", "en")
        text = _norm(c["text"])
        if not (_LEAVE_BASE_RE[lang].search(text) and _LEAVE_EXT_RE[lang].search(text)
                and _LEAVE_THRESHOLD_RE[lang].search(text)):
            continue
        result = {
            "law": "labor96", "article": "61", "language": lang,
            "base_rate_days": _LEAVE_BASE_RATE, "extended_rate_days": _LEAVE_EXT_RATE,
            "extended_threshold_years": _LEAVE_THRESHOLD_YEARS,
            "years": years, "accrued_days": None, "threshold_crossed": None,
        }
        if years is not None and years > 0:
            crossed = years >= _LEAVE_THRESHOLD_YEARS
            result["threshold_crossed"] = crossed
            if not crossed:
                result["accrued_days"] = _times(_LEAVE_BASE_RATE, years)
        return result
    return None


_LEAVE_LABELS = {
    "en": {
        "head": "COMPUTED FIGURES -- annual leave entitlement (Article 61, rates already read from the text)",
        "rate": ("Rate: {b:g} days per year of service, rising to {e:g} days per year once {t:g} "
                 "consecutive years have been completed with the same employer."),
        "total": ("Service stated: {y:g} years, below the {t:g}-year threshold -- accrued at "
                  "{b:g} days/year: {d:g} days."),
        "crossed": ("Service stated: {y:g} years, past the {t:g}-year threshold. The current "
                    "entitlement is {e:g} days/year, and the years before the threshold accrued at "
                    "{b:g} days/year. DO NOT calculate a total by splitting the {y:g} years into a "
                    "block at {b:g} days/year and a remaining block at {e:g} days/year -- the exact "
                    "date the {t:g}-year mark was reached is not known here, so that split is an "
                    "assumption you would be inventing, not something the documents state. State "
                    "only the two rates and the {t:g}-year threshold, and say plainly that a single "
                    "total cannot be given without knowing when the threshold was reached."),
        "nototal": "Years of service were not stated, so give the rate only, not a total.",
        "nocash": ("The cash value of unused leave (Article 63) is not computed here -- it needs a "
                   "daily wage, and the documents do not state how to convert a monthly wage into one."),
    },
    "ar": {
        "head": "أرقام محسوبة مسبقاً -- استحقاق الإجازة السنوية (المادة 61، النسب مقروءة من النص)",
        "rate": ("المعدل: {b:g} يوماً عن كل سنة خدمة، ويرتفع إلى {e:g} يوماً بعد إتمام {t:g} سنوات "
                 "متصلة لدى صاحب العمل نفسه."),
        "total": ("مدة الخدمة المذكورة: {y:g} سنة، أقل من عتبة الـ {t:g} سنوات -- محسوبة بمعدل "
                  "{b:g} يوماً/سنة: {d:g} يوماً."),
        "crossed": ("مدة الخدمة المذكورة: {y:g} سنة، تجاوزت عتبة الـ {t:g} سنوات. الاستحقاق الحالي "
                   "{e:g} يوماً/سنة، والسنوات التي سبقت العتبة كانت بمعدل {b:g} يوماً/سنة. لا تحسب "
                   "رقماً إجمالياً بتقسيم الـ {y:g} سنة إلى قسم بمعدل {b:g} وقسم بمعدل {e:g} -- "
                   "التاريخ الفعلي لتجاوز عتبة الـ {t:g} سنوات غير معروف هنا، فهذا التقسيم افتراض "
                   "من عندك وليس منصوصاً عليه في الوثائق. اذكر المعدلين وعتبة الـ {t:g} سنوات فقط، "
                   "وقل بوضوح إنه لا يمكن إعطاء رقم إجمالي واحد دون معرفة تاريخ بلوغ العتبة."),
        "nototal": "لم تُذكر مدة الخدمة، فاذكر المعدل فقط دون رقم إجمالي.",
        "nocash": ("القيمة النقدية للإجازة غير المستخدمة (المادة 63) غير محسوبة هنا -- فهي تحتاج "
                   "أجراً يومياً، والوثائق لا تنص على طريقة تحويل الراتب الشهري إليه."),
    },
}


def format_leave_block(result, lang="en"):
    """Render `compute_leave_accrual` output for injection into the model's context."""
    lb = _LEAVE_LABELS.get(lang, _LEAVE_LABELS["en"])
    b, e, t = result["base_rate_days"], result["extended_rate_days"], result["extended_threshold_years"]
    lines = [f"[{lb['head']}]", lb["rate"].format(b=b, e=e, t=t)]
    if result["years"] is None:
        lines.append(lb["nototal"])
    elif result["threshold_crossed"]:
        lines.append(lb["crossed"].format(y=result["years"], t=t, e=e, b=b))
    else:
        lines.append(lb["total"].format(y=result["years"], t=t, b=b, d=result["accrued_days"]))
    lines.append(lb["nocash"])
    return "\n".join(lines)
