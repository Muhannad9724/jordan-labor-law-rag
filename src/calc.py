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


def extract_service(question):
    """Years of service the person stated, months folded in, or None.

    "3 years and 6 months" -> 3.5. Months alone -> a fraction of a year, since
    Article 32 gives "a proportional remuneration" for parts of a year.
    """
    q = str(question)
    y = _YEARS_RE.search(q)
    m = _MONTHS_RE.search(q)
    total = 0.0
    if y:
        total += float(y.group(1).translate(AR_DIGITS).replace(",", ".").replace("،", "."))
    if m:
        total += float(m.group(1).translate(AR_DIGITS)) / 12.0
    return round(total, 4) if (y or m) and total > 0 else None


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
