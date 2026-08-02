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
    cents = (Decimal(str(base)) * Decimal(str(rate)) / Decimal(100))
    return float(cents.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


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
    """(amount, article) from Article 20(a) if it was retrieved, else (None, None).

    Not retrieved means not stated -- inventing 3000 would be exactly the kind of
    ungrounded number the whole system is built to avoid.
    """
    for c in chunks:
        if c.get("law") != "ssl" or str(c.get("article_number")) != "20":
            continue
        text = str(c["text"])
        m = _CEILING_EN.search(text)
        if m:
            return float(m.group(1).replace(",", "")), "20"
        m = _CEILING_AR.search(text)
        if m:
            return float(_AR_THOUSANDS[m.group(1)]), "20"
    return None, None


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
            for c in article_components(arts[art]):
                if c["scope"] == "private" and c["bearer"]:
                    comps.append({"article": art, **c})
        if not comps:
            continue

        emp = round(sum(c["rate"] for c in comps if c["bearer"] == "employee"), 4)
        er = round(sum(c["rate"] for c in comps if c["bearer"] == "employer"), 4)

        cap, cap_art = _ceiling(chunks)
        base = salary
        capped = bool(salary and cap and salary > cap)
        if capped:
            base = cap

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
            "capped": capped,
            "employee_amount": _amount(base, emp) if base else None,
            "employer_amount": _amount(base, er) if base else None,
            "total_amount": _amount(base, emp + er) if base else None,
        }
    return None


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
            line = f"  {art} ({c['article']}): {c['rate']:g}%"
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
    return "\n".join(lines)
