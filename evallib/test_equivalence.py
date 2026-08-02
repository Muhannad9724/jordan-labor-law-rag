"""Proof that evallib reproduces the inline helpers it replaced, exactly.

Run:  python evallib/test_equivalence.py

Seven notebooks each defined their own `metrics`, and three their own
`rank_of_correct`. They all agreed on the arithmetic, but that was luck rather
than design -- `LESSONS.md` records a scoring bug fixed in one place while other
runs stayed wrong forever, because nothing tied the copies together.

Replacing seven copies with one shared function is only safe if the shared one
returns identical values. This file is that proof.

## Why the originals are pasted in below rather than read from the notebooks

The first version of this test extracted the implementations live from the
notebook JSON. That worked exactly once -- after rewiring, the notebooks no
longer contain the old code, so the test had nothing left to compare against and
would have silently degraded into checking the new code against itself.

The seven historical implementations are therefore frozen here verbatim. This
file is now the record of what the published numbers were computed with, and it
keeps testing that even after every notebook has moved on.

## Deliberately not covered

Empty input. `metrics` is called on empty frames by none of these notebooks, and
the originals disagree there anyway (some raise, some return NaN, some guard).
evallib returns `{}` / `{"n": 0}`. Nothing depends on the difference.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT = next((d for d in [Path.cwd(), *Path.cwd().parents]
                if (d / "EXPLAINER.md").exists()), None)
assert PROJECT, "run from inside the project tree"
sys.path.insert(0, str(PROJECT))

from evallib import JUDGE_MODEL as CANON_MODEL          # noqa: E402
from evallib import JUDGE_PROMPT as CANON_PROMPT        # noqa: E402
from evallib import metrics as canon_metrics            # noqa: E402
from evallib import rank_of_correct as canon_rank       # noqa: E402

SCRIPTS = PROJECT / "scripts"

# ---------------------------------------------------------------------------
# The seven historical `metrics`, verbatim as they stood before extraction.
# ---------------------------------------------------------------------------
ORIGINALS = {
    "06": ('''
def metrics(df):
    n = len(df)
    r = df["rank"]
    return {"n": n, "recall@1": (r == 1).sum() / n, "recall@5": (r.notna() & (r <= 5)).sum() / n,
            "recall@10": r.notna().sum() / n, "mrr": (1 / r.dropna()).sum() / n}
''', "df", dict()),

    "11": ('''
def metrics(df, col):
    n = len(df)
    r = df[col]
    return {"n": n, "recall@1": (r == 1).sum() / n, "recall@5": (r.notna() & (r <= 5)).sum() / n,
            "recall@10": r.notna().sum() / n, "mrr": (1 / r.dropna()).sum() / n}
''', "df_col", dict()),

    "13": ('''
def metrics(df, col="rank"):
    n = len(df); r = df[col]
    return {"n": n,
            "recall@1": round((r == 1).sum() / n, 3),
            "recall@5": round((r.notna() & (r <= 5)).sum() / n, 3),
            "recall@10": round(r.notna().sum() / n, 3),
            "mrr": round((1 / r.dropna()).sum() / n, 3)}
''', "df", dict(ndigits=3)),

    "14": ('''
def metrics(r):
    n = len(r)
    return {"recall@1": round((r == 1).sum() / n, 3),
            "recall@5": round((r.notna() & (r <= 5)).sum() / n, 3),
            "recall@10": round(r.notna().sum() / n, 3),
            "mrr": round((1 / r.dropna()).sum() / n, 3)}
''', "series", dict(ndigits=3, include_n=False)),

    "19": ('''
def metrics(df):
    n = len(df)
    if n == 0:
        return {"n": 0}
    r = df["rank"]
    return {
        "n": n,
        "recall@1": (r == 1).sum() / n,
        "recall@5": (r.notna() & (r <= 5)).sum() / n,
        "recall@10": r.notna().sum() / n,
        "mrr": (1 / r.dropna()).sum() / n,
        "full_coverage@10": df["full_coverage"].sum() / n,
    }
''', "df", dict(full_coverage=True)),

    "21": ('''
def metrics(df):
    r, n = df["rank"], len(df)
    return {"recall@1": (r == 1).sum() / n, "recall@5": (r.notna() & (r <= 5)).sum() / n,
            "recall@10": r.notna().sum() / n, "mrr": (1 / r.dropna()).sum() / n,
            "full_coverage@10": df["full_coverage"].sum() / n}
''', "df", dict(include_n=False, full_coverage=True)),

    "22": ('''
def metrics(df):
    r, n = df["rank"], len(df)
    return {"recall@1": (r == 1).sum() / n, "recall@5": (r.notna() & (r <= 5)).sum() / n,
            "recall@10": r.notna().sum() / n, "mrr": (1 / r.dropna()).sum() / n}
''', "df", dict(include_n=False)),
}

# All three copies were logically identical; parameter names differed only.
ORIGINAL_RANK = '''
def rank_of_correct(results, row):
    correct_articles = set(row.answer_article_numbers)
    for pos, r in enumerate(results, start=1):
        if r["law"] == row.answer_law_code and r["language"] == row.answer_language and r["article_number"] in correct_articles:
            return pos
    return None
'''


def compile_fn(src, name):
    ns = {"pd": pd, "np": np}
    exec(compile(src, f"<original {name}>", "exec"), ns)
    return ns[name]


def frames():
    """Rank vectors covering the shapes these metrics actually see."""
    N = np.nan
    cases = {
        "all rank 1":      [1, 1, 1, 1],
        "spread 1..11":    [1, 2, 3, 5, 6, 10, N, N],
        "boundary 5 vs 6": [5, 6, 5, 6],
        "all missing":     [N, N, N],
        "single hit":      [1],
        "single miss":     [N],
        "realistic 60":    ([1] * 40 + [2] * 8 + [3] * 5 + [4] * 3 + [N] * 4),
    }
    for label, ranks in cases.items():
        df = pd.DataFrame({"rank": pd.Series(ranks, dtype="float64")})
        df["full_coverage"] = [bool(i % 3) for i in range(len(df))]
        yield label, df


def close(a, b):
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, (int, float, np.floating)) and isinstance(y, (int, float, np.floating)):
            if not (abs(float(x) - float(y)) < 1e-12 or (pd.isna(x) and pd.isna(y))):
                return False
        elif x != y:
            return False
    return True


failures = []
cases = list(frames())

print("== metrics: evallib vs the seven historical implementations ==")
for nb, (src, kind, canon_kwargs) in sorted(ORIGINALS.items()):
    fn = compile_fn(src, "metrics")
    ok = 0
    for label, df in cases:
        if kind == "df":
            got, want = fn(df), canon_metrics(df, **canon_kwargs)
        elif kind == "df_col":
            got, want = fn(df, "rank"), canon_metrics(df, "rank", **canon_kwargs)
        else:
            got, want = fn(df["rank"]), canon_metrics(df["rank"], **canon_kwargs)
        if close(got, want):
            ok += 1
        else:
            failures.append((nb, label, f"{got} != {want}"))
    print(f"   notebook {nb} original  {ok}/{len(cases)} inputs match")

print("\n== rank_of_correct: evallib vs the historical implementation ==")
RESULTS = [
    {"law": "ssl", "language": "ar", "article_number": 12},
    {"law": "labor96", "language": "en", "article_number": 61},
    {"law": "ssl", "language": "en", "article_number": 90},
    {"law": "ssl", "language": "ar", "article_number": 90},
]
QUESTIONS = [
    ("hit at 3", SimpleNamespace(answer_article_numbers=[90], answer_law_code="ssl", answer_language="en")),
    ("hit at 1", SimpleNamespace(answer_article_numbers=[12], answer_law_code="ssl", answer_language="ar")),
    ("wrong language is not a hit", SimpleNamespace(answer_article_numbers=[61], answer_law_code="labor96", answer_language="ar")),
    ("absent", SimpleNamespace(answer_article_numbers=[999], answer_law_code="ssl", answer_language="ar")),
    ("multi-article, second matches", SimpleNamespace(answer_article_numbers=[999, 61], answer_law_code="labor96", answer_language="en")),
]
orig_rank = compile_fn(ORIGINAL_RANK, "rank_of_correct")
ok = 0
for label, q in QUESTIONS:
    got, want = orig_rank(RESULTS, q), canon_rank(RESULTS, q)
    if got == want:
        ok += 1
    else:
        failures.append(("rank_of_correct", label, f"{got} != {want}"))
print(f"   {ok}/{len(QUESTIONS)} cases match")

print("\n== judge constants: evallib vs whatever the notebooks still declare ==")
# This check exists because the first draft of evallib/judge.py contained a
# JUDGE_PROMPT written from memory rather than copied. It graded the same two
# axes with different wording and different JSON keys -- it would have replaced
# the grader silently while every other test still passed.
lits = {}
for p in sorted(SCRIPTS.glob("*.ipynb")):
    nb = json.loads(p.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if getattr(t, "id", "") in ("JUDGE_PROMPT", "JUDGE_MODEL"):
                        try:
                            lits.setdefault(t.id, {})[p.name[:2]] = ast.literal_eval(node.value)
                        except Exception:
                            pass

for name, canon in (("JUDGE_PROMPT", CANON_PROMPT), ("JUDGE_MODEL", CANON_MODEL)):
    found = lits.get(name, {})
    bad = [nb for nb, v in found.items() if v != canon]
    state = "none left inline (all import from evallib)" if not found else (
        "all match evallib" if not bad else "MISMATCH in " + ", ".join(bad))
    print(f"   {name:13s} {len(found)} notebook literal(s), {state}")
    for nb in bad:
        failures.append((nb, name, "differs from evallib -- the grader would change"))

print()
if failures:
    print(f"FAILED -- {len(failures)} mismatch(es):")
    for nb, label, detail in failures:
        print(f"   {nb}  {label}: {detail}")
    sys.exit(1)
print("PASS -- evallib reproduces every historical implementation on every input tested.")
