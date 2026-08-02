"""Shared evaluation helpers for the notebooks in `scripts/`.

Deliberately a separate package from `src/`. `src/` is the deployed application and
ships to the host; nothing here is needed at runtime, and putting evaluation code
there would install it on the server for no reason.
"""

from evallib.metrics import metrics, mrr, rank_of_correct, recall_at
from evallib.judge import (
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_PROMPT,
    format_context_for_judge,
    make_judge,
)

__all__ = [
    "metrics", "mrr", "rank_of_correct", "recall_at",
    "JUDGE_MODEL", "JUDGE_PROMPT", "JUDGE_MAX_TOKENS",
    "format_context_for_judge", "make_judge",
]
