# How this project works, explained from scratch

No prior knowledge assumed. If you know what RAG is, skip to [Part 2](#part-2--the-pipeline-step-by-step).

`ROADMAP.md` is the formal build log with every number. This is the plain-language
version of the same story. `LESSONS.md` is the companion piece about what went wrong.

---

## Part 1 — What this is and why it exists

### The thing itself

A website where someone types a question about Jordanian employment law — in Arabic
or English — and gets an answer that quotes the actual article of the actual law,
with a citation.

It covers three laws:

- **Labour Law 1996** — the main employment law
- **Professional Work Law 2019** — covers licensed professions
- **Social Security Law** — pensions, injury compensation, unemployment

It is live at `jordan-legal-guide.up.railway.app`.

### Why not just ask ChatGPT?

Ask a general chatbot "how much end-of-service compensation am I owed in Jordan"
and it will give you a fluent, confident, plausible answer. The problem is you have
no way to tell whether it's true.

Language models generate text that *sounds* right. When they don't know something,
they don't stop — they produce something plausible anyway. For legal questions
that's not a small flaw. An invented article number is worse than no answer,
because it looks exactly like a real one.

There's a second problem: the model's training data has a cutoff date and doesn't
reliably include Jordan's Social Security Law in full.

### The fix: give it the book

**RAG** stands for *Retrieval-Augmented Generation*. The idea is simple:

> Instead of asking the model to answer from memory, **find the relevant law text
> first**, hand it to the model, and tell it: answer using only this.

It's the difference between a closed-book exam and an open-book one.

In a closed-book exam you're relying on memory, and when memory fails you guess.
In an open-book exam you look up the page first, then write the answer from what's
in front of you — and you can point at the page you used.

That "point at the page" part is what makes RAG worth the effort here. Every answer
this system produces comes with the article it came from, and the user can read the
original text themselves.

So the whole system is two jobs:

1. **Retrieval** — given a question, find the right articles. *(the hard part)*
2. **Generation** — given those articles, write an answer that only uses them.

Most of the work in this project went into job #1, because a perfect writer given
the wrong page still writes a wrong answer.

---

## Part 2 — The pipeline, step by step

Here's the whole thing end to end. Each step is explained below.

```
  PDF of the law
       ↓  (1) extract the text
  clean article-by-article tables
       ↓  (2) cut into pieces
  531 articles
       ↓  (3) turn each into numbers
  531 vectors
       ↓  (4) store them
  Pinecone database
       ⋮
  user's question  ──(5)──→  find the 5 best articles
       ↓
  (6) write an answer using only those 5
       ↓
  answer + citations shown in the browser
```

### Step 1 — Getting the text out of the PDFs

The laws come as PDFs. English ones had a real text layer, so the text could be
pulled straight out. The Arabic ones were scans — images of pages — so they needed
**OCR** (optical character recognition: software that reads text out of a picture).

That produced raw text, which then had to be cut into individual articles, since
"Article 61" is the unit people actually cite.

This step was messier than it sounds and produced a list of genuine defects that
had to be found and fixed: duplicate keys silently losing content, cross-references
being mistaken for headings, chapter titles and footnotes leaking into article
bodies, a missing article header, and a misread digit.

**Output:** six spreadsheets, one per law per language, one row per article.

### Step 2 — Cutting the law into pieces ("chunking")

You can't hand an entire law to a language model — it's too long, and burying one
relevant sentence in 300 pages of irrelevant text makes the answer worse, not
better. So the text gets cut into pieces.

**The question is: how big should a piece be?**

Think of it as writing index cards. Too small and a card is meaningless on its own.
Too big and one card covers five unrelated topics, so pulling it up doesn't tell
you much.

The first attempt split long articles at their lettered sub-clauses — (a), (b), (c)
— producing **739 pieces**. The reasoning: a whole article covers several rules, so
one "summary" of it would be blurry.

**That reasoning turned out to be wrong**, and finding out why was one of the more
interesting moments in the project. Two things went wrong:

**First, the clauses didn't make sense alone.** Here is a real one:

> *"b. Pension stipulated in Paragraph (a) of this Sub-paragraph shall be increased
> by (0.5%)…"*

Which pension? The card never says. "Paragraph (a)" points at a different card. On
its own this text is nearly meaningless — and search works on what the text says.

**Second, one article could hog the results.** A question about pension increases
returned ten results, and *seven of them were separate clauses of the same
article*, all saying near-identical things. The genuinely correct article — a
different one — got pushed off the list entirely. It wasn't ranked badly. It was
crowded out.

So the corpus was rebuilt at **whole-article granularity: 531 pieces**. Every
retrieval measure improved. The original assumption ("whole articles are blurry")
was wrong for this specific corpus because these articles are *single-topic* —
Social Security Article 71 is entirely about military pensions, Article 64 entirely
about early retirement. Merging clauses restored one coherent subject rather than
blending several.

**Lesson:** the right chunk size is a property of your documents, not a general
rule. It has to be tested.

### Step 3 — Turning text into numbers ("embeddings")

This is the concept most people find strangest, so here's the long version.

A computer can't compare meanings directly. What it *can* do is compare numbers. So
we use a model that converts a piece of text into a long list of numbers — a
**vector** — with one crucial property:

> **Text with similar meaning produces similar numbers.**

Picture a map. Every article in the law gets placed somewhere on it. Articles about
maternity leave land in one neighbourhood; articles about work injuries land in
another, far away. The map doesn't care what words were used — it cares what the
text is *about*.

Now, when someone asks a question, the question gets placed on the same map. The
answer is: whatever's nearest.

This is why RAG works across languages. The same model places Arabic and English
text on the *same* map, so an Arabic question can land next to an English article
about the same topic — even though they share not a single word.

(The real map has 3,072 dimensions rather than 2. Nobody can picture that, and you
don't need to. The intuition — *similar meaning, nearby location* — is the part
that matters.)

**Which model draws the map matters a lot**, and this project got it wrong the
first time. Seven models were tested. The story is in Part 4.

### Step 4 — Storing the vectors

531 articles → 531 vectors. These live in **Pinecone**, a hosted database built for
exactly one job: given a query vector, find the nearest stored ones, fast.

At 531 items you could honestly do this with a spreadsheet. Pinecone was chosen
partly for the deployment path (it's hosted, so there's no server to keep alive)
and partly for resume value — it's the vector database that appears most often by
name in job postings.

The full text of each article is stored alongside its vector, so there's no second
database to keep in sync.

### Step 5 — Finding the right articles (retrieval)

This is where most of the engineering went. **Two different searches run in
parallel, and their results get merged.**

**Search A — semantic (meaning).** The map from Step 3. Good at understanding that
"I got hurt at work" and "occupational injury" are the same topic. Weak spot: it
can be vague about exact terms and specific numbers.

**Search B — keyword (BM25).** Old-fashioned word matching, like Ctrl+F with a
ranking formula. Good at exact terms. Weak spot: it's completely blind across
languages — an Arabic question and an English article share no words at all, so it
scores zero every time.

The two have opposite strengths, so both are run and the results combined.

#### How the results get merged: RRF

The obvious way to merge would be to add up the two scores. **That doesn't work**,
because the numbers aren't comparable — semantic similarity is roughly 0-1, while
BM25 produces unbounded numbers like 14.2. Adding them is adding centimetres to
kilograms.

**Reciprocal Rank Fusion (RRF)** dodges the problem entirely: *ignore the scores,
use only the positions.*

Think of two judges scoring a competition. One is harsh, one is generous, so their
raw scores can't be compared. But you can compare their *rankings* — first, second,
third. Position is position, regardless of scoring style.

An article gets points for placing well in either list, and points add up:

```python
score = w/(k + rank_semantic) + (1-w)/(k + rank_keyword)
```

Being ranked #1 is worth a lot; being ranked #40 is worth almost nothing. `w`
controls how much the meaning-search counts versus the word-search. It is currently
**0.90** — 90% meaning, 10% keywords.

The top **5** articles go to the writer.

> **An honest note about BM25:** it was measured, and it contributes almost nothing
> here — it helped 2 questions, hurt 0, which isn't statistically meaningful. The
> argument that it would catch exact article-number lookups was *tested and
> refuted*: all 12 such probes were already found at rank 1-2 by the meaning-search
> alone. It's kept because it costs 1 millisecond per query and can only help. But
> the honest statement is "kept as cheap insurance," not "proven to help."

#### A few laws have hidden "only if" articles

Some articles don't answer a question themselves — they change whether a *different*
article even applies. A plain example: the Labour Law promises a lump-sum payment
for a workplace injury (Article 90) — but a separate article (Article 86) says that
whole rule only applies to a worker who is **not** covered by Social Security. Most
workers *are* covered, so for most people the real answer is a completely different
article, in a different law (Social Security's own Article 30), not Article 90 at
all.

The problem: the meaning-search doesn't know Article 86 exists, because a question
about "what do I get paid if I'm injured" is about pay amounts, not about the
"who does this apply to" article — so the two rarely rank near each other.

The fix is a small, hand-checked list of these "only if" articles. Whenever the
articles it depends on are about to be picked, the "only if" article is added too —
never anything more than that one extra article, and never at the cost of bumping
anything else out. This list is short on purpose: it only has entries that were
actually checked and shown to matter, not every article that theoretically could.

### Step 6 — Writing the answer (generation)

The retrieved articles (5, sometimes 6-7 when one of the "only if" articles above
gets added) get formatted into a prompt along with the question and a set of rules,
and sent to a language model (`gpt-4o-mini`). The rules are the part that does the
work — currently 22 of them, in seven groups:

- **Grounding** — use only these documents; if they don't answer it, say so
- **Coverage** — address *every* article that bears on the question, and every
  lettered paragraph of one you cite
- **Precision** — say which specific clause applies, don't blur two together
- **Usefulness** — answer first, then explain; ask for a missing personal fact
  rather than guessing at it
- **Judgment questions** — for "should I…?" lay out benefits and costs neutrally,
  and don't tell the person what to decide
- **Totals and calculations** — for anything involving arithmetic (contributions,
  end-of-service pay, overtime), don't do the maths yourself — see the box below
- **Multiple laws, and who a law covers** — when two laws both talk about the same
  event, don't blend them into one answer; and some laws only cover certain people
  (see the box above), so say that plainly when it matters

One rule matters more than it looks:

> *State what each provision says on its own terms. Do not explain how two
> provisions combine, qualify or override one another unless a document says so.*

That's a guardrail learned the hard way. When the model was told to be thorough, it
started *reasoning* about how two articles interact — and legal reasoning that
isn't in the text is exactly the kind of confident invention this whole system
exists to prevent. The one narrow exception: if a document *itself* states a
condition (like Article 86 above), stating that condition back is reading the text,
not the forbidden kind of reasoning — so a later rule spells that out as the one
case where connecting two articles is allowed.

> **Why the model doesn't do the maths.** Early on, asked "how much does my company
> pay into social security," the model tried to add up several percentages itself
> and got it wrong — sometimes right total, wrong split between employer and
> employee; sometimes a rate quietly dropped. The fix: a small piece of plain Python
> code reads the actual numbers out of the retrieved article text, does the
> addition, and hands the model a ready-made answer block it's told to copy rather
> than recompute. This exists for social security contributions, end-of-service pay,
> overtime pay, paid-leave accrual, and pension age/eligibility — each one reads its
> numbers from the text every time, rather than having them typed in once, so an
> amendment to the law is picked up automatically instead of silently going stale.

### Step 7 — Follow-up questions (multi-turn)

If the model needs a fact it wasn't given, it asks: *"how many years of service do
you have?"* The user replies **"12."**

That reply breaks the search, and understanding why is the key insight:

**The search is stateless and works on meaning.** The word "12" has no meaning
related to pensions or compensation. Put it on the map and it lands nowhere useful.
The search would return random articles, and the answer would be built on the wrong
documents.

So before searching, a small extra step rewrites the fragment into a full question
using the conversation so far:

> "12" **→** *"How much end-of-service compensation am I owed with 12 years of
> service under the Social Security Law?"*

Verified with a control — the same fragment, run both ways:

| | articles retrieved |
|---|---|
| **with** history (rewritten) | the actual compensation and pension articles |
| **without** history (raw) | Articles 1, 2, 4, 7 — definitions and scope boilerplate |

That second row is what a contentless query lands on.

**The rewrite goes only to the search, never to the writer.** The writer gets the
user's real words plus the conversation. The reason: the rewrite is machine-written
and can drift, adding terms the user never said. A slightly-too-specific *search*
is harmless. A slightly-wrong *answer*, restating a question the user never asked,
is not.

The underlying asymmetry: **the writer has memory, the search doesn't.**

---

## Part 3 — How we know it works

Anyone can build the above. The part that makes it a real project is being able to
say how well it works, with evidence.

### The measuring stick

**60 questions** with known correct answers — for each one, which article *should*
be found. Written to a deliberate brief: covering all three laws, both languages,
a range of difficulty.

### Retrieval scores: did we find the right article?

| Metric | Plain meaning | Current |
|---|---|---|
| **recall@1** | right article was the #1 result | **0.883** |
| **recall@5** | right article was somewhere in the top 5 | **1.000** |
| **MRR** | how high up it landed, averaged | **0.926** |

**recall@5 = 1.000 means the correct article is in the top 5 for all 60 questions.**
Not most. All of them.

MRR ("Mean Reciprocal Rank") rewards being higher up: rank 1 scores 1.0, rank 2
scores 0.5, rank 3 scores 0.33. 0.922 means results are almost always at the very
top, not merely present.

### Generation scores: was the answer any good?

Retrieval scores are arithmetic. Answer quality isn't — so a **separate, independent
model** (Claude Sonnet, deliberately from a different company than the one writing
the answers, so it can't favour its own style) reads each answer against the
documents and scores two things 1-5:

- **Faithfulness** — is every claim actually supported by the documents? *(catches
  invention)*
- **Relevancy** — does it actually answer what was asked? *(catches waffle and
  omission)*

| | Faithfulness | Relevancy |
|---|---|---|
| before the coverage rules | 4.70 | 4.55 |
| after the coverage rules | 4.80 | 4.80 |
| latest full 60-question run | 4.70 | 4.72 |

And the honest reading of that table: **the relevancy gain from the coverage rules
was real, the faithfulness gain was not**, and the drop in the latest full run
(measured against a harder, changed retrieval setup) was checked and judged as
"held," not "got worse" — the same 1-point-either-way noise every single-run
comparison in this project has to account for, not a real regression.

Relevancy improved on 14 questions and got worse on 3 when the coverage rules were
added (p = 0.013 — roughly a 1.3% chance of seeing a split that lopsided by luck).
Faithfulness was 12 up, 8 down — that's a coin flip. So it's reported as "held," not
"improved."

That distinction is the whole point of measuring. Numbers moving isn't the same as
numbers meaning something.

**Not yet in this table:** two new prompt rules and a code fix added after the
numbers above were measured (see `CLAUDE.md`'s latest phase) — checked on a small,
targeted set of questions, not the full 60, so their effect on these numbers isn't
known yet.

### Also tracked

- **Correct article cited:** 58 of 60
- **Invented citations:** 0 of 60

Zero is the number that matters most, since inventing a legal citation is the worst
thing this system could do.

---

## Part 4 — The decisions, and the ones that got overturned

Every significant choice was tested against alternatives. Several were later
re-tested and reversed. Here's the short version.

### Kept after testing

| Decision | Beat | Why |
|---|---|---|
| `gpt-4o-mini` writes the answers | Cohere Command, Qwen-72B | Best faithfulness, zero invented citations, ~17× cheaper |
| RRF merges the two searches | weighted-sum scoring | Works with partial result lists, which is what the real APIs return |
| Pinecone | Qdrant Cloud | Hosted, and better resume visibility |
| Hand-built HTML/CSS/JS frontend | Streamlit, Gradio | Shows more skill than wrapping a demo tool |

### Tested and rejected

**A reranker** (a second model that re-orders results). It fixed the cross-language
weakness but *hurt* same-language precision — and same-language is 80% of real use.
Traced to a concrete cause: several leave provisions open with near-identical
boilerplate, and a general-purpose reranker matches on that surface phrasing instead
of the clause that actually distinguishes them.

**Temperature = 0** (making the model deterministic). Sounds obviously right for a
legal tool. It measurably *increased* invented citations. Not adopted.

**Re-tuning the search-merge weight.** In-sample, `w = 0.83` looked better than the
production `0.85`. Testing it on held-out data showed the "improvement" was
overfitting — it beat production in **0%** of 500 held-out splits. Nothing changed.

### Reversed on re-review

The whole project was re-reviewed phase by phase, and two early decisions did not
survive.

**The embedding model.** Cohere was chosen in Phase 4 and looked fine for nine
phases. Re-testing on a corpus and eval set that had both changed underneath it,
Google's `gemini-embedding-001` won on 11 of 13 contested questions (p = 0.022) —
worth **+0.093 MRR**, about seven times more than every search-tuning knob combined.
The system was switched.

**The chunk size.** Covered in Step 2 — clause-splitting lost to whole articles on
every measure.

The pattern in both: *the original decision was defensible when it was made, and
stopped being right when its inputs changed.* Nothing was wrong with the reasoning.
The corpus, the eval set, and the rest of the stack had all moved.

---

## Part 5 — What actually runs in production

```
Browser (Arabic RTL chat UI)
        │
        ▼
FastAPI  ── serves the frontend and the API from one service
        │
        ├─ POST /ask        the main endpoint
        ├─ POST /followups  suggests 3 grounded next questions
        └─ GET  /health
        │
        ▼
Retriever ── Gemini embeddings → Pinecone (meaning)
          └─ BM25 in memory              (keywords)
          └─ RRF merge, w=0.90, top 5
        │
        ▼
Generator ── gpt-4o-mini via OpenRouter, 13 rules
        │
        ▼
answer + citations + the source articles
```

Deployed on **Railway**. Rate-limited to 10 questions/minute and 100/day per
visitor, because every question costs real money in API calls.

**The server keeps no memory of anyone.** Conversation history lives in the
browser and is re-sent with each question. That means no database, no sessions, and
no user data stored anywhere.

### Repo layout

| | |
|---|---|
| `src/retrieval.py` | search: embeddings + BM25 + RRF |
| `src/generation.py` | the prompt, the answer, the query rewriter |
| `src/api.py` | web service |
| `frontend/` | the chat UI |
| `scripts/01…23` | every decision notebook, in the order the work happened |
| `evallib/` | shared evaluation helpers, with an equivalence proof |
| `eval/` | the question sets and every run's results |
| `ROADMAP.md` | the formal build log |
| `LESSONS.md` | what went wrong and what it taught |

---

## Part 6 — Known limits

Stated plainly, because a project that claims no weaknesses hasn't been looked at
hard enough.

- **It is not legal advice**, and the interface says so on every screen.
- **The judge scores faithfulness to the retrieved documents, not to the law
  itself.** An answer that is perfectly faithful to the wrong article scores well.
- **Multi-turn has never been scored by the harness.** It was verified with a
  control, not measured — the 60-question set is single-turn by construction, so
  measuring it needs new evaluation data.
- **The Professional Work Law English text is the unamended 2019 original**, while
  the Arabic is amended as of 2024. They aren't fully parallel documents.
- **Gemini silently truncates** inputs over 2,048 tokens — no error, no warning. It
  affects 4 of 531 articles. Only one is ever a correct answer, and it still ranks
  #1, but nothing in the system would tell you if that changed.
- **Spending is capped per visitor, not globally.** Enough visitors at once could
  run up a real bill.

---

*Written after Phase 23. Numbers are from the Phase 20/21 evaluation runs.*
