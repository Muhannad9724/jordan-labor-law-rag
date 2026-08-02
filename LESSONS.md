# Mistakes, and what they taught

Every item here actually happened in this project. Most were caught and fixed;
a few are permanent damage that can't be undone.

They're kept written down for one reason: **almost none of them announced
themselves.** Nothing crashed. No error appeared. The numbers looked fine — they
were just wrong, and stayed wrong until something forced a second look.

Companion to `EXPLAINER.md` (how the system works) and `ROADMAP.md` (the formal
build log).

---

## Part 1 — Mistakes about measuring

This is the biggest category by far, and the one worth reading if you only read
one section.

### 1.1 The eval set that graded itself

**The single most expensive mistake in the project.**

**What happened.** The 60 test questions were written by reading each article and
then writing a question about it. Natural, careful, and it quietly poisoned every
number for nine phases.

Because the questions were written *while looking at the answer*, they reused the
law's own vocabulary. The article said "unemployment allowance," so the question
said "unemployment allowance." Search then had a free ride — matching a term to
itself, not understanding a question.

**How it surfaced.** Not through suspicion of the questions. It showed up as a
*symptom*: **5 of 7 completely different embedding models scored a perfect
recall@5 = 1.000.** Seven models with different architectures, different training,
different sizes — and the total spread between them was four questions.

That's not seven excellent models. That's a test that can't tell anything apart.

**The fix.** Every question was rewritten to describe a *situation* in ordinary
words, while keeping the correct answers byte-for-byte identical:

> before: *"What are the conditions for entitlement to the unemployment allowance?"*
> after: *"I just lost my job. Do I qualify for unemployment payments?"*

Because only the wording changed, the old and new scores are directly comparable —
so the gap between them *measures the inflation*:

| | change |
|---|---|
| mean recall@1 | **−0.233** |
| mean MRR | **−0.155** |
| word overlap with the correct article | 0.708 → **0.285** |

**Roughly a quarter of every score up to that point was the test grading itself.**

And the new set works: questions where models disagree went 4 → 10, the spread
between models grew 2.5×, and **zero questions are now missed by every model** —
harder, but not broken.

> **The rule:** if you write your test questions by reading the answers, you are
> measuring vocabulary overlap, not retrieval. Write the question from the
> *situation* a real user is in, and never with the answer on screen.
>
> **The warning sign:** when very different systems all score the same, suspect
> your ruler before you celebrate your systems.

### 1.2 A stable optimum is not a good optimum

**What happened.** The search-merge weight `w` was at 0.85. Sweeping it showed a
peak at 0.83 — slightly better. Was that peak real or noise?

The check was a **bootstrap**: resample the 60 questions 2,000 times, re-find the
peak each time. The answer came back reassuring — the optimum sat around 0.82 in a
tight range, and the production value 0.85 was chosen in **0 of 2,000** resamples.

That looks like overwhelming evidence to switch. It isn't.

**Why it's a trap.** A bootstrap resamples *the same 60 questions*. It can tell you
the peak is reliably **located** on this data. It can say nothing about whether
moving there helps on *different* data — which is the only question that matters.

**What actually settled it.** Cross-validation: fit `w` on half the questions,
score it on the other half, 500 times. The tuned value was **worse than the
untuned one for both embedding models** — it beat production in 0% of splits for
one of them.

The peak was real and reliably located. It was also **overfitting**. The curve
near the top is flat, and the "peak" was chasing the particular quirks of these 60
questions.

**Nothing was changed.**

> **The rule:** a bootstrap confirms *where* the optimum is. Only held-out data
> confirms that *moving there helps.* Those are different questions, and only the
> second one matters.

### 1.3 Choosing a metric that was already maxed out

**What happened.** Before running an embedding comparison, a decision rule was
written down in advance — good practice, deliberately done so the conclusion
couldn't be rationalised afterwards:

> *Switch only if a challenger beats the incumbent on **both** recall@5 and MRR by
> ≥ 0.05.*

The rule then failed to fire. But not because the challenger lost — because
**recall@5 was already 1.000 for five of seven models.** You cannot beat 1.000 by
0.05. Half the decision rule was dead on arrival.

**Why it's a trap.** Pre-committing to a rule is right. But a saturated metric in
that rule doesn't act as a high bar — it acts as an *automatic veto for the
incumbent*, regardless of evidence. The rule wasn't strict. It was broken.

**What was done.** The rule was **not** quietly amended mid-flight, because
rewriting your own decision rule after seeing results is precisely how
pre-committing loses its value. It was flagged as a flaw in the rule, the decision
was left open, and the switch happened later, deliberately, with the reasoning
stated.

> **The rule:** before committing to a metric, check it still has room to move. A
> metric already at its ceiling can't measure anything.

### 1.4 Reading a 1-question gap as a result

**What happened.** For several phases, config A scored 4.90 and config B scored
4.78, and B was declared worse. Perfectly reasonable-looking.

Nobody had ever checked **how much these scores move for no reason at all.** The
answer-writing model runs at a non-zero temperature — the same question asked twice
gives different wording, which an LLM judge can score differently.

**The fix.** Run the *identical* configuration twice and see how far apart the two
runs land. That's the **noise floor** — the amount of movement that means nothing.

Once measured, it changed the reading of a real result: a relevancy gain of +0.14
was about **2× the noise floor** (real), while a faithfulness gain of +0.02 sat
well inside it (nothing). Both had gone up. Only one meant anything.

> **The rule:** before comparing two configurations, run one of them twice. Any
> gap smaller than the gap between two identical runs is not a finding.

### 1.5 Comparing two things that differed in three ways

**What happened.** A prompt change was tested against an old baseline. The new arm
scored better. But the baseline had also been run with a different `top_k`, on a
different chunk granularity.

Three variables moved. One number came out. Nothing can be concluded about the
prompt.

**How it's done now.** A later prompt test **replayed the previous run's saved
retrieval results** instead of re-querying. Both arms saw byte-identical documents,
so the *only* difference was the prompt. As a bonus, the retrieval half cost
nothing to re-run.

> **The rule:** change one thing. If you can't, freeze everything else by feeding
> both arms literally the same saved inputs.

### 1.6 A single trial is not a measurement

**What happened.** A behavioural fix was made, tested once, and reported as
"Fixed."

It wasn't. Repeated testing showed **0 out of 6** locally and **0 out of 6** in
production. The one apparent success was the model's natural variation.

The real fix, once found, went from 0/6 to 4/4 — but that only became visible
because the claim was re-tested rather than trusted.

> **The rule:** language models are sampling processes. One trial tells you
> something is *possible*, never that it's *fixed*. Run it six times.

### 1.7 Reading results before the run finished

**What happened.** Halfway through an evaluation, one question looked like the
flagship success — the new prompt had picked up exactly the article the old one was
criticised for missing.

The finished run told a different story. It had picked up the right article and
**misapplied it** — using a clause about combining two pensions to answer a question
about a pension plus wages. Faithfulness went *down*, 4 → 3.

> **The rule:** more coverage is not automatically better coverage, and a partial
> result read mid-run is not a result.

### 1.8 Reading a sorted table as a ranking

**What happened.** A results table, sorted by score, showed model A above model B.
Easy to read as "A is better."

The paired test said **9–8, p = 1.000.** A statistical dead heat. The sort order was
meaningless for that pair — the table was sorted, so *something* had to be on top.

Meanwhile a different pair in the same table was 12–3, p = 0.035 — a genuine
difference. Both pairs looked equally like "ordered rows."

> **The rule:** a sorted table shows order, not significance. For close pairs, run
> the paired test before believing the rows.

---

## Part 2 — The scoring code is code, and it has bugs

An under-appreciated failure mode: **the thing measuring your system can be broken
while your system is fine.** You then "fix" a model that was never wrong.

### 2.1 The Arabic regex that missed a third of correct citations

**What happened.** A pattern checked whether answers cited the right article by
looking for `المادة` ("the article"). It scored a lot of correct Arabic answers as
misses.

**The cause is a grammar rule.** In Arabic, when the preposition `ل` attaches to the
definite article `ال`, the alef drops out:

```
ل + المادة  →  للمادة        ("according to the article")
```

`المادة` **does not appear inside** `للمادة`. So the ordinary way to write "according
to Article 90" scored as a failure to cite anything.

**Damage: 6 of 29 Arabic citations were silently marked wrong.** One evaluation's
true score was 87.5%, not the 82.5% reported.

**The fix** anchors on the invariant stem `مادة` with optional prefixes, and is
tested against real citation forms *and* negative cases — including `مادة كيميائية`
("chemical substance"), where the same word means something else entirely.

### 2.2 The digits bug that was invisible until it wasn't

**What happened.** The citation checker compared article numbers as text. Arabic
has its own digits:

```
٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩
```

`المادة (٦٥)` and "Article 65" refer to the same article. To the checker, `٦٥` and
`65` were different strings.

**The interesting part:** this bug existed for a long time and caused zero damage,
because the model had been answering Arabic questions with English citations. The
moment a *different* fix made Arabic citations work properly, this latent bug
activated and started scoring correct answers as wrong.

> **The lesson:** fixing one thing can wake up a bug that was dormant because the
> broken behaviour was hiding it.

### 2.3 Counting quoted numbers as invented ones

**What happened.** The "invented citation" check flagged answers mentioning an
article number that wasn't in the retrieved documents.

But **51 of 531 articles cross-reference other articles in their own text.**
Article 25's text literally says *"…subject to articles (32) and (33)."* An answer
quoting that sentence got flagged for inventing citations it was faithfully copying.

**Every single flagged case turned out to be a false positive. Real inventions:
zero.**

Worse: as answers got *more* faithful and quoted source text more often, the
"invention" count went **up**. The metric moved backwards from what it claimed to
measure.

### 2.4 My own detector, broken the same way

While testing whether the model asked clarifying questions, the detector looked for
`نحتاج` ("we need"). The model actually writes `أحتاج` ("I need").

A run where **5 of 6 answers plainly asked a question** scored 0 of 6. That nearly
produced a second wrong conclusion on top of the first.

### 2.5 Saving scores instead of saving answers

**Permanent, unfixable damage.**

Early evaluation runs saved *extracted citations* rather than the *answer text*.
When the citation regex (2.1) turned out to be broken, only one run could be
re-scored — the one that happened to save raw answers.

Every other run's numbers are stuck at their broken values forever. Re-measuring
them would mean paying for the generation again.

> **The rule:** persist the expensive artefact, not the cheap conclusion. Answer
> text, retrieved documents, and raw vectors are expensive. Scores are cheap and
> can always be recomputed — *if* you kept what produced them.
>
> This paid off later: caching the embedding vectors made an entire model
> re-evaluation cost **nothing** instead of re-embedding the corpus seven times.

---

## Part 3 — Shipping mistakes

### 3.1 The default that was shadowed

**What happened.** A production setting was changed from 10 to 5 in
`src/retrieval.py`.

It would have had **zero effect on the live app.** The web layer declared its own
default in the request model, and that value was passed explicitly on every
request — overriding the function default entirely.

Caught before deploying, but only by checking the whole call path rather than the
one file the setting "lived" in.

> **The rule:** a default is only a default if nobody passes the argument. Trace
> the value from the HTTP request to the function, not from the function outward.

### 3.2 Dependencies that build fine and crash at startup

**What happened.** After switching embedding providers, `requirements.txt` was
never updated. The two new packages were installed locally, so everything worked on
the development machine.

A deploy would have **built successfully** and then crashed on the first request —
the worst kind of failure, because the build going green looks like proof it works.

> **The rule:** after changing an import, check the requirements file. "It works on
> my machine" is exactly the condition that hides this.

### 3.3 The token limit that reserved money it never spent

**What happened.** An evaluation run died at question 31 of 60 with a payment
error, on an account with a healthy balance.

**The cause:** the API provider reserves credit against the request's `max_tokens`
**ceiling**, not actual usage. Leaving it unset defaults to the model's maximum —
65,536 tokens — so every request tried to reserve the cost of a maximum-length
response, despite each answer using a tiny fraction of that.

**The second failure was worse than the first.** The eval set is ordered by law, so
every question scored before the crash was Social Security Law. **The entire Labour
Law section went unmeasured**, and the partial average looked like a real result
while being a systematically biased sample.

Both extremes are now on record: unset → payment error; set too low (300) → empty
responses. It sits at 2,000.

> **The rules:** always set `max_tokens` explicitly. And if a run can die partway,
> **shuffle or stratify the order** — otherwise a partial result is a biased
> result, not a smaller one. Persist incrementally so a resume costs only the
> missing calls.

### 3.4 The browser serving old code

**What happened.** A frontend fix was deployed. Nothing changed. The fix was
rewritten, redeployed, still nothing.

The code was correct the entire time. The static file server sent an ETag but no
`Cache-Control` header, so browsers fell back to *heuristic caching* — a rule that
lets them guess a cache duration from how old the file looks. The browser kept
serving the old JavaScript.

Hours were spent debugging correct code.

> **The rule:** when a deploy appears to do nothing, verify the browser is running
> the new code *before* touching the code. Set `Cache-Control` explicitly.

### 3.5 The flag that disabled itself forever

**What happened.** A UI feature was supposed to nudge the view when new content
appeared — unless the user was scrolling themselves.

Implemented as `userScrolled = true` on any scroll. Once set, it was never unset.
**The first ordinary scroll permanently disabled the feature** for the rest of the
session.

Fixed by replacing the one-way boolean with a timestamp: *did the user scroll in
the last 600ms?* — a question that can become false again.

> **The rule:** a boolean that only ever flips one way is a latch, not a state.
> "Is the user scrolling *now*" needs recency, not history.

### 3.6 An empty `catch` block

A rendering failure was swallowed by an empty exception handler, so a broken
feature looked like a feature that simply never triggered. Adding one
`console.warn` turned an invisible mystery into a one-line diagnosis.

---

## Part 4 — Working with Arabic

Bilingual RAG has failure modes that never appear in an English-only system.

### 4.1 The English labels that switched the answer's language

**What happened.** Answers to Arabic questions had correct Arabic bodies but
**English citations** — "Labour Law 1996, Article 61" in the middle of Arabic
text. 8 of 30 Arabic answers did this.

**The cause was not the prompt.** It was the document formatting. The retrieved
articles were labelled with English scaffolding:

```
Law: Labour Law 1996
Article: 61
```

The model copied the language it saw its sources labelled in. Fixing the law names
alone wasn't enough — **the words "Law:" and "Article:" themselves had to be
localised.**

**Result: 8 of 30 → 0 of 30.**

> **The rule:** in a multilingual system, the scaffolding is content. Every label
> the model sees teaches it what language to answer in.

### 4.2 Silent truncation

The embedding provider accepts inputs longer than its 2,048-token limit **without
any error**. It quietly ignores the excess. A 6× oversized input is accepted just
as happily as a correct one.

4 of 531 articles are affected. Only one is ever a correct answer, and it still
ranks #1 — so the impact is nil today. But **nothing in the system would report it
if that changed**, so it's documented in the module itself.

> **The rule:** check input limits against your longest document explicitly.
> Assume no warning will be given.

---

## Part 5 — Prompting mistakes

### 5.1 Blaming the model for following instructions

**What happened.** Answers were leaving out relevant articles. Nine of ten
"forgotten" articles were sitting right there in the context, retrieved correctly.

The obvious diagnosis: the model isn't good enough. Upgrade it.

**That would have been wrong.** The prompt contained *three separate rules* pushing
toward short, single-source answers — including a literal "Keep it brief" — and
**not one rule** telling it to check every document.

The omissions weren't a failure. They were **compliance.** A more capable model
told to be brief would also have been brief, and swapping models first would have
created exactly the confound described in 1.5 — a real gain, attributed to entirely
the wrong cause.

Fixing the instructions instead: relevancy 4.55 → 4.80, articles cited per answer
1.22 → 1.55, **19 answers cited more sources and 0 cited fewer.**

> **The rule:** before blaming the model, read your own prompt as if you were
> following it literally. Very often the "bug" is written down in your own words.

### 5.2 "Be thorough" turning into "make things up"

**What happened.** A rule was added telling the model to cover every consequence
and exception. It improved coverage — and introduced a *new* failure: the model
began **reasoning about how provisions interact**, producing legal inferences no
document supported.

**The fix wasn't removing the rule.** It was pairing it with an explicit boundary,
placed immediately after it:

> *State what each provision says on its own terms. Do not explain how two
> provisions combine, qualify or override one another unless a document says so.*

Restate what's written. Don't reason about how the pieces relate.

It's attenuated, not eliminated — 5 of 8 remaining faithfulness regressions are
still the model relating two provisions. Stated plainly rather than glossed over.

> **The rule:** "be more complete" and "don't invent" pull against each other. A
> completeness instruction needs its boundary written next to it, not in a
> different section.

### 5.3 The instruction that lost to an example

**The subtlest one in the whole project.**

**What happened.** A rule said: if a personal fact is missing, ask for it. It
worked in single-turn conversations.

The moment conversation history was added, **it stopped firing completely.**
Reworded three times. Measured **0 of 6**, twice, locally and in production. The
rule was clear, correct, and utterly ignored.

**The cause.** With history included, the model could see *its own previous
answer* — an answer that had confidently answered a similar question without
asking anything. That prior answer is an **in-context demonstration**, and a
demonstration of what to do beats an instruction sitting thousands of tokens
earlier in a system prompt.

The model wasn't disobeying. It was pattern-matching on the strongest, most recent
example available — which happened to be its own past behaviour.

**The fix was position, not wording.** The same instruction was re-injected as a
short message placed *after* the history and *immediately before* the new question,
so the last thing read is "judge this question on its own terms" rather than a
worked example of not doing that.

**0 of 6 → 4 of 4.** The words barely changed. Only their location did.

> **The rule:** in a conversation, what the model has *seen done* outweighs what it
> has been *told to do*. If a system-prompt rule stops working once history is
> present, move it — don't rewrite it.

### 5.4 Instructing a language model to be a calculator

**What happened.** Asked "my salary is 1000 JD, how much do I pay and how much does
the company pay", the app got it right **3 times in 6**. Every article it needed was
in the context. Five prompt rules told it exactly how to do the sum, including
"show each component on its own line, then the total".

The instinct was to write a sixth rule. That was wrong for a reason worth naming:
answering meant finding six rates across four articles → deciding which side bears
each one → summing each side → multiplying by the wage. **Four chained steps, no
calculator, at non-zero temperature.** No wording fixes that. 3/6 is simply what
instructing a language model to be a calculator buys.

Moving the arithmetic into code took it to **6/6**, on every case tested.

> **The rule:** prompts change what a model *tries* to do. They don't make an
> unreliable capability reliable. When the failure is arithmetic, lookup or anything
> else code does perfectly, the fix is to do it in code and let the model narrate the
> result — not to describe the procedure more carefully.

### 5.5 The model was right; my figure was wrong

**What happened.** After moving the arithmetic into code, one case still failed:
850 JD gave an employer share of **121.12** where the answer should be 121.13. It
failed in **3 trials out of 3** — suspiciously consistent for a sampling model.

That consistency was the clue. The model wasn't computing anything; it was copying
what I handed it, exactly as instructed. `850 × 14.25% = 121.125`, and Python's
`round()` rounds half to **even**, not up. Every payslip convention rounds half up.

The calculator was wrong and the model was faithfully reproducing it. Had this been
scored as a model failure, the fix would have gone into the prompt and never worked.

> **The rule:** a wrong answer that repeats *identically* across trials is not a
> sampling failure — it is deterministic, so it comes from your code, not the model.
> Use the variance as the diagnostic: noisy means the model, stable means you.

---

## Part 6 — Deciding what to work on

### 6.1 Choosing without comparing

The first answer-writing model was picked because it was convenient — same vendor
as an existing key, nice API — and **never compared to anything**, in a project
where every other decision was measured.

Testing it later against two alternatives: it lost on faithfulness, produced an
invented citation, and cost **17× more**. The replacement won every measure.

> **The rule:** "it works" is not a decision. It's the absence of one.

### 6.2 Polishing the small knob while the big one is undecided

Considerable effort went into tuning how the two searches merge. Careful work,
properly cross-validated.

Then the sizes were compared:

| | worth |
|---|---|
| every search-merge knob, combined | ≤ **0.013** MRR |
| the still-undecided embedding model | **0.093** MRR |

**A ratio of about 7:1.** The better embedding model with *untuned* merging beat
the worse one with the best merge settings found anywhere.

All fusion tuning was parked until the model was settled — and once it was, the
optimum moved anyway, so most of that work would have needed redoing.

> **The rule:** size the prize before optimising. Order your work by how much each
> decision is worth, and settle the upstream one first — it moves the optimum of
> everything downstream.

### 6.3 Not asking whether the old numbers still apply

By phase 20, every published generation number had been measured on a different
embedding model, a different chunk size, a different `top_k`, a different fusion
weight, and a different eval set.

**Not one of those conditions still existed.** The numbers were quoted for months
after everything underneath them had changed.

> **The rule:** a measurement is only valid for the system it measured. When a
> component changes, mark every number downstream of it as stale — immediately,
> not when someone notices.

---

## The short version

If this document were one page:

1. **Never write test questions with the answer on screen.**
2. **When everything scores the same, your ruler is broken, not your systems.**
3. **Run one config twice before comparing two configs.**
4. **A stable optimum in-sample is not a better optimum out-of-sample.**
5. **One trial is not a measurement.**
6. **Your scoring code has bugs too — test it against real output and negative cases.**
7. **Save the expensive artefact (answers, vectors), not just the score.**
8. **Set `max_tokens` explicitly, and never let a run die in sorted order.**
9. **Read your own prompt literally before blaming the model.**
10. **In a conversation, an example beats an instruction — fix it by position.**
11. **Size the prize before optimising; settle the upstream decision first.**
12. **When a component changes, everything measured downstream is stale.**
13. **A prompt cannot make an unreliable capability reliable — do it in code and let
    the model narrate.**
14. **An identical wrong answer across trials is deterministic, so it is yours, not
    the model's.**

---

*Compiled after Phase 24, from the failures recorded across all 24 phases in
`ROADMAP.md`.*
