// Same-origin -- FastAPI serves this frontend directly (src/api.py), so no CORS.
const API_BASE_URL = "";

const LAW_NAMES_AR = {
  ssl: "قانون الضمان الاجتماعي",
  labor96: "قانون العمل لسنة ١٩٩٦",
  prof19: "قانون العمل المهني لسنة ٢٠١٩",
};

const SEED_QUESTIONS = [
  "كم يوم إجازة سنوية مدفوعة الأجر أستحق؟",
  "ما مدة الإشعار المطلوبة قبل إنهاء عقد العمل؟",
  "هل يحق لصاحب العمل فصلي من دون إنذار؟",
  "متى يحق لي الحصول على راتب التقاعد المبكر؟",
];

const thread = document.getElementById("thread");
const composer = document.getElementById("composer");
const input = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const railBody = document.getElementById("rail-body");
const railCount = document.getElementById("rail-count");

const AR_DIGITS = "٠١٢٣٤٥٦٧٨٩";
const toArabicDigits = (v) => String(v).replace(/\d/g, (d) => AR_DIGITS[+d]);

// Arabic script blocks (incl. presentation forms). A message is rendered RTL when a
// meaningful share of its letters are Arabic -- lets an Arabic and an English answer
// sit in the same thread, each set correctly.
const ARABIC_RE = /[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]/g;
function isArabic(text) {
  const letters = text.replace(/[^\p{L}]/gu, "");
  if (!letters) return false;
  return (letters.match(ARABIC_RE) || []).length / letters.length > 0.3;
}

// The page scrolls, not an inner pane. Bring a new question up near the top so the
// answer that follows it renders into open space instead of below the fold.
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function scrollIntoReading(el, offset = 88) {
  const top = el.getBoundingClientRect().top + window.scrollY - offset;
  window.scrollTo({ top, behavior: reducedMotion ? "auto" : "smooth" });
}

// Height of the sticky composer plus its breathing room -- the part of the viewport
// that is not actually available for reading. Keep in step with .composer and the
// .thread padding-bottom in style.css.
const COMPOSER_CLEARANCE = 92;
const READING_GUTTER = 20;

// Scrolling the *question* to the top was only ever half the job: the answer then
// rendered below it and, at the length answers now run, started below the fold. Put
// the answer itself at the top of the viewport instead, which is the most reading
// space the layout can give it.
//
// Two deliberate refusals:
//   - Do nothing when the answer already fits on screen. Scrolling a fully visible
//     answer is a jump with no payoff, and it throws away the question above it.
//   - Never scroll past the answer's own top. A long answer is read from its first
//     line, not its last, so this is not the usual chat "pin to bottom" behaviour.
function revealAnswer(el) {
  const rect = el.getBoundingClientRect();
  const room = window.innerHeight - COMPOSER_CLEARANCE;
  if (rect.top >= 0 && rect.bottom <= room) return;   // already fully readable
  scrollIntoReading(el, READING_GUTTER);
}

// The masthead is a landing device. Collapse it once the thread has real content so
// the space goes to the answer; see "Reading mode" in style.css.
function enterReadingMode() {
  document.body.classList.add("reading");
}

// Suggestions land about a second after the answer, by which time the reader may be
// mid-scroll. Nudging the page under someone's thumb is worse than a chip they have to
// scroll to find, so track *user-initiated* scrolling -- but as a timestamp, not a
// one-way flag. A flag stays true for the rest of the answer's life the instant it is
// set, so a single ordinary scroll while reading -- which a longer answer, a taller
// font, or a higher browser zoom all make more likely -- permanently disables the
// nudge, even though the actual scroll gesture finished a second ago. Only refuse to
// nudge if a scroll happened *recently*, i.e. the reader is scrolling right now.
//
// Deliberately not the "scroll" event: that also fires for our own smooth scrolling,
// which would immediately read as "the user just scrolled" and disable the nudge.
let lastUserScrollAt = 0;
const SCROLL_RECENCY_MS = 600;
for (const evt of ["wheel", "touchmove"]) {
  window.addEventListener(evt, () => { lastUserScrollAt = Date.now(); }, { passive: true });
}
window.addEventListener("keydown", (e) => {
  if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(e.key)
      && e.target !== input) {
    lastUserScrollAt = Date.now();
  }
});

/* ------------------------- sources rail ------------------------- */

function renderRailEmpty(message) {
  railCount.textContent = "";
  railBody.innerHTML = "";
  const p = document.createElement("p");
  p.className = "rail-empty";
  p.textContent = message;
  railBody.appendChild(p);
}

function renderSources(sources) {
  railBody.innerHTML = "";
  railCount.textContent = toArabicDigits(sources.length);

  for (const s of sources) {
    const card = document.createElement("article");
    card.className = "article-card";

    const top = document.createElement("div");
    top.className = "card-top";

    const numeral = document.createElement("div");
    numeral.className = "card-numeral";
    numeral.textContent = toArabicDigits(s.article_number);
    top.appendChild(numeral);

    const meta = document.createElement("div");
    meta.className = "card-meta";

    const law = document.createElement("div");
    law.className = "card-law";
    law.textContent = LAW_NAMES_AR[s.law] || s.law;
    meta.appendChild(law);

    const label = document.createElement("div");
    label.className = "card-article-label";
    label.textContent = `المادة ${toArabicDigits(s.article_number)}${s.clause ? " — الفقرة " + s.clause : ""}`;
    meta.appendChild(label);

    top.appendChild(meta);

    const chip = document.createElement("span");
    chip.className = `lang-chip ${s.language}`;
    chip.textContent = s.language === "ar" ? "عربي" : "EN";
    top.appendChild(chip);

    card.appendChild(top);

    const body = document.createElement("div");
    body.className = "card-text";
    if (!isArabic(s.text)) body.dir = "ltr";
    body.textContent = s.text;
    card.appendChild(body);

    const more = document.createElement("button");
    more.type = "button";
    more.className = "card-more";
    more.textContent = "عرض النص كاملاً";
    more.addEventListener("click", () => {
      const open = body.classList.toggle("expanded");
      more.textContent = open ? "طيّ النص" : "عرض النص كاملاً";
    });
    card.appendChild(more);

    railBody.appendChild(card);
  }
  railBody.scrollTop = 0;
}

/* ------------------------- thread entries ------------------------- */

function clearOpening() {
  const opening = thread.querySelector(".opening");
  if (opening) opening.remove();
}

function addQuestion(text) {
  clearOpening();
  const block = document.createElement("div");
  block.className = "question";
  if (!isArabic(text)) block.dir = "ltr";

  const label = document.createElement("div");
  label.className = "entry-label";
  label.textContent = "سؤالك";
  block.appendChild(label);

  const p = document.createElement("p");
  p.textContent = text;
  block.appendChild(p);

  thread.appendChild(block);
  enterReadingMode();
  lastUserScrollAt = 0;      // a new question restarts the reading position
  scrollIntoReading(block);
}

function addThinking() {
  const el = document.createElement("div");
  el.className = "thinking";
  el.textContent = "جارٍ البحث في نصوص القانون";
  thread.appendChild(el);
  return el;
}

function addAnswer(text, sources) {
  const block = document.createElement("div");
  block.className = "answer";
  if (!isArabic(text)) block.dir = "ltr";

  const label = document.createElement("div");
  label.className = "entry-label";
  label.textContent = "الإجابة";
  block.appendChild(label);

  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  block.appendChild(body);

  if (sources && sources.length) {
    const actions = document.createElement("div");
    actions.className = "answer-actions";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "cite-btn";
    btn.textContent = `عرض المواد المستند إليها (${toArabicDigits(sources.length)})`;
    btn.addEventListener("click", () => renderSources(sources));
    actions.appendChild(btn);

    block.appendChild(actions);
  }

  thread.appendChild(block);
  revealAnswer(block);
  return block;
}

// Suggestions arrive after the answer is already on screen, so they must never move
// text the user has started reading. The strip is appended below the answer and the
// page is not scrolled -- if the chips are below the fold, they are found by scrolling
// down, which is where someone looks when they have finished reading anyway.
function addSuggestions(answerBlock, questions) {
  if (!questions || !questions.length) return;

  const strip = document.createElement("div");
  strip.className = "suggestions";

  const label = document.createElement("div");
  label.className = "suggestions-label";
  label.textContent = "أسئلة ذات صلة";
  strip.appendChild(label);

  const row = document.createElement("div");
  row.className = "suggestion-row";

  for (const q of questions) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "suggestion";
    chip.textContent = q;
    if (!isArabic(q)) chip.dir = "ltr";
    chip.addEventListener("click", () => {
      if (sendBtn.disabled) return;      // a question is already in flight
      strip.remove();                    // a chip that has been taken is no longer a suggestion
      submitQuestion(q);
    });
    row.appendChild(chip);
  }

  strip.appendChild(row);
  answerBlock.appendChild(strip);

  // Chips that render below the fold are, from the reader's side, chips that did not
  // render. Nudge by the minimum needed to bring them into view -- scrollBy, not
  // scrollIntoView, because the latter would jump the strip to the top of the screen
  // and throw away the answer above it. Skipped only if the reader is scrolling right
  // now, not if they scrolled at any point while reading -- see the note above
  // SCROLL_RECENCY_MS for why that distinction matters.
  if (Date.now() - lastUserScrollAt < SCROLL_RECENCY_MS) return;
  const overshoot = strip.getBoundingClientRect().bottom
                    - (window.innerHeight - COMPOSER_CLEARANCE);
  if (overshoot > 0) {
    window.scrollBy({ top: overshoot + 12, behavior: reducedMotion ? "auto" : "smooth" });
  }
}

function addFailure(message) {
  const el = document.createElement("div");
  el.className = "failure";
  el.textContent = message;
  thread.appendChild(el);
  revealAnswer(el);   // an error is as useless below the fold as an answer is
}

function renderOpening() {
  thread.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "opening";

  const h2 = document.createElement("h2");
  h2.textContent = "اسأل عن حقوقك في العمل أو الضمان الاجتماعي، وستأتيك الإجابة من نصّ القانون نفسه.";
  wrap.appendChild(h2);

  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = "أو ابدأ بأحد هذه الأسئلة";
  wrap.appendChild(hint);

  const seeds = document.createElement("div");
  seeds.className = "seeds";
  for (const q of SEED_QUESTIONS) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "seed";
    b.textContent = q;
    b.addEventListener("click", () => {
      input.value = q;
      composer.requestSubmit();
    });
    seeds.appendChild(b);
  }
  wrap.appendChild(seeds);

  thread.appendChild(wrap);
}

/* ------------------------- request ------------------------- */

async function askQuestion(question) {
  const pending = addThinking();
  try {
    const resp = await fetch(`${API_BASE_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    pending.remove();

    if (!resp.ok) {
      const payload = await resp.json().catch(() => ({}));
      addFailure(typeof payload.detail === "string" ? payload.detail : "تعذّر الحصول على إجابة لهذا السؤال.");
      return;
    }

    const data = await resp.json();
    const block = addAnswer(data.answer, data.sources);
    if (data.sources && data.sources.length) renderSources(data.sources);

    // Not awaited: the answer is complete without it, and a slow or failed
    // suggestion call must not hold the composer disabled.
    fetchSuggestions(question, data, block);
  } catch (err) {
    pending.remove();
    addFailure("تعذّر الوصول إلى الخادم — تأكد من أن الخدمة قيد التشغيل.");
  }
}

async function fetchSuggestions(question, data, block) {
  if (!data.sources || !data.sources.length) return;
  try {
    const resp = await fetch(`${API_BASE_URL}/followups`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        answer: data.answer,
        chunk_ids: data.sources.map((s) => s.chunk_id),
      }),
    });
    if (!resp.ok) {                       // includes 429 -- silence is right for the user
      console.warn("[followups] HTTP", resp.status);
      return;
    }
    const payload = await resp.json();
    addSuggestions(block, payload.suggestions);
  } catch (err) {
    // Silent for the user, never for whoever has to debug it -- an empty catch here
    // swallowed rendering errors as indistinguishably as network ones.
    console.warn("[followups] failed:", err);
  }
}

function submitQuestion(question) {
  addQuestion(question);
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;

  askQuestion(question).finally(() => {
    sendBtn.disabled = false;
    input.focus();
  });
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  submitQuestion(question);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});

renderOpening();
renderRailEmpty("ستظهر هنا نصوص المواد القانونية التي استندت إليها الإجابة");
