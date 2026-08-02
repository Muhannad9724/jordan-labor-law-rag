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
const resetBtn = document.getElementById("reset-btn");
const railBody = document.getElementById("rail-body");
const railCount = document.getElementById("rail-count");
const suggestionsDock = document.getElementById("suggestions-dock");
const composerDock = document.querySelector(".composer-dock");

const RAIL_EMPTY_MESSAGE = "ستظهر هنا نصوص المواد القانونية التي استندت إليها الإجابة";

// Conversation history, oldest first. Held here rather than on the server: the backend
// stays stateless (no session store, nothing to expire or leak between users), which is
// what lets this run on one container with no database.
//
// Trimmed to the last MAX_HISTORY_TURNS entries because every turn is re-sent with each
// request -- an untrimmed thread would grow the payload, and the cost per question,
// without bound. The server caps it again independently; this is not a trust boundary.
let history = [];
const MAX_HISTORY_TURNS = 12;

function recordTurn(role, content) {
  history.push({ role, content });
  if (history.length > MAX_HISTORY_TURNS) {
    history = history.slice(-MAX_HISTORY_TURNS);
  }
}

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

// The part of the viewport the sticky dock covers, and so is not available for reading.
// Measured rather than hardcoded: the dock's height changes when suggestion chips appear
// and when the textarea grows, and a stale constant would either waste space or let text
// hide underneath.
function dockClearance() {
  return (composerDock ? composerDock.offsetHeight : 78) + 24;
}
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
  const room = window.innerHeight - dockClearance();
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

// Rendered into the sticky dock above the question bar, not into the answer. They
// belong to what you'd type next, so they live with the input.
//
// This is what removed the chip scroll-into-view logic entirely: the dock is sticky, so
// the chips are on screen at any scroll position and there is nothing to reveal. Every
// previous placement (end of the answer, then top of the answer) needed its own scrolling
// rules to be seen at all, and each had a case where they still weren't.
function addSuggestions(questions) {
  if (!questions || !questions.length) return;

  suggestionsDock.innerHTML = "";

  const label = document.createElement("div");
  label.className = "suggestions-label";
  label.textContent = "أسئلة ذات صلة";
  suggestionsDock.appendChild(label);

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
      submitQuestion(q);                 // clearSuggestions() runs inside
    });
    row.appendChild(chip);
  }

  suggestionsDock.appendChild(row);
  suggestionsDock.hidden = false;
  syncDockClearance();
}

// The dock floats over the thread, so the thread needs bottom padding equal to its
// height or the last lines of an answer sit underneath it -- the same failure the
// composer alone caused before, except the dock's height changes as chips come and go.
function syncDockClearance() {
  const h = composerDock.offsetHeight;
  document.documentElement.style.setProperty("--dock-clearance", `${h + 24}px`);
}

function clearSuggestions() {
  suggestionsDock.innerHTML = "";
  suggestionsDock.hidden = true;
  syncDockClearance();
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
  // Snapshot before this turn is recorded -- the server wants the conversation *prior*
  // to the current question, which it receives separately.
  const priorHistory = history.slice();
  try {
    const resp = await fetch(`${API_BASE_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: priorHistory }),
    });

    pending.remove();

    if (!resp.ok) {
      const payload = await resp.json().catch(() => ({}));
      addFailure(typeof payload.detail === "string" ? payload.detail : "تعذّر الحصول على إجابة لهذا السؤال.");
      return;
    }

    const data = await resp.json();
    // Recorded only on success, and as a pair. A failed turn leaves no trace, so the
    // next question is not rewritten against an exchange the user never actually saw.
    recordTurn("user", question);
    recordTurn("assistant", data.answer);

    addAnswer(data.answer, data.sources);
    if (data.sources && data.sources.length) renderSources(data.sources);

    // Not awaited: the answer is complete without it, and a slow or failed
    // suggestion call must not hold the composer disabled.
    fetchSuggestions(question, data);
  } catch (err) {
    pending.remove();
    addFailure("تعذّر الوصول إلى الخادم — تأكد من أن الخدمة قيد التشغيل.");
  }
}

async function fetchSuggestions(question, data) {
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
    addSuggestions(payload.suggestions);
  } catch (err) {
    // Silent for the user, never for whoever has to debug it -- an empty catch here
    // swallowed rendering errors as indistinguishably as network ones.
    console.warn("[followups] failed:", err);
  }
}

function submitQuestion(question) {
  clearSuggestions();   // they belonged to the previous answer
  addQuestion(question);
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;
  // Resetting mid-answer would wipe #thread out from under a fetch that is still
  // going to append its result into it -- see resetConversation()'s comment.
  resetBtn.disabled = true;

  askQuestion(question).finally(() => {
    sendBtn.disabled = false;
    resetBtn.disabled = false;
    input.focus();
  });
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  submitQuestion(question);
});

// Still purely client-side: the server holds no session, so forgetting the conversation
// means clearing the history array this page sends with each request -- there is nothing
// to tell the backend. Since turns now carry over, this is the only way to start a
// genuinely fresh topic; without it a new question keeps getting rewritten against an
// unrelated earlier exchange.
// Guarded by resetBtn.disabled (see submitQuestion) rather than only hiding the button,
// because the button is reachable the whole time a request is in flight -- without the
// guard, resetting mid-answer would clear #thread out from under askQuestion(), and its
// pending .then() would go on to append the answer into a thread that looks freshly
// reset, which reads as a stale answer appearing from nowhere.
function resetConversation() {
  history = [];                  // the actual point of reset now that turns carry over
  clearSuggestions();
  renderOpening();               // also clears #thread
  renderRailEmpty(RAIL_EMPTY_MESSAGE);
  document.body.classList.remove("reading");
  lastUserScrollAt = 0;
  input.value = "";
  input.style.height = "auto";
  window.scrollTo({ top: 0, behavior: reducedMotion ? "auto" : "smooth" });
  input.focus();
}

resetBtn.addEventListener("click", resetConversation);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
  syncDockClearance();   // a grown textarea makes the dock taller
});

window.addEventListener("resize", syncDockClearance);

renderOpening();
renderRailEmpty(RAIL_EMPTY_MESSAGE);
syncDockClearance();
