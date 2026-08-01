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
function scrollIntoReading(el) {
  const top = el.getBoundingClientRect().top + window.scrollY - 88;
  window.scrollTo({ top, behavior: reducedMotion ? "auto" : "smooth" });
}

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
}

function addFailure(message) {
  const el = document.createElement("div");
  el.className = "failure";
  el.textContent = message;
  thread.appendChild(el);
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
    addAnswer(data.answer, data.sources);
    if (data.sources && data.sources.length) renderSources(data.sources);
  } catch (err) {
    pending.remove();
    addFailure("تعذّر الوصول إلى الخادم — تأكد من أن الخدمة قيد التشغيل.");
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addQuestion(question);
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;

  askQuestion(question).finally(() => {
    sendBtn.disabled = false;
    input.focus();
  });
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
