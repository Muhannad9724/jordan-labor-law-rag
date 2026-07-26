// Same-origin now that FastAPI serves this frontend directly (src/api.py) -- no CORS needed.
const API_BASE_URL = "";

const LAW_DISPLAY_NAMES = {
  ssl: "Social Security Law",
  labor96: "Labor Law 1996",
  prof19: "Professional Work Law 2019",
};

const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const input = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

const EXAMPLE_QUESTIONS = [
  "How much notice does my employer need to give before ending my contract?",
  "How many paid vacation days am I entitled to per year?",
  "كم يوم إجازة سنوية مدفوعة الأجر أستحق؟",
  "What happens to my pension if I retire early?",
];

function renderEmptyState() {
  chat.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "empty-state";
  wrap.innerHTML = `
    <div class="mark"><span></span></div>
    <p>Ask a question about Jordanian labor or social security law to get started.</p>
    <div class="example-chips"></div>
  `;
  const chipRow = wrap.querySelector(".example-chips");
  for (const q of EXAMPLE_QUESTIONS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "example-chip";
    chip.textContent = q;
    if (isArabic(q)) chip.dir = "rtl";
    chip.addEventListener("click", () => {
      input.value = q;
      composer.requestSubmit();
    });
    chipRow.appendChild(chip);
  }
  chat.appendChild(wrap);
}

// Arabic script block (with presentation forms) -- if a meaningful share of a
// string's letters fall in this range, render that message right-to-left.
const ARABIC_RE = /[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]/g;
function isArabic(text) {
  const letters = text.replace(/[^\p{L}]/gu, "");
  if (!letters) return false;
  const arabicCount = (letters.match(ARABIC_RE) || []).length;
  return arabicCount / letters.length > 0.3;
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function appendMessage(role, text) {
  const emptyState = chat.querySelector(".empty-state");
  if (emptyState) emptyState.remove();

  const el = document.createElement("div");
  el.className = `msg ${role}`;
  if (isArabic(text)) el.dir = "rtl";
  el.textContent = text;
  chat.appendChild(el);
  scrollToBottom();
  return el;
}

function renderSources(container, sources) {
  if (!sources || sources.length === 0) return;
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `Sources (${sources.length})`;
  details.appendChild(summary);

  for (const s of sources) {
    const item = document.createElement("div");
    item.className = "source-item";
    if (isArabic(s.text)) item.dir = "rtl";

    const label = document.createElement("div");
    label.className = "source-label";
    const lawName = LAW_DISPLAY_NAMES[s.law] || s.law;
    label.textContent = `${lawName}, Article ${s.article_number}${s.clause ? "." + s.clause : ""} (${s.language})`;
    item.appendChild(label);

    const body = document.createElement("div");
    body.textContent = s.text;
    item.appendChild(body);

    details.appendChild(item);
  }
  container.appendChild(details);
}

async function askQuestion(question) {
  const pending = appendMessage("assistant pending", "Thinking...");
  try {
    const resp = await fetch(`${API_BASE_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      const detail = typeof body.detail === "string" ? body.detail : "The server couldn't answer that question.";
      pending.remove();
      appendMessage("error", detail);
      return;
    }

    const data = await resp.json();
    pending.remove();
    const answerEl = appendMessage("assistant", data.answer);
    renderSources(answerEl, data.sources);
    scrollToBottom();
  } catch (err) {
    pending.remove();
    appendMessage("error", "Couldn't reach the server. Is the backend running?");
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  appendMessage("user", question);
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
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

renderEmptyState();
