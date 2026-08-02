/* ------------------------------------------------------------------ markdown
   The model writes markdown -- "**Employee total:** 7.5%", per-article lists, and
   since the calculator landed, component tables. Setting .textContent showed all of
   that raw, asterisks and pipes included.

   Everything below builds DOM nodes and puts text through createTextNode/.textContent.
   Nothing is ever assigned to innerHTML, so model output cannot become markup --
   stronger than escaping-then-injecting, because there is no injection point to get
   wrong. Anything unrecognised falls through as literal text.                        */

function appendInline(parent, text) {
  const re = /(\*\*|__)([\s\S]+?)\1|`([^`]+)`|(?:\*|_)([^*_\n]+)(?:\*|_)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
    let el;
    if (m[2] !== undefined) { el = document.createElement("strong"); el.textContent = m[2]; }
    else if (m[3] !== undefined) { el = document.createElement("code"); el.textContent = m[3]; }
    else { el = document.createElement("em"); el.textContent = m[4]; }
    parent.appendChild(el);
    last = re.lastIndex;
  }
  if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
}

const RE_HEADING = /^\s{0,3}(#{1,4})\s+(.*)$/;
const RE_RULE = /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/;
const RE_BULLET = /^\s*[-*•]\s+(.*)$/;
const RE_NUMBER = /^\s*(\d+)[.)]\s+(.*)$/;
const RE_ROW = /^\s*\|.*\|\s*$/;
const RE_DIVIDER = /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/;

function splitRow(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
}

function isBlockStart(line) {
  return RE_HEADING.test(line) || RE_RULE.test(line) || RE_BULLET.test(line)
    || RE_NUMBER.test(line) || RE_ROW.test(line);
}

function renderMarkdown(text, container) {
  const lines = String(text).replace(/\r\n?/g, "\n").split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) { i += 1; continue; }

    if (RE_RULE.test(line)) {
      container.appendChild(document.createElement("hr"));
      i += 1;
      continue;
    }

    const heading = line.match(RE_HEADING);
    if (heading) {
      const h = document.createElement(`h${Math.min(heading[1].length + 2, 6)}`);
      appendInline(h, heading[2]);
      container.appendChild(h);
      i += 1;
      continue;
    }

    // A table needs its divider row; without one the pipes are just text.
    if (RE_ROW.test(line) && i + 1 < lines.length && RE_DIVIDER.test(lines[i + 1])
        && lines[i + 1].includes("-")) {
      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const hrow = document.createElement("tr");
      splitRow(line).forEach((cell) => {
        const th = document.createElement("th");
        appendInline(th, cell);
        hrow.appendChild(th);
      });
      thead.appendChild(hrow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      i += 2;
      while (i < lines.length && RE_ROW.test(lines[i])) {
        const tr = document.createElement("tr");
        splitRow(lines[i]).forEach((cell) => {
          const td = document.createElement("td");
          appendInline(td, cell);
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
        i += 1;
      }
      table.appendChild(tbody);

      // Own scroll container: a wide table must never make the page scroll sideways.
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      wrap.appendChild(table);
      container.appendChild(wrap);
      continue;
    }

    if (RE_BULLET.test(line) || RE_NUMBER.test(line)) {
      const ordered = RE_NUMBER.test(line);
      const re = ordered ? RE_NUMBER : RE_BULLET;
      const list = document.createElement(ordered ? "ol" : "ul");
      while (i < lines.length) {
        const m = lines[i].match(re);
        if (m) {
          const li = document.createElement("li");
          appendInline(li, ordered ? m[2] : m[1]);
          list.appendChild(li);
          i += 1;
          continue;
        }
        // A blank line only ends the list if what follows isn't another item of
        // it -- otherwise each item lands in its own <ol> and every <li> renders
        // as "1." (numbering restarts per list; the model writes items separated
        // by blank lines when each has multiple sentences).
        if (!lines[i].trim()) {
          let j = i + 1;
          while (j < lines.length && !lines[j].trim()) j += 1;
          if (j < lines.length && re.test(lines[j])) {
            i = j;
            continue;
          }
        }
        break;
      }
      container.appendChild(list);
      continue;
    }

    const p = document.createElement("p");
    let first = true;
    while (i < lines.length && lines[i].trim() && !(first === false && isBlockStart(lines[i]))) {
      if (!first) p.appendChild(document.createElement("br"));
      appendInline(p, lines[i].trim());
      first = false;
      i += 1;
    }
    container.appendChild(p);
  }

  if (!container.childNodes.length) container.textContent = String(text);
}
