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

// Builds one <ol>/<ul> starting at lines[start]. Returns [element, nextIndex].
//
// Two things a flat parser gets wrong on real model output:
//  1. Items are often multi-sentence, so the model separates them with a
//     blank line. A flat parser closes the list on any non-matching line,
//     so a blank line splits "1. A" / "1. B" / "1. C" into three separate
//     one-item <ol>s -- and since <li> numbering restarts per <ol>, every
//     item renders as "1.". Fix: swallow a blank line and keep going if
//     another item of the SAME type follows it.
//  2. A numbered item is sometimes followed by a bullet list elaborating
//     it ("1. Header:\n- detail\n- detail\n\n2. Header:..."). The bullets
//     are a different marker, so a flat parser closes the outer <ol> right
//     there -- same "1." bug, one level up. Fix: when a differently-typed
//     list immediately follows an item (blank line allowed), nest it
//     inside that item's <li> instead of ending the outer list.
function parseList(lines, start, ordered) {
  const re = ordered ? RE_NUMBER : RE_BULLET;
  const nestedRe = ordered ? RE_BULLET : RE_NUMBER;
  const list = document.createElement(ordered ? "ol" : "ul");
  let i = start;
  while (i < lines.length) {
    if (!lines[i].trim()) {
      let j = i + 1;
      while (j < lines.length && !lines[j].trim()) j += 1;
      if (j < lines.length && re.test(lines[j])) {
        i = j;
        continue;
      }
      break;
    }
    const m = lines[i].match(re);
    if (!m) break;
    const li = document.createElement("li");
    appendInline(li, ordered ? m[2] : m[1]);
    list.appendChild(li);
    i += 1;

    let k = i;
    while (k < lines.length && !lines[k].trim()) k += 1;
    if (k < lines.length && nestedRe.test(lines[k])) {
      const [sub, ni] = parseList(lines, k, !ordered);
      li.appendChild(sub);
      i = ni;
    }
  }
  return [list, i];
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
      const [list, ni] = parseList(lines, i, ordered);
      container.appendChild(list);
      i = ni;
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
