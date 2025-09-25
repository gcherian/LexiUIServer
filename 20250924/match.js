// src/lib/match.js
// Robust text span matcher over pdf.js tokens

function norm(s) {
  return String(s || "")
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[\u00A0]/g, " ")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// simple Levenshtein ratio
function levRatio(a, b) {
  const A = a.length, B = b.length;
  if (!A && !B) return 1;
  const dp = new Array(B + 1);
  for (let j = 0; j <= B; j++) dp[j] = j;
  for (let i = 1; i <= A; i++) {
    let prev = dp[0];
    dp[0] = i;
    for (let j = 1; j <= B; j++) {
      const tmp = dp[j];
      dp[j] = Math.min(
        dp[j] + 1,
        dp[j - 1] + 1,
        prev + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
      prev = tmp;
    }
  }
  return 1 - dp[B] / Math.max(1, Math.max(A, B));
}

function unionRect(span) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const t of span) {
    x0 = Math.min(x0, t.x0);
    y0 = Math.min(y0, t.y0);
    x1 = Math.max(x1, t.x1);
    y1 = Math.max(y1, t.y1);
  }
  return { x0: Math.floor(x0), y0: Math.floor(y0), x1: Math.ceil(x1), y1: Math.ceil(y1) };
}

/**
 * tokens: [{page,x0,y0,x1,y1,text}]
 * value: raw string from DocAI element.content
 * returns {page, rect:{x0,y0,x1,y1}, score}
 */
export function locateValue(value, tokens, maxWindow = 20) {
  const raw = (value || "").trim();
  if (!raw) return null;

  const looksNumeric = /^[\s\-,$€£₹.\d/]+$/.test(raw);
  const target = looksNumeric
    ? raw.replace(/[,\s]/g, "")
    : norm(raw);

  // group tokens per page and sort reading-order
  const byPage = new Map();
  for (const t of tokens || []) {
    if (!t || !t.text) continue;
    if (!byPage.has(t.page)) byPage.set(t.page, []);
    byPage.get(t.page).push(t);
  }
  byPage.forEach((arr) =>
    arr.sort((a, b) => (a.y0 === b.y0 ? a.x0 - b.x0 : a.y0 - b.y0))
  );

  let best = null;

  function scoreSpan(span) {
    const txt = span.map(t => (t.text || "")).join(" ");
    const s = looksNumeric
      ? levRatio(txt.replace(/[,\s]/g, ""), target)
      : levRatio(norm(txt), target);
    // penalize multi-line spreads a bit
    const ys = span.map(t => (t.y0 + t.y1) / 2);
    const spread = Math.max(...ys) - Math.min(...ys);
    const hs = span.map(t => t.y1 - t.y0);
    const avgH = hs.reduce((a,b)=>a+b,0) / Math.max(1, hs.length);
    const penalty = Math.max(0, spread - avgH * 0.7) / Math.max(1, avgH); // 0..?
    return s - Math.min(0.25, penalty * 0.15);
  }

  byPage.forEach((toks, pg) => {
    const n = toks.length;
    for (let i = 0; i < n; i++) {
      const span = [];
      for (let w = 0; w < maxWindow && i + w < n; w++) {
        const tok = toks[i + w];
        const s = (tok.text || "").trim();
        if (!s) continue;
        span.push(tok);

        // early prune: first token should somewhat match
        if (span.length === 1) {
          const r = levRatio(norm(s), norm(raw).split(" ")[0] || s);
          if (r < 0.45) continue;
        }

        const score = scoreSpan(span);
        if (!best || score > best.score) {
          best = { score, page: pg, rect: unionRect(span) };
        }
      }
    }
  });

  // require a minimum score
  if (!best || best.score < 0.55) return null;
  return best;
}