// File: src/tsp4/components/lasso/FieldLevelEditor.tsx
import React, { useEffect, useMemo, useState } from "react";
import "../ocr.css";
import PdfEditCanvas, { type EditRect, type TokenBox } from "./PdfEditCanvas";
import {
  API,
  uploadPdf,
  getMeta,
  getBoxes,
  docIdFromUrl,
  ocrPreview,
} from "../../../lib/api";

type KVRect = { page: number; x0: number; y0: number; x1: number; y1: number };
type AnyJson = Record<string, any>;
type FieldRow = { key: string; value: string; rects?: KVRect[] };

/* ---------- helpers ---------- */
const norm = (s: string) =>
  (s || "")
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[\u00A0]/g, " ")
    .replace(/[^\p{L}\p{N}\s]/gu, "")
    .replace(/\s+/g, " ")
    .trim();

const normNum = (s: string) =>
  (s || "").toLowerCase().normalize("NFKC").replace(/[,$]/g, "").replace(/\s+/g, " ").trim();

function levRatio(a: string, b: string): number {
  const m = a.length, n = b.length;
  if (!m && !n) return 1;
  const dp: number[] = new Array(n + 1);
  for (let j = 0; j <= n; j++) dp[j] = j;
  for (let i = 1; i <= m; i++) {
    let prev = dp[0];
    dp[0] = i;
    for (let j = 1; j <= n; j++) {
      const tmp = dp[j];
      dp[j] = Math.min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = tmp;
    }
  }
  return 1 - dp[n] / Math.max(1, Math.max(m, n));
}

function unionRect(span: TokenBox[]) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const t of span) {
    x0 = Math.min(x0, t.x0);
    y0 = Math.min(y0, t.y0);
    x1 = Math.max(x1, t.x1);
    y1 = Math.max(y1, t.y1);
  }
  return { x0: Math.floor(x0), y0: Math.floor(y0), x1: Math.ceil(x1), y1: Math.ceil(y1) };
}
function linePenalty(span: TokenBox[]) {
  if (span.length <= 1) return 0;
  const ys = span.map((t) => (t.y0 + t.y1) / 2).sort((a, b) => a - b);
  const spread = ys[ys.length - 1] - ys[0];
  const hs = span.map((t) => t.y1 - t.y0);
  const avg = hs.reduce((a, b) => a + b, 0) / Math.max(1, hs.length);
  return Math.max(0, spread - avg * 0.6) / Math.max(1, avg);
}

/** robust token matching across all pages */
function autoLocateByValue(valueRaw: string, allTokens: TokenBox[], maxWindow = 12) {
  const value = (valueRaw || "").trim();
  if (!value) return null;

  const looksNumeric = /^[\s\-$€£₹,.\d/]+$/.test(value);
  const target = looksNumeric ? normNum(value) : norm(value);
  if (!target) return null;

  // group & sort tokens per page
  const byPage = new Map<number, TokenBox[]>();
  for (const t of allTokens) {
    (byPage.get(t.page) || byPage.set(t.page, []).get(t.page)!).push(t);
  }
  byPage.forEach((arr) => arr.sort((a, b) => (a.y0 === b.y0 ? a.x0 - b.x0 : a.y0 - b.y0)));

  let best: { score: number; page: number; span: TokenBox[] } | null = null;

  byPage.forEach((toks, pg) => {
    const n = toks.length;
    for (let i = 0; i < n; i++) {
      let accum = "";
      const span: TokenBox[] = [];
      for (let w = 0; w < maxWindow && i + w < n; w++) {
        const t = toks[i + w];
        const txt = (t.text || "").trim();
        if (!txt) continue;
        span.push(t);
        accum = (accum ? accum + " " : "") + txt;

        // a couple of quick accept checks
        const cand = looksNumeric ? normNum(accum) : norm(accum);
        if (target.length >= 2 && !cand.includes(target.slice(0, 2))) continue;
        const sim = levRatio(cand, target);
        if (sim < 0.62) continue;

        const score = sim - Math.min(0.25, linePenalty(span) * 0.12);
        if (!best || score > best.score) best = { score, page: pg, span: [...span] };
      }
    }
  });

  if (!best) return null;
  const rect = unionRect(best.span);
  return { page: best.page, rect, score: best.score };
}

/* JSON → FieldRow[] (handles several common shapes incl. optional bboxes) */
function parseExtractionToRows(obj: AnyJson): FieldRow[] {
  if (!obj) return [];
  if (Array.isArray(obj.fields)) {
    return obj.fields.map((f: AnyJson) => ({
      key: String(f.key ?? ""),
      value: f.value != null ? String(f.value) : "",
      rects: normalizeRects(f.bboxes || f.boxes || f.bbox || f.rects),
    }));
  }
  const rows: FieldRow[] = [];
  for (const [k, v] of Object.entries(obj)) {
    if (v == null) { rows.push({ key: k, value: "" }); continue; }
    if (typeof v === "object" && !Array.isArray(v)) {
      rows.push({
        key: k,
        value: v.value != null ? String(v.value) : tryStr(v),
        rects: normalizeRects(v.bboxes || v.boxes || v.bbox || v.rects),
      });
    } else {
      rows.push({ key: k, value: Array.isArray(v) ? v.map(String).join(" ") : String(v) });
    }
  }
  return rows;
}
function normalizeRects(x: any): KVRect[] | undefined {
  if (!x) return undefined;
  const arr = Array.isArray(x) ? x : [x];
  const out: KVRect[] = [];
  for (const b of arr) {
    if (b && Number.isFinite(+b.page)) {
      out.push({ page: +b.page, x0: +b.x0, y0: +b.y0, x1: +b.x1, y1: +b.y1 });
    }
  }
  return out.length ? out : undefined;
}
const tryStr = (v: any) => {
  try { return typeof v === "string" ? v : JSON.stringify(v); } catch { return String(v); }
};

/* ---------- component ---------- */
export default function FieldLevelEditor() {
  // doc + page
  const [docUrl, setDocUrl] = useState("");
  const [docId, setDocId] = useState("");
  const [meta, setMeta] = useState<{ w: number; h: number }[]>([]);
  const [page, setPage] = useState(1);

  // tokens (orange)
  const [tokens, setTokens] = useState<TokenBox[]>([]);
  const tokensPage = useMemo(() => tokens.filter((t) => t.page === page), [tokens, page]);

  // left table
  const [rows, setRows] = useState<FieldRow[]>([]);
  const [focusedKey, setFocusedKey] = useState("");

  // overlays
  const [rect, setRect] = useState<EditRect | null>(null);
  const [showBoxes, setShowBoxes] = useState(false); // default OFF
  const [lastCrop, setLastCrop] = useState<{ url?: string; text?: string } | null>(null);

  async function onUploadPdf(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if (!f) return;
    try {
      const res = await uploadPdf(f);
      setDocId(res.doc_id);
      setDocUrl(res.annotated_tokens_url);
      const m = await getMeta(res.doc_id);
      setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
      setTokens((await getBoxes(res.doc_id)) as any);
      setPage(1);
      setRect(null);
    } finally { (e.target as HTMLInputElement).value = ""; }
  }

  async function onUploadJson(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if (!f) return;
    try {
      const parsed = JSON.parse(await f.text()) as AnyJson;
      setRows(parseExtractionToRows(parsed));
      setFocusedKey("");
      setRect(null);
    } catch { alert("Invalid JSON"); }
    finally { (e.target as HTMLInputElement).value = ""; }
  }

  useEffect(() => {
    const id = docIdFromUrl(docUrl);
    if (!id) return;
    (async () => {
      setDocId(id);
      const m = await getMeta(id);
      setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
      setTokens((await getBoxes(id)) as any);
      setPage(1);
      setRect(null);
    })();
  }, [docUrl]);

  // left click → pink highlight (prefer JSON rects; else token match)
  function onRowClick(r: FieldRow) {
    setFocusedKey(r.key);
    const rects = r.rects || [];
    if (rects.length) {
      // choose page with most coverage, then union same-page rects
      const byPg: Record<number, KVRect[]> = {};
      rects.forEach(b => (byPg[b.page] = byPg[b.page] ? [...byPg[b.page], b] : [b]));
      const pg = Number(Object.keys(byPg).sort((a, b) => byPg[+b].length - byPg[+a].length)[0]);
      const same = byPg[pg];
      const uni = same.reduce((acc, rr) => ({
        page: pg,
        x0: Math.min(acc.x0, rr.x0),
        y0: Math.min(acc.y0, rr.y0),
        x1: Math.max(acc.x1, rr.x1),
        y1: Math.max(acc.y1, rr.y1),
      }), { page: pg, x0: same[0].x0, y0: same[0].y0, x1: same[0].x1, y1: same[0].y1 });
      setPage(pg);
      setRect(uni);
      return;
    }
    const hit = autoLocateByValue(r.value, tokens);
    if (hit) {
      setPage(hit.page);
      setRect({ page: hit.page, ...hit.rect });
    } else {
      setRect(null);
    }
  }

  async function onRectCommitted(rr: EditRect) {
    if (!focusedKey) return;
    setRect(rr); // keep it
    try {
      if (docId) {
        const res = await ocrPreview(docId, rr.page, rr);
        setLastCrop({ url: res?.crop_url, text: (res?.text || "").trim() });
      }
    } catch { /* preview best-effort */ }
  }

  const serverW = meta[page - 1]?.w || 1;
  const serverH = meta[page - 1]?.h || 1;

  return (
    <div className="workbench">
      <div className="wb-toolbar">
        <input type="file" accept="application/pdf" onChange={onUploadPdf} />
        <input type="file" accept="application/json" onChange={onUploadJson} style={{ marginLeft: 8 }} />
        <input
          className="input"
          placeholder="Paste /data/{doc_id}/original.pdf"
          value={docUrl}
          onChange={(e) => setDocUrl(e.target.value || "")}
          style={{ marginLeft: 8 }}
        />
        <label className={showBoxes ? "btn toggle active" : "btn toggle"} style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={showBoxes} onChange={() => setShowBoxes(v => !v)} /> Boxes
        </label>
        <span className="spacer" />
        <span className="muted">API: {API}</span>
      </div>

      <div className="wb-split" style={{ display: "flex", gap: 12 }}>
        {/* LEFT 30% */}
        <div className="wb-left" style={{ flexBasis: "30%", flexGrow: 0, flexShrink: 0, overflow: "auto" }}>
          <div className="section-title">Extraction</div>
          {!rows.length ? (
            <div className="placeholder">Upload extraction JSON to see fields.</div>
          ) : (
            <table>
              <thead>
                <tr><th style={{ width: "40%" }}>Key</th><th>Value</th></tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const focused = r.key === focusedKey;
                  return (
                    <tr key={r.key + ":" + i}
                        onClick={() => onRowClick(r)}
                        style={focused ? { outline: "2px solid #ec4899", outlineOffset: -2 } : undefined}>
                      <td><code>{r.key}</code></td>
                      <td style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.value}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}

          {lastCrop?.url && (
            <div className="last-ocr" style={{ marginTop: 12 }}>
              <img src={lastCrop.url} alt="last-crop" />
              <div className="caption">
                <div className="title">Last OCR</div>
                <pre>{lastCrop.text || ""}</pre>
              </div>
            </div>
          )}
        </div>

        {/* RIGHT 70% */}
        <div className="wb-right" style={{ flexBasis: "70%", overflow: "auto" }}>
          {docUrl ? (
            <>
              <div className="toolbar-inline">
                <button disabled={page <= 1} onClick={() => setPage(p => p - 1)}>Prev</button>
                <span className="page-indicator">Page {page} {meta.length ? `/ ${meta.length}` : ""}</span>
                <button disabled={meta.length > 0 && page >= meta.length} onClick={() => setPage(p => p + 1)}>Next</button>
              </div>
              <PdfEditCanvas
                docUrl={docUrl}
                page={page}
                serverW={serverW}
                serverH={serverH}
                tokens={tokensPage}
                rect={rect}
                showTokenBoxes={showBoxes}
                editable={true}
                onRectChange={setRect}
                onRectCommit={onRectCommitted}
              />
            </>
          ) : (
            <div className="placeholder">Upload a PDF to begin.</div>
          )}
        </div>
      </div>
    </div>
  );
}