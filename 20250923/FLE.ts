// File: src/tsp4/components/lasso/FieldLevelEditor.tsx
import React, { useEffect, useMemo, useState } from "react";
import PdfEditCanvas, { type EditRect, type TokenBox } from "./PdfEditCanvas";
import "../ocr.css";

import {
  API,
  uploadPdf,
  getMeta,
  getBoxes,
  docIdFromUrl,
  ocrPreview,
  matchField,
  type MatchResp,
} from "../../../lib/api";

/* =============== Types =============== */
type KVRect = { page: number; x0: number; y0: number; x1: number; y1: number };
type FieldRow = { key: string; value: string; rects?: KVRect[] };
type AnyJson = Record<string, any>;

const COLORS: Record<string, string> = {
  fuzzy: "#22c55e",
  tfidf: "#3b82f6",
  minilm: "#a855f7",
  distilbert: "#facc15",
};

/* =============== Small helpers =============== */
function normNum(s: string) {
  return (s || "").toLowerCase().normalize("NFKC").replace(/[,$]/g, "").replace(/\s+/g, " ").trim();
}
function levRatio(a: string, b: string): number {
  const m = a.length, n = b.length;
  if (!m && !n) return 1;
  const dp = new Array(n + 1).fill(0).map((_, j) => j);
  for (let i = 1; i <= m; i++) {
    let prev = dp[0]; dp[0] = i;
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
  for (const t of span) { x0 = Math.min(x0, t.x0); y0 = Math.min(y0, t.y0); x1 = Math.max(x1, t.x1); y1 = Math.max(y1, t.y1); }
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

/** Quick local fallback: approximate by value (kept from your previous build) */
function autoLocateByValue(valueRaw: string, allTokens: TokenBox[], maxWindow = 16) {
  const raw = (valueRaw || "").trim();
  if (!raw) return null;
  const looksNumeric = /^[\s\-$€£₹,.\d/]+$/.test(raw);
  const words = looksNumeric ? [normNum(raw)] : raw.toLowerCase().normalize("NFKC").replace(/[^\p{L}\p{N}\s]/gu, " ").replace(/\s+/g, " ").trim().split(" ");

  const byPage = new Map<number, TokenBox[]>();
  for (const t of allTokens) (byPage.get(t.page) || byPage.set(t.page, []).get(t.page)!).push(t);
  byPage.forEach((arr) => arr.sort((a, b) => (a.y0 === b.y0 ? a.x0 - b.x0 : a.y0 - b.y0)));

  let best: { score: number; page: number; span: TokenBox[] } | null = null;

  function scoreSpan(span: TokenBox[]) {
    const txt = span.map((t) => (t.text || "")).join(" ").toLowerCase().normalize("NFKC").replace(/[^\p{L}\p{N}\s]/gu, " ");
    const spanWords = txt.split(/\s+/).filter(Boolean);

    if (looksNumeric) {
      const fuzz = levRatio(spanWords.join(" "), words.join(" "));
      return fuzz - Math.min(0.25, linePenalty(span) * 0.12);
    }

    let covered = 0, j = 0;
    for (let i = 0; i < words.length && j < spanWords.length;) {
      if (words[i] === spanWords[j] || levRatio(words[i], spanWords[j]) >= 0.8) { covered++; i++; j++; }
      else { j++; }
    }
    const coverage = covered / Math.max(1, words.length);
    const fuzz = levRatio(spanWords.join(" "), words.join(" "));
    return coverage * 0.75 + fuzz * 0.35 - Math.min(0.25, linePenalty(span) * 0.12);
  }

  byPage.forEach((toks, pg) => {
    const n = toks.length;
    for (let i = 0; i < n; i++) {
      const span: TokenBox[] = [];
      for (let w = 0; w < maxWindow && i + w < n; w++) {
        const t = toks[i + w];
        const token = (t.text || "").trim();
        if (!token) continue;
        span.push(t);
        const s = scoreSpan(span);
        if (!best || s > best.score) best = { score: s, page: pg, span: [...span] };
      }
    }
  });

  if (!best) return null;
  const rect = unionRect(best.span);
  return { page: best.page, rect, score: best.score };
}

function refineWithTokens(union: EditRect, pageTokens: TokenBox[]): EditRect {
  const minx = Math.min(union.x0, union.x1), maxx = Math.max(union.x0, union.x1);
  const miny = Math.min(union.y0, union.y1), maxy = Math.max(union.y0, union.y1);
  const inBox = (t: TokenBox) => {
    const cx = (t.x0 + t.x1) / 2, cy = (t.y0 + t.y1) / 2;
    return cx >= minx && cx <= maxx && cy >= miny && cy <= maxy;
  };
  const pool = pageTokens.filter(inBox);
  if (!pool.length) return union;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const t of pool) { x0 = Math.min(x0, t.x0); y0 = Math.min(y0, t.y0); x1 = Math.max(x1, t.x1); y1 = Math.max(y1, t.y1); }
  return { page: union.page, x0, y0, x1, y1 };
}

/* =============== DocAI JSON → rows =============== */
/**
 * We accept the shape shown on your screenshots:
 * documents[*].properties[*].metadata: {
 *   parser:"OCR", mimeType:"application/octet-stream", executionDateTime:"..",
 *   metadataMap: { <k>: [values...] },
 *   pages:[ { elements:[ {elementType:"paragraph", content:string, boundingBox:{x,y,width,height}, page:n}, ... ] } ... ]
 * }
 */
function normalizeDocAIBBox(b: any): KVRect | null {
  if (!b) return null;
  // DocAI gives origin top-left with width/height
  const x = Number(b.x ?? b.left ?? b.X ?? b.x0 ?? NaN);
  const y = Number(b.y ?? b.top ?? b.Y ?? b.y0 ?? NaN);
  const w = Number(b.width ?? b.w ?? (b.x1 != null && b.x0 != null ? b.x1 - b.x0 : NaN));
  const h = Number(b.height ?? b.h ?? (b.y1 != null && b.y0 != null ? b.y1 - b.y0 : NaN));
  const page = Number((b.page ?? b.Page ?? b.p) ?? NaN);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(w) || !Number.isFinite(h)) return null;
  if (!Number.isFinite(page)) return null;
  return { page, x0: x, y0: y, x1: x + w, y1: y + h };
}

function flattenDocAI(doc: AnyJson): FieldRow[] {
  const rows: FieldRow[] = [];

  // 1) metadataMap (turn into simple rows)
  const metaBlocks =
    doc?.documents?.[0]?.properties?.[0]?.metadata ||
    doc?.documents?.[0]?.metadata ||
    doc?.metadata ||
    {};

  const mm = metaBlocks?.metadataMap;
  if (mm && typeof mm === "object") {
    for (const k of Object.keys(mm)) {
      const val = mm[k];
      const s = Array.isArray(val) ? val.join(" | ") : String(val ?? "");
      rows.push({ key: `metadataMap.${k}`, value: s });
    }
  }

  // 2) pages/elements
  const pages = metaBlocks?.pages || doc?.pages || [];
  pages.forEach((p: any, i: number) => {
    const els = p?.elements || [];
    els.forEach((el: any, j: number) => {
      const content = String(el?.content ?? el?.text ?? "");
      const bb = normalizeDocAIBBox(el?.boundingBox);
      if (bb) {
        rows.push({
          key: `pages[${i}].elements[${j}]${el?.elementType ? `:${el.elementType}` : ""}`,
          value: content,
          rects: [bb],
        });
      } else {
        rows.push({
          key: `pages[${i}].elements[${j}]${el?.elementType ? `:${el.elementType}` : ""}`,
          value: content,
        });
      }
    });
  });

  return rows;
}

/* =============== Component =============== */
export default function FieldLevelEditor() {
  const [docUrl, setDocUrl] = useState("");
  const [docId, setDocId] = useState("");
  const [meta, setMeta] = useState<{ w: number; h: number }[]>([]);
  const [page, setPage] = useState(1);

  const [tokens, setTokens] = useState<TokenBox[]>([]);
  const tokensPage = useMemo(() => tokens.filter((t) => t.page === page), [tokens, page]);

  const [rows, setRows] = useState<FieldRow[]>([]);
  const [focusedKey, setFocusedKey] = useState("");

  const [rect, setRect] = useState<EditRect | null>(null);
  const [overlays, setOverlays] = useState<{ label: string; color: string; rect: EditRect | null }[]>([]);
  const [showBoxes, setShowBoxes] = useState(false);

  const [zoom, setZoom] = useState(1);
  const [editValue, setEditValue] = useState(false);
  const [loadingBoxes, setLoadingBoxes] = useState(false);

  /* ------ PDF upload ------ */
  async function onUploadPdf(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const res = await uploadPdf(f);
      setDocId(res.doc_id);
      setDocUrl(res.annotated_tokens_url);
      const m = await getMeta(res.doc_id);
      setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
      setTokens((await getBoxes(res.doc_id)) as any);
      setPage(1);
      setRect(null);
      setOverlays([]);
    } finally {
      (e.target as HTMLInputElement).value = "";
    }
  }

  /* ------ DocAI JSON upload (replaces ECM) ------ */
  async function onUploadDocAI(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const parsed = JSON.parse(await f.text()) as AnyJson;
      const fr = flattenDocAI(parsed);
      setRows(fr);
      setFocusedKey("");
      setRect(null);
      setOverlays([]);
    } catch (err) {
      console.error(err);
      alert("Invalid DocAI JSON");
    } finally {
      (e.target as HTMLInputElement).value = "";
    }
  }

  /* ------ Paste /data/{doc_id}/original.pdf ------ */
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
      setOverlays([]);
    })();
  }, [docUrl]);

  /* ------ Row click: show DocAI box immediately; then fetch model overlays ------ */
  function onRowClick(r: FieldRow) {
    setFocusedKey(r.key);
    setOverlays([]);
    setLoadingBoxes(true);

    // If DocAI provided a rect, show it NOW (pink)
    const rr = r.rects?.[0];
    if (rr) {
      const uiRect: EditRect = { page: rr.page, x0: rr.x0, y0: rr.y0, x1: rr.x1, y1: rr.y1 };
      setPage(rr.page);
      // optional refinement against tokens (keeps the touch-feeling tight)
      setRect(refineWithTokens(uiRect, tokens.filter(t => t.page === rr.page)));
    } else {
      // no DocAI rect → try local fuzzy fallback for a quick pink box
      const hit = autoLocateByValue(r.value, tokens);
      if (hit) {
        const refined = refineWithTokens({ page: hit.page, ...hit.rect }, tokens.filter(t => t.page === hit.page));
        setPage(hit.page);
        setRect(refined);
      } else {
        setRect(null);
      }
    }

    // Don’t block UI: get 4-method overlays
    (async () => {
      try {
        if (!docId) return;
        const res: MatchResp = await matchField(docId, r.key, r.value, 12);
        const pick = (m: any): EditRect | null =>
          !m ? null : ({ page: m.page, x0: m.rect.x0, y0: m.rect.y0, x1: m.rect.x1, y1: m.rect.y1 });
        const ovs = (["fuzzy", "tfidf", "minilm", "distilbert"] as const).map((k) => ({
          label: k,
          color: COLORS[k],
          rect: pick((res as any).methods?.[k]),
        }));
        setOverlays(ovs);
      } catch (e) {
        console.warn("matchField failed", e);
        setOverlays([]);
      } finally {
        setLoadingBoxes(false);
      }
    })();
  }

  /* ------ Commit (kept as-is; you can ignore edit mode for this phase) ------ */
  async function onRectCommitted(rr: EditRect) {
    if (!focusedKey) return;
    try {
      if (docId) {
        const res = await ocrPreview(docId, rr.page, rr);
        const text = (res?.text || "").trim();
        setRect(rr);
        if (editValue) {
          setRows(prev =>
            prev.map(row =>
              row.key === focusedKey
                ? { ...row, value: text, rects: [{ page: rr.page, x0: rr.x0, y0: rr.y0, x1: rr.x1, y1: rr.y1 }] }
                : row
            )
          );
        }
      }
    } catch {
      /* best-effort */
    }
  }

  const serverW = meta[page - 1]?.w || 1;
  const serverH = meta[page - 1]?.h || 1;

  /* =============== UI =============== */
  return (
    <div className="workbench">
      <div className="wb-toolbar" style={{ gap: 8 }}>
        <span style={{ fontWeight: 600 }}>Choose:</span>

        <label className="btn">
          <input type="file" accept="application/pdf" onChange={onUploadPdf} style={{ display: "none" }} />
          PDF
        </label>

// in FieldLevelEditor.tsx toolbar
<label className="btn">
  <input type="file" accept="application/json"
         onChange={async (e) => {
           const f = e.target.files?.[0]; if (!f) return;
           try {
             const res = await importDocAI(f);
             setDocId(res.doc_id);
             setDocUrl(`${API}/data/${res.doc_id}/original.pdf`); // stays blank until you also upload a PDF; ok.
             setTokens(res.boxes as any);
             setMeta(res.pages.map(p => ({w:p.width, h:p.height})));
             setRows(res.rows.map(kv => ({ key: kv.key, value: kv.value, rects: kv.rects })));
             setPage(1); setRect(null); setOverlays([]);
           } finally { (e.target as HTMLInputElement).value = ""; }
         }}
         style={{ display: "none" }} />
  DOCAI JSON
</label>

        <input
          className="input"
          placeholder="...or paste /data/{doc_id}/original.pdf"
          value={docUrl}
          onChange={(e) => setDocUrl(e.target.value || "")}
          style={{ marginLeft: 8, minWidth: 360 }}
        />

        <label className={showBoxes ? "btn toggle active" : "btn toggle"} style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={showBoxes} onChange={() => setShowBoxes((v) => !v)} /> Boxes
        </label>

        <label className={editValue ? "btn toggle active" : "btn toggle"} style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={editValue} onChange={() => setEditValue(v => !v)} /> Edit value
        </label>

        <span className="spacer" />

        <div className="toolbar-inline" style={{ gap: 6 }}>
          {Object.entries(COLORS).map(([k, c]) => (
            <span key={k} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <span style={{ width: 12, height: 12, background: c, border: `1px solid ${c}`, display: "inline-block", opacity: 0.7 }} />
              {k}
            </span>
          ))}
          {loadingBoxes ? <span className="muted" style={{ marginLeft: 10 }}>loading boxes…</span> : null}
        </div>

        <span className="spacer" />

        <div className="toolbar-inline" style={{ gap: 4 }}>
          <button onClick={() => setZoom((z) => Math.max(0.5, Math.round((z - 0.1) * 10) / 10))}>–</button>
          <span style={{ width: 44, textAlign: "center" }}>{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom((z) => Math.min(3, Math.round((z + 0.1) * 10) / 10))}>+</button>
          <button onClick={() => setZoom(1)}>Reset</button>
        </div>

        <span className="muted" style={{ marginLeft: 12 }}>API: {API}</span>
      </div>

      <div className="wb-split" style={{ display: "flex", gap: 12 }}>
        {/* LEFT 30% — DocAI rows */}
        <div className="wb-left" style={{ flexBasis: "30%", flexGrow: 0, flexShrink: 0, overflow: "auto" }}>
          <div className="section-title">DocAI</div>
          {!rows.length ? (
            <div className="placeholder">Upload DocAI JSON to see elements.</div>
          ) : (
            <table>
              <thead>
                <tr><th style={{ width: "42%" }}>Key / Element</th><th>Text</th></tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const focused = r.key === focusedKey;
                  return (
                    <tr
                      key={r.key + ":" + i}
                      onClick={() => onRowClick(r)}
                      style={focused ? { outline: "2px solid #ec4899", outlineOffset: -2 } : undefined}
                    >
                      <td><code>{r.key}</code></td>
                      <td style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {r.value}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* RIGHT 70% — PDF */}
        <div className="wb-right" style={{ flexBasis: "70%", overflow: "auto" }}>
          {docUrl ? (
            <>
              <div className="toolbar-inline">
                <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Prev</button>
                <span className="page-indicator">Page {page} {meta.length ? `/ ${meta.length}` : ""}</span>
                <button disabled={meta.length > 0 && page >= meta.length} onClick={() => setPage((p) => p + 1)}>Next</button>
              </div>
              <PdfEditCanvas
                docUrl={docUrl}
                page={page}
                serverW={serverW}
                serverH={serverH}
                tokens={tokensPage}
                rect={rect}
                overlays={overlays}
                showTokenBoxes={showBoxes}
                editable={true}
                onRectChange={setRect}
                onRectCommit={onRectCommitted}
                zoom={zoom}
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
