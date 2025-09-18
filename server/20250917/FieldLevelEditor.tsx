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
  distilExtract,
  matchField,
  type DistilField,
} from "../../../lib/api";

/* ========================= Types & Helpers ========================== */
type KVRect = { page: number; x0: number; y0: number; x1: number; y1: number };
type AnyJson = Record<string, any>;
type FieldRow = { key: string; value: string; rects?: KVRect[] };

const COLORS: Record<string,string> = {
  fuzzy: "#22c55e",
  tfidf: "#3b82f6",
  minilm: "#a855f7",
  distilbert: "#ef4444",
  layoutlmv3: "#f59e0b",
};

const ABBREV: Record<string, string> = {
  rd: "road", "rd.": "road",
  ave: "avenue", "ave.": "avenue", av: "avenue",
  st: "street", "st.": "street",
  blvd: "boulevard", "blvd.": "boulevard",
  dr: "drive", "dr.": "drive",
  ln: "lane", "ln.": "lane",
  hwy: "highway", "hwy.": "highway",
  ct: "court", "ct.": "court",
};

const norm = (s: string) =>
  (s || "")
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[\u00A0]/g, " ")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
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

/** Local fuzzy by value (always available, so the UI reacts instantly on click) */
function autoLocateByValue(valueRaw: string, allTokens: TokenBox[], maxWindow = 16) {
  const raw = (valueRaw || "").trim();
  if (!raw) return null;

  const looksNumeric = /^[\s\-$€£₹,.\d/]+$/.test(raw);
  const words = looksNumeric ? [normNum(raw)] : norm(raw).split(" ").map((w) => ABBREV[w] ?? w);

  const byPage = new Map<number, TokenBox[]>();
  for (const t of allTokens) {
    (byPage.get(t.page) || byPage.set(t.page, []).get(t.page)!).push(t);
  }
  byPage.forEach((arr) => arr.sort((a, b) => (a.y0 === b.y0 ? a.x0 - b.x0 : a.y0 - b.y0)));

  let best: { score: number; page: number; span: TokenBox[] } | null = null;

  function scoreSpan(span: TokenBox[]) {
    const txt = span
      .map((t) => (t.text || ""))
      .join(" ")
      .toLowerCase()
      .normalize("NFKC")
      .replace(/[^\p{L}\p{N}\s]/gu, " ");
    const spanWords = txt.split(/\s+/).filter(Boolean).map((w) => ABBREV[w] ?? w);

    if (looksNumeric) {
      const fuzz = levRatio(spanWords.join(" "), words.join(" "));
      return fuzz - Math.min(0.25, linePenalty(span) * 0.12);
    }

    let covered = 0;
    let j = 0;
    for (let i = 0; i < words.length && j < spanWords.length; ) {
      if (words[i] === spanWords[j] || levRatio(words[i], spanWords[j]) >= 0.8) {
        covered++; i++; j++;
      } else {
        j++;
      }
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

        if (span.length === 1 && !looksNumeric) {
          const first = words[0];
          const tokenN = token.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, "");
          if (levRatio(first, tokenN) < 0.6) continue;
        }

        const s = scoreSpan(span);
        if (!best || s > best.score) best = { score: s, page: pg, span: [...span] };
      }
    }
  });

  if (!best) return null;
  const rect = unionRect(best.span);
  return { page: best.page, rect, score: best.score };
}

/** Token refinement: clip union to tokens whose centers fall inside */
function refineWithTokens(
  union: { page: number; x0: number; y0: number; x1: number; y1: number },
  pageTokens: TokenBox[]
) {
  const inBox = (t: TokenBox) => {
    const cx = (t.x0 + t.x1) / 2, cy = (t.y0 + t.y1) / 2;
    const minx = Math.min(union.x0, union.x1), maxx = Math.max(union.x0, union.x1);
    const miny = Math.min(union.y0, union.y1), maxy = Math.max(union.y0, union.y1);
    return cx >= minx && cx <= maxx && cy >= miny && cy <= maxy;
  };
  const pool = pageTokens.filter(inBox);
  if (!pool.length) return union;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const t of pool) {
    x0 = Math.min(x0, t.x0); y0 = Math.min(y0, t.y0);
    x1 = Math.max(x1, t.x1); y1 = Math.max(y1, t.y1);
  }
  return { page: union.page, x0, y0, x1, y1 };
}

/* Flatten ECM-like JSON to rows */
function normalizeRects(x: any): KVRect[] | undefined {
  if (!x) return undefined;
  const arr = Array.isArray(x) ? x : [x];
  const out: KVRect[] = [];
  for (const b of arr) {
    if (b && Number.isFinite(+b.page)) {
      if (b.x != null && b.y != null && b.w != null && b.h != null) {
        out.push({ page: +b.page, x0: +b.x, y0: +b.y, x1: +b.x + +b.w, y1: +b.y + +b.h });
      } else {
        out.push({ page: +b.page, x0: +b.x0, y0: +b.y0, x1: +b.x1, y1: +b.y1 });
      }
    }
  }
  return out.length ? out : undefined;
}
function tryStr(v: any) { try { return typeof v === "string" ? v : JSON.stringify(v); } catch { return String(v); } }
type FlatKV = { key: string; value: string; rects?: KVRect[] };
function flattenJson(obj: any, prefix = ""): FlatKV[] {
  const rows: FlatKV[] = [];
  const push = (k: string, v: any) => rows.push({ key: k, value: v == null ? "" : String(v) });
  if (obj === null || obj === undefined) return [{ key: prefix || "(null)", value: "" }];

  if (Array.isArray(obj)) {
    obj.forEach((v, i) => {
      const path = prefix ? `${prefix}[${i}]` : `[${i}]`;
      if (typeof v === "object" && v !== null) rows.push(...flattenJson(v, path));
      else push(path, v);
    });
    return rows;
  }
  if (typeof obj !== "object") return [{ key: prefix || "(value)", value: String(obj) }];

  for (const k of Object.keys(obj)) {
    const path = prefix ? `${prefix}.${k}` : k;
    const v = obj[k];
    if (v && typeof v === "object" && !Array.isArray(v)) {
      const rects = normalizeRects(v.bboxes || v.boxes || v.bbox || v.rects);
      if ("value" in v || rects) {
        rows.push({ key: path, value: "value" in v ? String(v.value ?? "") : tryStr(v), rects });
      } else {
        rows.push(...flattenJson(v, path));
      }
    } else if (Array.isArray(v)) {
      rows.push(...flattenJson(v, path));
    } else {
      push(path, v);
    }
  }
  return rows;
}

/* ========================= Component ========================== */

export default function FieldLevelEditor() {
  // doc + page
  const [docUrl, setDocUrl] = useState("");
  const [docId, setDocId] = useState("");
  const [meta, setMeta] = useState<{ w: number; h: number }[]>([]);
  const [page, setPage] = useState(1);

  // tokens (orange)
  const [tokens, setTokens] = useState<TokenBox[]>([]);
  const tokensPage = useMemo(() => tokens.filter((t) => t.page === page), [tokens, page]);

  // left table (extraction)
  const [rows, setRows] = useState<FieldRow[]>([]);
  const [focusedKey, setFocusedKey] = useState("");

  // overlays
  const [rect, setRect] = useState<EditRect | null>(null);
  const [overlays, setOverlays] = useState<{label:string;color:string;rect:EditRect|null}[]>([]);
  const [showBoxes, setShowBoxes] = useState(false);
  const [lastCrop, setLastCrop] = useState<{ url?: string; text?: string } | null>(null);

  // zoom
  const [zoom, setZoom] = useState(1);

  // Distil results cache (optional)
  const [distil, setDistil] = useState<DistilField[]>([]);
  const DISTIL_OK = 0.45;

  /* -------- Uploads -------- */
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
      setLastCrop(null);
    } finally {
      (e.target as HTMLInputElement).value = "";
    }
  }

  async function onUploadEcm(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const parsed = JSON.parse(await f.text()) as AnyJson;
      const flat = flattenJson(parsed);
      const fr: FieldRow[] = flat.map((kv) => ({ key: kv.key, value: kv.value, rects: kv.rects }));
      setRows(fr);
      setFocusedKey("");
      setRect(null);
      setOverlays([]);
    } catch {
      alert("Invalid ECM JSON");
    } finally {
      (e.target as HTMLInputElement).value = "";
    }
  }

  /* -------- Paste /data/{doc}/original.pdf -------- */
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
      setLastCrop(null);
    })();
  }, [docUrl]);

  /* -------- Distil Runner (best-effort; no hard dependency) -------- */
  async function runDistilNow() {
    if (!docId || !rows.length) return;
    const specs = rows.map((r) => {
      const k = r.key.toLowerCase();
      const type =
        k.includes("zip") ? "zip" :
        k.includes("date") ? "date" :
        k.includes("amount") || k.includes("total") ? "amount" : "text";
      const label = r.key.replace(/\./g, " ").replace(/\[\d+\]/g, " ").replace(/\s+/g, " ").trim();
      return { key: r.key, label, type };
    });
    try {
      const res = await distilExtract(docId, specs, 12, 260);
      setDistil(res.fields || []);
    } catch (e) {
      console.warn("distil failed", e);
      setDistil([]);
    }
  }

  useEffect(() => {
    if (docId && rows.length) runDistilNow();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docId, rows.length]);

  /* -------- Row click: instant local fuzzy → server 5-method overlays -------- */
  function onRowClick(r: FieldRow) {
    console.log("[UI] row-click", r.key, r.value?.slice?.(0, 120));
    setFocusedKey(r.key);

    // Local fuzzy (instant)
    const hit = autoLocateByValue(r.value, tokens);
    if (hit) {
      const refined = refineWithTokens(
        { page: hit.page, ...hit.rect },
        tokens.filter((t) => t.page === hit.page)
      );
      setPage(hit.page);
      setRect(refined);
      // seed overlays with just fuzzy while server responds
      setOverlays([{ label: "fuzzy", color: COLORS.fuzzy, rect: refined }]);
    } else {
      setRect(null);
      setOverlays([]);
    }

    // Server 5-method overlays (best-effort)
    (async () => {
      try {
        if (!docId) return;
        const res = await matchField(docId, r.key, r.value);
        console.log("[UI] /lasso/match/field ->", res);

        const pick = (m: any): EditRect | null =>
          !m ? null : ({ page: m.page, x0: m.rect.x0, y0: m.rect.y0, x1: m.rect.x1, y1: m.rect.y1 });

        const ovs = (["fuzzy","tfidf","minilm","distilbert","layoutlmv3"] as const)
          .map((k) => ({ label: k, color: COLORS[k], rect: pick((res as any).methods?.[k]) }))
          .filter(Boolean) as {label:string;color:string;rect:EditRect|null}[];

        // If we didn't have a pink rect yet, jump to whichever overlay has a page
        if (!hit) {
          const pg = ovs.find(o => o.rect)?.rect?.page;
          if (pg) setPage(pg);
        }
        setOverlays(ovs);
      } catch (e) {
        console.warn("matchField failed", e);
      }
    })();
  }

  /* -------- Lasso/move/resize → OCR preview + update KV -------- */
  async function onRectCommitted(rr: EditRect) {
    if (!focusedKey) {
      console.log("[UI] onRectCommit skipped: no focused key");
      return;
    }
    try {
      if (docId) {
        const res = await ocrPreview(docId, rr.page, rr);
        const text = (res?.text || "").trim();
        setLastCrop({ url: res?.crop_url, text });

        // Update KV list inline for the focused key
        setRows((prev) =>
          prev.map((row) =>
            row.key === focusedKey
              ? { ...row, value: text, rects: [{ page: rr.page, x0: rr.x0, y0: rr.y0, x1: rr.x1, y1: rr.y1 }] }
              : row
          )
        );
      }
    } catch (e) {
      console.warn("ocrPreview failed", e);
    }
  }

  const serverW = meta[page - 1]?.w || 1;
  const serverH = meta[page - 1]?.h || 1;

  /* ========================= Render ========================== */

  return (
    <div className="workbench">
      <div className="wb-toolbar" style={{ gap: 8 }}>
        <span style={{ fontWeight: 600 }}>Choose:</span>

        <label className="btn">
          <input type="file" accept="application/pdf" onChange={onUploadPdf} style={{ display: "none" }} />
          PDF
        </label>

        <label className="btn">
          <input type="file" accept="application/json" onChange={onUploadEcm} style={{ display: "none" }} />
          ECM JSON
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

        <button onClick={runDistilNow} disabled={!docId || !rows.length}>Refresh Distil</button>

        <span className="spacer" />

        <div className="toolbar-inline" style={{ gap: 6 }}>
          {Object.entries(COLORS).map(([k, c]) => (
            <span key={k} style={{ display:"inline-flex", alignItems:"center", gap:6, fontSize:12 }}>
              <span style={{ width:12, height:12, background:c, border:`1px solid ${c}`, display:"inline-block", opacity:0.7 }} />
              {k}
            </span>
          ))}
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
        {/* LEFT 30% — KV table */}
        <div className="wb-left" style={{ flexBasis: "30%", flexGrow: 0, flexShrink: 0, overflow: "auto" }}>
          <div className="section-title">Extraction</div>
          {!rows.length ? (
            <div className="placeholder">Upload ECM JSON to see fields.</div>
          ) : (
            <table>
              <thead>
                <tr><th style={{ width: "42%" }}>Key</th><th>Value</th></tr>
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
