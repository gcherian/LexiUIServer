// File: src/tsp4/components/lasso/FieldLevelEditor.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import "../ocr.css";
import PdfEditCanvas, { type EditRect, type TokenBox } from "./PdfEditCanvas";
import {
  API,
  uploadPdf,
  getMeta,
  getBoxes,
  docIdFromUrl,
  ocrPreview, // optional: just to show last OCR from the lasso
} from "../../../lib/api";

/* ------------ Types ------------ */
type KVRect = { page: number; x0: number; y0: number; x1: number; y1: number };
type AnyJson = Record<string, any>;

type FieldRow = {
  key: string;
  value: string;
  rects?: KVRect[]; // if JSON gives boxes; else we compute by matching
};

/* ------------ String helpers + matching ------------ */
function norm(s: string): string {
  return (s || "")
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[\u00A0]/g, " ")
    .replace(/[^\p{L}\p{N}\s]/gu, "")
    .replace(/\s+/g, " ")
    .trim();
}
function normKeepDigits(s: string): string {
  return (s || "").toLowerCase().normalize("NFKC").replace(/[,$]/g, "").replace(/\s+/g, " ").trim();
}
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
function unionRect(span: TokenBox[]): { x0: number; y0: number; x1: number; y1: number } {
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
/** Find best window of tokens (across all pages) that matches value. Returns page + rect union. */
function autoLocateByValue(valueRaw: string, allTokens: TokenBox[], maxWindow = 8): { page: number; rect: { x0: number; y0: number; x1: number; y1: number }; score: number } | null {
  const value = valueRaw?.trim();
  if (!value) return null;

  const looksNumeric = /^[\s\-$€£₹,.\d/]+$/.test(value);
  const target = looksNumeric ? normKeepDigits(value) : norm(value);
  if (!target) return null;

  // group tokens by page, order roughly by reading order
  const byPage = new Map<number, TokenBox[]>();
  for (const t of allTokens) {
    const arr = byPage.get(t.page) || [];
    arr.push(t);
    byPage.set(t.page, arr);
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

        const cand = looksNumeric ? normKeepDigits(accum) : norm(accum);
        if (target.length >= 2 && !cand.includes(target.slice(0, 2))) continue;

        const sim = levRatio(cand, target);
        if (sim < 0.6) continue;

        const score = sim - Math.min(0.25, linePenalty(span) * 0.12);
        if (!best || score > best.score) best = { score, page: pg, span: [...span] };
      }
    }
  });

  if (!best) return null;
  const rect = unionRect(best.span);
  return { page: best.page, rect, score: best.score };
}

/* -------------------------------------------
   JSON → rows[] adapter
   Supports common shapes that may include bboxes.
--------------------------------------------*/
function parseExtractionToRows(obj: AnyJson): FieldRow[] {
  if (!obj) return [];
  // Prefer: { fields:[{key, value, bboxes:[{page,x0,y0,x1,y1}]}] }
  if (Array.isArray(obj.fields)) {
    return obj.fields.map((f: AnyJson) => ({
      key: String(f.key ?? ""),
      value: f.value != null ? String(f.value) : "",
      rects: normalizeBoxes(f.bboxes || f.boxes || f.bbox || f.rects),
    }));
  }
  const rows: FieldRow[] = [];
  for (const [k, v] of Object.entries(obj)) {
    if (v == null) { rows.push({ key: k, value: "" }); continue; }
    if (typeof v === "object" && !Array.isArray(v)) {
      const vv = v as AnyJson;
      rows.push({
        key: k,
        value: vv.value != null ? String(vv.value) : stringy(v),
        rects: normalizeBoxes(vv.bboxes || vv.boxes || vv.bbox || vv.rects),
      });
      continue;
    }
    if (Array.isArray(v)) {
      const rects = normalizeBoxes(v);
      if (rects?.length) { rows.push({ key: k, value: "", rects }); continue; }
      rows.push({ key: k, value: v.map(String).join(" ") });
      continue;
    }
    rows.push({ key: k, value: String(v) });
  }
  return rows;
}
function normalizeBoxes(input: any): KVRect[] | undefined {
  if (!input) return undefined;
  const arr = Array.isArray(input) ? input : [input];
  const out: KVRect[] = [];
  for (const b of arr) {
    if (b && Number.isFinite(Number(b.page))) {
      out.push({ page: Number(b.page), x0: Number(b.x0), y0: Number(b.y0), x1: Number(b.x1), y1: Number(b.y1) });
    }
  }
  return out.length ? out : undefined;
}
function stringy(v: any): string {
  try { if (typeof v === "string") return v; return JSON.stringify(v); } catch { return String(v); }
}

/* ------------ Component ------------ */
export default function FieldLevelEditor() {
  // Document/page
  const [docUrl, setDocUrl] = useState("");
  const [docId, setDocId] = useState("");
  const [meta, setMeta] = useState<{ w: number; h: number }[]>([]);
  const [page, setPage] = useState(1);

  // OCR tokens (orange)
  const [tokens, setTokens] = useState<TokenBox[]>([]);
  const tokensThisPage = useMemo(() => tokens.filter((t) => t.page === page), [tokens, page]);

  // Left table (extraction JSON)
  const [rows, setRows] = useState<FieldRow[]>([]);
  const [focusedKey, setFocusedKey] = useState("");

  // Pink highlight on PDF (single union rect, editable via lasso/move/resize)
  const [rect, setRect] = useState<EditRect | null>(null);

  // Last OCR preview (from lasso)
  const [lastCrop, setLastCrop] = useState<{ url?: string; text?: string } | null>(null);

  // Upload PDF
  async function onUploadPdf(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f) return;
    try {
      const res = await uploadPdf(f);
      setDocId(res.doc_id);
      setDocUrl(res.annotated_tokens_url);
      const m = await getMeta(res.doc_id);
      setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
      const b = await getBoxes(res.doc_id);
      setTokens((b as any) || []);
      setPage(1);
      setRect(null);
    } finally {
      (ev.target as HTMLInputElement).value = "";
    }
  }

  // Upload extraction JSON
  async function onUploadJson(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f) return;
    try {
      const text = await f.text();
      const parsed = JSON.parse(text) as AnyJson;
      setRows(parseExtractionToRows(parsed));
      setFocusedKey("");
      setRect(null);
    } catch {
      alert("Invalid JSON file");
    } finally {
      (ev.target as HTMLInputElement).value = "";
    }
  }

  // Paste URL → load meta + tokens
  useEffect(() => {
    const id = docIdFromUrl(docUrl);
    if (!id) return;
    (async () => {
      setDocId(id);
      const m = await getMeta(id);
      setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
      const b = await getBoxes(id);
      setTokens((b as any) || []);
      setPage(1);
      setRect(null);
    })();
  }, [docUrl]);

  // Click a row → link to boxes: prefer JSON boxes; else auto-locate via tokens
  function onRowClick(r: FieldRow) {
    setFocusedKey(r.key);
    const rects = r.rects || [];
    if (rects.length) {
      // Merge same-page rects; jump to first page
      const firstPg = rects[0].page;
      const same = rects.filter((rr) => rr.page === firstPg);
      const union = same.reduce(
        (acc, rr) => ({
          page: acc.page,
          x0: Math.min(acc.x0, rr.x0),
          y0: Math.min(acc.y0, rr.y0),
          x1: Math.max(acc.x1, rr.x1),
          y1: Math.max(acc.y1, rr.y1),
        }),
        { page: firstPg, x0: rects[0].x0, y0: rects[0].y0, x1: rects[0].x1, y1: rects[0].y1 }
      );
      setPage(firstPg);
      setRect(union);
      return;
    }
    // No rects in JSON → try matching tokens
    const hit = autoLocateByValue(r.value, tokens);
    if (hit) {
      setPage(hit.page);
      setRect({ page: hit.page, ...hit.rect });
    } else {
      setRect(null);
    }
  }

  // Lasso commit → update link locally + optional OCR preview
  async function onRectCommitted(rr: EditRect) {
    if (!focusedKey) return;
    // Update the row's rects to this single rect
    setRows((prev) =>
      prev.map((row) =>
        row.key === focusedKey
          ? { ...row, rects: [{ page: rr.page, x0: rr.x0, y0: rr.y0, x1: rr.x1, y1: rr.y1 }] }
          : row
      )
    );
    setRect(rr);

    // Optional OCR preview (non-destructive; does not overwrite JSON value)
    try {
      if (docId) {
        const res = await ocrPreview(docId, rr.page, rr);
        setLastCrop({ url: res?.crop_url, text: (res?.text || "").trim() });
      }
    } catch {
      /* ignore preview errors */
    }
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
        <span className="spacer" />
        <span className="muted">API: {API}</span>
      </div>

      <div className="wb-split" style={{ display: "flex", gap: 12 }}>
        {/* LEFT: 30% — extraction fields */}
        <div className="wb-left" style={{ flexBasis: "30%", flexGrow: 0, flexShrink: 0, overflow: "auto" }}>
          <div className="section-title">Extraction</div>
          {!rows.length ? (
            <div className="placeholder">Upload extraction JSON to see fields.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th style={{ width: "38%" }}>Key</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, idx) => {
                  const focused = r.key === focusedKey;
                  return (
                    <tr
                      key={r.key + ":" + idx}
                      onClick={() => onRowClick(r)}
                      style={focused ? { outline: "2px solid #ec4899", outlineOffset: -2 } : undefined}
                    >
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

        {/* RIGHT: 70% — PDF (stable size) */}
        <div className="wb-right" style={{ flexBasis: "70%", overflow: "auto" }}>
          {docUrl ? (
            <>
              <div className="toolbar-inline">
                <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Prev</button>
                <span className="page-indicator">
                  Page {page} {meta.length ? `/ ${meta.length}` : ""}
                </span>
                <button disabled={meta.length > 0 && page >= meta.length} onClick={() => setPage((p) => p + 1)}>Next</button>
              </div>

              <PdfEditCanvas
                docUrl={docUrl}
                page={page}
                serverW={serverW}
                serverH={serverH}
                tokens={tokensThisPage}
                rect={rect}
                showTokenBoxes={true}
                editable={true}              // lasso + move/resize enabled
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