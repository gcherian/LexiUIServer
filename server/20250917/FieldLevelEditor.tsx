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
  locateAll,
  type LocateResp,
} from "../../../lib/api";

type KVRect = { page:number; x0:number; y0:number; x1:number; y1:number };
type FieldRow = { key:string; value:string; rects?:KVRect[] };
type AnyJson = Record<string, any>;

const COLORS: Record<string,string> = {
  autolocate:  "#ec4899",
  tfidf:       "#f59e0b",
  minilm:      "#10b981",
  distilbert:  "#3b82f6",
  layoutlmv3:  "#a855f7",
};

export default function FieldLevelEditor() {
  const [docUrl, setDocUrl] = useState("");
  const [docId, setDocId] = useState("");
  const [meta, setMeta] = useState<{ w:number; h:number }[]>([]);
  const [page, setPage] = useState(1);

  const [tokens, setTokens] = useState<TokenBox[]>([]);
  const tokensPage = useMemo(() => tokens.filter(t => t.page === page), [tokens, page]);

  const [rows, setRows] = useState<FieldRow[]>([]);
  const [focusedKey, setFocusedKey] = useState("");

  const [rect, setRect] = useState<EditRect | null>(null);
  const [showBoxes, setShowBoxes] = useState(false);
  const [zoom, setZoom] = useState(1);

  const [overlays, setOverlays] = useState<
    Array<{page:number;x0:number;y0:number;x1:number;y1:number;color:string;label:string}>
  >([]);

  /* -------- uploads -------- */
  async function onUploadPdf(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if (!f) return;
    try {
      const res = await uploadPdf(f);
      setDocId(res.doc_id);
      setDocUrl(res.annotated_tokens_url);
      const m = await getMeta(res.doc_id);
      setMeta(m.pages.map(p => ({ w: p.width, h: p.height })));
      setTokens((await getBoxes(res.doc_id)) as any);
      setPage(1); setRect(null); setOverlays([]);
    } finally { (e.target as HTMLInputElement).value = ""; }
  }
  async function onUploadEcm(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if (!f) return;
    try {
      const parsed = JSON.parse(await f.text()) as AnyJson;
      const flat = flattenJson(parsed);
      const fr: FieldRow[] = flat.map(kv => ({ key: kv.key, value: kv.value, rects: kv.rects }));
      setRows(fr); setFocusedKey(""); setRect(null); setOverlays([]);
    } catch { alert("Invalid ECM JSON"); }
    finally { (e.target as HTMLInputElement).value = ""; }
  }

  useEffect(() => {
    const id = docIdFromUrl(docUrl);
    if (!id) return;
    (async () => {
      setDocId(id);
      const m = await getMeta(id);
      setMeta(m.pages.map(p => ({ w:p.width, h:p.height })));
      setTokens((await getBoxes(id)) as any);
      setPage(1); setRect(null); setOverlays([]);
    })();
  }, [docUrl]);

  /* -------- row click -> draw five boxes -------- */
  async function onRowClick(r: FieldRow) {
    setFocusedKey(r.key);
    setOverlays([]);
    if (!docId) return;

    try {
      const resp: LocateResp = await locateAll(docId, r.key, r.value, 12);
      const hits = resp.hits;
      const ov: Array<{page:number;x0:number;y0:number;x1:number;y1:number;color:string;label:string}> = [];
      (["autolocate","tfidf","minilm","distilbert","layoutlmv3"] as const).forEach(m => {
        const h = (hits as any)[m];
        if (h && h.page != null && h.rect) {
          ov.push({ page: h.page, ...h.rect, color: COLORS[m], label: m });
        }
      });
      setOverlays(ov);
      if (ov.length) setPage(ov[0].page);

      const order = ["autolocate","tfidf","minilm","distilbert","layoutlmv3"] as const;
      let best:any = null;
      for (const m of order) if ((hits as any)[m]) { best = (hits as any)[m]; break; }
      if (best) setRect({ page: best.page, x0: best.rect.x0, y0: best.rect.y0, x1: best.rect.x1, y1: best.rect.y1 });
      else setRect(null);
    } catch (e) {
      console.warn("/lasso/locate failed", e);
      setOverlays([]);
    }
  }

  /* -------- lasso commit -> OCR preview & update KV -------- */
  async function onRectCommitted(rr: EditRect) {
    if (!focusedKey) return;
    try {
      if (docId) {
        const res = await ocrPreview(docId, rr.page, rr);
        const text = (res?.text || "").trim();
        setRows(prev =>
          prev.map(row =>
            row.key === focusedKey
              ? { ...row, value: text, rects: [{ page: rr.page, x0: rr.x0, y0: rr.y0, x1: rr.x1, y1: rr.y1 }] }
              : row
          )
        );
      }
    } catch {}
  }

  const serverW = meta[page - 1]?.w || 1;
  const serverH = meta[page - 1]?.h || 1;

  return (
    <div className="workbench">
      <div className="wb-toolbar" style={{ gap: 8 }}>
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

        {/* token boxes toggle */}
        <label className={showBoxes ? "btn toggle active" : "btn toggle"} style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={showBoxes} onChange={() => setShowBoxes((v) => !v)} /> Boxes
        </label>

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
        {/* LEFT */}
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
                    <tr key={r.key + ":" + i} onClick={() => onRowClick(r)}
                        style={focused ? { outline: "2px solid #ec4899", outlineOffset: -2 } : undefined}>
                      <td><code>{r.key}</code></td>
                      <td style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.value}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* RIGHT */}
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
                showTokenBoxes={showBoxes}
                editable={true}
                onRectChange={setRect}
                onRectCommit={onRectCommitted}
                zoom={zoom}
                overlayRects={overlays.filter(o => o.page === page)}
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

/* ---------- helpers: flatten JSON (unchanged) ---------- */
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
      if ("value" in v || rects) rows.push({ key: path, value: "value" in v ? String(v.value ?? "") : tryStr(v), rects });
      else rows.push(...flattenJson(v, path));
    } else if (Array.isArray(v)) {
      rows.push(...flattenJson(v, path));
    } else push(path, v);
  }
  return rows;
}
