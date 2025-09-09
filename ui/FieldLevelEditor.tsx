import React, { useEffect, useMemo, useState } from "react";
import "../ocr.css";
import PdfEditCanvas, { type EditRect } from "./PdfEditCanvas";

import {
  API,
  uploadPdf,
  getMeta,
  getBoxes,
  listProms,
  getProm,
  setDoctype,
  getFields,
  ecmExtract,
  putFields,
  bindField,
  ocrPreview,
  docIdFromUrl,
  type FieldDocState,
  type PromCatalog,
} from "../../../lib/api";

type TokenBox = { page:number; x0:number; y0:number; x1:number; y1:number; text?:string };

function isEditableForCatalogKey(cat: PromCatalog | null, key: string): boolean {
  if (!cat) return true;
  const f = cat.fields.find(x => x.key === key);
  if (!f) return true;
  const opts = (f as any)["enum"] as string[] | undefined;
  if (Array.isArray(opts) && opts.length > 0) return false;
  const t = (f as any).type ?? "string";
  return t === "string";
}

/* ---------- VALUE LOCATOR HELPERS (client-side) ---------- */

function norm(s: string): string {
  return (s || "")
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[\u00A0]/g, " ")              // nbsp → space
    .replace(/[^\p{L}\p{N}\s]/gu, "")       // drop punctuation
    .replace(/\s+/g, " ")                   // collapse spaces
    .trim();
}

// Keep numbers/dates recognizable
function normKeepDigits(s: string): string {
  return (s || "")
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[,$]/g, "")         // strip commas/currency
    .replace(/\s+/g, " ")
    .trim();
}

// Simple, fast Levenshtein ratio (0..1)
function levRatio(a: string, b: string): number {
  const m = a.length, n = b.length;
  if (m === 0 && n === 0) return 1;
  const dp = new Array(n + 1);
  for (let j = 0; j <= n; j++) dp[j] = j;
  for (let i = 1; i <= m; i++) {
    let prev = dp[0]; dp[0] = i;
    for (let j = 1; j <= n; j++) {
      const tmp = dp[j];
      dp[j] = Math.min(
        dp[j] + 1,
        dp[j - 1] + 1,
        prev + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
      prev = tmp;
    }
  }
  const dist = dp[n];
  const maxLen = Math.max(1, Math.max(m, n));
  return 1 - dist / maxLen;
}

// Union bbox for a span of tokens (server coords)
function unionRect(span: TokenBox[]): { x0:number; y0:number; x1:number; y1:number } {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const t of span) {
    x0 = Math.min(x0, t.x0); y0 = Math.min(y0, t.y0);
    x1 = Math.max(x1, t.x1); y1 = Math.max(y1, t.y1);
  }
  return { x0: Math.floor(x0), y0: Math.floor(y0), x1: Math.ceil(x1), y1: Math.ceil(y1) };
}

// Linearity penalty: prefer spans that look like one line
function linePenalty(span: TokenBox[]): number {
  if (span.length <= 1) return 0;
  // vertical spread vs median height
  const ys = span.map(t => (t.y0 + t.y1)/2).sort((a,b)=>a-b);
  const yspread = ys[ys.length-1] - ys[0];
  const heights = span.map(t => (t.y1 - t.y0));
  const avgH = heights.reduce((a,b)=>a+b,0)/Math.max(1,heights.length);
  return Math.max(0, yspread - avgH*0.6) / Math.max(1, avgH); // 0 if roughly one line
}

// Try to locate a value across all pages/tokens
function autoLocateByValue(valueRaw: string, allTokens: TokenBox[], maxWindow = 8) {
  const value = valueRaw?.trim();
  if (!value) return null;

  // Choose normalization strategy
  const looksNumeric = /^[\s\-$€£₹,.\d/]+$/.test(value);
  const target = looksNumeric ? normKeepDigits(value) : norm(value);
  if (!target) return null;

  // Group tokens by page to keep spans on one page
  const byPage = new Map<number, TokenBox[]>();
  for (const t of allTokens) {
    if (!byPage.has(t.page)) byPage.set(t.page, []);
    byPage.get(t.page)!.push(t);
  }
  // Sort each page's tokens left-to-right, top-to-bottom (roughly)
  for (const [, arr] of byPage) {
    arr.sort((a, b) => (a.y0 === b.y0 ? a.x0 - b.x0 : a.y0 - b.y0));
  }

  let best: { score: number; page: number; span: TokenBox[] } | null = null;

  for (const [pg, toks] of byPage) {
    const n = toks.length;
    for (let i = 0; i < n; i++) {
      let textAccum = "";
      const span: TokenBox[] = [];
      for (let w = 0; w < maxWindow && i + w < n; w++) {
        const t = toks[i + w];
        const tokenText = (t.text || "").trim();
        if (!tokenText) continue;
        span.push(t);

        // build candidate text with spaces; normalize
        textAccum = (textAccum ? textAccum + " " : "") + tokenText;

        const cand = looksNumeric ? normKeepDigits(textAccum) : norm(textAccum);
        if (!cand) continue;

        // quick filter: if target's first 2 chars aren't in cand, skip early
        if (target.length >= 2 && !cand.includes(target.slice(0, 2))) continue;

        const sim = levRatio(cand, target);
        if (sim < 0.6) continue; // ignore weak matches early

        // penalize multi-line or tall areas a bit
        const penalty = linePenalty(span);
        const score = sim - Math.min(0.25, penalty * 0.12);

        if (!best || score > best.score) {
          best = { score, page: pg, span: [...span] };
        }
      }
    }
  }

  if (!best) return null;
  const rect = unionRect(best.span);
  return { page: best.page, rect, score: best.score };
}
/* ---------- END LOCATOR HELPERS ---------- */

export default function FieldLevelEditor() {
  // document
  const [docUrl, setDocUrl] = useState("");
  const [docId, setDocId] = useState("");
  const [meta, setMeta] = useState<{w:number;h:number}[]>([]);
  const [page, setPage] = useState(1);
  const [tokens, setTokens] = useState<TokenBox[]>([]);
  const [loading, setLoading] = useState(false);

  // editor
  const [rect, setRect] = useState<EditRect | null>(null);
  const [showBoxes, setShowBoxes] = useState(true);

  // PROM / fields
  const [proms, setProms] = useState<Array<{doctype:string; file:string}>>([]);
  const [doctype, setDoctypeSel] = useState("");
  const [catalog, setCatalog] = useState<PromCatalog | null>(null);
  const [fields, setFields] = useState<FieldDocState | null>(null);
  const [focusedKey, setFocusedKey] = useState<string>("");

  // ---------- upload / paste ----------
  async function onUpload(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f) return;
    setLoading(true);
    try {
      const res = await uploadPdf(f);
      await bootstrapFromDocId(res.doc_id, res.annotated_tokens_url);
    } finally {
      setLoading(false);
      (ev.target as HTMLInputElement).value = "";
    }
  }

  async function bootstrapFromDocId(id: string, url: string) {
    setDocUrl(url);
    setDocId(id);
    const m = await getMeta(id);
    setMeta(m.pages.map(p => ({ w: p.width, h: p.height })));
    const b = await getBoxes(id);
    setTokens(b as any);
    setPage(1);
    setFields(null);
    setCatalog(null);
    setDoctypeSel("");
    setRect(null);
    setFocusedKey("");
  }

  useEffect(() => {
    const id = docIdFromUrl(docUrl);
    if (!id) return;
    setLoading(true);
    (async () => {
      try { await bootstrapFromDocId(id, docUrl); }
      finally { setLoading(false); }
    })();
  }, [docUrl]);

  // PROM list
  useEffect(() => { (async () => {
    try { setProms(await listProms()); } catch {}
  })(); }, []);

  // Doctype
  async function onSelectDoctype(dt: string) {
    setDoctypeSel(dt);
    if (!docId) return;
    await setDoctype(docId, dt);

    try {
      const st = await getFields(docId);
      setFields(st);
    } catch {
      setFields({ doc_id: docId, doctype: dt, fields: [], audit: [] });
    }

    try { setCatalog(await getProm(dt)); } catch { setCatalog(null); }
  }

  // Extract via ECM mock
  async function onExtract() {
    if (!docId || !doctype) return;
    const st = await ecmExtract(docId, doctype);
    setFields(st);
  }

  // Focus a field -> show pink box (if bbox saved) OR auto-locate by value
  function focusKey(k: string) {
    setFocusedKey(k);

    const f = fields?.fields.find(x => x.key === k);
    const b = (f as any)?.bbox;

    if (b && Number.isFinite(Number(b.page))) {
      const rr: EditRect = {
        page: Number(b.page),
        x0: Number(b.x0), y0: Number(b.y0), x1: Number(b.x1), y1: Number(b.y1),
      };
      setPage(rr.page || 1);
      setRect(rr);
      return;
    }

    // No bbox yet -> try to auto-locate value in tokens
    const value = (f?.value || "").toString();
    if (value) {
      const found = autoLocateByValue(value, tokens);
      if (found) {
        setPage(found.page);
        const rr: EditRect = { page: found.page, ...found.rect };
        setRect(rr);
        return;
      }
    }

    // Still nothing -> clear; user can draw
    setRect(null);
  }

  // Persist entire FieldDocState (manual edits)
  async function saveAllFields() {
    if (!fields) return;
    const saved = await putFields(fields.doc_id, fields);
    setFields(saved);
  }

  // After drag/resize/draw finishes -> OCR + bind to focused field
  async function onRectCommitted(rr: EditRect) {
    if (!focusedKey) {
      alert("Select a field on the left, then adjust the box.");
      return;
    }
    const editable = isEditableForCatalogKey(catalog, focusedKey);
    if (!editable) {
      alert("This field is read-only.");
      return;
    }

    try {
      const res = await ocrPreview(docId, rr.page, rr);
      const text = (res?.text || "").trim();

      // Update UI immediately
      setFields(prev => prev ? {
        ...prev,
        fields: prev.fields.map(f => f.key === focusedKey
          ? { ...f, value: text, source: "ocr", confidence: 0.8,
              bbox: { page: rr.page, x0: rr.x0, y0: rr.y0, x1: rr.x1, y1: rr.y1 } }
          : f)
      } : prev);

      // Persist to server
      const st = await bindField(docId, focusedKey, rr.page, rr);
      setFields(st);
      setRect(rr);
    } catch (e:any) {
      alert(`OCR failed: ${e?.message || e}`);
    }
  }

  // helpers
  const serverW = meta[page-1]?.w || 1;
  const serverH = meta[page-1]?.h || 1;
  const tokensThisPage = useMemo(() => tokens.filter(t => t.page === page), [tokens, page]);

  return (
    <div className="workbench">
      {/* toolbar */}
      <div className="wb-toolbar">
        <input type="file" accept="application/pdf" onChange={onUpload} />
        <input
          className="input"
          placeholder="Paste /data/{doc_id}/original.pdf"
          value={docUrl}
          onChange={(e)=>setDocUrl(e.target.value)}
        />
        <label className={showBoxes ? "btn toggle active" : "btn toggle"} style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={showBoxes} onChange={()=>setShowBoxes(v=>!v)} /> Boxes
        </label>
        <span className="spacer" />
        <span className="muted">API: {API}</span>
      </div>

      <div className="wb-split">
        {/* LEFT: fields list */}
        <div className="wb-right">
          <div className="row">
            <label>Doctype</label>
            <select value={doctype} onChange={(e)=>onSelectDoctype(e.target.value)} disabled={!docId}>
              <option value="">(select)</option>
              {proms.map(p => <option key={p.doctype} value={p.doctype}>{p.doctype}</option>)}
            </select>
          </div>

          <div className="row">
            <label>Actions</label>
            <div style={{ display:"flex", gap:8 }}>
              <button className="primary" onClick={onExtract} disabled={!doctype || !docId || loading}>Extract</button>
              <button onClick={saveAllFields} disabled={!fields}>Save</button>
            </div>
          </div>

          <div style={{ marginTop:12 }}>
            <div className="section-title">Fields</div>
            {!fields || fields.fields.length === 0 ? (
              <div className="placeholder">Choose a doctype and click <b>Extract</b>.</div>
            ) : (
              <table>
                <thead><tr><th>Key</th><th>Value</th><th>Source</th><th>Conf</th><th>Edit</th></tr></thead>
                <tbody>
                  {fields.fields.map((f, idx) => {
                    const editable = isEditableForCatalogKey(catalog, f.key);
                    const focused = f.key === focusedKey;
                    return (
                      <tr key={(f.key||"k")+":"+idx}
                          onClick={() => focusKey(f.key)}
                          style={focused ? { outline:"2px solid #ec4899", outlineOffset:-2 } : undefined}>
                        <td><code>{f.key}</code></td>
                        <td>
                          <input
                            value={f.value || ""}
                            onChange={(e)=>setFields(s=>s?{...s,fields:s.fields.map(x=>x.key===f.key?{...x,value:e.target.value,source:"user"}:x)}:s)}
                            onBlur={saveAllFields}
                            disabled={!editable}
                          />
                        </td>
                        <td>{f.source || ""}</td>
                        <td>{f.confidence ? f.confidence.toFixed(2) : ""}</td>
                        <td>{editable ? <span className="badge">Editable</span> : <span className="badge warn">Locked</span>}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
            <div className="hint">
              Select a field. On the right, adjust the <span style={{color:"#ec4899"}}>pink box</span> or draw a new one.
              Release to OCR & update the value.
            </div>
          </div>
        </div>

        {/* RIGHT: single pink box editor */}
        <div className="wb-left">
          {docUrl ? (
            <>
              <div className="toolbar-inline">
                <button disabled={page<=1} onClick={()=>setPage(p=>p-1)}>Prev</button>
                <span className="page-indicator">Page {page} {meta.length?`/ ${meta.length}`:""}</span>
                <button disabled={meta.length>0 && page>=meta.length} onClick={()=>setPage(p=>p+1)}>Next</button>
              </div>

              <PdfEditCanvas
                docUrl={docUrl}
                page={page}
                serverW={serverW}
                serverH={serverH}
                tokens={tokensThisPage}
                rect={rect}
                showTokenBoxes={showBoxes}
                editable={!!focusedKey && isEditableForCatalogKey(catalog, focusedKey)}
                onRectChange={setRect}
                onRectCommit={onRectCommitted}
              />

              <div className="hint">
                Drag inside to move. Drag corners/sides to resize. Snaps to word edges.  
                Drawing a new box anywhere will rebind the focused field.
              </div>
            </>
          ) : (
            <div className="placeholder">Upload or paste a PDF URL to begin.</div>
          )}
        </div>
      </div>
    </div>
  );
}