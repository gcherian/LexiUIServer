import React, { useEffect, useMemo, useRef, useState } from "react";
import "../ocr.css";
import PdfEditCanvas, { type EditRect, type TokenBox } from "./PdfEditCanvas";

import {
  API,
  uploadPdf,
  getMeta,
  getBoxes,
  listProms,
  getProm,
  setDoctype,
  getFields,
  putFields,
  ecmExtract,
  bindField,
  ocrPreview,
  docIdFromUrl,
  type FieldDocState,
  type PromCatalog,
} from "../../../lib/api";

/* ----------------------- helpers & types ----------------------- */

type LocateRect = { x0: number; y0: number; x1: number; y1: number };
type LocateHit = { page: number; rect: LocateRect; score: number };
type Candidate = { score: number; page: number; span: TokenBox[] };

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
  const m = a.length,
    n = b.length;
  if (!m && !n) return 1;
  const dp = new Array(n + 1);
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
  let x0 = Infinity,
    y0 = Infinity,
    x1 = -Infinity,
    y1 = -Infinity;
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
  const yspread = ys[ys.length - 1] - ys[0];
  const hs = span.map((t) => t.y1 - t.y0);
  const avg = hs.reduce((a, b) => a + b, 0) / Math.max(1, hs.length);
  return Math.max(0, yspread - avg * 0.6) / Math.max(1, avg);
}
function autoLocateByValue(valueRaw: string, allTokens: TokenBox[], maxWindow = 8): LocateHit | null {
  const value = valueRaw?.trim();
  if (!value) return null;
  const looksNumeric = /^[\s\-$€£₹,.\d/]+$/.test(value);
  const target = looksNumeric ? normKeepDigits(value) : norm(value);
  if (!target) return null;

  const byPage = new Map<number, TokenBox[]>();
  for (const t of allTokens) {
    const arr = byPage.get(t.page) || [];
    arr.push(t);
    byPage.set(t.page, arr);
  }
  byPage.forEach((arr) => arr.sort((a, b) => (a.y0 === b.y0 ? a.x0 - b.x0 : a.y0 - b.y0)));

  let best: Candidate | null = null;

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

function isEditableForCatalogKey(cat: PromCatalog | null, key: string): boolean {
  if (!cat) return true;
  const f = cat.fields.find((x) => x.key === key);
  if (!f) return true;
  const opts = (f as any)["enum"] as string[] | undefined;
  if (Array.isArray(opts) && opts.length > 0) return false;
  const t = (f as any).type ?? "string";
  return t === "string";
}

/* ----------------------- component ----------------------------- */

export default function FieldLevelEditor() {
  // Document/page
  const [docUrl, setDocUrl] = useState("");
  const [docId, setDocId] = useState("");
  const [meta, setMeta] = useState<{ w: number; h: number }[]>([]);
  const [page, setPage] = useState(1);

  const [tokens, setTokens] = useState<TokenBox[]>([]);
  const tokensThisPage = useMemo(() => tokens.filter((t) => t.page === page), [tokens, page]);

  const [showBoxes, setShowBoxes] = useState(true);
  const [lastCrop, setLastCrop] = useState<{ url?: string; text?: string } | null>(null);

  // Rect (pink)
  const [rect, setRect] = useState<EditRect | null>(null);

  // PROM + field state
  const [proms, setProms] = useState<Array<{ doctype: string; file: string }>>([]);
  const [doctype, setDoctypeSel] = useState("");
  const [catalog, setCatalog] = useState<PromCatalog | null>(null);
  const [fields, setFields] = useState<FieldDocState | null>(null);
  const [focusedKey, setFocusedKey] = useState("");

  // Split sizing
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [pdfPct, setPdfPct] = useState(70);
  const draggingSplit = useRef(false);

  // Upload
  async function onUpload(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f) return;
    try {
      const res = await uploadPdf(f);
      await bootstrap(res.doc_id, res.annotated_tokens_url);
    } finally {
      (ev.target as HTMLInputElement).value = "";
    }
  }

  async function bootstrap(id: string, url: string) {
    setDocId(id);
    setDocUrl(url);
    const m = await getMeta(id);
    setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
    const b = await getBoxes(id);
    setTokens(b as any);
    setPage(1);
    setRect(null);
    setFields(null);
    setCatalog(null);
    setDoctypeSel("");
    setFocusedKey("");
    setLastCrop(null);
  }

  // Paste /data/{doc}/original.pdf
  useEffect(() => {
    const id = docIdFromUrl(docUrl);
    if (!id) return;
    (async () => {
      await bootstrap(id, docUrl);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl]);

  // PROM list once
  useEffect(() => {
    (async () => {
      try {
        setProms(await listProms());
      } catch {
        setProms([]);
      }
    })();
  }, []);

  // Select doctype
  async function onSelectDoctype(dt: string) {
    setDoctypeSel(dt);
    if (!docId) return;
    await setDoctype(docId, dt);
    try {
      setFields(await getFields(docId));
    } catch {
      // seed
      const cat = await getProm(dt);
      setCatalog(cat);
      setFields({
        doc_id: docId,
        doctype: dt,
        fields: cat.fields.map((f) => ({ key: f.key, value: "", source: "user", confidence: 0 })),
        audit: [],
      });
      return;
    }
    setCatalog(await getProm(dt));
  }

  // Extract via mock
  async function onExtract() {
    if (!docId || !doctype) return;
    const st = await ecmExtract(docId, doctype);
    setFields(st);
  }

  // Focus a field to show its pink box / auto locate
  function focusKey(k: string) {
    setFocusedKey(k);
    const f = fields?.fields.find((x) => x.key === k);
    const bbox: any = f && (f as any).bbox ? (f as any).bbox : null;

    if (bbox && Number.isFinite(Number(bbox.page))) {
      const rr: EditRect = {
        page: Number(bbox.page),
        x0: Number(bbox.x0),
        y0: Number(bbox.y0),
        x1: Number(bbox.x1),
        y1: Number(bbox.y1),
      };
      setPage(rr.page || 1);
      setRect(rr);
      return;
    }

    const value = (f?.value || "").toString();
    if (value) {
      const found: LocateHit | null = autoLocateByValue(value, tokens);
      if (found) {
        setPage(found.page);
        setRect({ page: found.page, ...found.rect });
        return;
      }
    }
    setRect(null);
  }

  async function saveAllFields() {
    if (!fields) return;
    setFields(await putFields(fields.doc_id, fields));
  }

  // Lasso commit -> OCR -> update + bind
  async function onRectCommitted(rr: EditRect) {
    if (!focusedKey) {
      alert("Select a field on the right first.");
      return;
    }
    if (!isEditableForCatalogKey(catalog, focusedKey)) {
      alert("This field is read-only.");
      return;
    }
    try {
      const res = await ocrPreview(docId, rr.page, rr);
      const text = (res?.text || "").trim();
      setLastCrop({ url: res?.crop_url, text });

      // live update
      setFields((prev) =>
        prev
          ? {
              ...prev,
              fields: prev.fields.map((f) =>
                f.key === focusedKey
                  ? {
                      ...f,
                      value: text,
                      source: "ocr",
                      confidence: 0.8,
                      bbox: { page: rr.page, x0: rr.x0, y0: rr.y0, x1: rr.x1, y1: rr.y1 },
                    }
                  : f
              ),
            }
          : prev
      );

      // persist
      const st = await bindField(docId, focusedKey, rr.page, rr);
      setFields(st);
      setRect(rr);
    } catch (e: any) {
      alert(`OCR failed: ${e?.message || e}`);
    }
  }

  // Split: drag divider
  function onDividerMouseDown(e: React.MouseEvent) {
    e.preventDefault();
    draggingSplit.current = true;
    document.body.style.cursor = "col-resize";
    window.addEventListener("mousemove", onDividerMove);
    window.addEventListener("mouseup", onDividerUp, { once: true });
  }
  function onDividerMove(e: MouseEvent) {
    if (!draggingSplit.current || !containerRef.current) return;
    const r = containerRef.current.getBoundingClientRect();
    const x = e.clientX - r.left;
    const pct = Math.max(45, Math.min(90, (x / r.width) * 100));
    setPdfPct(Math.round(pct));
  }
  function onDividerUp() {
    draggingSplit.current = false;
    document.body.style.cursor = "";
    window.removeEventListener("mousemove", onDividerMove);
  }

  const serverW = meta[page - 1]?.w || 1;
  const serverH = meta[page - 1]?.h || 1;

  return (
    <div className="workbench">
      <div className="wb-toolbar">
        <input type="file" accept="application/pdf" onChange={onUpload} />
        <input
          className="input"
          placeholder="Paste /data/{doc_id}/original.pdf"
          value={docUrl}
          onChange={(e) => setDocUrl(e.target.value || "")}
        />
        <label className={showBoxes ? "btn toggle active" : "btn toggle"} style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={showBoxes} onChange={() => setShowBoxes((v) => !v)} /> Boxes
        </label>
        <span className="spacer" />
        <span className="muted">API: {API}</span>
      </div>

      <div className="wb-split resizable" ref={containerRef}>
        {/* LEFT: PDF */}
        <div className="wb-left" style={{ flexBasis: `${pdfPct}%` }}>
          {docUrl ? (
            <>
              <div className="toolbar-inline">
                <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                  Prev
                </button>
                <span className="page-indicator">
                  Page {page} {meta.length ? `/ ${meta.length}` : ""}
                </span>
                <button disabled={meta.length > 0 && page >= meta.length} onClick={() => setPage((p) => p + 1)}>
                  Next
                </button>
              </div>

              <PdfEditCanvas
                docUrl={docUrl}
                page={page}
                serverW={serverW}
                serverH={serverH}
                tokens={tokensThisPage}
                rect={rect}
                showTokenBoxes={showBoxes}
                editable={!!focusedKey}
                onRectChange={setRect}
                onRectCommit={onRectCommitted}
              />

              {lastCrop?.url && (
                <div className="last-ocr">
                  <img src={lastCrop.url} alt="last-crop" />
                  <div className="caption">
                    <div className="title">Last OCR</div>
                    <pre>{lastCrop.text || ""}</pre>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="placeholder">Upload or paste a PDF URL to begin.</div>
          )}
        </div>

        {/* Divider */}
        <div className="wb-divider" onMouseDown={onDividerMouseDown} title="Drag to resize" />

        {/* RIGHT: Fields */}
        <div className="wb-right" style={{ flexBasis: `${100 - pdfPct}%` }}>
          <div className="row">
            <label>Doctype</label>
            <select value={doctype} onChange={(e) => onSelectDoctype(e.target.value || "")} disabled={!docId}>
              <option value="">(select)</option>
              {proms.map((p) => (
                <option key={p.doctype} value={p.doctype}>
                  {p.doctype}
                </option>
              ))}
            </select>
          </div>

          <div className="row">
            <label>Actions</label>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="primary" onClick={onExtract} disabled={!doctype || !docId}>
                Extract
              </button>
              <button onClick={saveAllFields} disabled={!fields}>
                Save
              </button>
            </div>
          </div>

          <div className="section-title" style={{ marginTop: 12 }}>
            Fields
          </div>

          {!fields || fields.fields.length === 0 ? (
            <div className="placeholder">
              Choose a doctype and click <b>Extract</b>.
            </div>
          ) : (
            <div className="field-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Key</th>
                    <th>Value</th>
                    <th>Source</th>
                    <th>Conf</th>
                    <th>Edit</th>
                  </tr>
                </thead>
                <tbody>
                  {fields.fields.map((f, idx) => {
                    const editable = isEditableForCatalogKey(catalog, f.key);
                    const focused = f.key === focusedKey;
                    return (
                      <tr
                        key={(f.key || "k") + ":" + idx}
                        onClick={() => focusKey(f.key)}
                        style={focused ? { outline: "2px solid #ec4899", outlineOffset: -2 } : undefined}
                      >
                        <td>
                          <code>{f.key}</code>
                        </td>
                        <td>
                          <input
                            value={f.value || ""}
                            onChange={(e) =>
                              setFields((s) =>
                                s
                                  ? {
                                      ...s,
                                      fields: s.fields.map((x) =>
                                        x.key === f.key ? { ...x, value: e.target.value, source: "user" } : x
                                      ),
                                    }
                                  : s
                              )
                            }
                            disabled={!editable}
                            onBlur={saveAllFields}
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
            </div>
          )}

          <div className="hint">Select a field → lasso on the PDF → value updates & saves.</div>
        </div>
      </div>
    </div>
  );
}
