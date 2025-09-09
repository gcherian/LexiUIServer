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

type BoxLike = { page:number; x0:number; y0:number; x1:number; y1:number; text?:string };

function isEditableForCatalogKey(cat: PromCatalog | null, key: string): boolean {
  if (!cat) return true;
  const f = cat.fields.find(x => x.key === key);
  // Heuristic: allow editing for strings unless enum is provided (assume controlled)
  if (!f) return true;
  if (f.enum && f.enum.length > 0) return false;
  return (f.type || "string") === "string";
}

export default function FieldLevelEditor() {
  // document
  const [docUrl, setDocUrl] = useState("");
  const [docId, setDocId] = useState("");
  const [meta, setMeta] = useState<{w:number;h:number}[]>([]);
  const [page, setPage] = useState(1);
  const [tokens, setTokens] = useState<BoxLike[]>([]); // word boxes from /boxes
  const [loading, setLoading] = useState(false);

  // right-pane editor state
  const [rect, setRect] = useState<EditRect | null>(null); // current pink rect (server coords)
  const [showBoxes, setShowBoxes] = useState(true);

  // left pane (fields & prom)
  const [proms, setProms] = useState<Array<{doctype:string; file:string}>>([]);
  const [doctype, setDoctypeSel] = useState("");
  const [catalog, setCatalog] = useState<PromCatalog | null>(null);
  const [fields, setFields] = useState<FieldDocState | null>(null);
  const [focusedKey, setFocusedKey] = useState<string>("");

  // ---------- upload/paste ----------
  async function onUpload(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0]; if (!f) return;
    setLoading(true);
    try {
      const res = await uploadPdf(f);
      setDocUrl(res.annotated_tokens_url);
      setDocId(res.doc_id);
      const m = await getMeta(res.doc_id);
      setMeta(m.pages.map(p => ({w:p.width, h:p.height})));
      const b = await getBoxes(res.doc_id);
      setTokens(b as any);
      setPage(1); setFields(null); setCatalog(null); setDoctypeSel(""); setRect(null); setFocusedKey("");
    } finally {
      setLoading(false);
      (ev.target as HTMLInputElement).value = "";
    }
  }
  useEffect(() => { // support pasted /data/{doc_id}/original.pdf
    const id = docIdFromUrl(docUrl);
    if (!id) return;
    (async () => {
      setLoading(true);
      try {
        setDocId(id);
        const m = await getMeta(id);
        setMeta(m.pages.map(p => ({w:p.width, h:p.height})));
        const b = await getBoxes(id);
        setTokens(b as any);
        setPage(1); setFields(null); setCatalog(null); setDoctypeSel(""); setRect(null); setFocusedKey("");
      } finally { setLoading(false); }
    })();
  }, [docUrl]);

  // proms
  useEffect(() => { (async () => { try { setProms(await listProms()); } catch {} })(); }, []);
  async function onSelectDoctype(dt: string) {
    setDoctypeSel(dt);
    if (!docId) return;
    await setDoctype(docId, dt);
    try {
      const st = await getFields(docId);
      setFields(st);
    } catch {
      // seed later via Extract
      setFields({ doc_id: docId, doctype: dt, fields: [], audit: [] });
    }
    try { setCatalog(await getProm(dt)); } catch { setCatalog(null); }
  }

  // extract (mock ECM)
  async function onExtract() {
    if (!docId || !doctype) return;
    setFields(await ecmExtract(docId, doctype));
  }

  // change focus -> jump page & show saved bbox as pink rect
  function focusKey(k: string) {
    setFocusedKey(k);
    const f = fields?.fields.find(x => x.key === k);
    const b = (f as any)?.bbox;
    if (b && typeof b.page === "number") {
      setPage(b.page);
      setRect({ page: b.page, x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1 });
    } else {
      setRect(null);
    }
  }

  // save field value & bbox
  async function saveBinding(key: string, rr: EditRect, newText?: string) {
    if (!docId || !key) return;
    // If server bindField already OCRs via /bind, prefer that. We still preview with /lasso/lasso in the canvas.
    const st = await bindField(docId, key, rr.page, rr);
    setFields(st);
  }

  // when user finishes drag/resize/draw
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
    // live preview + save
    try {
      const res = await ocrPreview(docId, rr.page, rr);
      // Update value immediately in left panel for responsiveness
      setFields(prev => prev ? {
        ...prev,
        fields: prev.fields.map(f => f.key === focusedKey
          ? { ...f, value: (res?.text || "").trim(), source: "ocr", confidence: 0.8,
              bbox: { page: rr.page, x0: rr.x0, y0: rr.y0, x1: rr.x1, y1: rr.y1 } }
          : f)
      } : prev);
      // Persist via /bind
      await saveBinding(focusedKey, rr, (res?.text || "").trim());
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
        <input className="input" placeholder="Paste /data/{doc_id}/original.pdf" value={docUrl} onChange={e=>setDocUrl(e.target.value)} />
        <label className={showBoxes ? "btn toggle active" : "btn toggle"} style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={showBoxes} onChange={() => setShowBoxes(v=>!v)} /> Boxes
        </label>
        <span className="spacer" />
        <span className="muted">API: {API}</span>
      </div>

      <div className="wb-split">
        {/* LEFT: field list w/ statuses */}
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
                    const focused = f.key === focusedKey;
                    const editable = isEditableForCatalogKey(catalog, f.key);
                    return (
                      <tr key={(f.key||"k")+":"+idx} onClick={() => focusKey(f.key)}
                          style={focused ? { outline:"2px solid #ec4899", outlineOffset:-2 } : undefined}>
                        <td><code>{f.key}</code></td>
                        <td>
                          <input
                            value={f.value || ""}
                            onChange={(e)=>setFields(s=>s?{...s,fields:s.fields.map(x=>x.key===f.key?{...x,value:e.target.value,source:"user"}:x)}:s)}
                            onBlur={(e)=>fields && putFields(fields.doc_id, fields)}
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
            <div className="hint">Select a field to edit its bounding box on the PDF.</div>
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
                onRectChange={(r)=>setRect(r)}
                onRectCommit={onRectCommitted}
              />
              <div className="hint">
                Pink box = current field. Drag edges to resize (snaps to words). Drag inside to move.
                Release to OCR & save.
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