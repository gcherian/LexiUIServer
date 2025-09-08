import { useEffect, useMemo, useState } from "react";
import {
  API,
  uploadPdf,
  getMetaByDocId,
  docUrlFromId,
  guessDocIdFromUrl,
  listPromDoctypes,
  getPromCatalog,
  setDoctype,
  ecmExtract,
  listFields,
  saveFieldState,
  tokenSearch,
  type FieldState,
  type PromCatalog,
  type MetaResp,
  type Box,
} from "../../lib/api";
import PdfCanvasWithBoxes from "./PdfCanvasWithBoxes";

function useQuery() {
  const [q] = useState(() => new URLSearchParams(window.location.search));
  return q;
}

export default function OcrWorkbench() {
  const query = useQuery();

  const [status, setStatus] = useState<"idle"|"loading"|"ready"|"error">("idle");
  const [error, setError] = useState<string | null>(null);

  const [docId, setDocId] = useState<string>("");
  const [docUrl, setDocUrl] = useState<string>(query.get("doc_url") || "");
  const [meta, setMeta] = useState<MetaResp | null>(null);

  const [proms, setProms] = useState<Array<{ doctype: string; file: string }>>([]);
  const [catalog, setCatalog] = useState<PromCatalog | null>(null);
  const [doctype, setDoctypeLocal] = useState<string>("");

  const [fields, setFields] = useState<FieldState[]>([]);
  const [searchQ, setSearchQ] = useState<string>("invoice");
  const [page, setPage] = useState<number>(1);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);

  const resolvedDocId = useMemo(() => docId || guessDocIdFromUrl(docUrl) || "", [docId, docUrl]);

  useEffect(() => { (async () => { try { setProms(await listPromDoctypes()); } catch {} })(); }, []);

  useEffect(() => {
    if (!docUrl) return;
    const maybe = guessDocIdFromUrl(docUrl);
    if (maybe && !docId) setDocId(maybe);
  }, [docUrl]); // eslint-disable-line

  useEffect(() => {
    (async () => {
      if (!resolvedDocId) return;
      setStatus("loading"); setError(null);
      try {
        const m = await getMetaByDocId(resolvedDocId);
        setMeta(m);
        const url = docUrlFromId(resolvedDocId);
        setDocUrl(url);
        const fs = await listFields({ doc_url: url });
        setFields(fs || []);
        setStatus("ready");
      } catch (e: any) {
        setStatus("error"); setError(String(e?.message || e));
      }
    })();
  }, [resolvedDocId]); // eslint-disable-line

  async function onUpload(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f) return;
    setStatus("loading"); setError(null);
    try {
      const res = await uploadPdf(f);
      setDocId(res.doc_id);
      setDocUrl(res.annotated_tokens_url);
      const m = await getMetaByDocId(res.doc_id);
      setMeta(m);
      const fs = await listFields({ doc_url: res.annotated_tokens_url });
      setFields(fs || []); setPage(1); setShowBoxes(true);
      setStatus("ready");
    } catch (e: any) {
      setStatus("error"); setError(String(e?.message || e));
    } finally {
      (ev.target as HTMLInputElement).value = "";
    }
  }

  async function onChooseDoctype(dt: string) {
    try {
      setDoctypeLocal(dt);
      const cat = await getPromCatalog(dt); setCatalog(cat);
      if (resolvedDocId) {
        await setDoctype(resolvedDocId, dt);
        // <-- populate fields via ECM mock so users see something immediately
        const populated = await ecmExtract(resolvedDocId, dt);
        setFields(populated.fields || []);
      }
    } catch {}
  }

  // click a box to create/assign a field
  function onBoxClick(b: Box) {
    const id = b.id || `${b.page}:${b.x0}:${b.y0}`;
    const f: FieldState = {
      id, name: (b.label || "field").toLowerCase().replace(/\s+/g, "_"),
      value: "", page: b.page,
      bbox: { x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1, page: b.page },
      confidence: b.confidence ?? 0.8, source: "bbox",
    };
    setFields(prev => [f, ...prev]);
  }

  async function onSaveField(idx: number) {
    if (!docUrl) return;
    const f = fields[idx]; if (!f) return;
    const payload: FieldState = { ...f };
    if (!payload.key && payload.name) payload.key = payload.name;
    await saveFieldState({ doc_url: docUrl, field: payload });
  }

  async function onSearch() {
    if (!resolvedDocId || !searchQ) return;
    const hits = await tokenSearch(resolvedDocId, searchQ, 50);
    if (hits.length) setPage(hits[0].page); // jump to the best page
  }

  const pages = meta?.pages?.length ?? 0;

  return (
    <div className="workbench">
      <div className="wb-toolbar">
        <input type="file" accept="application/pdf" onChange={onUpload} />
        <input
          className="input"
          placeholder="Paste /data/{doc_id}/original.pdf or any PDF URL"
          value={docUrl}
          onChange={(e) => setDocUrl(e.target.value)}
        />
        <label className={showBoxes ? "btn toggle active" : "btn toggle"} style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={showBoxes} onChange={() => setShowBoxes(v=>!v)} /> Boxes
        </label>
        <span style={{ marginLeft: "auto" }}>API: <code>{API}</code></span>
      </div>

      {status === "error" && <div style={{ color: "crimson" }}>Error: {error}</div>}

      <div className="wb-split">
        {/* LEFT — PDF + boxes + search */}
        <div className="wb-left">
          {docUrl ? (
            <PdfCanvasWithBoxes
              docUrl={docUrl}
              page={page}
              showBoxes={showBoxes}
              onPageChange={(p)=>setPage(p)}
              onReady={()=>{}}
              onBoxClick={onBoxClick}
            />
          ) : (
            <div className="placeholder">Upload or paste a PDF URL to begin.</div>
          )}

          <div className="row" style={{ marginTop: 8 }}>
            <label>Search</label>
            <div style={{ display: "flex", gap: 8 }}>
              <input
                placeholder="Search tokens (e.g., 'invoice total')"
                value={searchQ}
                onChange={(e) => setSearchQ(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && onSearch()}
              />
              <button onClick={onSearch}>Search</button>
              {docUrl && (
                <a className="btn-link" href={`/bbox_workbench?doc_url=${encodeURIComponent(docUrl)}`} style={{ marginLeft: "auto" }}>
                  Open BBox Workbench →
                </a>
              )}
            </div>
          </div>
          <div className="meta-hint">
            {resolvedDocId ? <>Document: <code>{resolvedDocId}</code> • Pages: {pages || "?"}</> : null}
          </div>
        </div>

        {/* RIGHT — doctype + fields */}
        <div className="wb-right">
          <div className="row">
            <label>Doctype</label>
            <select value={doctype} onChange={(e) => onChooseDoctype(e.target.value)} disabled={!resolvedDocId}>
              <option value="">(select)</option>
              {proms.map((p) => (<option key={p.doctype} value={p.doctype}>{p.doctype}</option>))}
            </select>
          </div>
          {catalog && (
            <div className="row">
              <label>Catalog</label>
              <div className="muted">{catalog.doctype} v{catalog.version}</div>
            </div>
          )}

          <div style={{ marginTop: 12 }}>
            <div className="section-title">Fields</div>
            {fields.length === 0 ? (
              <div className="placeholder">No fields yet. Pick a doctype to pull ECM values or click a box to start a field.</div>
            ) : (
              <div className="field-list">
                {fields.map((f, i) => (
                  <div className="field-row" key={(f.key || f.name || "f")+":"+i}>
                    <input
                      className="mono"
                      value={f.name || f.key || ""}
                      onChange={(e)=>setFields(prev=>prev.map((x,idx)=>idx===i?{...x, name:e.target.value, key:e.target.value}:x))}
                    />
                    <input
                      value={f.value || ""}
                      onChange={(e)=>setFields(prev=>prev.map((x,idx)=>idx===i?{...x, value:e.target.value}:x))}
                    />
                    <button onClick={()=>onSaveField(i)}>Save</button>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="hint">Tip: Clicking a box on the left seeds a field with its bounding box.</div>
        </div>
      </div>
    </div>
  );
}