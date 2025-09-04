import { useEffect, useMemo, useState, type ChangeEvent } from "react";
import PdfViewerFS from "./PdfViewerFS";
import {
  uploadDoc, getMeta, getBoxes, search,
  listProms, getProm, setDocType, getFieldState, saveFieldState,
  ecmExtract, bindField, semanticSearch
} from "../../lib/api";
import "./ocr.css";

type Rect = { x0:number; y0:number; x1:number; y1:number };
type Box  = Rect & { page:number; id?:string; label?:string; text?:string };
type Match = { page:number; bbox:Rect; text:string; score:number };

type FieldState = { key:string; value?:string|null; bbox?: (Rect & {page:number})|null; source:string; confidence:number };
type FieldDocState = { doc_id:string; doctype:string; fields: FieldState[]; audit:any[] };

const API = "http://localhost:8000/lasso";

export default function OcrWorkbench(){
  const [doc, setDoc] = useState<{doc_id:string; pdfUrl:string} | null>(null);
  const [meta, setMeta] = useState<{pages:{page:number;width:number;height:number}[]} | null>(null);

  const [page, setPage] = useState(1);
  const [scale, setScale] = useState(1.25);
  const [viewerOpen, setViewerOpen] = useState(false);

  const [doctype, setDoctypeState] = useState("invoice");
  const [doctypeOptions, setDoctypeOptions] = useState<string[]>(["invoice","lease_loan"]);

  const [fstate, setFstate] = useState<FieldDocState|null>(null);
  const [highlights, setHighlights] = useState<Box[]>([]);

  const [filterKV, setFilterKV] = useState("");
  const [filterHL, setFilterHL] = useState("");

  const [bindKey, setBindKey] = useState<string|null>(null);

  useEffect(() => {
    (async () => {
      try {
        const list = await listProms(API);
        const names = (list.doctypes || []).map((d:any)=> d.doctype).filter(Boolean);
        if (names.length) setDoctypeOptions(names);
        await getProm(API, doctype); // sanity check default exists
      } catch {/* keep defaults */}
    })();
  }, []);

  function toAbs(u: string){ return u.startsWith("http") ? u : `http://localhost:8000${u}`; }

async function doSemantic(q: string){
  if(!doc || !q.trim()) return;
  const r = await semanticSearch(API, doc.doc_id, q, 5);
  if (r.results?.length) setPage(r.results[0].page);  // jump to best page
}

  async function doUpload(e: ChangeEvent<HTMLInputElement>){
    const f = e.target.files?.[0]; if(!f) return;
    const res = await uploadDoc(API, f, "tesseract");
    const pdfUrl = toAbs(res.annotated_tokens_url);
    setDoc({ doc_id: res.doc_id, pdfUrl });
    const m = await getMeta(API, res.doc_id); setMeta(m);

    // init fields for selected doctype
    await setDocType(API, res.doc_id, doctype);
    const s = await getFieldState(API, res.doc_id); setFstate(s);

    // prime highlights from OCR tokens (includes .text)
    try { const bx: Box[] = await getBoxes(API, res.doc_id); setHighlights(bx); } catch { setHighlights([]); }

    setPage(1); setScale(1.25);
  }

  async function runECM(){
    if(!doc) return;
    const s = await ecmExtract(API, doc.doc_id, doctype); // auto-inits if missing
    setFstate(s);
  }

  async function saveFields(){
    if(!doc || !fstate) return;
    const saved = await saveFieldState(API, doc.doc_id, fstate);
    setFstate(saved);
    alert("Saved extraction JSON.");
  }

  function startBind(key: string){
    setBindKey(key);
    setViewerOpen(true);
  }

  async function onLasso(pageNum:number, rect:Rect){
    if(!doc || !bindKey || !fstate) return;
    const s = await bindField(API, doc.doc_id, bindKey, pageNum, rect);
    setFstate(s);
    const val = s.fields.find(f=>f.key===bindKey)?.value || "";
    setBindKey(null);
    setViewerOpen(false);
    setPage(pageNum);
    alert(`Bound "${bindKey}" to:\n${val}`);
  }

  async function doSearch(q: string){
    if(!doc || !q.trim()) return;
    const r = await search(API, doc.doc_id, q, 60);
    const m: Match[] = r.matches || [];
    setHighlights(m.map(({page,bbox,text}) => ({ page, ...bbox, text, label: text })));
    if (m.length) setPage(m[0].page);
  }

  async function doSemantic(q: string){
    if(!doc || !q.trim()) return;
    const r = await semanticSearch(API, doc.doc_id, q, 5);
    // Move to best page if found
    if (r.results?.length) setPage(r.results[0].page);
  }

  const kvRows = useMemo(()=>{
    if(!fstate) return [];
    const q = filterKV.trim().toLowerCase();
    return fstate.fields.filter(f =>
      !q || f.key.toLowerCase().includes(q) || (f.value??"").toLowerCase().includes(q)
    );
  }, [fstate, filterKV]);

  const hlRows = useMemo(()=>{
    const q = filterHL.trim().toLowerCase();
    return highlights.filter(b =>
      !q || (b.text??"").toLowerCase().includes(q) || (""+b.page).includes(q)
    );
  }, [highlights, filterHL]);

  const ocrSize = meta?.pages.find(p => p.page === page);

  return (
    <div className="ocr-app">
      <header className="ocr-header">
        <div className="brand">
          <span className="wf">WELLS FARGO</span><span className="pipe">|</span>
          <span className="app">EDIP · Extraction Review</span>
        </div>
        <div className="toolbar">
          <div className="toolseg">
            <label>Upload</label>
            <input type="file" accept="application/pdf" onChange={doUpload} />
          </div>

          <div className="toolseg">
            <label>DocType</label>
            <select value={doctype} onChange={e=> setDoctypeState(e.target.value)}>
              {doctypeOptions.map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>

          <div className="toolseg">
            <button onClick={runECM} disabled={!doc}>Run ECM</button>
            <button className="primary" onClick={saveFields} disabled={!doc || !fstate}>Save JSON</button>
          </div>

          <div className="toolseg">
            <label>PDF</label>
            <div className="seg">
              <button
                type="button"
                onClick={()=>{ if(doc){ setViewerOpen(true); } }}
                disabled={!doc}
              >
                Open Viewer
              </button>
              {doc && <span className="meter">p{page}{meta?`/${meta.pages.length}`:""}</span>}
            </div>
          </div>

          <div className="toolseg">
            <label>Find</label>
            <div className="seg">
              <input
                id="q"
                className="w200"
                placeholder="Token search…"
                onKeyDown={(e)=>{ if(e.key==="Enter"){ const v=(e.target as HTMLInputElement).value; doSearch(v); }}}
              />
              <button onClick={()=>{ const q=(document.getElementById("q") as HTMLInputElement).value; doSearch(q); }}>Go</button>
            </div>
          </div>


<div className="toolseg">
  <label>Semantic</label>
  <div className="seg">
    <input
      id="qs"
      className="w200"
      placeholder="Meaning-based…"
      onKeyDown={(e)=>{ if(e.key==="Enter"){ const v=(e.target as HTMLInputElement).value; doSemantic(v); }}}
    />
    <button onClick={()=>{ const v=(document.getElementById("qs") as HTMLInputElement).value; doSemantic(v); }}>Jump</button>
  </div>
</div>

          <div className="toolseg">
            <label>Semantic</label>
            <div className="seg">
              <input
                id="qs"
                className="w200"
                placeholder="Meaning-based…"
                onKeyDown={(e)=>{ if(e.key==="Enter"){ const v=(e.target as HTMLInputElement).value; doSemantic(v); }}}
              />
              <button onClick={()=>{ const q=(document.getElementById("qs") as HTMLInputElement).value; doSemantic(q); }}>Jump</button>
            </div>
          </div>

          {bindKey && (
            <div className="toolseg">
              <label>Binding</label>
              <div className="seg">
                <span className="meter">{bindKey}</span>
                <button onClick={()=> setBindKey(null)}>Cancel</button>
              </div>
            </div>
          )}
        </div>
      </header>

      <main className="ocr-main" style={{gridTemplateColumns:"1fr 1fr"}}>
        {/* Left: KV table */}
        <section className="panel">
          <div className="panel-title">Key ↔ Value</div>
          <div className="table-tools">
            <input className="filter" placeholder="Filter keys/values…" value={filterKV} onChange={e=> setFilterKV(e.target.value)} />
          </div>
          <div className="table-wrap">
            <table className="grid">
              <thead>
                <tr>
                  <th style={{width:220}}>Key</th>
                  <th>Value</th>
                  <th style={{width:120}}>Source / Len</th>
                  <th style={{width:140}}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {kvRows.map((f)=>(
                  <tr key={f.key} className={bindKey===f.key ? "active" : ""}>
                    <td className="mono">{f.key}</td>
                    <td>
                      <div style={{display:"flex", alignItems:"center", gap:8}}>
                        <input
                          className="cell-input"
                          style={{flex:1, minWidth:240}}
                          value={f.value ?? ""}
                          title={f.value ?? ""}
                          onChange={e=>{
                            if (!fstate) return;
                            setFstate({
                              ...fstate,
                              fields: fstate.fields.map(g =>
                                g.key===f.key ? {...g, value:e.target.value, source:"user"} : g
                              )
                            });
                          }}
                        />
                        <button
                          className="secondary"
                          onClick={()=> navigator.clipboard.writeText(String(f.value ?? ""))}
                          title="Copy value"
                        >
                          Copy
                        </button>
                      </div>
                    </td>
                    <td style={{whiteSpace:"nowrap"}}>
                      <span className="tag">{f.source||"user"}</span>
                      <span className="mono dim" style={{marginLeft:8}}>{String(f.value ?? "").length} ch</span>
                    </td>
                    <td>
                      <div className="row-actions">
                        <button
                          type="button"
                          onClick={() => {
                            startBind(f.key);
                            setViewerOpen(true);
                            if (f.bbox?.page) setPage(f.bbox.page);
                          }}
                          disabled={!doc}
                        >
                          Bind from PDF
                        </button>
                        {f.bbox && (
                          <button
                            className="secondary"
                            onClick={()=>{
                              setPage(f.bbox!.page);
                              setViewerOpen(true);
                            }}
                          >
                            View bbox
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
                {kvRows.length===0 && (<tr><td colSpan={4} className="empty">No fields yet. Upload a PDF and Run ECM.</td></tr>)}
              </tbody>
            </table>
          </div>
        </section>

        {/* Right: Highlights */}
        <section className="panel">
          <div className="panel-title">Highlights</div>
          <div className="table-tools">
            <input className="filter" placeholder="Filter by text or page…" value={filterHL} onChange={e=> setFilterHL(e.target.value)} />
          </div>
          <div className="table-wrap">
            <table className="grid">
              <thead>
                <tr>
                  <th style={{width:80}}>Page</th>
                  <th>Text</th>
                  <th className="mono" style={{width:260}}>BBox (x0,y0 → x1,y1)</th>
                </tr>
              </thead>
              <tbody>
                {hlRows.map((b, i)=>(
                  <tr key={`${b.page}-${i}`}>
                    <td>p{b.page}</td>
                    <td className="mono">{b.text ?? ""}</td>
                    <td className="mono">
                      ({Math.round(b.x0)},{Math.round(b.y0)}) → ({Math.round(b.x1)},{Math.round(b.y1)})
                    </td>
                  </tr>
                ))}
                {hlRows.length===0 && (<tr><td colSpan={3} className="empty">No highlights yet. Try Search.</td></tr>)}
              </tbody>
            </table>
          </div>
        </section>
      </main>

      {viewerOpen && doc && ocrSize && (
        <PdfViewerFS
          url={doc.pdfUrl}                      // absolute URL
          page={page}
          onClose={()=> setViewerOpen(false)}
          onChangePage={(p)=> setPage(Math.max(1, p))}
          scale={scale}
          onZoom={(s)=> setScale(s)}
          ocrSize={{ width: ocrSize.width, height: ocrSize.height }}
          bindKey={bindKey}
          onLasso={onLasso}
        />
      )}
    </div>
  );
}

