import React, { useEffect, useMemo, useState } from "react";
import "./../ocr.css";
import {
  API, uploadPdf, getMeta, getBoxes, docIdFromUrl, docUrlFromId,
  listProms, getProm, setDoctype, ecmExtract,
  getFields, putFields, type Box, type FieldDocState, type PromCatalog
} from "../../lib/api";
import PdfCanvas from "./PdfCanvas";
import BindModal from "./BindModal";

export default function LassoExtractor() {
  const [docUrl, setDocUrl] = useState<string>("");
  const [docId, setDocId] = useState<string>("");
  const [meta, setMeta] = useState<{w:number; h:number}[]>([]);
  const [page, setPage] = useState<number>(1);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);

  // prom / doctype / fields
  const [proms, setProms] = useState<Array<{doctype:string;file:string}>>([]);
  const [doctype, setDt] = useState<string>("");
  const [catalog, setCatalog] = useState<PromCatalog | null>(null);
  const [state, setState] = useState<FieldDocState | null>(null);

  // modal for binding
  const [modalOpen, setModalOpen] = useState(false);
  const [modalFromBox, setModalFromBox] = useState<Box | null>(null);
  const allKeys = useMemo(()=>catalog?.fields?.map(f=>f.key) || [], [catalog]);

  // upload
  async function onUpload(ev: React.ChangeEvent<HTMLInputElement>){
    const f = ev.target.files?.[0]; if(!f) return;
    try{
      const res = await uploadPdf(f);
      const url = res.annotated_tokens_url;
      setDocUrl(url); setDocId(res.doc_id);
      const m = await getMeta(res.doc_id);
      setMeta(m.pages.map(p=>({ w:p.width, h:p.height })));
      const b = await getBoxes(res.doc_id); setBoxes(b);
      setPage(1); setState(null); setDt(""); setCatalog(null);
    } finally { (ev.target as HTMLInputElement).value = ""; }
  }

  // load by pasted URL (optional)
  useEffect(()=>{
    const id = docIdFromUrl(docUrl); if(!id) return;
    (async ()=>{
      setDocId(id);
      const m = await getMeta(id); setMeta(m.pages.map(p=>({ w:p.width, h:p.height })));
      const b = await getBoxes(id); setBoxes(b); setPage(1);
      setState(null); setDt(""); setCatalog(null);
    })();
  }, [docUrl]);

  // proms
  useEffect(()=>{ (async()=>{ try{ setProms(await listProms()); }catch{} })(); }, []);

  // when doctype selected → initialize fields from PROM (via setDoctype) then we can Extract
  async function onSelectDoctype(dt:string){
    setDt(dt);
    if(!docId) return;
    await setDoctype(docId, dt);
    try {
      const st = await getFields(docId); setState(st);
      const cat = await getProm(dt); setCatalog(cat);
    } catch {
      const cat = await getProm(dt); setCatalog(cat);
      setState({ doc_id: docId, doctype: dt, fields: cat.fields.map(f=>({ key:f.key, value:"", source:"user", confidence:0 })) });
    }
  }

  async function onExtract(){
    if(!docId || !doctype) return;
    const st = await ecmExtract(docId, doctype);
    setState(st);
  }

  // save single cell
  async function saveField(k:string, val:string){
    if(!state) return;
    const next: FieldDocState = { ...state, fields: state.fields.map(f=>f.key===k?{...f, value:val, source:"user"}:f) };
    const saved = await putFields(docId, next);
    setState(saved);
  }

  // click a box → open modal to bind (user picks a key)
  function onBoxClick(b:Box){ setModalFromBox(b); setModalOpen(true); }

  const serverW = meta[page-1]?.w || 1, serverH = meta[page-1]?.h || 1;

  return (
    <div className="workbench">
      <div className="wb-toolbar">
        <input type="file" accept="application/pdf" onChange={onUpload}/>
        <input className="input" placeholder="Paste /data/{doc_id}/original.pdf" value={docUrl} onChange={(e)=>setDocUrl(e.target.value)}/>
        <label className={showBoxes? "btn toggle active":"btn toggle"} style={{marginLeft:8}}>
          <input type="checkbox" checked={showBoxes} onChange={()=>setShowBoxes(v=>!v)}/> Boxes
        </label>
        <span className="spacer"/>
        <span className="muted">API: {API}</span>
      </div>

      <div className="wb-split">
        {/* LEFT: PDF */}
        <div className="wb-left">
          {docUrl ? (
            <>
              <div className="toolbar-inline">
                <button disabled={page<=1} onClick={()=>setPage(p=>p-1)}>Prev</button>
                <span className="page-indicator">Page {page} {meta.length?`/ ${meta.length}`:""}</span>
                <button disabled={meta.length>0 && page>=meta.length} onClick={()=>setPage(p=>p+1)}>Next</button>
              </div>
              <PdfCanvas
                docUrl={docUrl}
                page={page}
                serverW={serverW}
                serverH={serverH}
                boxes={boxes.filter(b=>b.page===page)}
                showBoxes={showBoxes}
                lasso={false}
                onBoxClick={onBoxClick}
              />
              <div className="hint">Click a highlighted box to bind it to a field.</div>
            </>
          ) : (
            <div className="placeholder">Upload or paste a PDF URL to begin.</div>
          )}
        </div>

        {/* RIGHT: controls + fields table */}
        <div className="wb-right">
          <div className="row">
            <label>Doctype</label>
            <select value={doctype} onChange={(e)=>onSelectDoctype(e.target.value)} disabled={!docId}>
              <option value="">(select)</option>
              {proms.map(p=><option key={p.doctype} value={p.doctype}>{p.doctype}</option>)}
            </select>
          </div>

          <div className="row">
            <label>Actions</label>
            <div style={{display:"flex", gap:8}}>
              <button className="primary" onClick={onExtract} disabled={!doctype || !docId}>Extract</button>
            </div>
          </div>

          <div style={{marginTop:12}}>
            <div className="section-title">Fields</div>
            {!state || state.fields.length===0 ? (
              <div className="placeholder">Choose a doctype, then click <b>Extract</b> to populate expected keys.</div>
            ) : (
              <table style={{ width:"100%", borderCollapse:"collapse" }}>
                <thead>
                  <tr><th style={th}>Key</th><th style={th}>Value</th><th style={th}>Source</th><th style={th}>Conf</th><th style={th}>Bind</th></tr>
                </thead>
                <tbody>
                  {state.fields.map((f,idx)=>{
                    const missing = !f.value;
                    return (
                      <tr key={(f.key||"k")+":"+idx} style={{ background: missing ? "#fff7ed" : "transparent" }}>
                        <td style={td}><code>{f.key}</code></td>
                        <td style={td}>
                          <input value={f.value || ""} onChange={(e)=>{
                            const v = e.target.value;
                            setState(s=>s?{...s, fields:s.fields.map(x=>x.key===f.key?{...x, value:v}:x)}:s);
                          }} onBlur={(e)=>saveField(String(f.key), e.target.value)} />
                        </td>
                        <td style={td}>{f.source || ""}</td>
                        <td style={td}>{f.confidence ? f.confidence.toFixed(2) : ""}</td>
                        <td style={td}><button onClick={()=>{ setModalFromBox(null); setModalOpen(true); setDt(doctype); /* bind via lasso */ }}>Bind via Lasso</button></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
          <div className="hint">Orange rows need values. Use <b>Bind via Lasso</b> or click a box on the PDF.</div>
        </div>
      </div>

      {/* Bind modal */}
      <BindModal
        open={modalOpen}
        onClose={()=>setModalOpen(false)}
        docId={docId}
        docUrl={docUrl}
        page={page}
        serverW={serverW}
        serverH={serverH}
        allKeys={allKeys}
        box={modalFromBox}
        initialKey={state?.fields?.find(f=>!f.value)?.key || ""}
        onBound={(st)=>setState(st)}
      />
    </div>
  );
}

const th:React.CSSProperties={ textAlign:"left", padding:"6px 8px", borderBottom:"1px solid #e5e7eb" };
const td:React.CSSProperties={ padding:"6px 8px", borderBottom:"1px solid #f1f5f9" };
