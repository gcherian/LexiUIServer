import React, { useEffect, useMemo, useState } from "react";
import "../../assets/stylesheets/lasso.css"; // your local CSS (avoid pdfjs css)
import PdfEditCanvas, { type EditRect, type TokenBox, type OverlayRect } from "./PdfEditCanvas";
import { API, getMeta, getBoxes, matchField, saveGT, type Box as TBox } from "../../../lib/api";

type KVRect = { page:number;x0:number;y0:number;x1:number;y1:number };
type FieldRow = { key:string; value:string; rects?:KVRect[] };

const COLORS: Record<string,string> = {
  fuzzy:"#22c55e", tfidf:"#3b82f6", minilm:"#a855f7", distilbert:"#f59e0b"
};

function unionRect(span: TokenBox[]) {
  let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
  for(const t of span){ x0=Math.min(x0,t.x0); y0=Math.min(y0,t.y0); x1=Math.max(x1,t.x1); y1=Math.max(y1,t.y1); }
  return { x0:Math.floor(x0),y0:Math.floor(y0),x1:Math.ceil(x1),y1:Math.ceil(y1) };
}

export default function FieldLevelEditor() {
  // doc
  const [docUrl,setDocUrl]=useState("");
  const [docId,setDocId]=useState("");
  const [meta,setMeta]=useState<{w:number;h:number}[]>([]);
  const [page,setPage]=useState(1);
  // tokens
  const [tokens,setTokens]=useState<TokenBox[]>([]);
  const tokensPage = useMemo(()=>tokens.filter(t=>t.page===page),[tokens,page]);
  // kv
  const [rows,setRows]=useState<FieldRow[]>([]);
  const [focusedKey,setFocusedKey]=useState("");
  // overlays
  const [rect,setRect]=useState<EditRect|null>(null);
  const [overlays,setOverlays]=useState<OverlayRect[]>([]);
  const [showBoxes,setShowBoxes]=useState(false);
  const [editValue,setEditValue]=useState(false);
  const [zoom,setZoom]=useState(1);

  const [loadingFast,setLoadingFast]=useState(false);
  const [loadingHeavy,setLoadingHeavy]=useState(false);

  // helpers
  const serverW = meta[page-1]?.w || 1;
  const serverH = meta[page-1]?.h || 1;

  // Load from pasted /data/{doc}/original.pdf
  useEffect(()=>{
    const m=docUrl.match(/\/data\/([A-Za-z0-9_-]+)\/original\.pdf/i);
    const id = m ? m[1] : "";
    if(!id) return;
    (async()=>{
      setDocId(id);
      const m = await getMeta(id);
      setMeta(m.pages.map(p=>({w:p.width,h:p.height})));
      setTokens((await getBoxes(id)) as any);
      setPage(1); setRect(null); setOverlays([]);
    })();
  },[docUrl]);

  function mapOverlays(methods:Record<string, any>): OverlayRect[] {
    const pick = (m:any): EditRect|null => m && m.rect ? ({ page:m.page, x0:m.rect.x0, y0:m.rect.y0, x1:m.rect.x1, y1:m.rect.y1 }) : null;
    return (["fuzzy","tfidf","minilm","distilbert"] as const).map(k => ({
      label: k, color: COLORS[k], rect: pick(methods?.[k])
    }));
  }

  async function onRowClick(r: FieldRow) {
    if(!docId) return;
    setFocusedKey(r.key);
    setOverlays([]); setLoadingFast(true); setLoadingHeavy(false);

    // FAST first (fuzzy/tfidf)
    try{
      const fast = await matchField(docId, r.key, r.value, 12, undefined, true);
      const ovs = mapOverlays(fast.methods||{});
      setOverlays(ovs);
      const firstPg = ovs.find(o=>o.rect)?.rect?.page;
      if(firstPg) setPage(firstPg);
    }catch(e){ console.warn("FAST locate failed", e); }
    finally{ setLoadingFast(false); }

    // HEAVY in the background (minilm/distilbert)
    setLoadingHeavy(true);
    matchField(docId, r.key, r.value, 12)
      .then((all)=>{
        const merged = mapOverlays(all.methods||{});
        setOverlays(merged);
      })
      .catch((e)=>console.warn("HEAVY locate failed", e))
      .finally(()=>setLoadingHeavy(false));
  }

  async function onRectCommitted(rr: EditRect) {
    if(!focusedKey) return;
    // Update KV rect inline
    setRows(prev => prev.map(row => row.key===focusedKey ? ({...row, rects:[{page:rr.page,x0:rr.x0,y0:rr.y0,x1:rr.x1,y1:rr.y1}]}) : row));
    if(editValue){
      // Optional: on commit, you can OCR on server and update row.value.
      // If you already have /lasso/lasso OCR preview, call it here. For now leave as rect-only editing.
    }
  }

  async function onSaveGT(){
    if(!focusedKey || !rect || !docId) return;
    try{
      const preds: Record<string, any> = {};
      for(const o of overlays){
        preds[o.label] = o.rect ? { page:o.rect.page, rect:{x0:o.rect.x0,y0:o.rect.y0,x1:o.rect.x1,y1:o.rect.y1}, score:0 } : null;
      }
      const row = rows.find(r=>r.key===focusedKey)!;
      const res = await saveGT({
        doc_id: docId,
        key: focusedKey,
        value: row?.value || "",
        rect: { page: rect.page, x0: rect.x0, y0: rect.y0, x1: rect.x1, y1: rect.y1 },
        preds
      });
      console.log("GT saved", res);
      alert("Saved ground truth.\nIoU:\n" + JSON.stringify(res?.report || {}, null, 2));
    }catch(e:any){
      alert("Save GT failed: " + (e?.message || String(e)));
    }
  }

  // Minimal KV table from uploaded ECM JSON
  async function onUploadEcm(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if(!f) return;
    try{
      const parsed = JSON.parse(await f.text()) as Record<string, any>;
      const flat: FieldRow[] = [];
      const stack = (o:any, prefix="")=>{
        if(o===null||o===undefined) return;
        if(Array.isArray(o)){
          o.forEach((v,i)=>stack(v, `${prefix}[${i}]`)); return;
        }
        if(typeof o==="object"){
          for(const k of Object.keys(o)){
            const path = prefix ? `${prefix}.${k}` : k;
            const v = (o as any)[k];
            if(v && typeof v==="object" && !Array.isArray(v)){
              if("value" in v || "rects" in v || "bboxes" in v || "bbox" in v){
                const rects = (v.rects||v.bboxes||v.bbox) as any[]|undefined;
                flat.push({ key:path, value:String(v.value ?? ""), rects:rects?.map((b:any)=>({page:+b.page,x0:+b.x0,y0:+b.y0,x1:+b.x1,y1:+b.y1})) });
              }else stack(v, path);
            }else{
              flat.push({ key:path, value: v==null ? "" : String(v) });
            }
          }
          return;
        }
        flat.push({ key: prefix||"(value)", value: String(o) });
      };
      stack(parsed, "");
      setRows(flat);
      setFocusedKey(""); setRect(null); setOverlays([]);
    }catch{ alert("Invalid ECM JSON"); }
    finally{ (e.target as HTMLInputElement).value=""; }
  }

  return (
    <div className="workbench">
      <div className="wb-toolbar">
        <span style={{fontWeight:600}}>Choose:</span>

        <label className="btn"><input type="file" accept="application/json" onChange={onUploadEcm} style={{display:"none"}}/>ECM JSON</label>

        <input className="input" placeholder="...or paste http://localhost:8080/data/{doc_id}/original.pdf"
               value={docUrl} onChange={e=>setDocUrl(e.target.value)} style={{minWidth:380,marginLeft:8}}/>

        <label className={showBoxes ? "btn toggle active":"btn toggle"} style={{marginLeft:8}}>
          <input type="checkbox" checked={showBoxes} onChange={()=>setShowBoxes(v=>!v)} /> Boxes
        </label>

        <label className={editValue ? "btn toggle active":"btn toggle"} style={{marginLeft:8}}>
          <input type="checkbox" checked={editValue} onChange={()=>setEditValue(v=>!v)} /> Edit Value
        </label>

        <button className="btn" onClick={onSaveGT} disabled={!rect || !focusedKey}>Save GT</button>

        <span className="spacer" />

        <div className="toolbar-inline" style={{gap:6}}>
          {Object.entries(COLORS).map(([k,c])=>(
            <span key={k} style={{display:"inline-flex",alignItems:"center",gap:6,fontSize:12}}>
              <span style={{width:12,height:12,background:c,border:`1px solid ${c}`,display:"inline-block",opacity:0.7}}/>
              {k}
            </span>
          ))}
        </div>

        <span className="spacer" />

        <div className="toolbar-inline" style={{gap:4}}>
          <button onClick={()=>setZoom(z=>Math.max(0.5, Math.round((z-0.1)*10)/10))}>–</button>
          <span style={{width:44,textAlign:"center"}}>{Math.round(zoom*100)}%</span>
          <button onClick={()=>setZoom(z=>Math.min(3, Math.round((z+0.1)*10)/10))}>+</button>
          <button onClick={()=>setZoom(1)}>Reset</button>
        </div>

        <span className="muted" style={{marginLeft:12}}>API: {API}</span>
      </div>

      <div className="wb-split" style={{display:"flex",gap:12}}>
        {/* LEFT: KV */}
        <div className="wb-left">
          <div className="section-title">Extraction</div>
          {!rows.length ? (
            <div className="placeholder">Upload ECM JSON to see fields.</div>
          ) : (
            <table>
              <thead><tr><th style={{width:"42%"}}>Key</th><th>Value</th></tr></thead>
              <tbody>
                {rows.map((r,i)=>{
                  const focused = r.key===focusedKey;
                  return (
                    <tr key={r.key+":"+i} onClick={()=>onRowClick(r)}
                        style={focused ? { outline:"2px solid #00b4ff", outlineOffset:-2 } : undefined}>
                      <td><code>{r.key}</code></td>
                      <td style={{whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{r.value}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* RIGHT: PDF */}
        <div className="wb-right">
          {!docUrl ? (
            <div className="placeholder">Upload a PDF (in data/) and paste its URL above to begin.</div>
          ) : (
            <>
              <div className="toolbar-inline">
                <button disabled={page<=1} onClick={()=>setPage(p=>p-1)}>Prev</button>
                <span className="page-indicator">Page {page} {meta.length? `/ ${meta.length}`:""}</span>
                <button disabled={meta.length>0 && page>=meta.length} onClick={()=>setPage(p=>p+1)}>Next</button>
                {(loadingFast || loadingHeavy) && (
                  <span className="muted" style={{marginLeft:8}}>
                    {loadingFast ? "loading: fuzzy/tfidf…" : loadingHeavy ? "loading: minilm/distilbert…" : ""}
                  </span>
                )}
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
          )}
        </div>
      </div>
    </div>
  );
}
