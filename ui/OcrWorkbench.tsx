import { useMemo, useState, type ChangeEvent } from "react";
import PdfCanvas, { Box, Rect } from "./PdfCanvas";
import {
  uploadDoc, getMeta, getBoxes, saveBoxes, search, lasso, rebuild, audit
} from "../../lib/api";
import "./ocr.css";

type Match = { page:number; bbox:Rect; text:string; score:number };

const SERVER = "http://localhost:8000";
const API    = "http://localhost:8000/lasso";

type Tool = "select" | "lasso";
type OCRParams = {
  dpi: number; psm: number; oem: number; lang: string;
  binarize: boolean; deskew: boolean; dilate: number; erode: number;
};

export default function OcrWorkbench(){
  const [doc, setDoc] = useState<any>(null);
  const [meta, setMeta] = useState<{pages:{page:number;width:number;height:number}[]} | null>(null);
  const [page, setPage] = useState(1);
  const [scale, setScale] = useState(1.25);
  const [tool, setTool] = useState<Tool>("select");

  const [params, setParams] = useState<OCRParams>({
    dpi: 220, psm: 6, oem: 1, lang: "eng",
    binarize: true, deskew: true, dilate: 0, erode: 0,
  });

  const [boxes, setBoxes] = useState<Box[]>([]);
  const [highlights, setHighlights] = useState<Box[]>([]);
  const [selected, setSelected] = useState<number[]>([]);
  const [filter, setFilter] = useState("");

  async function doUpload(e: ChangeEvent<HTMLInputElement>){
    const f = e.target.files?.[0]; if(!f) return;
    const res = await uploadDoc(API, f, "tesseract");
    const pdfUrl = `${SERVER}${res.annotated_tokens_url}?t=${Date.now()}`;
    setDoc({ ...res, pdfUrl });

    const m = await getMeta(API, res.doc_id); setMeta(m);
    // hydrate boxes
    const b = await getBoxes(API, res.doc_id); setBoxes(b);
    setPage(1); setScale(1.25); setSelected([]); setHighlights([]);
  }

  async function rerunOCR(){
    if(!doc) return;
    const m = await rebuild(API, doc.doc_id, params);
    setMeta(m);
    const b = await getBoxes(API, doc.doc_id);
    setBoxes(b);
  }

  async function doSearch(q: string){
    if(!doc || !q.trim()) return;
    const r = await search(API, doc.doc_id, q, 60);
    const m: Match[] = r.matches || [];
    setHighlights(m.map(({page,bbox}) => ({ page, ...bbox })));
    if (m.length) setPage(m[0].page);
  }

  async function onLassoRect(rect: Rect){
    if(!doc) return;
    const out = await lasso(API, doc.doc_id, page, rect);
    const added: Box = { id: `new_${Date.now()}`, label: out?.text?.slice(0,40) ?? "", page, ...rect };
    setBoxes(prev => [...prev, added]);
    setTool("select");
    await audit(API, { event:"lasso", payload: { doc_id:doc.doc_id, page, rect, result: out }});
  }

  function focusBox(idx:number){
    const b = boxes[idx]; if(!b) return;
    setPage(b.page); setSelected([idx]);
  }
  function updateBox(idx:number, patch:Partial<Box>){
    setBoxes(prev => { const n = prev.slice(); n[idx] = {...prev[idx], ...patch}; return n; });
  }
  async function persistBoxes(){
    if(!doc) return;
    await saveBoxes(API, doc.doc_id, boxes);
    alert("Saved");
  }

  const pageBoxIndices = useMemo(()=>{
    const ixs = boxes.map((b,i)=> b.page===page ? i : -1).filter(i=>i>=0);
    const f = filter.trim().toLowerCase();
    return f ? ixs.filter(i => (boxes[i].label ?? "").toLowerCase().includes(f) || (boxes[i].id ?? "").toLowerCase().includes(f)) : ixs;
  }, [boxes, page, filter]);

  const stats = useMemo(()=>{
    const total = boxes.length;
    const onPage = boxes.filter(b => b.page===page).length;
    return { total, onPage };
  }, [boxes, page]);

  const ocrSize = meta?.pages.find(p => p.page === page);

  return (
    <div className="ocr-app">
      <header className="ocr-header">
        <div className="brand">
          <span className="wf">WELLS FARGO</span><span className="pipe">|</span>
          <span className="app">EDIP Platform Web · OCR Workbench</span>
        </div>

        <div className="toolbar">
          <div className="toolseg">
            <label>Tool</label>
            <div className="seg">
              <button className={tool==="select"?"active":""} onClick={()=>setTool("select")}>Select</button>
              <button className={tool==="lasso"?"active":""} onClick={()=>setTool("lasso")}>Lasso</button>
            </div>
          </div>

          <div className="toolseg">
            <label>Page</label>
            <div className="pager">
              <button onClick={()=> setPage(p=> Math.max(1, p-1))}>Prev</button>
              <span>{page}{meta?` / ${meta.pages.length}`:""}</span>
              <button onClick={()=> setPage(p=> p+1)}>Next</button>
            </div>
          </div>

          <div className="toolseg">
            <label>Zoom</label>
            <div className="seg">
              <button onClick={()=> setScale(s=> Math.max(0.5, +(s-0.25).toFixed(2)))}>-</button>
              <span className="meter">{scale.toFixed(2)}x</span>
              <button onClick={()=> setScale(s=> +(s+0.25).toFixed(2))}>+</button>
            </div>
          </div>

          <div className="toolseg">
            <button className="primary" onClick={persistBoxes} disabled={!doc}>Save</button>
          </div>
        </div>
      </header>

      <main className="ocr-main">
        <aside className="left">
          <div className="panel">
            <div className="panel-title">Upload</div>
            <input type="file" accept="application/pdf" onChange={doUpload} />
          </div>

          <div className="panel">
            <div className="panel-title">Search (fuzzy)</div>
            <div className="searchrow">
              <input id="q" placeholder="Find any token…" onKeyDown={(e)=>{ if(e.key==="Enter"){ const v=(e.target as HTMLInputElement).value; doSearch(v); }}}/>
              <button onClick={()=>{ const q=(document.getElementById("q") as HTMLInputElement).value; doSearch(q); }}>Go</button>
            </div>
          </div>

          <div className="panel">
            <div className="panel-title">OCR Settings</div>
            <div className="grid2">
              <label>DPI <b>{params.dpi}</b></label>
              <input type="range" min={120} max={400} step={10} value={params.dpi} onChange={e=> setParams({...params, dpi: +e.target.value})}/>
              <label>PSM</label>
              <select value={params.psm} onChange={e=> setParams({...params, psm: +e.target.value})}>
                {[3,4,6,11,12,13].map(p=><option key={p} value={p}>{p}</option>)}
              </select>
              <label>Lang</label>
              <input value={params.lang} onChange={e=> setParams({...params, lang: e.target.value})}/>
              <label>Deskew</label>
              <input type="checkbox" checked={params.deskew} onChange={e=> setParams({...params, deskew: e.target.checked})}/>
              <label>Binarize</label>
              <input type="checkbox" checked={params.binarize} onChange={e=> setParams({...params, binarize: e.target.checked})}/>
              <label>Dilate</label>
              <input type="number" min={0} max={5} value={params.dilate} onChange={e=> setParams({...params, dilate: +e.target.value})}/>
              <label>Erode</label>
              <input type="number" min={0} max={5} value={params.erode} onChange={e=> setParams({...params, erode: +e.target.value})}/>
            </div>
            <button onClick={rerunOCR} disabled={!doc}>Re-run OCR</button>
            <div className="hint">Tip: If you see too few boxes, try DPI 300–350 and PSM 6 or 11.</div>
          </div>

          <div className="panel">
            <div className="panel-title">Boxes on Page {page} <span className="muted">({stats.onPage}/{stats.total})</span></div>
            <input className="filter" placeholder="Filter by label/id" value={filter} onChange={e=> setFilter(e.target.value)} />
            <div className="list">
              {pageBoxIndices.length===0 && <div className="empty">No boxes. Try Search or use Lasso.</div>}
              {pageBoxIndices.map(idx=>{
                const b = boxes[idx]; const active = selected.includes(idx);
                return (
                  <div key={b.id ?? idx} className={"item"+(active?" active":"")} onClick={()=>focusBox(idx)}>
                    <div className="row">
                      <input className="label" placeholder="Label" value={b.label ?? ""} onChange={e=> updateBox(idx, {label: e.target.value})} onClick={e=> e.stopPropagation()}/>
                    </div>
                    <div className="meta">
                      <span className="mono">{b.id ?? `#${idx}`}</span>
                      <span className="coords">({Math.round(b.x0)},{Math.round(b.y0)})→({Math.round(b.x1)},{Math.round(b.y1)})</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </aside>

        <section className="stage">
          {doc && ocrSize ? (
            <PdfCanvas
              url={doc.pdfUrl}
              page={page}
              scale={scale}
              boxes={boxes}
              highlights={highlights}
              selected={selected}
              ocrSize={{ width: ocrSize.width, height: ocrSize.height }}
              tool={tool}
              onLasso={onLassoRect}
              onSelectBox={focusBox}
            />
          ) : (
            <div className="placeholder">Upload a PDF to begin.</div>
          )}
        </section>
      </main>
    </div>
  );
}
