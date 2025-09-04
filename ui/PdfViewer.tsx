import { useEffect, useRef, useState } from "react";
import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

type Rect = { x0:number; y0:number; x1:number; y1:number };

type Props = {
  url: string;
  page: number;
  scale: number;
  ocrSize: {width:number; height:number};
  bindKey: string | null;
  onClose: () => void;
  onChangePage: (p:number) => void;
  onZoom: (s:number) => void;
  onLasso: (page:number, rect:Rect) => void;
};

export default function PdfViewerFS({
  url, page, scale, ocrSize, bindKey,
  onClose, onChangePage, onZoom, onLasso
}: Props){
  const pdfRef = useRef<any>(null);
  const base = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const [vp, setVp] = useState<any>(null);
  const [err, setErr] = useState<string|null>(null);

  // inline styles so CSS can’t shrink the modal
  const modal = { position:"fixed" as const, inset:0, background:"rgba(0,0,0,0.55)", display:"flex", alignItems:"center", justifyContent:"center", zIndex: 9999 };
  const frame = { width:"92vw", height:"88vh", background:"#fff", borderRadius:14, boxShadow:"0 10px 40px rgba(0,0,0,.4)", display:"grid", gridTemplateRows:"56px 1fr", overflow:"hidden" };
  const header = { display:"flex", alignItems:"center", justifyContent:"space-between", padding:"0 14px", borderBottom:"1px solid #e5e5e5", background:"#fafafa" };
  const stage = { position:"relative" as const, width:"100%", height:"100%", overflow:"auto" };
  const cBase = { display:"block", margin:"0 auto", background:"#fff" };
  const cOv = { position:"absolute" as const, inset:0, cursor:"crosshair", display:"block", margin:"0 auto" };

  async function render(pn:number, sc:number){
    try{
      setErr(null);
      if(!pdfRef.current){
        // IMPORTANT: URL must be absolute (http://localhost:8000/data/...)
        pdfRef.current = await getDocument(url).promise;
      }
      const p = await pdfRef.current.getPage(pn);
      const viewport = p.getViewport({ scale: sc });
      const c = base.current!; 
      c.width = viewport.width; c.height = viewport.height;
      const ctx = c.getContext("2d"); if(!ctx) return;
      await p.render({ canvasContext: ctx, viewport }).promise;
      setVp(viewport);
      const ov = overlay.current!; 
      ov.width = viewport.width; ov.height = viewport.height;
    }catch(e:any){ setErr(e?.message || "Failed to render PDF"); console.error(e); }
  }

  useEffect(()=>{ pdfRef.current=null; render(page, scale); }, [url]);
  useEffect(()=>{ if(pdfRef.current) render(page, scale); }, [page, scale]);

  const [drag, setDrag] = useState<{x:number;y:number}|null>(null);
  const toOcr = (x:number,y:number) => {
    if (!vp) return { x, y };
    const sx = ocrSize.width  / vp.width;
    const sy = ocrSize.height / vp.height;
    return { x: x * sx, y: y * sy };
  };

  return (
    <div style={modal}>
      <div style={frame} onClick={e=> e.stopPropagation()}>
        <div style={header}>
          <div style={{fontWeight:700}}>
            PDF Viewer {bindKey ? <span style={{marginLeft:8, fontWeight:500, fontSize:12, padding:"4px 8px", background:"#eef", borderRadius:12}}>Binding: {bindKey}</span> : null}
          </div>
          <div style={{display:"flex", gap:8, alignItems:"center"}}>
            <button onClick={()=> onChangePage(Math.max(1, page-1))}>Prev</button>
            <span style={{fontFamily:"monospace"}}>p{page}</span>
            <button onClick={()=> onChangePage(page+1)}>Next</button>
            <span style={{width:8}}/>
            <button onClick={()=> onZoom(Math.max(0.5, +(scale-0.25).toFixed(2)))}>-</button>
            <span style={{fontFamily:"monospace"}}>{scale.toFixed(2)}x</span>
            <button onClick={()=> onZoom(+(scale+0.25).toFixed(2))}>+</button>
            <span style={{width:8}}/>
            <button onClick={onClose}>Close</button>
          </div>
        </div>
        <div style={stage}>
          {err ? <div style={{padding:16, color:"#b00020", fontFamily:"monospace"}}>{err}</div> : <>
            <canvas ref={base} style={cBase as any}/>
            <canvas
              ref={overlay}
              style={cOv as any}
              onMouseDown={(e)=>{
                const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
                setDrag({ x: e.clientX - r.left, y: e.clientY - r.top });
              }}
              onMouseUp={(e)=>{
                if (!drag) return;
                const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
                const a = toOcr(drag.x, drag.y);
                const b = toOcr(e.clientX - r.left, e.clientY - r.top);
                onLasso(page, {
                  x0: Math.min(a.x,b.x), y0: Math.min(a.y,b.y),
                  x1: Math.max(a.x,b.x), y1: Math.max(a.y,b.y),
                });
                setDrag(null);
              }}
              width={vp?.width||0} height={vp?.height||0}
            />
          </>}
        </div>
      </div>
    </div>
  );
}
