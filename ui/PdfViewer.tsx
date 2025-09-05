import { useEffect, useRef, useState } from "react";
import { getDocument, GlobalWorkerOptions, type PDFDocumentProxy, type RenderTask } from "pdfjs-dist";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

type Rect = { x0:number; y0:number; x1:number; y1:number };

type Props = {
  url: string;                       // absolute URL to PDF
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
  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  const baseRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const renderTaskRef = useRef<RenderTask | null>(null);
  const [vp, setVp] = useState<{ width:number; height:number } | null>(null);
  const [err, setErr] = useState<string|null>(null);
  const mounted = useRef(true);

  // inline styles (no external CSS conflicts)
  const modal = { position:"fixed" as const, inset:0, background:"rgba(0,0,0,0.55)", display:"flex", alignItems:"center", justifyContent:"center", zIndex: 9999 };
  const frame = { width:"92vw", height:"88vh", background:"#fff", borderRadius:14, boxShadow:"0 10px 40px rgba(0,0,0,.4)", display:"grid", gridTemplateRows:"56px 1fr", overflow:"hidden" };
  const header = { display:"flex", alignItems:"center", justifyContent:"space-between", padding:"0 14px", borderBottom:"1px solid #e5e5e5", background:"#fafafa" };
  const stage = { position:"relative" as const, width:"100%", height:"100%", overflow:"auto" };
  const cBase = { display:"block", margin:"0 auto", background:"#fff" };
  const cOv = { position:"absolute" as const, inset:0, cursor:"crosshair", display:"block", margin:"0 auto" };

  // Cancel any in-flight render safely
  const cancelCurrentRender = async () => {
    const t = renderTaskRef.current;
    if (t) {
      try { t.cancel(); } catch {}
      try { await t.promise; } catch {}
      renderTaskRef.current = null;
    }
  };

  // Main render function with single-flight and guards
  async function renderPage(pn:number, sc:number){
    try{
      setErr(null);
      // Ensure canvases exist
      const base = baseRef.current;
      const overlay = overlayRef.current;
      if (!base || !overlay) return;

      // Lazy load / reload PDF when URL changes
      if (!pdfRef.current) {
        pdfRef.current = await getDocument({ url }).promise;
      }

      // Cancel any previous render BEFORE starting a new one
      await cancelCurrentRender();

      // Get page + viewport
      const p = await pdfRef.current.getPage(pn);
      const viewport = p.getViewport({ scale: sc });

      // Size the canvases
      base.width = viewport.width;  base.height = viewport.height;
      overlay.width = viewport.width; overlay.height = viewport.height;

      // Kick off render and await it (so we don't overlap)
      const ctx = base.getContext("2d");
      if (!ctx) throw new Error("No 2D context");
      const task = p.render({ canvasContext: ctx, viewport });
      renderTaskRef.current = task;
      await task.promise; // if canceled, this throws; caught below
      renderTaskRef.current = null;

      if (!mounted.current) return;
      setVp({ width: viewport.width, height: viewport.height });
    }catch(e:any){
      if (e?.name === "RenderingCancelledException") return; // expected on cancel
      setErr(e?.message || "Failed to render PDF");
      // Reset vp so overlay doesn't try to use stale dims
      setVp(null);
    }
  }

  // Load/reset when URL changes
  useEffect(()=>{
    mounted.current = true;
    (async ()=>{
      // Tear down old doc and render task
      await cancelCurrentRender();
      pdfRef.current = null;
      // Kick initial render
      await renderPage(page, scale);
    })();
    return () => { mounted.current = false; cancelCurrentRender(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // Re-render on page/scale change
  useEffect(()=>{
    if (!pdfRef.current) return; // will render on URL effect
    renderPage(page, scale);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, scale]);

  // Lasso helpers
  const [drag, setDrag] = useState<{x:number;y:number}|null>(null);
  const toOcr = (x:number,y:number) => {
    if (!vp) return { x, y };
    const sx = ocrSize.width  / vp.width;
    const sy = ocrSize.height / vp.height;
    return { x: x * sx, y: y * sy };
  };

  return (
    <div style={modal} onClick={onClose}>
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
          {err ? (
            <div style={{padding:16, color:"#b00020", fontFamily:"monospace"}}>{err}</div>
          ) : (
            <>
              <canvas ref={baseRef} style={cBase as any}/>
              <canvas
                ref={overlayRef}
                style={cOv as any}
                onMouseDown={(e)=>{
                  const el = overlayRef.current; if (!el) return;
                  const r = el.getBoundingClientRect();
                  setDrag({ x: e.clientX - r.left, y: e.clientY - r.top });
                }}
                onMouseUp={(e)=>{
                  const el = overlayRef.current; if (!el || !drag) return;
                  const r = el.getBoundingClientRect();
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
            </>
          )}
        </div>
      </div>
    </div>
  );
}
