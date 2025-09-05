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
  const renderTaskRef = useRef<RenderTask | null>(null);
  const renderIdRef = useRef(0);
  const mounted = useRef(true);

  // The visible canvas lives inside this wrapper. We replace it on every render.
  const baseWrapRef = useRef<HTMLDivElement | null>(null);
  const currentCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Overlay is a div (not a canvas) so we never resize a rendering target.
  const overlayRef = useRef<HTMLDivElement | null>(null);

  const [vp, setVp] = useState<{ width:number; height:number } | null>(null);
  const [err, setErr] = useState<string|null>(null);

  // ---------- styles ----------
  const modal = { position:"fixed" as const, inset:0, background:"rgba(0,0,0,0.55)", display:"flex", alignItems:"center", justifyContent:"center", zIndex: 9999 };
  const frame = { width:"92vw", height:"88vh", background:"#fff", borderRadius:14, boxShadow:"0 10px 40px rgba(0,0,0,.4)", display:"grid", gridTemplateRows:"56px 1fr", overflow:"hidden" };
  const header = { display:"flex", alignItems:"center", justifyContent:"space-between", padding:"0 14px", borderBottom:"1px solid #e5e5e5", background:"#fafafa" };
  const stage  = { position:"relative" as const, width:"100%", height:"100%", overflow:"auto" };
  const baseWrap = { position:"relative" as const, display:"block", margin:"0 auto", background:"#fff" };
  const overlayStyle = { position:"absolute" as const, inset:0, cursor:"crosshair" };

  // ---------- helpers ----------
  const cancelCurrentRender = async () => {
    const t = renderTaskRef.current;
    if (t) {
      try { t.cancel(); } catch {}
      try { await t.promise; } catch {}
      renderTaskRef.current = null;
    }
  };

  async function renderPage(pn:number, sc:number){
    const myId = ++renderIdRef.current;
    try{
      setErr(null);

      // lazy load PDF for given URL
      if (!pdfRef.current) {
        pdfRef.current = await getDocument({ url }).promise;
      }

      // cancel any in-progress task first
      await cancelCurrentRender();

      const p = await pdfRef.current.getPage(pn);
      const viewport = p.getViewport({ scale: sc });

      // --- render into a brand-new offscreen canvas ---
      const work = document.createElement("canvas");
      work.width = viewport.width;
      work.height = viewport.height;
      const wctx = work.getContext("2d");
      if (!wctx) throw new Error("No 2D context");

      const task = p.render({ canvasContext: wctx, viewport });
      renderTaskRef.current = task;
      await task.promise;                 // if canceled, throws
      renderTaskRef.current = null;

      // if a newer render started, drop this one
      if (myId !== renderIdRef.current) return;
      if (!mounted.current) return;

      // --- swap the visible canvas node ---
      const host = baseWrapRef.current;
      if (!host) return;

      // Remove previous canvas (if any)
      if (currentCanvasRef.current && currentCanvasRef.current.parentNode === host) {
        host.removeChild(currentCanvasRef.current);
      }

      // Insert the freshly rendered canvas
      host.appendChild(work);
      currentCanvasRef.current = work;

      // Size overlay to match the new canvas (via CSS; it's absolute over host)
      // host acts as the sizing box; ensure its size matches the canvas
      (host as HTMLDivElement).style.width  = `${work.width}px`;
      (host as HTMLDivElement).style.height = `${work.height}px`;

      setVp({ width: viewport.width, height: viewport.height });
    }catch(e:any){
      if (e?.name === "RenderingCancelledException") return;
      setErr(e?.message || "Failed to render PDF");
      setVp(null);
    }
  }

  // Reload on URL change
  useEffect(()=>{
    mounted.current = true;
    (async ()=>{
      await cancelCurrentRender();
      pdfRef.current = null;
      // clear any existing canvas immediately
      if (baseWrapRef.current && currentCanvasRef.current) {
        try { baseWrapRef.current.removeChild(currentCanvasRef.current); } catch {}
        currentCanvasRef.current = null;
      }
      await renderPage(page, scale);
    })();
    return () => { mounted.current = false; cancelCurrentRender(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // Re-render on page/scale change
  useEffect(()=>{
    if (!pdfRef.current) return; // URL effect will render initial
    renderPage(page, scale);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, scale]);

  // Lasso (overlay is a DIV; we compute offsets relative to it)
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
            <div style={{position:"relative", width:"100%", height:"100%"}}>
              {/* Visible canvas container (we swap the child canvas node each render) */}
              <div ref={baseWrapRef} style={baseWrap as any} />

              {/* Overlay DIV for lasso */}
              <div
                ref={overlayRef}
                style={overlayStyle as any}
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
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
