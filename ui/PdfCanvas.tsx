import { useEffect, useRef, useState } from "react";
import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";

// v4 worker (ES module)
GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

export type Box = { x0:number; y0:number; x1:number; y1:number; page:number };

type Props = {
  url?: string;                // PDF URL to render
  page: number;                // current page (1-based)
  boxes?: Box[];               // boxes to draw (in OCR px space)
  ocrSize?: {width:number;height:number}; // OCR page size for scaling
  selected?: number[];         // indexes of boxes to accent
  scale?: number;              // optional zoom
  onLasso?: (rect:{x0:number;y0:number;x1:number;y1:number})=>void; // OCR space
};

export default function PdfCanvas({
  url, page, boxes = [], ocrSize, selected = [], scale = 1.5, onLasso
}: Props) {
  const pdfRef = useRef<any>(null);
  const [vp, setVp] = useState<any>(null);
  const baseCanvas = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);

  // load doc once
  useEffect(() => {
    if (!url) return;
    (async () => {
      pdfRef.current = await getDocument(url).promise;
      // render first page immediately (will change again below)
      const p = await pdfRef.current.getPage(page);
      const viewport = p.getViewport({ scale });
      const c = baseCanvas.current!;
      c.width = viewport.width; c.height = viewport.height;
      const ctx = c.getContext("2d"); if (!ctx) return;
      await p.render({ canvasContext: ctx, viewport }).promise;
      setVp(viewport);
    })();
  }, [url]);

  // render when page or scale changes
  useEffect(() => {
    (async () => {
      if (!pdfRef.current) return;
      const p = await pdfRef.current.getPage(page);
      const viewport = p.getViewport({ scale });
      const c = baseCanvas.current!;
      c.width = viewport.width; c.height = viewport.height;
      const ctx = c.getContext("2d"); if (!ctx) return;
      await p.render({ canvasContext: ctx, viewport }).promise;
      setVp(viewport);
    })();
  }, [page, scale]);

  // draw overlays
  useEffect(() => {
    if (!vp) return;
    const ov = overlay.current!;
    const ctx = ov.getContext("2d"); if (!ctx) return;
    ov.width = vp.width; ov.height = vp.height;
    ctx.clearRect(0,0,ov.width,ov.height);

    if (!ocrSize) return; // we need OCR size to scale correctly

    const sx = vp.width / ocrSize.width;
    const sy = vp.height / ocrSize.height;

    // draw all boxes faintly; selected ones stronger
    boxes.filter(b => b.page === page).forEach((b, i) => {
      const x = b.x0 * sx, y = b.y0 * sy, w = (b.x1-b.x0)*sx, h = (b.y1-b.y0)*sy;
      const isSel = selected.includes(i);
      ctx.lineWidth = isSel ? 3 : 2;
      ctx.strokeStyle = isSel ? "rgba(255,120,0,0.95)" : "rgba(0,120,200,0.9)";
      ctx.fillStyle   = isSel ? "rgba(255,180,0,0.20)" : "rgba(0,160,255,0.18)";
      ctx.fillRect(x,y,w,h);
      ctx.strokeRect(x,y,w,h);
    });
  }, [boxes, selected, page, vp, ocrSize]);

  // lasso => convert viewport px back to OCR px
  const [drag, setDrag] = useState<{x:number;y:number}|null>(null);
  const toOcr = (x:number,y:number) => {
    if (!vp || !ocrSize) return { x, y };
    const sx = ocrSize.width  / vp.width;
    const sy = ocrSize.height / vp.height;
    return { x: x * sx, y: y * sy };
  };

  return (
    <div style={{position:"relative", display:"inline-block"}}>
      <canvas ref={baseCanvas} style={{display:"block"}} />
      <canvas
        ref={overlay}
        style={{position:"absolute", inset:0, cursor:"crosshair"}}
        onMouseDown={(e)=>{
          const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
          setDrag({ x: e.clientX - r.left, y: e.clientY - r.top });
        }}
        onMouseUp={(e)=>{
          if (!drag || !onLasso) return;
          const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
          const a = toOcr(drag.x, drag.y);
          const b = toOcr(e.clientX - r.left, e.clientY - r.top);
          onLasso({ x0: Math.min(a.x,b.x), y0: Math.min(a.y,b.y), x1: Math.max(a.x,b.x), y1: Math.max(a.y,b.y) });
          setDrag(null);
        }}
        width={vp?.width||0} height={vp?.height||0}
      />
    </div>
  );
}
