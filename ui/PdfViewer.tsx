import { useEffect, useRef, useState } from "react";
import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";
import "./ocr.css";

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

export default function PdfViewer({
  url, page, scale, ocrSize, bindKey,
  onClose, onChangePage, onZoom, onLasso
}: Props){
  const pdfRef = useRef<any>(null);
  const [vp, setVp] = useState<any>(null);
  const baseCanvas = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    (async () => {
      pdfRef.current = await getDocument(url).promise;
      const p = await pdfRef.current.getPage(page);
      const viewport = p.getViewport({ scale });
      const c = baseCanvas.current!; c.width = viewport.width; c.height = viewport.height;
      const ctx = c.getContext("2d"); if (!ctx) return;
      await p.render({ canvasContext: ctx, viewport }).promise;
      setVp(viewport);
    })();
  }, [url]);

  useEffect(() => {
    (async () => {
      if (!pdfRef.current) return;
      const p = await pdfRef.current.getPage(page);
      const viewport = p.getViewport({ scale });
      const c = baseCanvas.current!; c.width = viewport.width; c.height = viewport.height;
      const ctx = c.getContext("2d"); if (!ctx) return;
      await p.render({ canvasContext: ctx, viewport }).promise;
      setVp(viewport);
    })();
  }, [page, scale]);

  // lasso
  const [drag, setDrag] = useState<{x:number;y:number}|null>(null);
  const toOcr = (x:number,y:number) => {
    if (!vp) return { x, y };
    const sx = ocrSize.width  / vp.width;
    const sy = ocrSize.height / vp.height;
    return { x: x * sx, y: y * sy };
  };

  return (
    <div className="modal">
      <div className="modal-frame">
        <div className="modal-header">
          <div className="title">PDF Viewer {bindKey ? <span className="pill">Binding: {bindKey}</span> : null}</div>
          <div className="actions">
            <button onClick={()=> onChangePage(page-1)} disabled={page<=1}>Prev</button>
            <span className="mono">p{page}</span>
            <button onClick={()=> onChangePage(page+1)}>Next</button>
            <span style={{width:8}}/>
            <button onClick={()=> onZoom(Math.max(0.5, +(scale-0.25).toFixed(2)))}>-</button>
            <span className="mono">{scale.toFixed(2)}x</span>
            <button onClick={()=> onZoom(+(scale+0.25).toFixed(2))}>+</button>
            <span style={{width:8}}/>
            <button className="secondary" onClick={onClose}>Close</button>
          </div>
        </div>

        <div className="modal-body">
          <div className="pdf-stage">
            <canvas ref={baseCanvas} className="pdf-base" />
            <canvas
              ref={overlay}
              className="pdf-overlay lasso"
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
                  x0: Math.min(a.x,b.x),
                  y0: Math.min(a.y,b.y),
                  x1: Math.max(a.x,b.x),
                  y1: Math.max(a.y,b.y),
                });
                setDrag(null);
              }}
              width={vp?.width||0}
              height={vp?.height||0}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
