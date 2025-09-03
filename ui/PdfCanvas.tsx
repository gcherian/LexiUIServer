import { useRef, useEffect, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
// @ts-ignore
import pdfjsWorker from "pdfjs-dist/build/pdf.worker.min.js?url";
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorker;

type Box = { x0:number; y0:number; x1:number; y1:number; page:number };
type Props = {
  url: string;
  boxes?: Box[];
  onLasso?: (rect:{x0:number;y0:number;x1:number;y1:number})=>void;
};

export default function PdfCanvas({ url, boxes=[], onLasso }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [viewport, setViewport] = useState<any>(null);
  const [down, setDown] = useState<{x:number;y:number}|null>(null);

  useEffect(()=>{
    (async()=>{
      const doc = await pdfjsLib.getDocument(url).promise;
      const page = await doc.getPage(1);
      const vp = page.getViewport({ scale: 1.5 });
      const canvas = canvasRef.current!;
      canvas.width = vp.width; canvas.height = vp.height;
      await page.render({ canvasContext: canvas.getContext("2d"), viewport: vp }).promise;
      setViewport(vp);
    })();
  }, [url]);

  useEffect(()=>{
    if (!viewport) return;
    const ov = overlayRef.current!, ctx = ov.getContext("2d")!;
    ov.width = viewport.width; ov.height = viewport.height;
    ctx.clearRect(0,0,ov.width,ov.height);
    ctx.fillStyle = "rgba(0,160,255,.25)"; ctx.strokeStyle = "rgba(0,120,200,.9)"; ctx.lineWidth = 2;
    for (const b of boxes){
      const r = viewport.convertToViewportRectangle([b.x0,b.y0,b.x1,b.y1]);
      const x = Math.min(r[0], r[2]), y = Math.min(r[1], r[3]);
      const w = Math.abs(r[2]-r[0]), h = Math.abs(r[3]-r[1]);
      ctx.fillRect(x,y,w,h); ctx.strokeRect(x,y,w,h);
    }
  }, [boxes, viewport]);

  const toPdf = (x:number,y:number)=>{
    const pt = viewport.convertToPdfPoint(x,y);
    return { x: pt[0], y: pt[1] };
  };

  return (
    <div style={{position:"relative", display:"inline-block"}}>
      <canvas ref={canvasRef} style={{display:"block"}} />
      <canvas ref={overlayRef}
        style={{position:"absolute", top:0, left:0, cursor:"crosshair"}}
        onMouseDown={(e)=>{
          const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
          setDown({ x: e.clientX - r.left, y: e.clientY - r.top });
        }}
        onMouseUp={(e)=>{
          if (!down || !onLasso) return;
          const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
          const a = toPdf(down.x, down.y);
          const b = toPdf(e.clientX - r.left, e.clientY - r.top);
          onLasso({ x0: Math.min(a.x,b.x), y0: Math.min(a.y,b.y), x1: Math.max(a.x,b.x), y1: Math.max(a.y,b.y) });
          setDown(null);
        }}
        width={viewport?.width||0} height={viewport?.height||0}
      />
    </div>
  );
}
