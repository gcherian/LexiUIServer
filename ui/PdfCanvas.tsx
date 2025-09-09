import React, { useEffect, useRef } from "react";
import type { Box, Rect } from "../../lib/api";
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy, type PDFPageProxy } from "pdfjs-dist";
import pdfjsWorker from "pdfjs-dist/build/pdf.worker.min?url";
GlobalWorkerOptions.workerSrc = pdfjsWorker as string;

type Props = {
  docUrl: string;
  page: number;
  serverW: number;
  serverH: number;
  boxes: Box[];
  showBoxes: boolean;
  lasso: boolean;

  onBoxClick?: (box: Box) => void;
  onLassoDone?: (rectServer: Rect) => void;
};

export default function PdfCanvas({
  docUrl, page, serverW, serverH, boxes, showBoxes, lasso, onBoxClick, onLassoDone
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement|null>(null);
  const overlayRef = useRef<SVGSVGElement|null>(null);
  const pdfRef = useRef<PDFDocumentProxy|null>(null);
  const scaleRef = useRef<{sx:number; sy:number}>({ sx:1, sy:1 });

  // load doc once
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const doc = await getDocument(docUrl).promise;
      if (cancelled) return;
      pdfRef.current = doc;
      await render();
    })().catch(()=>{});
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl]);

  // re-render when inputs change
  useEffect(() => { render().catch(()=>{}); }, [page, boxes, showBoxes, lasso, serverW, serverH]);

  async function render() {
    if (!pdfRef.current || !canvasRef.current || !overlayRef.current) return;
    const pdf = pdfRef.current;
    const p: PDFPageProxy = await pdf.getPage(page);
    const viewport = p.getViewport({ scale: 1.5 }); // any scale; we compute mapping
    const cvs = canvasRef.current, ctx = cvs.getContext("2d")!;
    cvs.width = viewport.width; cvs.height = viewport.height;
    await p.render({ canvasContext: ctx, viewport }).promise;

    const sx = viewport.width / Math.max(1, serverW);
    const sy = viewport.height / Math.max(1, serverH);
    scaleRef.current = { sx, sy };

    const svg = overlayRef.current;
    svg.setAttribute("viewBox", `0 0 ${viewport.width} ${viewport.height}`);
    svg.setAttribute("width", `${viewport.width}`);
    svg.setAttribute("height", `${viewport.height}`);
    svg.innerHTML = "";

    if (showBoxes) {
      boxes.forEach((b) => {
        if (b.page !== page) return;
        const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        r.setAttribute("x", String(b.x0 * sx));
        r.setAttribute("y", String(b.y0 * sy));         // Tesseract coords are top-left origin
        r.setAttribute("width", String((b.x1 - b.x0) * sx));
        r.setAttribute("height", String((b.y1 - b.y0) * sy));
        r.setAttribute("class", "bbox-rect");
        r.addEventListener("click", () => onBoxClick?.(b));
        svg.appendChild(r);
      });
    }
  }

  // lasso interactions
  function toSvg(e: React.MouseEvent<SVGSVGElement>) {
    const svg = overlayRef.current!;
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const m = svg.getScreenCTM(); const p = m ? pt.matrixTransform(m.inverse()) : ({ x:0, y:0 } as any);
    return { x: p.x as number, y: p.y as number };
  }
  function onDown(e: React.MouseEvent<SVGSVGElement>) {
    if (!lasso) return;
    const { x, y } = toSvg(e);
    overlayRef.current!.dataset["sx"] = String(x);
    overlayRef.current!.dataset["sy"] = String(y);
    const tmp = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    tmp.setAttribute("id", "__lasso__");
    tmp.setAttribute("class", "bbox-rect drawing");
    tmp.setAttribute("x", String(x)); tmp.setAttribute("y", String(y));
    tmp.setAttribute("width","0"); tmp.setAttribute("height","0");
    overlayRef.current!.appendChild(tmp);
  }
  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    if (!lasso) return;
    const sx = Number(overlayRef.current!.dataset["sx"]||NaN);
    const sy = Number(overlayRef.current!.dataset["sy"]||NaN);
    if (Number.isNaN(sx) || Number.isNaN(sy)) return;
    const { x, y } = toSvg(e);
    const x0 = Math.min(sx,x), y0 = Math.min(sy,y), x1 = Math.max(sx,x), y1 = Math.max(sy,y);
    const tmp = overlayRef.current!.querySelector("#__lasso__") as SVGRectElement | null;
    if (tmp) { tmp.setAttribute("x", String(x0)); tmp.setAttribute("y", String(y0)); tmp.setAttribute("width", String(x1-x0)); tmp.setAttribute("height", String(y1-y0)); }
  }
  function onUp() {
    if (!lasso) return;
    const sx = Number(overlayRef.current!.dataset["sx"]||NaN);
    const sy = Number(overlayRef.current!.dataset["sy"]||NaN);
    const tmp = overlayRef.current!.querySelector("#__lasso__"); if (tmp) tmp.remove();
    overlayRef.current!.dataset["sx"]=""; overlayRef.current!.dataset["sy"]="";
    if (Number.isNaN(sx) || Number.isNaN(sy)) return;

    // Convert to server coordinate space (top-left origin)
    const ex = Number((tmp as any)?.getAttribute?.("x") ?? sx);
    const ey = Number((tmp as any)?.getAttribute?.("y") ?? sy);
    const ew = Number((tmp as any)?.getAttribute?.("width") ?? 0);
    const eh = Number((tmp as any)?.getAttribute?.("height") ?? 0);
    const { sx:scx, sy:scy } = scaleRef.current;
    const rect: Rect = { x0: Math.round(ex / scx), y0: Math.round(ey / scy), x1: Math.round((ex+ew)/scx), y1: Math.round((ey+eh)/scy) };
    onLassoDone?.(rect);
  }

  return (
    <div className="pdf-stage">
      <canvas ref={canvasRef}/>
      <svg
        ref={overlayRef}
        className={lasso ? "overlay crosshair" : "overlay"}
        onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp}
      />
    </div>
  );
}
