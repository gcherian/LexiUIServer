import React, { useEffect, useRef } from "react";
import { type Box } from "../../lib/api";
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy, type PDFPageProxy } from "pdfjs-dist";
import pdfjsWorker from "pdfjs-dist/build/pdf.worker.min?url";
GlobalWorkerOptions.workerSrc = pdfjsWorker as string;

export type ViewportInfo = { width: number; height: number };
export type RectImg = { x0:number; y0:number; x1:number; y1:number }; // image coords, origin bottom-left

type Props = {
  docUrl: string;
  page: number;
  showBoxes: boolean;
  isLasso: boolean;
  boxes: Box[];
  selectedBoxId?: string | null;

  onPageRender?: (vp: ViewportInfo) => void;
  onBoxClick?: (b: Box) => void;
  onLasso?: (rect: RectImg) => void;
};

export default function PdfCanvasWithBoxes({
  docUrl, page, showBoxes, isLasso, boxes, selectedBoxId, onPageRender, onBoxClick, onLasso,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement|null>(null);
  const overlayRef = useRef<SVGSVGElement|null>(null);
  const pdfRef = useRef<PDFDocumentProxy|null>(null);
  const vpH = useRef<number>(0);

  // load or re-use the same document
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const doc = await getDocument(docUrl).promise;
      if (cancelled) return;
      pdfRef.current = doc;
      await renderPage();
    })().catch(()=>{});
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl]);

  // re-render on page / boxes / selection / toggle / lasso state
  useEffect(() => { renderPage().catch(()=>{}); }, [page, showBoxes, boxes, selectedBoxId, isLasso]);

  async function renderPage() {
    if (!pdfRef.current || !canvasRef.current || !overlayRef.current) return;
    const pdf = pdfRef.current;
    const p: PDFPageProxy = await pdf.getPage(page);
    const viewport = p.getViewport({ scale: 1.5 });

    const canvas = canvasRef.current; const ctx = canvas.getContext("2d")!;
    canvas.width = viewport.width; canvas.height = viewport.height;
    vpH.current = viewport.height;
    await p.render({ canvasContext: ctx, viewport }).promise;

    const svg = overlayRef.current;
    svg.setAttribute("viewBox", `0 0 ${viewport.width} ${viewport.height}`);
    svg.setAttribute("width", `${viewport.width}`);
    svg.setAttribute("height", `${viewport.height}`);
    svg.innerHTML = "";

    if (showBoxes) {
      for (const b of boxes) {
        const id = b.id || `${b.page}:${b.x0}:${b.y0}`;
        const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        r.setAttribute("x", `${b.x0}`);
        r.setAttribute("y", `${viewport.height - b.y1}`); // flip Y
        r.setAttribute("width", `${b.x1 - b.x0}`);
        r.setAttribute("height", `${b.y1 - b.y0}`);
        r.setAttribute("class", id === selectedBoxId ? "bbox-rect selected" : "bbox-rect");
        r.addEventListener("click", () => onBoxClick?.(b));
        svg.appendChild(r);
      }
    }
    onPageRender?.({ width: viewport.width, height: viewport.height });
  }

  // lasso
  function clientToSvg(e: React.MouseEvent<SVGSVGElement>) {
    const svg = overlayRef.current!;
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const m = svg.getScreenCTM(); const p = m ? pt.matrixTransform(m.inverse()) : ({ x:0, y:0 } as any);
    return { x: p.x as number, y: p.y as number };
  }
  function onMouseDown(e: React.MouseEvent<SVGSVGElement>) {
    if (!isLasso) return;
    const { x, y } = clientToSvg(e);
    // store as data attrs (only needed within component scope)
    overlayRef.current!.dataset["lassoStartX"] = String(x);
    overlayRef.current!.dataset["lassoStartY"] = String(y);
    // temp box
    const temp = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    temp.setAttribute("id", "__drawing__");
    temp.setAttribute("class", "bbox-rect drawing");
    temp.setAttribute("x", String(x));
    temp.setAttribute("y", String(y));
    temp.setAttribute("width", "0");
    temp.setAttribute("height", "0");
    overlayRef.current!.appendChild(temp);
  }
  function onMouseMove(e: React.MouseEvent<SVGSVGElement>) {
    if (!isLasso) return;
    const startX = Number(overlayRef.current!.dataset["lassoStartX"] || NaN);
    const startY = Number(overlayRef.current!.dataset["lassoStartY"] || NaN);
    if (Number.isNaN(startX) || Number.isNaN(startY)) return;
    const { x, y } = clientToSvg(e);
    const x0 = Math.min(startX, x), x1 = Math.max(startX, x);
    const y0 = Math.min(startY, y), y1 = Math.max(startY, y);
    const temp = overlayRef.current!.querySelector("#__drawing__") as SVGRectElement | null;
    if (temp) { temp.setAttribute("x", String(x0)); temp.setAttribute("y", String(y0)); temp.setAttribute("width", String(x1-x0)); temp.setAttribute("height", String(y1-y0)); }
  }
  function onMouseUp() {
    if (!isLasso) return;
    const startX = Number(overlayRef.current!.dataset["lassoStartX"] || NaN);
    const startY = Number(overlayRef.current!.dataset["lassoStartY"] || NaN);
    const temp = overlayRef.current!.querySelector("#__drawing__"); if (temp) temp.remove();
    overlayRef.current!.dataset["lassoStartX"] = ""; overlayRef.current!.dataset["lassoStartY"] = "";
    if (Number.isNaN(startX) || Number.isNaN(startY)) return;

    // compute final box from element last size
    // NOTE: overlay's y origin is top-left; image origin is bottom-left
    // We already stored SVG top-left dims; convert to image coords:
    const rect = (temp as any) as SVGRectElement | null;
    // Safety: if we removed temp already, fallback to start-only
    const x = startX, y = startY;
    const w = rect ? Number((rect as any).getAttribute?.("width") || 0) : 0;
    const h = rect ? Number((rect as any).getAttribute?.("height") || 0) : 0;
    const X0 = x, Y0Top = y, X1 = x + w, Y1Top = y + h;
    const img = { x0: X0, y0: vpH.current - Y1Top, x1: X1, y1: vpH.current - Y0Top };
    onLasso?.(img);
  }

  return (
    <div className="pdf-stage">
      <canvas ref={canvasRef} />
      <svg
        ref={overlayRef}
        className={isLasso ? "overlay crosshair" : "overlay"}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
      />
    </div>
  );
}
