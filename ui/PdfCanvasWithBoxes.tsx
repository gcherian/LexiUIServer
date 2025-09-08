import React, { useEffect, useRef, useState } from "react";
import { getBoxes, type Box } from "../../lib/api";
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy, type PDFPageProxy } from "pdfjs-dist";
import pdfjsWorker from "pdfjs-dist/build/pdf.worker.min?url";
GlobalWorkerOptions.workerSrc = pdfjsWorker as string;

export type LassoRect = { x0: number; y0: number; x1: number; y1: number };

type Props = {
  docUrl: string;
  page: number;
  showBoxes: boolean;
  onPageChange?: (next: number, total: number) => void;
  onReady?: (totalPages: number, viewportHeight: number) => void;
  onBoxClick?: (box: Box) => void;
  onLasso?: (rect: LassoRect) => void;
};

export default function PdfCanvasWithBoxes({
  docUrl, page, showBoxes, onPageChange, onReady, onBoxClick, onLasso
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<SVGSVGElement | null>(null);
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [total, setTotal] = useState<number>(0);
  const [viewportH, setViewportH] = useState<number>(0);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [lasso, setLasso] = useState<LassoRect | null>(null);
  const [isLasso, setIsLasso] = useState<boolean>(false);

  // load pdf
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!docUrl) return;
      const doc = await getDocument(docUrl).promise;
      if (cancelled) return;
      setPdf(doc);
      setTotal(doc.numPages);
      onReady?.(doc.numPages, viewportH);
      onPageChange?.(1, doc.numPages);
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line
  }, [docUrl]);

  // fetch boxes per page
  useEffect(() => {
    (async () => {
      if (!docUrl) return;
      const b = await getBoxes({ doc_url: docUrl, page });
      setBoxes(b || []);
    })();
  }, [docUrl, page]);

  // draw
  useEffect(() => {
    if (!pdf || !canvasRef.current || !overlayRef.current) return;
    (async () => {
      const ctx = canvasRef.current!.getContext("2d")!;
      const p: PDFPageProxy = await pdf.getPage(page);
      const viewport = p.getViewport({ scale: 1.5 });
      canvasRef.current!.width = viewport.width;
      canvasRef.current!.height = viewport.height;
      setViewportH(viewport.height);
      await p.render({ canvasContext: ctx, viewport }).promise;

      const svg = overlayRef.current!;
      svg.setAttribute("viewBox", `0 0 ${viewport.width} ${viewport.height}`);
      svg.setAttribute("width", `${viewport.width}`);
      svg.setAttribute("height", `${viewport.height}`);
      svg.innerHTML = "";

      if (showBoxes) {
        boxes.forEach((b) => {
          const id = b.id || `${b.page}:${b.x0}:${b.y0}`;
          const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          rect.setAttribute("x", `${b.x0}`);
          rect.setAttribute("y", `${viewport.height - b.y1}`);
          rect.setAttribute("width", `${b.x1 - b.x0}`);
          rect.setAttribute("height", `${b.y1 - b.y0}`);
          rect.setAttribute("class", "bbox-rect");
          rect.addEventListener("click", () => onBoxClick?.(b));
          svg.appendChild(rect);
        });
      }
      if (lasso) {
        const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        r.setAttribute("x", `${lasso.x0}`);
        r.setAttribute("y", `${viewport.height - lasso.y1}`);
        r.setAttribute("width", `${lasso.x1 - lasso.x0}`);
        r.setAttribute("height", `${lasso.y1 - lasso.y0}`);
        r.setAttribute("class", "bbox-rect drawing");
        svg.appendChild(r);
      }
    })();
  }, [pdf, page, boxes, showBoxes, lasso]);

  // lasso handlers
  function toSvgPoint(e: React.MouseEvent<SVGSVGElement>) {
    const svg = overlayRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const m = svg.getScreenCTM();
    const p = m ? pt.matrixTransform(m.inverse()) : ({ x: 0, y: 0 } as any);
    return { x: p.x as number, y: p.y as number };
  }
  function onMouseDown(e: React.MouseEvent<SVGSVGElement>) {
    if (!isLasso) return;
    const { x, y } = toSvgPoint(e);
    setLasso({ x0: x, y0: viewportH - y, x1: x, y1: viewportH - y });
  }
  function onMouseMove(e: React.MouseEvent<SVGSVGElement>) {
    if (!isLasso || !lasso) return;
    const { x, y } = toSvgPoint(e);
    setLasso(prev => prev ? { ...prev, x1: x, y1: viewportH - y } : prev);
  }
  function onMouseUp() {
    if (!isLasso || !lasso) return;
    onLasso?.(lasso);
    setIsLasso(false);
  }

  return (
    <div className="pdf-stage">
      <div className="toolbar-inline">
        <button disabled={!pdf || page<=1} onClick={() => onPageChange?.(page-1, total)}>Prev</button>
        <span className="page-indicator">Page {page}{pdf ? ` / ${total}` : ""}</span>
        <button disabled={!pdf || page>=total} onClick={() => onPageChange?.(page+1, total)}>Next</button>
        <label className={showBoxes ? "btn toggle active" : "btn toggle"} style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={showBoxes} readOnly /> Boxes
        </label>
        <button className={isLasso ? "btn toggle active" : "btn toggle"} onClick={() => setIsLasso(v=>!v)}>Lasso</button>
      </div>
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