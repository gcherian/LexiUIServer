// File: src/components/lasso/PdfCanvas.tsx
import React, { useEffect, useRef } from "react";
// TOP OF FILE: replace the worker wiring
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy, type PDFPageProxy } from "pdfjs-dist";
// Use vite-friendly worker path
GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.js", import.meta.url).toString();


/** Minimal box shape expected from the server. */
type BoxLike = {
  page: number;
  x0: number; y0: number; x1: number; y1: number; // TOP-LEFT origin (server space)
  id?: string | null;
  label?: string | null;
  text?: string | null;
  confidence?: number | null;
};

export type RectServer = { x0: number; y0: number; x1: number; y1: number };

type Props = {
  /** Absolute/relative URL to the PDF (e.g., /data/{doc_id}/original.pdf) */
  docUrl: string;
  /** 1-based page number to render */
  page: number;

  /**
   * Server coordinate-space dimensions for the current page (from /lasso/doc/{id}/meta).
   * These are required to scale between PDF canvas pixels and server OCR coordinates.
   */
  serverW: number;
  serverH: number;

  /** Boxes to draw (typically already filtered by page, but we re-check page for safety). */
  boxes: BoxLike[];

  /** Whether to render the bounding boxes overlay. */
  showBoxes: boolean;

  /** Enable lasso mode (drag to draw). Emits onLassoDone with server-space rect. */
  lasso: boolean;

  /** Optional: visually mark a selected box by id (or synthetic id) */
  selectedBoxId?: string | null;

  /** When a box is clicked */
  onBoxClick?: (box: BoxLike) => void;

  /** When a lasso drag completes (rect is in SERVER coordinates, top-left origin) */
  onLassoDone?: (rect: RectServer) => void;
};

export default function PdfCanvas({
  docUrl,
  page,
  serverW,
  serverH,
  boxes,
  showBoxes,
  lasso,
  selectedBoxId,
  onBoxClick,
  onLassoDone,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<SVGSVGElement | null>(null);
  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  const lastViewportSize = useRef<{ w: number; h: number }>({ w: 1, h: 1 });
  const scaleRef = useRef<{ sx: number; sy: number }>({ sx: 1, sy: 1 });

  // Load the PDF once per docUrl
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const doc = await getDocument(docUrl).promise;
      if (cancelled) return;
      pdfRef.current = doc;
      await renderPage();
    })().catch(() => {});
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl]);

  // Re-render when inputs change
  useEffect(() => {
    renderPage().catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, showBoxes, boxes, selectedBoxId, lasso, serverW, serverH]);

  // Repaint when the number of boxes or server dims change (first-load fix)
  useEffect(() => {
    if (pdfRef.current) { renderPage().catch(()=>{}); }
  }, [boxes.length, serverW, serverH]); // keeps first-load boxes from requiring a toggle

  async function renderPage() {
    if (!pdfRef.current || !canvasRef.current || !overlayRef.current) return;
    const pdf = pdfRef.current;
    const pg: PDFPageProxy = await pdf.getPage(page);

    // Choose a canvas scale. 1.5 is a good compromise of sharpness/perf.
    const viewport = pg.getViewport({ scale: 1.5 });
    lastViewportSize.current = { w: viewport.width, h: viewport.height };

    // Render canvas
    const cvs = canvasRef.current;
    const ctx = cvs.getContext("2d")!;
    cvs.width = viewport.width;
    cvs.height = viewport.height;
    await pg.render({ canvasContext: ctx, viewport }).promise;

    // Compute scale factors from server-space → canvas-space
    const sx = viewport.width / Math.max(1, serverW);
    const sy = viewport.height / Math.max(1, serverH);
    scaleRef.current = { sx, sy };

    // Prepare overlay SVG
    const svg = overlayRef.current;
    svg.setAttribute("viewBox", `0 0 ${viewport.width} ${viewport.height}`);
    svg.setAttribute("width", String(viewport.width));
    svg.setAttribute("height", String(viewport.height));
    // Clear previous children
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    if (showBoxes) drawBoxes(svg, sx, sy);
  }

  /** Draw all boxes + numeric labels. */
  function drawBoxes(svg: SVGSVGElement, sx: number, sy: number) {
    // Use a fragment for perf
    const frag = document.createDocumentFragment();
    let drawIdx = 0;

    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      if (b.page !== page) continue;

      const gx = b.x0 * sx;
      const gy = b.y0 * sy; // top-left origin stays top-left
      const gw = (b.x1 - b.x0) * sx;
      const gh = (b.y1 - b.y0) * sy;

      const syntheticId = b.id || `${b.page}:${b.x0}:${b.y0}:${i}`;
      const selected = selectedBoxId && syntheticId === selectedBoxId;

      // Rectangle
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", String(gx));
      rect.setAttribute("y", String(gy));
      rect.setAttribute("width", String(gw));
      rect.setAttribute("height", String(gh));
      rect.setAttribute("class", selected ? "bbox-rect selected" : "bbox-rect");
      rect.dataset["boxId"] = syntheticId;
      rect.addEventListener("click", (e) => {
        e.stopPropagation();
        onBoxClick?.(b);
      });
      frag.appendChild(rect);

      // Numeric label background
      const tag = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      tag.setAttribute("x", String(Math.max(gx - 1, 0)));
      tag.setAttribute("y", String(Math.max(gy - 14, 0)));
      tag.setAttribute("width", "16");
      tag.setAttribute("height", "14");
      tag.setAttribute("rx", "2");
      tag.setAttribute("ry", "2");
      tag.setAttribute("class", "box-tag");
      frag.appendChild(tag);

      // Numeric label text
      const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
      txt.setAttribute("x", String(Math.max(gx - 1, 0) + 8));
      txt.setAttribute("y", String(Math.max(gy - 14, 0) + 8));
      txt.setAttribute("class", "box-tag-text");
      txt.textContent = String(++drawIdx);
      frag.appendChild(txt);
    }

    svg.appendChild(frag);
  }

  // ---- Lasso handling (SVG overlay coords -> server coords) ----
  function svgPointFromClient(e: React.MouseEvent<SVGSVGElement>) {
    const svg = overlayRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const m = svg.getScreenCTM();
    const p = m ? pt.matrixTransform(m.inverse()) : ({ x: 0, y: 0 } as any);
    return { x: p.x as number, y: p.y as number };
  }

  function onMouseDown(e: React.MouseEvent<SVGSVGElement>) {
    if (!lasso) return;
    const { x, y } = svgPointFromClient(e);
    const svg = overlayRef.current!;
    svg.dataset["sx"] = String(x);
    svg.dataset["sy"] = String(y);

    // temp drawing rect
    const tmp = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    tmp.setAttribute("id", "__lasso__");
    tmp.setAttribute("class", "bbox-rect drawing");
    tmp.setAttribute("x", String(x));
    tmp.setAttribute("y", String(y));
    tmp.setAttribute("width", "0");
    tmp.setAttribute("height", "0");
    svg.appendChild(tmp);
  }

  function onMouseMove(e: React.MouseEvent<SVGSVGElement>) {
    if (!lasso) return;
    const svg = overlayRef.current!;
    const sx = Number(svg.dataset["sx"] || NaN);
    const sy = Number(svg.dataset["sy"] || NaN);
    if (Number.isNaN(sx) || Number.isNaN(sy)) return;

    const { x, y } = svgPointFromClient(e);
    const x0 = Math.min(sx, x), y0 = Math.min(sy, y);
    const x1 = Math.max(sx, x), y1 = Math.max(sy, y);

    const tmp = svg.querySelector("#__lasso__") as SVGRectElement | null;
    if (tmp) {
      tmp.setAttribute("x", String(x0));
      tmp.setAttribute("y", String(y0));
      tmp.setAttribute("width", String(x1 - x0));
      tmp.setAttribute("height", String(y1 - y0));
    }
  }

function onMouseUp() {
  if (!lasso) return;
  const svg = overlayRef.current!;
  const tmp = svg.querySelector("#__lasso__") as SVGRectElement | null;

  // Read before cleanup
  const sxData = Number(svg.dataset["sx"] || NaN);
  const syData = Number(svg.dataset["sy"] || NaN);

  // Canvas/SVG-space rect
  const ex = Number.isFinite(parseFloat(tmp?.getAttribute("x") || "")) ? parseFloat(tmp!.getAttribute("x")!) : sxData;
  const ey = Number.isFinite(parseFloat(tmp?.getAttribute("y") || "")) ? parseFloat(tmp!.getAttribute("y")!) : syData;
  const ew = Number.isFinite(parseFloat(tmp?.getAttribute("width") || ""))  ? parseFloat(tmp!.getAttribute("width")!)  : 0;
  const eh = Number.isFinite(parseFloat(tmp?.getAttribute("height") || "")) ? parseFloat(tmp!.getAttribute("height")!) : 0;

  // Cleanup
  svg.dataset["sx"] = "";
  svg.dataset["sy"] = "";
  if (tmp) tmp.remove();

  if (Number.isNaN(sxData) || Number.isNaN(syData)) return;

  // Normalize canvas-space corners
  let x0c = Math.min(ex, ex + ew), y0c = Math.min(ey, ey + eh);
  let x1c = Math.max(ex, ex + ew), y1c = Math.max(ey, ey + eh);

  // Convert canvas/svg px -> server px.
  // IMPORTANT: flip Y using canvas height so server crop matches visual selection.
  const { sx, sy } = scaleRef.current;               // canvas px per server px
  const vy = lastViewportSize.current.h;             // canvas (viewport) height in px

  // Bottom-left conversion for Y (fixes "north-west" drift):
  let X0 = Math.floor(x0c / sx);
  let X1 = Math.ceil (x1c / sx);
  let Y0 = Math.floor((vy - y1c) / sy);              // <-- flipped
  let Y1 = Math.ceil ((vy - y0c) / sy);              // <-- flipped

  // Clamp
  X0 = clamp(X0, 0, Math.max(0, serverW - 1));
  X1 = clamp(X1, 0, Math.max(0, serverW - 1));
  Y0 = clamp(Y0, 0, Math.max(0, serverH - 1));
  Y1 = clamp(Y1, 0, Math.max(0, serverH - 1));

  if (X1 - X0 < 2 || Y1 - Y0 < 2) return;

  onLassoDone?.({ x0: X0, y0: Y0, x1: X1, y1: Y1 });
}
  function clamp(n: number, lo: number, hi: number) {
    return Math.min(hi, Math.max(lo, n));
  }

  return (
    <div className="pdf-stage">
      <canvas ref={canvasRef} />
      <svg
        ref={overlayRef}
        className={lasso ? "overlay crosshair" : "overlay"}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
      />
    </div>
  );
}
