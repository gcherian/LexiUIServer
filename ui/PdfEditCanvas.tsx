import React, { useEffect, useRef } from "react";
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy, type PDFPageProxy } from "pdfjs-dist";
import "pdfjs-dist/web/pdf_viewer.css";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.js",
  import.meta.url
).toString();

export type Token = { page: number; x0: number; y0: number; x1: number; y1: number; text?: string };
export type EditRect = { page: number; x0: number; y0: number; x1: number; y1: number };

type Props = {
  docUrl: string;
  page: number;
  serverW: number; // OCR-space width for this page
  serverH: number; // OCR-space height for this page
  tokens: Token[];
  rect: EditRect | null;
  showTokenBoxes: boolean;
  editable: boolean;
  onRectChange: (r: EditRect | null) => void;
  onRectCommit: (r: EditRect) => void;
};

export default function PdfEditCanvas({
  docUrl,
  page,
  serverW,
  serverH,
  tokens,
  rect,
  showTokenBoxes,
  editable,
  onRectChange,
  onRectCommit,
}: Props) {
  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  const pageRef = useRef<PDFPageProxy | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null); // scroll container
  const baseCanvas = useRef<HTMLCanvasElement | null>(null); // the PDF page bitmap
  const overlay = useRef<HTMLDivElement | null>(null); // interaction layer

  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const dragNow = useRef<{ x: number; y: number } | null>(null);

  // Load + render the page into the base canvas
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!docUrl) return;
      const loading = await getDocument(docUrl).promise;
      if (cancelled) return;
      pdfRef.current = loading;
      const pg = await loading.getPage(page);
      if (cancelled) return;
      pageRef.current = pg;

      const viewport = pg.getViewport({ scale: 1.0, rotation: pg.rotate || 0 });
      const scale = Math.min(
        1.0,
        1400 / Math.max(viewport.width, viewport.height) // sanity cap for very large pages
      );

      const vp = pg.getViewport({ scale, rotation: pg.rotate || 0 });
      const canvas = baseCanvas.current!;
      const ctx = canvas.getContext("2d")!;
      canvas.width = Math.floor(vp.width);
      canvas.height = Math.floor(vp.height);
      canvas.style.width = `${Math.floor(vp.width)}px`;
      canvas.style.height = `${Math.floor(vp.height)}px`;

      await pg.render({ canvasContext: ctx, viewport: vp }).promise;

      // ensure overlay tracks size
      if (overlay.current) {
        overlay.current.style.width = canvas.style.width;
        overlay.current.style.height = canvas.style.height;
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [docUrl, page]);

  // Mouse handlers (lasso)
  function onMouseDown(e: React.MouseEvent) {
    if (!editable || !overlay.current) return;
    e.preventDefault();
    dragStart.current = { x: e.clientX, y: e.clientY };
    dragNow.current = { x: e.clientX, y: e.clientY };
    overlay.current.classList.add("dragging");
  }
  function onMouseMove(e: React.MouseEvent) {
    if (!editable || !dragStart.current) return;
    dragNow.current = { x: e.clientX, y: e.clientY };
    drawOverlay();
  }
  function onMouseUp(e: React.MouseEvent) {
    if (!editable) return;

    const start = dragStart.current;
    const now = dragNow.current;
    dragStart.current = null;
    dragNow.current = null;
    if (overlay.current) overlay.current.classList.remove("dragging");
    if (!start || !now || !baseCanvas.current || !overlay.current || !wrapRef.current) return;

    // --- compute CSS rect inside the canvas, ADD scroll offsets ----
    const canvRect = baseCanvas.current.getBoundingClientRect();
    const wrap = wrapRef.current;
    const scrollX = wrap.scrollLeft;
    const scrollY = wrap.scrollTop;

    const xCss0 = Math.min(start.x, now.x) - canvRect.left + scrollX;
    const yCss0 = Math.min(start.y, now.y) - canvRect.top + scrollY;
    const xCss1 = Math.max(start.x, now.x) - canvRect.left + scrollX;
    const yCss1 = Math.max(start.y, now.y) - canvRect.top + scrollY;

    // clamp to canvas css bounds
    const X0 = Math.max(0, Math.min(xCss0, canvRect.width));
    const Y0 = Math.max(0, Math.min(yCss0, canvRect.height));
    const X1 = Math.max(0, Math.min(xCss1, canvRect.width));
    const Y1 = Math.max(0, Math.min(yCss1, canvRect.height));

    // --- map CSS → OCR pixels, with rotation support -------------
    const rot = (pageRef.current?.rotate || 0) % 360;
    const cssW = canvRect.width;
    const cssH = canvRect.height;

    // CSS→server scale
    const sx = serverW / cssW;
    const sy = serverH / cssH;

    // helper: map one point
    function mapPoint(xc: number, yc: number) {
      let x = xc * sx;
      let y = yc * sy;
      // rotate **to** OCR coordinate frame (OCR is unrotated; pdf.js already rotated the bitmap we see)
      // If pdf.js applied a 90/270 render rotation, our CSS has swapped axes; undo to OCR space:
      switch (((rot % 360) + 360) % 360) {
        case 0:
          return { x, y };
        case 90:
          return { x: y, y: serverW - x };
        case 180:
          return { x: serverW - x, y: serverH - y };
        case 270:
          return { x: serverH - y, y: x };
        default:
          return { x, y };
      }
    }

    const p0 = mapPoint(X0, Y0);
    const p1 = mapPoint(X1, Y1);

    const x0 = Math.max(0, Math.min(Math.floor(Math.min(p0.x, p1.x)), serverW - 1));
    const y0 = Math.max(0, Math.min(Math.floor(Math.min(p0.y, p1.y)), serverH - 1));
    const x1 = Math.max(0, Math.min(Math.ceil(Math.max(p0.x, p1.x)), serverW - 1));
    const y1 = Math.max(0, Math.min(Math.ceil(Math.max(p0.y, p1.y)), serverH - 1));

    const rr: EditRect = { page, x0, y0, x1, y1 };
    onRectChange(rr);
    onRectCommit(rr);
  }

  // Draw interaction overlay (pink live rectangle + token boxes)
  function drawOverlay() {
    if (!overlay.current || !baseCanvas.current) return;
    const el = overlay.current;
    el.innerHTML = "";

    // tokens
    if (showTokenBoxes) {
      for (const t of tokens) {
        const d = document.createElement("div");
        d.className = "tok";
        positionCss(d, t.x0, t.y0, t.x1, t.y1);
        el.appendChild(d);
      }
    }

    // current rect
    if (rect && rect.page === page) {
      const d = document.createElement("div");
      d.className = "pink";
      positionCss(d, rect.x0, rect.y0, rect.x1, rect.y1);
      el.appendChild(d);
    }

    // live drag box
    if (dragStart.current && dragNow.current) {
      const r = baseCanvas.current.getBoundingClientRect();
      const wrap = wrapRef.current!;
      const sx = serverW / r.width;
      const sy = serverH / r.height;

      const xCss0 = Math.min(dragStart.current.x, dragNow.current.x) - r.left + wrap.scrollLeft;
      const yCss0 = Math.min(dragStart.current.y, dragNow.current.y) - r.top + wrap.scrollTop;
      const xCss1 = Math.max(dragStart.current.x, dragNow.current.x) - r.left + wrap.scrollLeft;
      const yCss1 = Math.max(dragStart.current.y, dragNow.current.y) - r.top + wrap.scrollTop;

      // show the live box in CSS space (no rotation for display)
      const live = document.createElement("div");
      live.className = "pink live";
      live.style.left = `${Math.min(xCss0, xCss1)}px`;
      live.style.top = `${Math.min(yCss0, yCss1)}px`;
      live.style.width = `${Math.abs(xCss1 - xCss0)}px`;
      live.style.height = `${Math.abs(yCss1 - yCss0)}px`;
      el.appendChild(live);
    }
  }

  // helper: convert OCR-space box → CSS position (display only, rotation already baked into bitmap)
  function positionCss(node: HTMLDivElement, x0: number, y0: number, x1: number, y1: number) {
    if (!baseCanvas.current) return;
    const r = baseCanvas.current.getBoundingClientRect();
    const cssW = r.width;
    const cssH = r.height;
    const sx = cssW / serverW;
    const sy = cssH / serverH;

    // pdf.js renders with rotation applied to the bitmap we see.
    // For display, map OCR coords back into that bitmap orientation.
    const rot = (pageRef.current?.rotate || 0) % 360;

    function fwd(x: number, y: number) {
      switch (((rot % 360) + 360) % 360) {
        case 0:
          return { x: x * sx, y: y * sy };
        case 90:
          return { x: (serverH - y) * sx, y: x * sy };
        case 180:
          return { x: (serverW - x) * sx, y: (serverH - y) * sy };
        case 270:
          return { x: y * sx, y: (serverW - x) * sy };
        default:
          return { x: x * sx, y: y * sy };
      }
    }

    const a = fwd(Math.min(x0, x1), Math.min(y0, y1));
    const b = fwd(Math.max(x0, x1), Math.max(y0, y1));

    node.style.left = `${Math.min(a.x, b.x)}px`;
    node.style.top = `${Math.min(a.y, b.y)}px`;
    node.style.width = `${Math.abs(b.x - a.x)}px`;
    node.style.height = `${Math.abs(b.y - a.y)}px`;
  }

  // redraw when inputs change
  useEffect(() => {
    drawOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokens, rect, showTokenBoxes, page, serverW, serverH]);

  return (
    <div ref={wrapRef} className="pdf-stage" style={{ position: "relative", overflow: "auto" }}>
      <canvas ref={baseCanvas} />
      <div
        ref={overlay}
        className="overlay"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        style={{
          position: "absolute",
          inset: 0,
          cursor: editable ? "crosshair" : "default",
          userSelect: "none",
        }}
      />
    </div>
  );
}
