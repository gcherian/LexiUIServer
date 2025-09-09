import React, { useEffect, useRef } from "react";
import {
  GlobalWorkerOptions,
  getDocument,
  type PDFDocumentProxy,
  type PDFPageProxy,
} from "pdfjs-dist";
import "pdfjs-dist/web/pdf_viewer.css";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.js",
  import.meta.url
).toString();

export type TokenBox = {
  page: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  text?: string;
};

export type EditRect = { page: number; x0: number; y0: number; x1: number; y1: number };

type Props = {
  docUrl: string;
  page: number;

  /** OCR/server coordinate space for this page (width/height from /lasso/doc/{id}/meta) */
  serverW: number;
  serverH: number;

  tokens: TokenBox[];
  rect: EditRect | null;
  showTokenBoxes: boolean;
  editable: boolean;

  /** Live rect while dragging and when restored (pink box). */
  onRectChange: (r: EditRect | null) => void;

  /** Commit: fire OCR/bind upstream. */
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
  const baseCanvas = useRef<HTMLCanvasElement | null>(null); // rendered pdf bitmap
  const overlayRef = useRef<HTMLDivElement | null>(null); // interactive overlay

  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const dragNow = useRef<{ x: number; y: number } | null>(null);

  // --- Render page to canvas -------------------------------------------------
  useEffect(() => {
    let stop = false;

    (async () => {
      if (!docUrl) return;
      const doc = await getDocument(docUrl).promise;
      if (stop) return;
      pdfRef.current = doc;

      const pg = await doc.getPage(page);
      if (stop) return;
      pageRef.current = pg;

      // NOTE: render with rotation = 0 so OCR coords match the bitmap directly.
      const v1 = pg.getViewport({ scale: 1, rotation: 0 });
      // keep the page reasonably sized for the UI; we only need a visual
      const scale = Math.min(1, 1400 / Math.max(v1.width, v1.height));
      const vp = pg.getViewport({ scale, rotation: 0 });

      const c = baseCanvas.current!;
      const ctx = c.getContext("2d")!;
      c.width = Math.floor(vp.width);
      c.height = Math.floor(vp.height);
      c.style.width = `${c.width}px`;
      c.style.height = `${c.height}px`;

      await pg.render({ canvasContext: ctx, viewport: vp }).promise;

      if (overlayRef.current) {
        overlayRef.current.style.width = c.style.width;
        overlayRef.current.style.height = c.style.height;
      }

      drawOverlay(); // initial overlay
    })();

    return () => {
      stop = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl, page]);

  // --- Mouse handlers (lasso) -----------------------------------------------
  function onMouseDown(e: React.MouseEvent) {
    if (!editable) return;
    if (!overlayRef.current) return;
    e.preventDefault();
    dragStart.current = { x: e.clientX, y: e.clientY };
    dragNow.current = { x: e.clientX, y: e.clientY };
    drawOverlay();
  }

  function onMouseMove(e: React.MouseEvent) {
    if (!editable) return;
    if (!dragStart.current) return;
    dragNow.current = { x: e.clientX, y: e.clientY };
    drawOverlay();
  }

  function onMouseUp() {
    if (!editable) return;
    const start = dragStart.current;
    const now = dragNow.current;
    dragStart.current = null;
    dragNow.current = null;
    drawOverlay();

    if (!start || !now) return;
    if (!overlayRef.current || !baseCanvas.current) return;

    // Use the overlay's client rect (NO manual scroll offsets) – it tracks the canvas.
    const r = overlayRef.current.getBoundingClientRect();

    // CSS coords inside overlay
    const x0css = Math.max(0, Math.min(Math.min(start.x, now.x) - r.left, r.width));
    const y0css = Math.max(0, Math.min(Math.min(start.y, now.y) - r.top, r.height));
    const x1css = Math.max(0, Math.min(Math.max(start.x, now.x) - r.left, r.width));
    const y1css = Math.max(0, Math.min(Math.max(start.y, now.y) - r.top, r.height));

    // CSS -> OCR pixels (simple scale; we rendered rotation=0)
    const sx = serverW / r.width;
    const sy = serverH / r.height;
    const X0 = Math.floor(Math.min(x0css, x1css) * sx);
    const Y0 = Math.floor(Math.min(y0css, y1css) * sy);
    const X1 = Math.ceil(Math.max(x0css, x1css) * sx);
    const Y1 = Math.ceil(Math.max(y0css, y1css) * sy);

    const rr: EditRect = {
      page,
      x0: Math.max(0, Math.min(X0, serverW - 1)),
      y0: Math.max(0, Math.min(Y0, serverH - 1)),
      x1: Math.max(0, Math.min(X1, serverW - 1)),
      y1: Math.max(0, Math.min(Y1, serverH - 1)),
    };

    onRectChange(rr);
    onRectCommit(rr);
  }

  // --- Overlay draw (tokens + pink boxes) -----------------------------------
  function drawOverlay() {
    const overlay = overlayRef.current;
    const canvas = baseCanvas.current;
    if (!overlay || !canvas) return;

    overlay.innerHTML = "";

    // Token boxes (orange)
    if (showTokenBoxes) {
      for (const t of tokens) {
        const d = document.createElement("div");
        d.className = "tok";
        placeCss(d, t.x0, t.y0, t.x1, t.y1);
        overlay.appendChild(d);
      }
    }

    // Persisted/selected rect (pink)
    if (rect && rect.page === page) {
      const d = document.createElement("div");
      d.className = "pink";
      placeCss(d, rect.x0, rect.y0, rect.x1, rect.y1);
      overlay.appendChild(d);
    }

    // Live drag rect
    if (dragStart.current && dragNow.current) {
      const d = document.createElement("div");
      d.className = "pink live";
      const R = overlay.getBoundingClientRect();
      const x0 = Math.min(dragStart.current.x, dragNow.current.x) - R.left;
      const y0 = Math.min(dragStart.current.y, dragNow.current.y) - R.top;
      const x1 = Math.max(dragStart.current.x, dragNow.current.x) - R.left;
      const y1 = Math.max(dragStart.current.y, dragNow.current.y) - R.top;
      d.style.left = `${Math.max(0, Math.min(x0, R.width))}px`;
      d.style.top = `${Math.max(0, Math.min(y0, R.height))}px`;
      d.style.width = `${Math.max(0, Math.min(x1 - x0, R.width))}px`;
      d.style.height = `${Math.max(0, Math.min(y1 - y0, R.height))}px`;
      overlay.appendChild(d);
    }
  }

  // OCR px -> CSS px for overlay (display only)
  function placeCss(node: HTMLDivElement, x0: number, y0: number, x1: number, y1: number) {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const R = overlay.getBoundingClientRect();
    const sx = R.width / serverW;
    const sy = R.height / serverH;

    const left = Math.min(x0, x1) * sx;
    const top = Math.min(y0, y1) * sy;
    const width = Math.abs(x1 - x0) * sx;
    const height = Math.abs(y1 - y0) * sy;

    node.style.left = `${left}px`;
    node.style.top = `${top}px`;
    node.style.width = `${width}px`;
    node.style.height = `${height}px`;
  }

  // Redraw when props change
  useEffect(() => {
    drawOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokens, rect, showTokenBoxes, page, serverW, serverH]);

  return (
    <div ref={wrapRef} className="pdf-stage" style={{ position: "relative", overflow: "auto" }}>
      <canvas ref={baseCanvas} />
      <div
        ref={overlayRef}
        className="overlay"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        style={{
          position: "absolute",
          inset: 0,
          // transparent; we never paint a solid background → no "black bar" effect
          background: "transparent",
          cursor: editable ? "crosshair" : "default",
          userSelect: "none",
        }}
      />
    </div>
  );
}
