import React, { useEffect, useRef } from "react";
import {
  GlobalWorkerOptions,
  getDocument,
  type PDFDocumentProxy,
  type PDFPageProxy,
} from "pdfjs-dist";
import "pdfjs-dist/web/pdf_viewer.css";

/** Vite + pdfjs-dist v4 worker (MJS!) */
GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
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

  /** OCR/server coordinate space for this page (from /lasso/doc/{id}/meta) */
  serverW: number;
  serverH: number;

  tokens: TokenBox[];
  rect: EditRect | null;
  showTokenBoxes: boolean;
  editable: boolean;

  onRectChange: (r: EditRect | null) => void; // live
  onRectCommit: (r: EditRect) => void;        // mouseup commit
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

  const baseCanvas = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);

  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const dragNow = useRef<{ x: number; y: number } | null>(null);

  // Render page bitmap. We render with rotation=0 so OCR coords == bitmap coords.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!docUrl) return;
      const doc = await getDocument(docUrl).promise;
      if (cancelled) return;
      pdfRef.current = doc;

      const pg = await doc.getPage(page);
      if (cancelled) return;
      pageRef.current = pg;

      const vp1 = pg.getViewport({ scale: 1, rotation: 0 });
      const scale = Math.min(1, 1400 / Math.max(vp1.width, vp1.height));
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

      drawOverlay();
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl, page]);

  function onMouseDown(e: React.MouseEvent) {
    if (!editable || !overlayRef.current) return;
    e.preventDefault();
    dragStart.current = { x: e.clientX, y: e.clientY };
    dragNow.current = { x: e.clientX, y: e.clientY };
    drawOverlay();
  }
  function onMouseMove(e: React.MouseEvent) {
    if (!editable || !dragStart.current) return;
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
    if (!overlayRef.current) return;

    // Use overlay client rect (tracks canvas; no manual scroll math)
    const R = overlayRef.current.getBoundingClientRect();
    const x0css = Math.max(0, Math.min(Math.min(start.x, now.x) - R.left, R.width));
    const y0css = Math.max(0, Math.min(Math.min(start.y, now.y) - R.top, R.height));
    const x1css = Math.max(0, Math.min(Math.max(start.x, now.x) - R.left, R.width));
    const y1css = Math.max(0, Math.min(Math.max(start.y, now.y) - R.top, R.height));

    // CSS -> OCR px (linear scale; rotation was 0 at render)
    const sx = serverW / R.width;
    const sy = serverH / R.height;

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

  // Draw tokens + pink rect + live drag
  function drawOverlay() {
    const overlay = overlayRef.current;
    if (!overlay) return;
    overlay.innerHTML = "";

    if (showTokenBoxes) {
      for (const t of tokens) {
        const d = document.createElement("div");
        d.className = "tok";
        placeCss(d, t.x0, t.y0, t.x1, t.y1);
        overlay.appendChild(d);
      }
    }

    if (rect && rect.page === page) {
      const d = document.createElement("div");
      d.className = "pink";
      placeCss(d, rect.x0, rect.y0, rect.x1, rect.y1);
      overlay.appendChild(d);
    }

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

  // OCR px -> overlay CSS px (display only)
  function placeCss(node: HTMLDivElement, x0: number, y0: number, x1: number, y1: number) {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const R = overlay.getBoundingClientRect();
    const sx = R.width / serverW;
    const sy = R.height / serverH;

    node.style.left = `${Math.min(x0, x1) * sx}px`;
    node.style.top = `${Math.min(y0, y1) * sy}px`;
    node.style.width = `${Math.abs(x1 - x0) * sx}px`;
    node.style.height = `${Math.abs(y1 - y0) * sy}px`;
  }

  useEffect(() => {
    drawOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokens, rect, showTokenBoxes, page, serverW, serverH]);

  return (
    <div className="pdf-stage" style={{ position: "relative", overflow: "auto" }}>
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
          background: "transparent",
          cursor: editable ? "crosshair" : "default",
          userSelect: "none",
        }}
      />
    </div>
  );
}
