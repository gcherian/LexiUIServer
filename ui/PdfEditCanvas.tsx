import React, { useEffect, useRef } from "react";
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy, type PDFPageProxy } from "pdfjs-dist";
import "pdfjs-dist/web/pdf_viewer.css";

GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.js", import.meta.url).toString();

export type EditRect = { page: number; x0: number; y0: number; x1: number; y1: number };
export type Token = { page: number; x0: number; y0: number; x1: number; y1: number; text?: string };

type Props = {
  docUrl: string;
  page: number;
  serverW: number;
  serverH: number;
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
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);

  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const dragNow = useRef<{ x: number; y: number } | null>(null);

  // Render PDF page
  useEffect(() => {
    let cancel = false;
    (async () => {
      if (!docUrl) return;
      const pdf = await getDocument(docUrl).promise;
      if (cancel) return;
      pdfRef.current = pdf;
      const pg = await pdf.getPage(page);
      if (cancel) return;
      pageRef.current = pg;

      const viewport = pg.getViewport({ scale: 1.0, rotation: pg.rotate || 0 });
      const scale = Math.min(1, 1400 / Math.max(viewport.width, viewport.height));
      const vp = pg.getViewport({ scale, rotation: pg.rotate || 0 });

      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d")!;
      canvas.width = Math.floor(vp.width);
      canvas.height = Math.floor(vp.height);
      canvas.style.width = `${canvas.width}px`;
      canvas.style.height = `${canvas.height}px`;

      await pg.render({ canvasContext: ctx, viewport: vp }).promise;

      // sync overlay size
      if (overlayRef.current) {
        overlayRef.current.style.width = canvas.style.width;
        overlayRef.current.style.height = canvas.style.height;
      }
      drawOverlay(); // draw any existing rect/tokens
    })();
    return () => {
      cancel = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl, page]);

  // Pointer handlers
  function onMouseDown(e: React.MouseEvent) {
    if (!editable) return;
    e.preventDefault();
    dragStart.current = { x: e.clientX, y: e.clientY };
    dragNow.current = { x: e.clientX, y: e.clientY };
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
    if (!start || !now || !canvasRef.current || !wrapRef.current) return;

    // 1) CSS rect (canvas bbox + scroll)
    const canvRect = canvasRef.current.getBoundingClientRect();
    const scx = wrapRef.current.scrollLeft;
    const scy = wrapRef.current.scrollTop;

    const xCss0 = Math.min(start.x, now.x) - canvRect.left + scx;
    const yCss0 = Math.min(start.y, now.y) - canvRect.top + scy;
    const xCss1 = Math.max(start.x, now.x) - canvRect.left + scx;
    const yCss1 = Math.max(start.y, now.y) - canvRect.top + scy;

    const cssW = canvRect.width;
    const cssH = canvRect.height;

    // 2) CSS → OCR scale
    const sx = serverW / cssW;
    const sy = serverH / cssH;

    const rot = ((pageRef.current?.rotate || 0) % 360 + 360) % 360;

    function toOCR(xc: number, yc: number) {
      const x = xc * sx;
      const y = yc * sy;
      switch (rot) {
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

    const p0 = toOCR(xCss0, yCss0);
    const p1 = toOCR(xCss1, yCss1);

    const x0 = Math.max(0, Math.min(Math.floor(Math.min(p0.x, p1.x)), serverW - 1));
    const y0 = Math.max(0, Math.min(Math.floor(Math.min(p0.y, p1.y)), serverH - 1));
    const x1 = Math.max(0, Math.min(Math.ceil(Math.max(p0.x, p1.x)), serverW - 1));
    const y1 = Math.max(0, Math.min(Math.ceil(Math.max(p0.y, p1.y)), serverH - 1));

    const rr: EditRect = { page, x0, y0, x1, y1 };
    onRectChange(rr);
    onRectCommit(rr);
    drawOverlay();
  }

  // Display helpers: map OCR→CSS for overlay
  function fromOCR(x: number, y: number) {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const canvRect = canvasRef.current.getBoundingClientRect();
    const cssW = canvRect.width;
    const cssH = canvRect.height;
    const sx = cssW / serverW;
    const sy = cssH / serverH;
    const rot = ((pageRef.current?.rotate || 0) % 360 + 360) % 360;
    switch (rot) {
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

  function positionCss(node: HTMLDivElement, x0: number, y0: number, x1: number, y1: number) {
    const a = fromOCR(Math.min(x0, x1), Math.min(y0, y1));
    const b = fromOCR(Math.max(x0, x1), Math.max(y0, y1));
    node.style.left = `${Math.min(a.x, b.x)}px`;
    node.style.top = `${Math.min(a.y, b.y)}px`;
    node.style.width = `${Math.abs(b.x - a.x)}px`;
    node.style.height = `${Math.abs(b.y - a.y)}px`;
  }

  function drawOverlay() {
    if (!overlayRef.current || !canvasRef.current) return;
    const el = overlayRef.current;
    el.innerHTML = "";

    // token boxes
    if (showTokenBoxes) {
      for (const t of tokens) {
        const d = document.createElement("div");
        d.className = "tok";
        positionCss(d, t.x0, t.y0, t.x1, t.y1);
        el.appendChild(d);
      }
    }

    // committed rect (pink)
    if (rect && rect.page === page) {
      const d = document.createElement("div");
      d.className = "pink";
      positionCss(d, rect.x0, rect.y0, rect.x1, rect.y1);
      el.appendChild(d);
    }

    // live drag feedback
    if (dragStart.current && dragNow.current) {
      const r = canvasRef.current.getBoundingClientRect();
      const scx = wrapRef.current?.scrollLeft || 0;
      const scy = wrapRef.current?.scrollTop || 0;
      const xCss0 = Math.min(dragStart.current.x, dragNow.current.x) - r.left + scx;
      const yCss0 = Math.min(dragStart.current.y, dragNow.current.y) - r.top + scy;
      const xCss1 = Math.max(dragStart.current.x, dragNow.current.x) - r.left + scx;
      const yCss1 = Math.max(dragStart.current.y, dragNow.current.y) - r.top + scy;

      const live = document.createElement("div");
      live.className = "pink live";
      live.style.left = `${Math.min(xCss0, xCss1)}px`;
      live.style.top = `${Math.min(yCss0, yCss1)}px`;
      live.style.width = `${Math.abs(xCss1 - xCss0)}px`;
      live.style.height = `${Math.abs(yCss1 - yCss0)}px`;
      el.appendChild(live);
    }
  }

  useEffect(() => {
    drawOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokens, rect, showTokenBoxes, page, serverW, serverH]);

  return (
    <div ref={wrapRef} className="pdf-stage" style={{ position: "relative", overflow: "auto" }}>
      <canvas ref={canvasRef} />
      <div
        ref={overlayRef}
        className="overlay"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        style={{
          position: "absolute",
          inset: 0,
          zIndex: 2,
          cursor: editable ? "crosshair" : "default",
          userSelect: "none",
          pointerEvents: editable ? "auto" : "none",
        }}
      />
    </div>
  );
}
