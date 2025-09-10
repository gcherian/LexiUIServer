// File: src/tsp4/components/lasso/PdfEditCanvas.tsx
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  GlobalWorkerOptions,
  getDocument,
  type PDFDocumentProxy,
  type PDFPageProxy,
} from "pdfjs-dist";
import "pdfjs-dist/web/pdf_viewer.css";

/** Vite + pdfjs-dist v4 worker (MJS) */
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
  serverW: number;
  serverH: number;
  tokens: TokenBox[];
  rect: EditRect | null;
  showTokenBoxes: boolean;
  editable: boolean;

  onRectChange: (r: EditRect | null) => void; // live (drag/resize/move)
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
  const baseCanvas = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);

  // interaction state
  const mode = useRef<"idle" | "lasso" | "move" | "resize">("idle");
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const dragNow = useRef<{ x: number; y: number } | null>(null);
  const resizeHandle = useRef<"nw"|"n"|"ne"|"e"|"se"|"s"|"sw"|"w"|null>(null);
  const moveBase = useRef<{ dx: number; dy: number } | null>(null);

  // Render page exactly at OCR resolution for 1:1 mapping
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!docUrl) return;
      const pdf: PDFDocumentProxy = await getDocument(docUrl).promise;
      if (cancelled) return;
      const pg: PDFPageProxy = await pdf.getPage(page);
      if (cancelled) return;

      const naturalWidth = pg.view[2];
      const scale = serverW / naturalWidth;
      const vp = pg.getViewport({ scale, rotation: 0 });

      const c = baseCanvas.current!;
      const ctx = c.getContext("2d")!;
      c.width = serverW;
      c.height = serverH;
      c.style.width = `${serverW}px`;
      c.style.height = `${serverH}px`;

      await pg.render({ canvasContext: ctx, viewport: vp }).promise;

      if (overlayRef.current) {
        overlayRef.current.style.width = c.style.width;
        overlayRef.current.style.height = c.style.height;
      }
      drawOverlay();
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl, page, serverW, serverH]);

  useLayoutEffect(() => {
    drawOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokens, rect, showTokenBoxes, serverW, serverH]);

  function cssToOcr(xcss: number, ycss: number) {
    const overlay = overlayRef.current!;
    const R = overlay.getBoundingClientRect();
    const sx = serverW / R.width;
    const sy = serverH / R.height;
    return { x: Math.max(0, Math.min(Math.round(xcss * sx), serverW - 1)), y: Math.max(0, Math.min(Math.round(ycss * sy), serverH - 1)) };
  }
  function ocrToCss(x: number, y: number) {
    const overlay = overlayRef.current!;
    const R = overlay.getBoundingClientRect();
    const sx = R.width / serverW;
    const sy = R.height / serverH;
    return { x: x * sx, y: y * sy };
  }
  function ocrRectToCss(r: { x0:number;y0:number;x1:number;y1:number }) {
    const a = ocrToCss(r.x0, r.y0), b = ocrToCss(r.x1, r.y1);
    return { left: Math.min(a.x,b.x), top: Math.min(a.y,b.y), width: Math.abs(b.x-a.x), height: Math.abs(b.y-a.y) };
  }

  // snapping
  function rectIntersects(r: EditRect, t: TokenBox) {
    const x0 = Math.max(Math.min(r.x0, r.x1), Math.min(t.x0, t.x1));
    const y0 = Math.max(Math.min(r.y0, r.y1), Math.min(t.y0, t.y1));
    const x1 = Math.min(Math.max(r.x0, r.x1), Math.max(t.x0, t.x1));
    const y1 = Math.min(Math.max(r.y0, r.y1), Math.max(t.y0, t.y1));
    return x1 > x0 && y1 > y0;
  }
  function unionTokens(span: TokenBox[]) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const t of span) { x0=Math.min(x0,t.x0); y0=Math.min(y0,t.y0); x1=Math.max(x1,t.x1); y1=Math.max(y1,t.y1); }
    return { x0: Math.floor(x0), y0: Math.floor(y0), x1: Math.ceil(x1), y1: Math.ceil(y1) };
  }
  function snapRectToTokens(r: EditRect): EditRect {
    const pageTokens = tokens.filter((t) => t.page === r.page);
    if (!pageTokens.length) return r;
    const inside = pageTokens.filter(
      (t) =>
        Math.min(r.x0, r.x1) <= Math.min(t.x0, t.x1) &&
        Math.min(r.y0, r.y1) <= Math.min(t.y0, t.y1) &&
        Math.max(r.x0, r.x1) >= Math.max(t.x0, t.x1) &&
        Math.max(r.y0, r.y1) >= Math.max(t.y0, t.y1)
    );
    const candidates = inside.length ? inside : pageTokens.filter((t) => rectIntersects(r, t));
    if (!candidates.length) return r;
    const u = unionTokens(candidates);
    return { page: r.page, ...u };
  }

  function drawOverlay() {
    const overlay = overlayRef.current;
    if (!overlay) return;
    overlay.innerHTML = "";

    if (showTokenBoxes) {
      for (const t of tokens) {
        if (t.page !== page) continue;
        const d = document.createElement("div");
        d.className = "tok";
        placeDiv(d, ocrRectToCss({ x0: t.x0, y0: t.y0, x1: t.x1, y1: t.y1 }));
        overlay.appendChild(d);
      }
    }

    if (rect && rect.page === page) {
      const d = document.createElement("div");
      d.className = "pink";
      placeDiv(d, ocrRectToCss({ x0: rect.x0, y0: rect.y0, x1: rect.x1, y1: rect.y1 }));
      // handles
      for (const pos of ["nw","n","ne","e","se","s","sw","w"] as const) {
        const h = document.createElement("div");
        h.className = `handle ${pos}`;
        d.appendChild(h);
      }
      overlay.appendChild(d);
    }

    // live lasso
    if (mode.current === "lasso" && dragStart.current && dragNow.current) {
      const overlayR = overlay.getBoundingClientRect();
      const x0 = Math.min(dragStart.current.x, dragNow.current.x) - overlayR.left;
      const y0 = Math.min(dragStart.current.y, dragNow.current.y) - overlayR.top;
      const x1 = Math.max(dragStart.current.x, dragNow.current.x) - overlayR.left;
      const y1 = Math.max(dragStart.current.y, dragNow.current.y) - overlayR.top;
      const d = document.createElement("div");
      d.className = "pink live";
      placeDiv(d, {
        left: clamp(x0, 0, overlayR.width),
        top: clamp(y0, 0, overlayR.height),
        width: clamp(x1 - x0, 0, overlayR.width),
        height: clamp(y1 - y0, 0, overlayR.height),
      });
      overlay.appendChild(d);
    }
  }

  function placeDiv(node: HTMLDivElement, css: { left:number; top:number; width:number; height:number }) {
    node.style.left = `${css.left}px`;
    node.style.top = `${css.top}px`;
    node.style.width = `${css.width}px`;
    node.style.height = `${css.height}px`;
  }
  function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(v, hi)); }

  function hitHandle(e: React.MouseEvent): typeof resizeHandle.current {
    if (!overlayRef.current || !rect || rect.page !== page) return null;
    const overlay = overlayRef.current;
    const R = overlay.getBoundingClientRect();
    const css = ocrRectToCss(rect);
    const x = e.clientX - R.left;
    const y = e.clientY - R.top;
    const pad = 8;
    const inside = (cx:number,cy:number, w:number,h:number) => x>=cx && x<=cx+w && y>=cy && y<=cy+h;
    const handles = {
      nw: { x: css.left - pad, y: css.top - pad },
      n:  { x: css.left + css.width/2 - pad, y: css.top - pad },
      ne: { x: css.left + css.width - pad, y: css.top - pad },
      e:  { x: css.left + css.width - pad, y: css.top + css.height/2 - pad },
      se: { x: css.left + css.width - pad, y: css.top + css.height - pad },
      s:  { x: css.left + css.width/2 - pad, y: css.top + css.height - pad },
      sw: { x: css.left - pad, y: css.top + css.height - pad },
      w:  { x: css.left - pad, y: css.top + css.height/2 - pad },
    } as const;
    for (const k of Object.keys(handles) as (keyof typeof handles)[]) {
      const p = handles[k];
      if (inside(p.x, p.y, pad*2, pad*2)) return k;
    }
    // inside rect for move?
    if (x >= css.left && x <= css.left + css.width && y >= css.top && y <= css.top + css.height) return "e"; // use "e" sentinel for move start
    return null;
  }

  function onMouseDown(e: React.MouseEvent) {
    if (!editable || !overlayRef.current) return;
    e.preventDefault();
    const h = hitHandle(e);
    if (h) {
      if (h === "e" && rect) {
        mode.current = "move";
        dragStart.current = { x: e.clientX, y: e.clientY };
        moveBase.current = { dx: 0, dy: 0 };
      } else {
        mode.current = "resize";
        resizeHandle.current = h;
        dragStart.current = { x: e.clientX, y: e.clientY };
      }
    } else {
      mode.current = "lasso";
      dragStart.current = { x: e.clientX, y: e.clientY };
      dragNow.current = { x: e.clientX, y: e.clientY };
    }
    drawOverlay();
  }
  function onMouseMove(e: React.MouseEvent) {
    if (!editable || !dragStart.current) return;
    if (mode.current === "lasso") {
      dragNow.current = { x: e.clientX, y: e.clientY };
      drawOverlay();
      return;
    }
    if (!overlayRef.current) return;
    const overlay = overlayRef.current;
    const R = overlay.getBoundingClientRect();
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;

    if (mode.current === "move" && rect) {
      const { x, y } = cssToOcr(dx, dy);
      const w = rect.x1 - rect.x0;
      const h = rect.y1 - rect.y0;
      const nx0 = clamp(rect.x0 + x, 0, serverW - w - 1);
      const ny0 = clamp(rect.y0 + y, 0, serverH - h - 1);
      const next = { page, x0: nx0, y0: ny0, x1: nx0 + w, y1: ny0 + h };
      onRectChange(next);
      drawOverlay();
      return;
    }

    if (mode.current === "resize" && rect) {
      const css = ocrRectToCss(rect);
      let x0css = css.left, y0css = css.top, x1css = css.left + css.width, y1css = css.top + css.height;
      const nx = clamp(e.clientX - R.left, 0, R.width);
      const ny = clamp(e.clientY - R.top,  0, R.height);

      switch (resizeHandle.current) {
        case "nw": x0css = nx; y0css = ny; break;
        case "n":  y0css = ny; break;
        case "ne": x1css = nx; y0css = ny; break;
        case "e":  x1css = nx; break;
        case "se": x1css = nx; y1css = ny; break;
        case "s":  y1css = ny; break;
        case "sw": x0css = nx; y1css = ny; break;
        case "w":  x0css = nx; break;
      }
      const { x: X0, y: Y0 } = cssToOcr(x0css, y0css);
      const { x: X1, y: Y1 } = cssToOcr(x1css, y1css);
      const next: EditRect = { page, x0: Math.min(X0, X1), y0: Math.min(Y0, Y1), x1: Math.max(X0, X1), y1: Math.max(Y0, Y1) };
      onRectChange(next);
      drawOverlay();
      return;
    }
  }
  function onMouseUp(e: React.MouseEvent) {
    if (!editable) return;
    const start = dragStart.current;
    dragStart.current = null;
    dragNow.current = null;

    if (mode.current === "lasso" && overlayRef.current && start) {
      const R = overlayRef.current.getBoundingClientRect();
      const x0css = clamp(Math.min(start.x, e.clientX) - R.left, 0, R.width);
      const y0css = clamp(Math.min(start.y, e.clientY) - R.top,  0, R.height);
      const x1css = clamp(Math.max(start.x, e.clientX) - R.left, 0, R.width);
      const y1css = clamp(Math.max(start.y, e.clientY) - R.top,  0, R.height);

      const a = cssToOcr(x0css, y0css);
      const b = cssToOcr(x1css, y1css);
      let rr: EditRect = { page, x0: Math.min(a.x,b.x), y0: Math.min(a.y,b.y), x1: Math.max(a.x,b.x), y1: Math.max(a.y,b.y) };

      // snap to words
      rr = snapRectToTokens(rr);
      onRectChange(rr);
      onRectCommit(rr);
    } else if (mode.current === "move" || mode.current === "resize") {
      if (rect) onRectCommit(rect);
    }
    mode.current = "idle";
    drawOverlay();
  }

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