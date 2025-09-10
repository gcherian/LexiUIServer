// File: src/tsp4/components/lasso/PdfEditCanvas.tsx
import React, { useEffect, useLayoutEffect, useRef } from "react";
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

  /** live change while dragging */
  onRectChange: (r: EditRect | null) => void;
  /** commit on mouseup */
  onRectCommit: (r: EditRect) => void;

  /** Optional display zoom (does not affect OCR coords). Default 1.0 */
  zoom?: number;
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
  zoom = 1,
}: Props) {
  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  const pageRef = useRef<PDFPageProxy | null>(null);

  const baseCanvas = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);

  /** For stable pointer math during drags (snapshot of overlay client rect) */
  const overlayBox = useRef<DOMRect | null>(null);

  /** Drag state eliminates “acceleration” by always diffing from the start */
  type DragState =
    | { mode: "none" }
    | { mode: "new"; startX: number; startY: number }
    | { mode: "move"; startX: number; startY: number; orig: EditRect };
  const drag = useRef<DragState>({ mode: "none" });

  // ---------------- Rendering ----------------
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

      // Use page's inherent rotation; mapping stays correct because we always convert via overlay snapshot.
      const rot = (pg.rotate || 0) % 360;
      const vp1 = pg.getViewport({ scale: 1, rotation: rot });
      const maxDisplay = 1400; // soft cap
      const baseScale = Math.min(1, maxDisplay / Math.max(vp1.width, vp1.height));
      const vp = pg.getViewport({ scale: baseScale * zoom, rotation: rot });

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
  }, [docUrl, page, zoom]);

  /** Keep overlay in lockstep with canvas size (resizes, split drag, etc.) */
  useLayoutEffect(() => {
    const c = baseCanvas.current;
    const o = overlayRef.current;
    if (!c || !o) return;
    const ro = new ResizeObserver(() => {
      const r = c.getBoundingClientRect();
      o.style.width = `${Math.floor(r.width)}px`;
      o.style.height = `${Math.floor(r.height)}px`;
      drawOverlay();
    });
    ro.observe(c);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------------- Pointer helpers ----------------
  function cssToOcr(xCss: number, yCss: number) {
    const R = overlayBox.current!;
    const sx = serverW / R.width;
    const sy = serverH / R.height;
    const X = Math.max(0, Math.min(serverW - 1, Math.round(xCss * sx)));
    const Y = Math.max(0, Math.min(serverH - 1, Math.round(yCss * sy)));
    return { X, Y };
  }

  function hitRectInCss(px: number, py: number): boolean {
    if (!overlayRef.current || !rect) return false;
    const R = overlayRef.current.getBoundingClientRect();
    const sx = R.width / serverW;
    const sy = R.height / serverH;
    const rx0 = Math.min(rect.x0, rect.x1) * sx;
    const ry0 = Math.min(rect.y0, rect.y1) * sy;
    const rx1 = Math.max(rect.x0, rect.x1) * sx;
    const ry1 = Math.max(rect.y0, rect.y1) * sy;
    return px >= rx0 && px <= rx1 && py >= ry0 && py <= ry1;
  }

  // ---------------- Mouse events (stable drag/move) ----------------
  function onMouseDown(e: React.MouseEvent) {
    if (!editable || !overlayRef.current) return;
    overlayBox.current = overlayRef.current.getBoundingClientRect();

    const px = e.clientX - overlayBox.current.left;
    const py = e.clientY - overlayBox.current.top;

    if (rect && hitRectInCss(px, py)) {
      drag.current = { mode: "move", startX: e.clientX, startY: e.clientY, orig: rect };
    } else {
      drag.current = { mode: "new", startX: e.clientX, startY: e.clientY };
    }
    e.preventDefault();
  }

  function onMouseMove(e: React.MouseEvent) {
    if (drag.current.mode === "none" || !overlayBox.current) return;

    const R = overlayBox.current;
    const dx = e.clientX - drag.current.startX;
    const dy = e.clientY - drag.current.startY;

    if (drag.current.mode === "new") {
      const sx = Math.min(drag.current.startX, e.clientX) - R.left;
      const sy = Math.min(drag.current.startY, e.clientY) - R.top;
      const w = Math.abs(dx);
      const h = Math.abs(dy);
      const a = cssToOcr(sx, sy);
      const b = cssToOcr(sx + w, sy + h);
      onRectChange({
        page,
        x0: Math.min(a.X, b.X),
        y0: Math.min(a.Y, b.Y),
        x1: Math.max(a.X, b.X),
        y1: Math.max(a.Y, b.Y),
      });
      return;
    }

    if (drag.current.mode === "move") {
      const sx = serverW / R.width;
      const sy = serverH / R.height;
      const dX = Math.round(dx * sx);
      const dY = Math.round(dy * sy);
      const o = drag.current.orig;
      const nr: EditRect = {
        page,
        x0: clamp(o.x0 + dX, 0, serverW - 1),
        y0: clamp(o.y0 + dY, 0, serverH - 1),
        x1: clamp(o.x1 + dX, 0, serverW - 1),
        y1: clamp(o.y1 + dY, 0, serverH - 1),
      };
      onRectChange(nr);
      return;
    }
  }

  function onMouseUp() {
    if (drag.current.mode !== "none" && rect) {
      onRectCommit(rect);
    }
    drag.current = { mode: "none" };
  }

  // ---------------- Overlay drawing ----------------
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

    // Live drag box (visual only) is handled by onRectChange -> re-render -> above
  }

  function placeCss(node: HTMLDivElement, x0: number, y0: number, x1: number, y1: number) {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const R = overlay.getBoundingClientRect();
    const sx = R.width / serverW;
    const sy = R.height / serverH;

    node.style.position = "absolute";
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
          cursor: editable
            ? drag.current.mode === "move"
              ? "grabbing"
              : "crosshair"
            : "default",
          userSelect: "none",
        }}
      />
      <style>{`
        .overlay .tok {
          border: 1px solid rgba(255, 165, 0, 0.85);
          background: rgba(255, 165, 0, 0.15);
          pointer-events: none;
        }
        .overlay .pink {
          border: 2px solid rgba(236, 72, 153, 0.95);
          background: rgba(236, 72, 153, 0.18);
          box-shadow: 0 0 0 1px rgba(236,72,153,0.25) inset;
          pointer-events: none;
        }
      `}</style>
    </div>
  );
}

// ---------------- utils ----------------
function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}
