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

  /** live change while dragging/resizing */
  onRectChange: (r: EditRect | null) => void;
  /** commit on mouseup (triggers OCR in parent) */
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

  /** Snapshot of overlay client rect (stable pointer math) */
  const overlayBox = useRef<DOMRect | null>(null);

  /** Handle size in CSS px (visual space) */
  const HANDLE = 8;

  type Handle =
    | "inside"
    | "nw" | "n" | "ne"
    | "e"
    | "se" | "s" | "sw"
    | "w"
    | "new";

  type DragState =
    | { mode: "none" }
    | { mode: "new"; startX: number; startY: number }
    | { mode: "move"; startX: number; startY: number; orig: EditRect }
    | { mode: "resize"; startX: number; startY: number; orig: EditRect; handle: Exclude<Handle,"inside"|"new"> };

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

  /** Keep overlay sized with the canvas (split drag, resizes, etc.) */
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

  // ---------------- Helpers ----------------
  function cssToOcr(xCss: number, yCss: number) {
    const R = overlayBox.current!;
    const sx = serverW / R.width;
    const sy = serverH / R.height;
    const X = clamp(Math.round(xCss * sx), 0, serverW - 1);
    const Y = clamp(Math.round(yCss * sy), 0, serverH - 1);
    return { X, Y };
  }

  function ocrToCss(x: number, y: number) {
    const R = overlayRef.current!.getBoundingClientRect();
    const sx = R.width / serverW;
    const sy = R.height / serverH;
    return { x: x * sx, y: y * sy };
  }

  function rectCss() {
    if (!rect || !overlayRef.current) return null;
    const { x: x0, y: y0 } = ocrToCss(Math.min(rect.x0, rect.x1), Math.min(rect.y0, rect.y1));
    const { x: x1, y: y1 } = ocrToCss(Math.max(rect.x0, rect.x1), Math.max(rect.y0, rect.y1));
    return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 };
  }

  function hitHandle(px: number, py: number): Handle {
    if (!rect) return "new";
    const rc = rectCss();
    if (!rc) return "new";

    // Corners
    if (dist(px, py, rc.x0, rc.y0) <= HANDLE) return "nw";
    if (dist(px, py, rc.x1, rc.y0) <= HANDLE) return "ne";
    if (dist(px, py, rc.x1, rc.y1) <= HANDLE) return "se";
    if (dist(px, py, rc.x0, rc.y1) <= HANDLE) return "sw";

    // Edges
    if (Math.abs(py - rc.y0) <= HANDLE && px >= rc.x0 && px <= rc.x1) return "n";
    if (Math.abs(px - rc.x1) <= HANDLE && py >= rc.y0 && py <= rc.y1) return "e";
    if (Math.abs(py - rc.y1) <= HANDLE && px >= rc.x0 && px <= rc.x1) return "s";
    if (Math.abs(px - rc.x0) <= HANDLE && py >= rc.y0 && py <= rc.y1) return "w";

    // Inside
    if (px >= rc.x0 && px <= rc.x1 && py >= rc.y0 && py <= rc.y1) return "inside";

    return "new";
  }

  function cursorForHandle(h: Handle) {
    switch (h) {
      case "nw": case "se": return "nwse-resize";
      case "ne": case "sw": return "nesw-resize";
      case "n": case "s": return "ns-resize";
      case "e": case "w": return "ew-resize";
      case "inside": return "move";
      case "new": default: return "crosshair";
    }
  }

  // ---------------- Mouse events ----------------
  function onMouseDown(e: React.MouseEvent) {
    if (!editable || !overlayRef.current) return;
    overlayBox.current = overlayRef.current.getBoundingClientRect();

    const px = e.clientX - overlayBox.current.left;
    const py = e.clientY - overlayBox.current.top;

    const h = hitHandle(px, py);
    if (h === "inside" && rect) {
      drag.current = { mode: "move", startX: e.clientX, startY: e.clientY, orig: rect };
    } else if (h === "new") {
      drag.current = { mode: "new", startX: e.clientX, startY: e.clientY };
    } else {
      // resize
      drag.current = {
        mode: "resize",
        startX: e.clientX,
        startY: e.clientY,
        orig: rect!,
        handle: h as Exclude<Handle,"inside"|"new">,
      };
    }
    e.preventDefault();
  }

  function onMouseMove(e: React.MouseEvent) {
    if (!overlayRef.current) return;

    // Update cursor shape when idle
    if (drag.current.mode === "none") {
      const R = overlayRef.current.getBoundingClientRect();
      const h = hitHandle(e.clientX - R.left, e.clientY - R.top);
      overlayRef.current.style.cursor = editable ? cursorForHandle(h) : "default";
      return;
    }

    const R = overlayBox.current!;
    const dxCss = e.clientX - drag.current.startX;
    const dyCss = e.clientY - drag.current.startY;

    if (drag.current.mode === "new") {
      const sx = Math.min(drag.current.startX, e.clientX) - R.left;
      const sy = Math.min(drag.current.startY, e.clientY) - R.top;
      const w = Math.abs(dxCss);
      const h = Math.abs(dyCss);
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
      const dX = Math.round(dxCss * sx);
      const dY = Math.round(dyCss * sy);
      const o = drag.current.orig;
      onRectChange({
        page,
        x0: clamp(o.x0 + dX, 0, serverW - 1),
        y0: clamp(o.y0 + dY, 0, serverH - 1),
        x1: clamp(o.x1 + dX, 0, serverW - 1),
        y1: clamp(o.y1 + dY, 0, serverH - 1),
      });
      return;
    }

    if (drag.current.mode === "resize") {
      const o = drag.current.orig;
      const base = rectCssFrom(o);
      // convert css delta -> ocr delta
      const sx = serverW / R.width;
      const sy = serverH / R.height;
      const dX = Math.round(dxCss * sx);
      const dY = Math.round(dyCss * sy);

      let nx0 = o.x0, ny0 = o.y0, nx1 = o.x1, ny1 = o.y1;
      switch (drag.current.handle) {
        case "nw": nx0 = o.x0 + dX; ny0 = o.y0 + dY; break;
        case "n":  ny0 = o.y0 + dY; break;
        case "ne": nx1 = o.x1 + dX; ny0 = o.y0 + dY; break;
        case "e":  nx1 = o.x1 + dX; break;
        case "se": nx1 = o.x1 + dX; ny1 = o.y1 + dY; break;
        case "s":  ny1 = o.y1 + dY; break;
        case "sw": nx0 = o.x0 + dX; ny1 = o.y1 + dY; break;
        case "w":  nx0 = o.x0 + dX; break;
      }
      const rr: EditRect = {
        page,
        x0: clamp(nx0, 0, serverW - 1),
        y0: clamp(ny0, 0, serverH - 1),
        x1: clamp(nx1, 0, serverW - 1),
        y1: clamp(ny1, 0, serverH - 1),
      };
      onRectChange(rr);
      return;
    }
  }

  function onMouseUp() {
    if (drag.current.mode !== "none" && rect) {
      onRectCommit(rect); // parent will OCR and update value
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
      const box = document.createElement("div");
      box.className = "pink";
      placeCss(box, rect.x0, rect.y0, rect.x1, rect.y1);
      overlay.appendChild(box);

      // Handles
      const rc = rectCss();
      if (rc) {
        const hs: Array<[Handle, number, number, string]> = [
          ["nw", rc.x0, rc.y0, "nwse-resize"],
          ["n", (rc.x0 + rc.x1) / 2, rc.y0, "ns-resize"],
          ["ne", rc.x1, rc.y0, "nesw-resize"],
          ["e", rc.x1, (rc.y0 + rc.y1) / 2, "ew-resize"],
          ["se", rc.x1, rc.y1, "nwse-resize"],
          ["s", (rc.x0 + rc.x1) / 2, rc.y1, "ns-resize"],
          ["sw", rc.x0, rc.y1, "nesw-resize"],
          ["w", rc.x0, (rc.y0 + rc.y1) / 2, "ew-resize"],
        ];
        for (const [, x, y, cur] of hs) {
          const h = document.createElement("div");
          h.className = "handle";
          h.style.left = `${x - HANDLE / 2}px`;
          h.style.top = `${y - HANDLE / 2}px`;
          h.style.width = `${HANDLE}px`;
          h.style.height = `${HANDLE}px`;
          h.style.cursor = cur;
          overlay.appendChild(h);
        }
      }
    }
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
          cursor:
            drag.current.mode === "move"
              ? "grabbing"
              : editable
              ? "crosshair"
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
        .overlay .handle {
          position: absolute;
          border-radius: 3px;
          background: rgba(236,72,153,0.95);
          box-shadow: 0 0 0 2px #fff inset, 0 0 0 1px rgba(236,72,153,0.8);
          pointer-events: none; /* purely visual; hit-testing is computed in code */
        }
      `}</style>
    </div>
  );
}

// ---------------- utils ----------------
function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}
function dist(x1: number, y1: number, x2: number, y2: number) {
  const dx = x1 - x2, dy = y1 - y2;
  return Math.hypot(dx, dy);
}
function rectCssFrom(r: EditRect) {
  return r; // kept for readability; we convert via ocrToCss where needed
}
