// File: PdfEditCanvas.tsx
import React, { useEffect, useLayoutEffect, useRef } from "react";
import {
  GlobalWorkerOptions,
  getDocument,
  type PDFDocumentProxy,
  type PDFPageProxy,
} from "pdfjs-dist";

// IMPORTANT: do NOT import any css from pdfjs-dist to avoid Sass errors.
// Provide worker URL for v4:
GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

/* =============== Types =============== */
export type TokenBox = {
  page: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  text?: string;
};

export type EditRect = {
  page: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
};

export type OverlayRect = {
  label: string;            // "fuzzy" | "tfidf" | "minilm" | "distilbert" | "layoutlmv3"
  color: string;            // css color
  rect: EditRect | null;    // if null → not drawn
};

type Props = {
  docUrl: string;
  page: number;

  /** OCR/server coordinate space for this page */
  serverW: number;
  serverH: number;

  tokens: TokenBox[];
  rect: EditRect | null;                 // editable GT box
  overlays?: OverlayRect[];              // non-editable colored overlays
  showTokenBoxes: boolean;
  editable: boolean;

  onRectChange: (r: EditRect | null) => void;
  onRectCommit: (r: EditRect) => void;

  zoom?: number; // default 1.0
};

export default function PdfEditCanvas({
  docUrl,
  page,
  serverW,
  serverH,
  tokens,
  rect,
  overlays = [],
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

  // Cached overlay bounding box used while dragging
  const overlayBox = useRef<DOMRect | null>(null);
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
    | { mode: "resize"; startX: number; startY: number; orig: EditRect; handle: Exclude<Handle, "inside" | "new"> };

  const drag = useRef<DragState>({ mode: "none" });

  /* ------------ Render PDF page to canvas ------------ */
  useEffect(() => {
    let cancelled = false;

    (async () => {
      if (!docUrl || !baseCanvas.current) return;

      const doc = await getDocument(docUrl).promise;
      if (cancelled) return;
      pdfRef.current = doc;

      const pg = await doc.getPage(page);
      if (cancelled) return;
      pageRef.current = pg;

      // Pick a base scale (try to cap width for performance) then apply zoom
      const vp1 = pg.getViewport({ scale: 1, rotation: 0 });
      const maxDisplay = 1400;
      const baseScale = Math.min(1, maxDisplay / Math.max(vp1.width, vp1.height));
      const vp = pg.getViewport({ scale: baseScale * zoom, rotation: 0 });

      const c = baseCanvas.current;
      const ctx = c!.getContext("2d")!;
      c!.width = Math.floor(vp.width);
      c!.height = Math.floor(vp.height);
      c!.style.width = `${c!.width}px`;
      c!.style.height = `${c!.height}px`;

      // pdfjs v4 render
      await pg.render({ canvasContext: ctx, viewport: vp }).promise;

      // Sync overlay size
      if (overlayRef.current) {
        overlayRef.current.style.width = c!.style.width;
        overlayRef.current.style.height = c!.style.height;
      }

      drawOverlay();
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl, page, zoom]);

  // Keep overlay div sized to the canvas even if layout changes
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

  /* ------------ Helpers: coord transforms ------------ */
  function cssToOcr(xCss: number, yCss: number) {
    const R = overlayRef.current!.getBoundingClientRect();
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

    if (dist(px, py, rc.x0, rc.y0) <= HANDLE) return "nw";
    if (dist(px, py, rc.x1, rc.y0) <= HANDLE) return "ne";
    if (dist(px, py, rc.x1, rc.y1) <= HANDLE) return "se";
    if (dist(px, py, rc.x0, rc.y1) <= HANDLE) return "sw";
    if (Math.abs(py - rc.y0) <= HANDLE && px >= rc.x0 && px <= rc.x1) return "n";
    if (Math.abs(px - rc.x1) <= HANDLE && py >= rc.y0 && py <= rc.y1) return "e";
    if (Math.abs(py - rc.y1) <= HANDLE && px >= rc.x0 && px <= rc.x1) return "s";
    if (Math.abs(px - rc.x0) <= HANDLE && py >= rc.y0 && py <= rc.y1) return "w";
    if (px >= rc.x0 && px <= rc.x1 && py >= rc.y0 && py <= rc.y1) return "inside";
    return "new";
  }

  function cursorForHandle(h: Handle) {
    switch (h) {
      case "nw":
      case "se":
        return "nwse-resize";
      case "ne":
      case "sw":
        return "nesw-resize";
      case "n":
      case "s":
        return "ns-resize";
      case "e":
      case "w":
        return "ew-resize";
      case "inside":
        return "move";
      case "new":
      default:
        return "crosshair";
    }
  }

  /* ------------ Mouse events ------------ */
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
      drag.current = { mode: "resize", startX: e.clientX, startY: e.clientY, orig: rect!, handle: h as any };
    }
    e.preventDefault();
  }

  function onMouseMove(e: React.MouseEvent) {
    if (!overlayRef.current) return;

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
      onRectChange({
        page,
        x0: clamp(nx0, 0, serverW - 1),
        y0: clamp(ny0, 0, serverH - 1),
        x1: clamp(nx1, 0, serverW - 1),
        y1: clamp(ny1, 0, serverH - 1),
      });
      return;
    }
  }

  function onMouseUp() {
    if (drag.current.mode !== "none" && rect) {
      onRectCommit(rect);
    }
    drag.current = { mode: "none" };
  }

  /* ------------ Overlay drawing ------------ */
  function drawOverlay() {
    const overlay = overlayRef.current;
    if (!overlay) return;
    overlay.innerHTML = "";

    // token boxes (orange)
    if (showTokenBoxes) {
      for (const t of tokens) {
        if (t.page !== page) continue;
        const d = document.createElement("div");
        d.style.position = "absolute";
        d.style.border = "1px solid rgba(255,165,0,0.55)";
        d.style.background = "rgba(255,165,0,0.10)";
        d.style.pointerEvents = "none";
        placeCss(d, t.x0, t.y0, t.x1, t.y1);
        overlay.appendChild(d);
      }
    }

    // editable main GT box (pink)
    if (rect && rect.page === page) {
      const box = document.createElement("div");
      box.style.position = "absolute";
      box.style.border = "2px solid rgba(236,72,153,0.95)";
      box.style.background = "rgba(236,72,153,0.18)";
      box.style.boxShadow = "0 0 0 1px rgba(236,72,153,0.25) inset";
      box.style.pointerEvents = "none";
      placeCss(box, rect.x0, rect.y0, rect.x1, rect.y1);
      overlay.appendChild(box);

      // resize handles
      if (editable) {
        const rc = rectCss();
        if (rc) {
          const hs: Array<[number, number, string]> = [
            [rc.x0, rc.y0, "nwse-resize"],
            [(rc.x0 + rc.x1) / 2, rc.y0, "ns-resize"],
            [rc.x1, rc.y0, "nesw-resize"],
            [rc.x1, (rc.y0 + rc.y1) / 2, "ew-resize"],
            [rc.x1, rc.y1, "nwse-resize"],
            [(rc.x0 + rc.x1) / 2, rc.y1, "ns-resize"],
            [rc.x0, rc.y1, "nesw-resize"],
            [rc.x0, (rc.y0 + rc.y1) / 2, "ew-resize"],
          ];
          for (const [x, y, cur] of hs) {
            const h = document.createElement("div");
            h.style.position = "absolute";
            h.style.borderRadius = "3px";
            h.style.background = "rgba(236,72,153,0.95)";
            h.style.boxShadow = "0 0 0 2px #fff inset, 0 0 0 1px rgba(236,72,153,0.8)";
            h.style.pointerEvents = "none";
            h.style.width = `${HANDLE}px`;
            h.style.height = `${HANDLE}px`;
            h.style.left = `${x - HANDLE / 2}px`;
            h.style.top = `${y - HANDLE / 2}px`;
            h.style.cursor = cur;
            overlay.appendChild(h);
          }
        }
      }
    }

    // model overlays (non-editable)
    if (overlays && overlays.length) {
      for (const ov of overlays) {
        if (!ov || !ov.rect || ov.rect.page !== page) continue;
        const d = document.createElement("div");
        d.style.position = "absolute";
        d.style.border = `2px solid ${ov.color}`;
        d.style.background = `${ov.color}22`;
        d.style.boxShadow = `0 0 0 1px ${ov.color}55 inset`;
        d.style.pointerEvents = "none";
        d.title = ov.label;
        placeCss(d, ov.rect.x0, ov.rect.y0, ov.rect.x1, ov.rect.y1);
        overlay.appendChild(d);
      }
    }
  }

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
  }, [tokens, rect, overlays, showTokenBoxes, page, serverW, serverH]);

  return (
    <div className="pdf-stage" style={{ position: "relative", overflow: "auto" }}>
      <canvas ref={baseCanvas} />
      <div
        ref={overlayRef}
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
    </div>
  );
}

/* =============== utils =============== */
function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}
function dist(x1: number, y1: number, x2: number, y2: number) {
  const dx = x1 - x2, dy = y1 - y2;
  return Math.hypot(dx, dy);
}