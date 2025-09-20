import React, { useEffect, useLayoutEffect, useRef } from "react";
import {
  GlobalWorkerOptions,
  getDocument,
  type PDFDocumentProxy,
  type PDFPageProxy,
} from "pdfjs-dist";

/** pdfjs v4 worker (MJS) */
GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

/* ========================= Types ========================= */
export type TokenBox = { page:number;x0:number;y0:number;x1:number;y1:number;text?:string };
export type EditRect = { page:number;x0:number;y0:number;x1:number;y1:number };
export type OverlayRect = { label:string;color:string;rect:EditRect|null };

type Props = {
  docUrl: string;
  page: number;
  serverW: number;
  serverH: number;
  tokens: TokenBox[];
  rect: EditRect | null;
  overlays?: OverlayRect[];
  showTokenBoxes: boolean;
  editable: boolean;
  onRectChange: (r: EditRect | null) => void;
  onRectCommit: (r: EditRect) => void;
  zoom?: number;
};

export default function PdfEditCanvas({
  docUrl, page, serverW, serverH,
  tokens, rect, overlays = [], showTokenBoxes,
  editable, onRectChange, onRectCommit, zoom = 1,
}: Props) {
  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  const pageRef = useRef<PDFPageProxy | null>(null);
  const baseCanvas = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const overlayBox = useRef<DOMRect | null>(null);

  const HANDLE = 8;

  type Handle = "inside"|"nw"|"n"|"ne"|"e"|"se"|"s"|"sw"|"w"|"new";
  type DragState =
    | { mode: "none" }
    | { mode: "new"; startX: number; startY: number }
    | { mode: "move"; startX: number; startY: number; orig: EditRect }
    | { mode: "resize"; startX: number; startY: number; orig: EditRect; handle: Exclude<Handle,"inside"|"new"> };
  const drag = useRef<DragState>({ mode: "none" });

  /* ---------------- Render page ---------------- */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!docUrl) return;

      try {
        const doc = await getDocument(docUrl).promise;
        if (cancelled) return;
        pdfRef.current = doc;

        const pg = await doc.getPage(page);
        if (cancelled) return;
        pageRef.current = pg;

        const vp1 = pg.getViewport({ scale: 1 });
        const maxDisplay = 1400;
        const baseScale = Math.min(1, maxDisplay / Math.max(vp1.width, vp1.height));
        const vp = pg.getViewport({ scale: baseScale * zoom });

        const c = baseCanvas.current!;
        const ctx = c.getContext("2d")!;

        // give the canvas a visible board even if render fails
        c.width = Math.floor(vp.width);
        c.height = Math.floor(vp.height);
        c.style.width = `${c.width}px`;
        c.style.height = `${c.height}px`;
        ctx.fillStyle = "#f6f7f9";
        ctx.fillRect(0, 0, c.width, c.height);

        await pg.render({ canvasContext: ctx, viewport: vp }).promise;
        // sync overlay layer
        if (overlayRef.current) {
          overlayRef.current.style.width = c.style.width;
          overlayRef.current.style.height = c.style.height;
        }
        drawOverlay();
        console.info("[PdfEditCanvas] render OK", { page, size: [c.width, c.height] });
      } catch (err) {
        console.warn("[PdfEditCanvas] render FAILED; falling back <object>", err);
        // If pdfjs fails, we still draw overlays over a blank canvas;
        // the parent will show a fallback <object> element below.
        drawOverlay();
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl, page, zoom]);

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

  /* ---------------- Coord helpers ---------------- */
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
    const rc = rectCss(); if (!rc) return "new";
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
      case "nw": case "se": return "nwse-resize";
      case "ne": case "sw": return "nesw-resize";
      case "n": case "s":   return "ns-resize";
      case "e": case "w":   return "ew-resize";
      case "inside":        return "move";
      default:              return "crosshair";
    }
  }

  /* ---------------- Mouse events ---------------- */
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
      const w = Math.abs(dxCss), h = Math.abs(dyCss);
      const a = cssToOcr(sx, sy), b = cssToOcr(sx + w, sy + h);
      onRectChange({ page, x0: Math.min(a.X, b.X), y0: Math.min(a.Y, b.Y), x1: Math.max(a.X, b.X), y1: Math.max(a.Y, b.Y) });
      return;
    }
    if (drag.current.mode === "move") {
      const sx = serverW / R.width, sy = serverH / R.height;
      const dX = Math.round(dxCss * sx), dY = Math.round(dyCss * sy);
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
      const sx = serverW / R.width, sy = serverH / R.height;
      const dX = Math.round(dxCss * sx), dY = Math.round(dyCss * sy);
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
    }
  }
  function onMouseUp() {
    if (drag.current.mode !== "none" && rect) onRectCommit(rect);
    drag.current = { mode: "none" };
  }

  /* ---------------- Draw overlays ---------------- */
  function drawOverlay() {
    const overlay = overlayRef.current; if (!overlay) return;
    overlay.innerHTML = "";

    if (showTokenBoxes) {
      for (const t of tokens) {
        if (t.page !== page) continue;
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

      const rc = rectCss();
      if (editable && rc) {
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
    if (overlays?.length) {
      for (const ov of overlays) {
        if (!ov?.rect || ov.rect.page !== page) continue;
        const d = document.createElement("div");
        placeCss(d, ov.rect.x0, ov.rect.y0, ov.rect.x1, ov.rect.y1);
        d.style.position = "absolute";
        d.style.border = `2px solid ${ov.color}`;
        d.style.background = `${ov.color}22`;
        d.style.boxShadow = `0 0 0 1px ${ov.color}55 inset`;
        d.style.pointerEvents = "none";
        d.title = ov.label;
        overlay.appendChild(d);
      }
    }
  }

  function placeCss(node: HTMLDivElement, x0: number, y0: number, x1: number, y1: number) {
    const overlay = overlayRef.current; if (!overlay) return;
    const R = overlay.getBoundingClientRect();
    const sx = R.width / serverW, sy = R.height / serverH;
    node.style.position = "absolute";
    node.style.left = `${Math.min(x0, x1) * sx}px`;
    node.style.top = `${Math.min(y0, y1) * sy}px`;
    node.style.width = `${Math.abs(x1 - x0) * sx}px`;
    node.style.height = `${Math.abs(y1 - y0) * sy}px`;
  }

  useEffect(() => { drawOverlay(); }, [tokens, rect, overlays, showTokenBoxes, page, serverW, serverH]);

  return (
    <div className="pdf-stage" style={{ position: "relative", overflow: "auto" }}>
      <canvas ref={baseCanvas} style={{ display: "block" }} />
      {/* Fallback viewer in case pdfjs render fails (keeps overlays usable) */}
      <object
        data={docUrl}
        type="application/pdf"
        style={{ width: "100%", height: 0, border: "0", display: "none" }}
        onLoad={(e) => {
          // if canvas ends up size 0 (rare), unhide the object viewer
          const c = baseCanvas.current!;
          const obj = e.currentTarget as HTMLObjectElement;
          if (c && (c.width === 0 || c.height === 0)) {
            obj.style.display = "block";
            obj.style.height = "88vh";
          }
        }}
      />
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
          cursor: drag.current.mode === "move" ? "grabbing" : (editable ? "crosshair" : "default"),
          userSelect: "none",
        }}
      />
      <style>{`
        .pdf-stage { background:#fafafa; }
        .overlay .tok {
          border: 1px solid rgba(255, 165, 0, 0.55);
          background: rgba(255, 165, 0, 0.10);
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
          pointer-events: none;
        }
      `}</style>
    </div>
  );
}

/* ========================= utils ========================= */
function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }
function dist(x1: number, y1: number, x2: number, y2: number) { return Math.hypot(x1 - x2, y1 - y2); }