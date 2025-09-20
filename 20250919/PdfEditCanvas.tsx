import React, { useEffect, useLayoutEffect, useRef } from "react";
import {
  GlobalWorkerOptions,
  getDocument,
  type PDFDocumentProxy,
  type PDFPageProxy,
} from "pdfjs-dist";
import "pdfjs-dist/web/pdf_viewer.css";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

export type TokenBox = { page:number; x0:number; y0:number; x1:number; y1:number; text?:string };
export type EditRect = { page:number; x0:number; y0:number; x1:number; y1:number };
export type OverlayRect = { label:string; color:string; rect:EditRect|null };

type Props = {
  docUrl: string;
  page: number;
  serverW: number;   // OCR coordinate space
  serverH: number;
  tokens: TokenBox[];

  rect: EditRect | null;                   // editable
  overlays?: OverlayRect[];                // non-editable model boxes
  showTokenBoxes: boolean;
  editable: boolean;

  onRectChange: (r: EditRect | null) => void;
  onRectCommit: (r: EditRect) => void;

  zoom?: number;
};

export default function PdfEditCanvas(props: Props) {
  const {
    docUrl, page, serverW, serverH, tokens,
    rect, overlays = [], showTokenBoxes, editable,
    onRectChange, onRectCommit, zoom = 1,
  } = props;

  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  const pageRef = useRef<PDFPageProxy | null>(null);
  const baseCanvas = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const overlayBox = useRef<DOMRect | null>(null);

  const HANDLE = 8;

  type Drag =
    | { mode:"none" }
    | { mode:"new"; sx:number; sy:number }
    | { mode:"move"; sx:number; sy:number; orig:EditRect }
    | { mode:"resize"; sx:number; sy:number; orig:EditRect; handle:"nw"|"n"|"ne"|"e"|"se"|"s"|"sw"|"w" };
  const drag = useRef<Drag>({ mode:"none" });

  /* ------------ PDF render ------------ */
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

      const v1 = pg.getViewport({ scale: 1 });
      const maxDisplay = 1400;
      const baseScale = Math.min(1, maxDisplay / Math.max(v1.width, v1.height));
      const vp = pg.getViewport({ scale: baseScale * zoom });

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
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl, page, zoom]);

  useLayoutEffect(() => {
    const c = baseCanvas.current, o = overlayRef.current;
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

  /* ------------ helpers ------------ */
  function clamp(v:number, lo:number, hi:number) { return Math.max(lo, Math.min(hi, v)); }

  function cssToOcr(xCss:number, yCss:number) {
    const R = overlayBox.current!;
    const sx = serverW / R.width;
    const sy = serverH / R.height;
    return { X: clamp(Math.round(xCss * sx), 0, serverW - 1),
             Y: clamp(Math.round(yCss * sy), 0, serverH - 1) };
  }
  function ocrToCss(x:number, y:number) {
    const R = overlayRef.current!.getBoundingClientRect();
    const sx = R.width / serverW, sy = R.height / serverH;
    return { x: x * sx, y: y * sy };
  }
  function rectCss() {
    if (!rect || !overlayRef.current) return null;
    const { x: x0, y: y0 } = ocrToCss(Math.min(rect.x0, rect.x1), Math.min(rect.y0, rect.y1));
    const { x: x1, y: y1 } = ocrToCss(Math.max(rect.x0, rect.x1), Math.max(rect.y0, rect.y1));
    return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 };
  }

  /* ------------ mouse ------------ */
  function hitHandle(px:number, py:number) {
    if (!rect) return "new";
    const rc = rectCss(); if (!rc) return "new";
    const d = (x:number, y:number) => Math.hypot(px - x, py - y);
    if (d(rc.x0, rc.y0) <= HANDLE) return "nw";
    if (d(rc.x1, rc.y0) <= HANDLE) return "ne";
    if (d(rc.x1, rc.y1) <= HANDLE) return "se";
    if (d(rc.x0, rc.y1) <= HANDLE) return "sw";
    if (Math.abs(py - rc.y0) <= HANDLE && px >= rc.x0 && px <= rc.x1) return "n";
    if (Math.abs(px - rc.x1) <= HANDLE && py >= rc.y0 && py <= rc.y1) return "e";
    if (Math.abs(py - rc.y1) <= HANDLE && px >= rc.x0 && px <= rc.x1) return "s";
    if (Math.abs(px - rc.x0) <= HANDLE && py >= rc.y0 && py <= rc.y1) return "w";
    if (px >= rc.x0 && px <= rc.x1 && py >= rc.y0 && py <= rc.y1) return "inside";
    return "new";
  }

  function onMouseDown(e:React.MouseEvent) {
    if (!editable || !overlayRef.current) return;
    overlayBox.current = overlayRef.current.getBoundingClientRect();
    const px = e.clientX - overlayBox.current.left;
    const py = e.clientY - overlayBox.current.top;
    const h = hitHandle(px, py) as any;

    if (h === "inside" && rect)      drag.current = { mode:"move",   sx:e.clientX, sy:e.clientY, orig:rect };
    else if (h === "new")            drag.current = { mode:"new",    sx:e.clientX, sy:e.clientY };
    else                             drag.current = { mode:"resize", sx:e.clientX, sy:e.clientY, orig:rect!, handle:h };
    e.preventDefault();
  }

  function onMouseMove(e:React.MouseEvent) {
    if (!overlayRef.current) return;

    if (drag.current.mode === "none") {
      const R = overlayRef.current.getBoundingClientRect();
      const h = hitHandle(e.clientX - R.left, e.clientY - R.top);
      overlayRef.current.style.cursor = editable
        ? (h === "inside" ? "move" : (h === "new" ? "crosshair" : "nwse-resize"))
        : "default";
      return;
    }

    const R = overlayBox.current!;
    const dxCss = e.clientX - drag.current.sx;
    const dyCss = e.clientY - drag.current.sy;

    const clampRect = (r:EditRect): EditRect => ({
      page,
      x0: clamp(Math.min(r.x0, r.x1), 0, serverW - 1),
      y0: clamp(Math.min(r.y0, r.y1), 0, serverH - 1),
      x1: clamp(Math.max(r.x0, r.x1), 0, serverW - 1),
      y1: clamp(Math.max(r.y0, r.y1), 0, serverH - 1),
    });

    if (drag.current.mode === "new") {
      const sx = Math.min(drag.current.sx, e.clientX) - R.left;
      const sy = Math.min(drag.current.sy, e.clientY) - R.top;
      const a = cssToOcr(sx, sy);
      const b = cssToOcr(sx + Math.abs(dxCss), sy + Math.abs(dyCss));
      onRectChange(clampRect({ page, x0:a.X, y0:a.Y, x1:b.X, y1:b.Y }));
      return;
    }

    if (drag.current.mode === "move") {
      const sX = serverW / R.width, sY = serverH / R.height;
      const dX = Math.round(dxCss * sX), dY = Math.round(dyCss * sY);
      const o = drag.current.orig;
      onRectChange(clampRect({ page, x0:o.x0 + dX, y0:o.y0 + dY, x1:o.x1 + dX, y1:o.y1 + dY }));
      return;
    }

    if (drag.current.mode === "resize") {
      const sX = serverW / R.width, sY = serverH / R.height;
      const dX = Math.round(dxCss * sX), dY = Math.round(dyCss * sY);
      const o = drag.current.orig;
      let nx0=o.x0, ny0=o.y0, nx1=o.x1, ny1=o.y1;
      switch (drag.current.handle) {
        case "nw": nx0=o.x0+dX; ny0=o.y0+dY; break;
        case "n":  ny0=o.y0+dY;             break;
        case "ne": nx1=o.x1+dX; ny0=o.y0+dY; break;
        case "e":  nx1=o.x1+dX;             break;
        case "se": nx1=o.x1+dX; ny1=o.y1+dY; break;
        case "s":  ny1=o.y1+dY;             break;
        case "sw": nx0=o.x0+dX; ny1=o.y1+dY; break;
        case "w":  nx0=o.x0+dX;             break;
      }
      onRectChange(clampRect({ page, x0:nx0, y0:ny0, x1:nx1, y1:ny1 }));
    }
  }

  function onMouseUp() {
    if (drag.current.mode !== "none" && rect) onRectCommit(rect);
    drag.current = { mode:"none" };
  }

  /* ------------ drawing ------------ */
  function placeCss(node:HTMLDivElement, x0:number, y0:number, x1:number, y1:number) {
    const R = overlayRef.current!.getBoundingClientRect();
    const sx = R.width / serverW, sy = R.height / serverH;
    node.style.position = "absolute";
    node.style.left   = `${Math.min(x0, x1) * sx}px`;
    node.style.top    = `${Math.min(y0, y1) * sy}px`;
    node.style.width  = `${Math.abs(x1 - x0) * sx}px`;
    node.style.height = `${Math.abs(y1 - y0) * sy}px`;
  }

  function drawOverlay() {
    const overlay = overlayRef.current; if (!overlay) return;
    overlay.innerHTML = "";

    if (showTokenBoxes) {
      for (const t of tokens) {
        const d = document.createElement("div");
        d.className = "tok";
        placeCss(d, t.x0, t.y0, t.x1, t.y1);
        overlay.appendChild(d);
      }
    }

    // Editable
    if (rect && rect.page === page) {
      const d = document.createElement("div");
      d.className = "pink";
      placeCss(d, rect.x0, rect.y0, rect.x1, rect.y1);
      overlay.appendChild(d);

      // handles
      const rc = rectCss();
      if (editable && rc) {
        const hs: Array<[number,number,string]> = [
          [rc.x0, rc.y0, "nwse-resize"], [(rc.x0+rc.x1)/2, rc.y0, "ns-resize"],
          [rc.x1, rc.y0, "nesw-resize"], [rc.x1,(rc.y0+rc.y1)/2,"ew-resize"],
          [rc.x1, rc.y1, "nwse-resize"], [(rc.x0+rc.x1)/2, rc.y1, "ns-resize"],
          [rc.x0, rc.y1, "nesw-resize"], [rc.x0,(rc.y0+rc.y1)/2,"ew-resize"],
        ];
        for (const [x,y,cur] of hs) {
          const h = document.createElement("div");
          h.className = "handle";
          h.style.left = `${x - HANDLE/2}px`;
          h.style.top  = `${y - HANDLE/2}px`;
          h.style.width = `${HANDLE}px`; h.style.height = `${HANDLE}px`;
          h.style.cursor = cur;
          overlay.appendChild(h);
        }
      }
    }

    // Overlays
    for (const ov of overlays) {
      if (!ov || !ov.rect || ov.rect.page !== page) continue;
      const d = document.createElement("div");
      placeCss(d, ov.rect.x0, ov.rect.y0, ov.rect.x1, ov.rect.y1);
      d.style.border = `2px solid ${ov.color}`;
      d.style.background = `${ov.color}22`;
      d.style.boxShadow = `0 0 0 1px ${ov.color}55 inset`;
      d.style.pointerEvents = "none";
      d.title = ov.label;
      overlay.appendChild(d);
    }
  }

  useEffect(() => { drawOverlay(); }, [tokens, rect, overlays, showTokenBoxes, page, serverW, serverH]);

  return (
    <div className="pdf-stage" style={{ position:"relative", overflow:"auto" }}>
      <canvas ref={baseCanvas} />
      <div
        ref={overlayRef}
        className="overlay"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        style={{ position:"absolute", inset:0, background:"transparent", userSelect:"none" }}
      />
      <style>{`
        .overlay .tok {
          border: 1px solid rgba(255,165,0,0.55);
          background: rgba(255,165,0,0.10);
          pointer-events: none;
        }
        .overlay .pink {
          border: 2px solid rgba(236,72,153,0.95);
          background: rgba(236,72,153,0.18);
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