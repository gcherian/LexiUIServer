import { useEffect, useRef, useState } from "react";
import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";

// pdfjs-dist v4 ESM worker (Vite-friendly)
GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

export type Rect = { x0: number; y0: number; x1: number; y1: number };
export type Box  = Rect & { page: number; id?: string; label?: string; text?: string };

type Props = {
  url?: string;
  page: number;
  boxes?: Box[];                         // base layer (blue)
  highlights?: Box[];                    // search results (orange)
  selected?: number[];                   // indices referring to boxes[]
  ocrSize?: { width: number; height: number };
  scale?: number;                        // UI zoom (1.25 etc.)
  tool?: "select" | "lasso";
  onLasso?: (rect: Rect) => void;        // OCR space
  onSelectBox?: (boxIndex: number) => void;
};

type VpInfo = { wCss:number; hCss:number; dpr:number };

export default function PdfCanvas({
  url,
  page,
  boxes = [],
  highlights = [],
  selected = [],
  ocrSize,
  scale = 1.25,
  tool = "select",
  onLasso,
  onSelectBox,
}: Props) {
  const pdfRef = useRef<any>(null);
  const [vp, setVp] = useState<VpInfo | null>(null);
  const baseCanvas = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);

  // lasso drag state (in overlay CSS pixels)
  const [dragStart, setDragStart] = useState<{ x:number; y:number } | null>(null);
  const [dragNow, setDragNow]     = useState<{ x:number; y:number } | null>(null);

  // ---------- Render PDF with DPR-aware sizing ----------
  async function renderPage(pnum:number, zoom:number){
    if (!pdfRef.current || !baseCanvas.current) return;
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    const p = await pdfRef.current.getPage(pnum);
    // render at device pixels
    const viewport = p.getViewport({ scale: zoom * dpr });

    const c = baseCanvas.current!;
    // set backing store size (device pixels)
    c.width  = Math.floor(viewport.width);
    c.height = Math.floor(viewport.height);
    // set CSS display size (CSS pixels)
    c.style.width  = `${Math.floor(viewport.width / dpr)}px`;
    c.style.height = `${Math.floor(viewport.height / dpr)}px`;
    // reset any stray transforms (prevents accidental mirror)
    const ctx = c.getContext("2d"); if (!ctx) return;
    ctx.setTransform(1,0,0,1,0,0);
    ctx.clearRect(0,0,c.width,c.height);

    await p.render({ canvasContext: ctx, viewport }).promise;

    // sync overlay canvas size/transform
    const ov = overlay.current!;
    ov.width  = c.width;
    ov.height = c.height;
    ov.style.width  = c.style.width;
    ov.style.height = c.style.height;

    setVp({ wCss: Math.floor(viewport.width / dpr), hCss: Math.floor(viewport.height / dpr), dpr });
  }

  // ---------- Load PDF once ----------
  useEffect(() => {
    if (!url) return;
    (async () => {
      pdfRef.current = await getDocument(url).promise;
      await renderPage(page, scale);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // ---------- Re-render on page / scale ----------
  useEffect(() => {
    if (!pdfRef.current) return;
    renderPage(page, scale);
    // reset any in-progress lasso on page change
    setDragStart(null); setDragNow(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, scale]);

  // ---------- Draw overlays (boxes + highlights + live lasso) ----------
  useEffect(() => {
    if (!vp || !overlay.current) return;
    const ov = overlay.current;
    const ctx = ov.getContext("2d"); if (!ctx) return;

    // overlay uses device pixels internally
    ctx.setTransform(1,0,0,1,0,0);
    ctx.clearRect(0,0,ov.width,ov.height);

    if (!ocrSize) return; // cannot scale without OCR dimensions
    const sx = (ov.width / vp.dpr) / ocrSize.width;   // device→ocr mapping split: ov.width is device px
    const sy = (ov.height / vp.dpr) / ocrSize.height;

    // helper to draw a rect in OCR space
    const draw = (r:Rect, mode:"base"|"sel"|"hl") => {
      const x = r.x0 * sx * vp.dpr, y = r.y0 * sy * vp.dpr;
      const w = (r.x1 - r.x0) * sx * vp.dpr, h = (r.y1 - r.y0) * sy * vp.dpr;
      drawStyledRect(ctx, x, y, w, h, mode);
    };

    // Base boxes (blue)
    boxes.forEach((b, i) => {
      if (b.page !== page) return;
      draw(b, selected.includes(i) ? "sel" : "base");
    });

    // Highlights (orange), drawn on top
    highlights.forEach((h) => {
      if (h.page !== page) return;
      draw(h, "hl");
    });

    // Live lasso (drag rectangle) — drawn in overlay CSS pixels → convert to device px
    if (tool === "lasso" && dragStart && dragNow) {
      const x0css = Math.min(dragStart.x, dragNow.x);
      const y0css = Math.min(dragStart.y, dragNow.y);
      const x1css = Math.max(dragStart.x, dragNow.x);
      const y1css = Math.max(dragStart.y, dragNow.y);
      const x = x0css * vp.dpr, y = y0css * vp.dpr;
      const w = (x1css - x0css) * vp.dpr, h = (y1css - y0css) * vp.dpr;

      // dashed purple rubber band
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(145, 66, 255, 0.95)";
      ctx.fillStyle   = "rgba(145, 66, 255, 0.15)";
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
    }
  }, [boxes, highlights, selected, page, vp, ocrSize, tool, dragStart, dragNow]);

  // ---------- Pointer logic ----------
  function hitTestClient(clientX: number, clientY: number): number | null {
    if (!overlay.current || !vp || !ocrSize) return null;
    const r = overlay.current.getBoundingClientRect();
    const pxCss = clientX - r.left;
    const pyCss = clientY - r.top;

    const sx = (vp.wCss) / ocrSize.width;
    const sy = (vp.hCss) / ocrSize.height;

    // Walk from end -> start to prefer later (likely recently added)
    for (let i = boxes.length - 1; i >= 0; i--) {
      const b = boxes[i];
      if (b.page !== page) continue;
      const x = b.x0 * sx, y = b.y0 * sy, w = (b.x1 - b.x0) * sx, h = (b.y1 - b.y0) * sy;
      if (pxCss >= x && pxCss <= x + w && pyCss >= y && pyCss <= y + h) return i;
    }
    return null;
  }

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const el = overlay.current; if (!el) return;
    const r = el.getBoundingClientRect();
    const ox = e.clientX - r.left, oy = e.clientY - r.top;

    if (tool === "lasso") {
      setDragStart({ x: ox, y: oy });
      setDragNow({ x: ox, y: oy });
    } else {
      const hit = hitTestClient(e.clientX, e.clientY);
      if (hit != null && onSelectBox) onSelectBox(hit);
    }
  };

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (tool !== "lasso" || !dragStart) return;
    const el = overlay.current; if (!el) return;
    const r = el.getBoundingClientRect();
    setDragNow({ x: e.clientX - r.left, y: e.clientY - r.top });
  };

  const onMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (tool !== "lasso" || !dragStart || !onLasso || !overlay.current || !vp || !ocrSize) {
      setDragStart(null); setDragNow(null); return;
    }
    const r = overlay.current.getBoundingClientRect();
    const ax = dragStart.x, ay = dragStart.y;
    const bx = e.clientX - r.left, by = e.clientY - r.top;

    // convert CSS px → OCR px
    const sx = ocrSize.width  / vp.wCss;
    const sy = ocrSize.height / vp.hCss;
    const a = { x: ax * sx, y: ay * sy };
    const b = { x: bx * sx, y: by * sy };

    onLasso({
      x0: Math.min(a.x, b.x),
      y0: Math.min(a.y, b.y),
      x1: Math.max(a.x, b.x),
      y1: Math.max(a.y, b.y),
    });

    setDragStart(null); setDragNow(null);
  };

  return (
    <div className="pdf-stage" style={{position:"relative", display:"inline-block"}}>
      <canvas
        ref={baseCanvas}
        className="pdf-base"
        style={{ display:"block", transform:"none" }}   // guard against accidental flips
      />
      <canvas
        ref={overlay}
        className={"pdf-overlay " + tool}
        style={{ position:"absolute", inset:0, cursor: tool==="lasso" ? "crosshair" : "default", transform:"none" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        width={vp ? Math.floor(vp.wCss * (vp.dpr)) : 0}
        height={vp ? Math.floor(vp.hCss * (vp.dpr)) : 0}
      />
    </div>
  );
}

// ---------- helpers ----------
function drawStyledRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number,
  mode: "base" | "sel" | "hl"
) {
  // Ensure no stray transform
  ctx.save();
  ctx.setTransform(1,0,0,1,0,0);

  if (mode === "hl") {
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(255,140,0,0.95)";
    ctx.fillStyle   = "rgba(255,170,0,0.18)";
  } else if (mode === "sel") {
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(0,180,255,1)";
    ctx.fillStyle   = "rgba(0,180,255,0.20)";
  } else {
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(0,120,200,0.9)";
    ctx.fillStyle   = "rgba(0,160,255,0.12)";
  }
  ctx.fillRect(x, y, w, h);
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}
