import { useEffect, useRef, useState } from "react";
import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

export type Rect = { x0: number; y0: number; x1: number; y1: number };
export type Box  = Rect & { page: number; id?: string; label?: string; text?: string };

type Props = {
  url?: string;
  page: number;

  // Layers
  ocrBoxes?: Box[];              // all OCR tokens (light blue)
  highlightBoxes?: Box[];        // search hits (orange)
  boundBoxes?: Box[];            // user-bound edits (green)

  // Selection/linking
  selectedBoxIds?: string[];
  onSelectBox?: (boxId: string, boxIndex: number) => void;

  // Geometry from OCR run
  ocrSize?: { width: number; height: number }; // OCR pixel size per page (pre-rotation)
  scale?: number;
  tool?: "select" | "lasso";
  onLasso?: (rect: Rect) => void;              // returns OCR-space rect (pre-rotation)
};

type VpInfo = { wCss:number; hCss:number; dpr:number; rotation:number };

export default function PdfCanvas({
  url,
  page,
  ocrBoxes = [],
  highlightBoxes = [],
  boundBoxes = [],
  selectedBoxIds = [],
  onSelectBox,
  ocrSize,
  scale = 1.25,
  tool = "select",
  onLasso,
}: Props) {
  const pdfRef = useRef<any>(null);
  const [vp, setVp] = useState<VpInfo | null>(null);
  const baseCanvas = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);

  const [dragStart, setDragStart] = useState<{ x:number; y:number } | null>(null);
  const [dragNow, setDragNow]     = useState<{ x:number; y:number } | null>(null);

  // ----- rotation helpers (OCR <-> viewport) -----
  function rotatePoint(x:number, y:number, W:number, H:number, rot:number) {
    // rot in {0,90,180,270} clockwise. Input: OCR coords (pre-rotation)
    switch (rot) {
      case 0:   return { x,           y };
      case 90:  return { x: y,        y: W - x };
      case 180: return { x: W - x,    y: H - y };
      case 270: return { x: H - y,    y: x };
      default:  return { x, y };
    }
  }
  function invRotatePoint(x:number, y:number, W:number, H:number, rot:number) {
    // inverse mapping: viewport -> OCR
    switch (rot) {
      case 0:   return { x,            y };
      case 90:  return { x: W - y,     y: x };
      case 180: return { x: W - x,     y: H - y };
      case 270: return { x: y,         y: H - x };
      default:  return { x, y };
    }
  }

  // ----- render PDF (offscreen, rotation-aware) -----
  async function renderPage(pnum:number, zoom:number){
    if (!pdfRef.current || !baseCanvas.current) return;
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    const pageObj = await pdfRef.current.getPage(pnum);
    const rotation: number = (pageObj.rotate || 0) % 360;              // PDF’s own rotation
    const viewport = pageObj.getViewport({ scale: zoom * dpr, rotation, dontFlip: true });

    const work = document.createElement("canvas");
    work.width  = Math.floor(viewport.width);
    work.height = Math.floor(viewport.height);
    const wctx = work.getContext("2d"); if (!wctx) return;
    wctx.setTransform(1,0,0,1,0,0);
    wctx.clearRect(0,0,work.width,work.height);

    await pageObj.render({ canvasContext: wctx, viewport }).promise;

    const c = baseCanvas.current!;
    c.width  = work.width;
    c.height = work.height;
    c.style.width  = `${Math.floor(work.width / dpr)}px`;
    c.style.height = `${Math.floor(work.height / dpr)}px`;

    const ctx = c.getContext("2d"); if (!ctx) return;
    ctx.setTransform(1,0,0,1,0,0);
    ctx.clearRect(0,0,c.width,c.height);
    ctx.drawImage(work, 0, 0);

    const ov = overlay.current!;
    ov.width  = c.width;
    ov.height = c.height;
    ov.style.width  = c.style.width;
    ov.style.height = c.style.height;

    setVp({ wCss: Math.floor(work.width / dpr), hCss: Math.floor(work.height / dpr), dpr, rotation });
  }

  // ----- load PDF -----
  useEffect(() => {
    if (!url) return;
    (async () => {
      pdfRef.current = await getDocument({ url }).promise;
      await renderPage(page, scale);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // ----- re-render on page/scale -----
  useEffect(() => {
    if (!pdfRef.current) return;
    renderPage(page, scale);
    setDragStart(null); setDragNow(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, scale]);

  // ----- draw overlays (OCR -> viewport with rotation) -----
  useEffect(() => {
    if (!vp || !overlay.current) return;
    const ov = overlay.current;
    const ctx = ov.getContext("2d"); if (!ctx) return;

    ctx.setTransform(1,0,0,1,0,0);
    ctx.clearRect(0,0,ov.width,ov.height);

    if (!ocrSize) return;

    // If page is rotated 90/270, viewport's width/height correspond to swapped OCR dims.
    const oW = ocrSize.width, oH = ocrSize.height;
    const vWcss = vp.wCss, vHcss = vp.hCss;

    // CSS→device scale
    const sx = (vWcss / (vp.rotation % 180 === 0 ? oW : oH)) * vp.dpr;
    const sy = (vHcss / (vp.rotation % 180 === 0 ? oH : oW)) * vp.dpr;

    const draw = (r:Rect, mode:"ocr"|"hl"|"bound"|"selected") => {
      // transform the four corners, then compute axis-aligned bounds in viewport space
      const p1 = rotatePoint(r.x0, r.y0, oW, oH, vp.rotation);
      const p2 = rotatePoint(r.x1, r.y1, oW, oH, vp.rotation);
      const x0 = Math.min(p1.x, p2.x) * sx;
      const y0 = Math.min(p1.y, p2.y) * sy;
      const x1 = Math.max(p1.x, p2.x) * sx;
      const y1 = Math.max(p1.y, p2.y) * sy;
      drawStyledRect(ctx, x0, y0, x1 - x0, y1 - y0, mode);
    };

    // 1) OCR (blue)
    ocrBoxes.forEach((b) => { if (b.page === page) draw(b, "ocr"); });
    // 2) Highlights (orange)
    highlightBoxes.forEach((h) => { if (h.page === page) draw(h, "hl"); });
    // 3) Bound (green)
    boundBoxes.forEach((g) => { if (g.page === page) draw(g, "bound"); });
    // 4) Selected (purple)
    if (selectedBoxIds?.length) {
      const ids = new Set(selectedBoxIds);
      [...ocrBoxes, ...highlightBoxes, ...boundBoxes].forEach((b) => {
        if (b.page !== page || !b.id) return;
        if (ids.has(b.id)) draw(b, "selected");
      });
    }

    // live lasso (viewport CSS → device px)
    if (tool === "lasso" && dragStart && dragNow) {
      const x0css = Math.min(dragStart.x, dragNow.x);
      const y0css = Math.min(dragStart.y, dragNow.y);
      const x1css = Math.max(dragStart.x, dragNow.x);
      const y1css = Math.max(dragStart.y, dragNow.y);
      const x = x0css * vp.dpr, y = y0css * vp.dpr;
      const w = (x1css - x0css) * vp.dpr, h = (y1css - y0css) * vp.dpr;

      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(145, 66, 255, 0.95)";
      ctx.fillStyle   = "rgba(145, 66, 255, 0.15)";
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
    }
  }, [ocrBoxes, highlightBoxes, boundBoxes, selectedBoxIds, page, vp, ocrSize, tool, dragStart, dragNow]);

  // ----- hit testing (select tool) -----
  function hitTestClient(clientX: number, clientY: number): { id?:string; idx:number } | null {
    if (!overlay.current || !vp || !ocrSize) return null;
    const r = overlay.current.getBoundingClientRect();
    const pxCss = clientX - r.left;
    const pyCss = clientY - r.top;

    // viewport CSS → OCR px (inverse rotation)
    const vWcss = vp.wCss, vHcss = vp.hCss;
    const oW = ocrSize.width, oH = ocrSize.height;

    const sxInv = (vp.rotation % 180 === 0 ? oW : oH) / vWcss;
    const syInv = (vp.rotation % 180 === 0 ? oH : oW) / vHcss;

    const vx = pxCss * sxInv;
    const vy = pyCss * syInv;
    const p = invRotatePoint(vx, vy, oW, oH, vp.rotation);

    // Prefer bound > highlight > ocr
    const layers: { arr: Box[] }[] = [
      { arr: boundBoxes.filter(b => b.page === page) },
      { arr: highlightBoxes.filter(b => b.page === page) },
      { arr: ocrBoxes.filter(b => b.page === page) }
    ];

    for (const { arr } of layers) {
      for (let i = arr.length - 1; i >= 0; i--) {
        const b = arr[i];
        if (p.x >= Math.min(b.x0, b.x1) && p.x <= Math.max(b.x0, b.x1) &&
            p.y >= Math.min(b.y0, b.y1) && p.y <= Math.max(b.y0, b.y1)) {
          return { id: b.id, idx: i };
        }
      }
    }
    return null;
  }

  // ----- pointer handlers -----
  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const el = overlay.current; if (!el) return;
    const r = el.getBoundingClientRect();
    const ox = e.clientX - r.left, oy = e.clientY - r.top;

    if (tool === "lasso") {
      setDragStart({ x: ox, y: oy });
      setDragNow({ x: ox, y: oy });
    } else {
      const hit = hitTestClient(e.clientX, e.clientY);
      if (hit && onSelectBox) onSelectBox(hit.id || "", hit.idx);
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
    const x0css = Math.min(dragStart.x, e.clientX - r.left);
    const y0css = Math.min(dragStart.y, e.clientY - r.top);
    const x1css = Math.max(dragStart.x, e.clientX - r.left);
    const y1css = Math.max(dragStart.y, e.clientY - r.top);

    // viewport CSS → OCR px rect (inverse rotation)
    const vWcss = vp.wCss, vHcss = vp.hCss;
    const oW = ocrSize.width, oH = ocrSize.height;
    const sxInv = (vp.rotation % 180 === 0 ? oW : oH) / vWcss;
    const syInv = (vp.rotation % 180 === 0 ? oH : oW) / vHcss;

    const a = invRotatePoint(x0css * sxInv, y0css * syInv, oW, oH, vp.rotation);
    const b = invRotatePoint(x1css * sxInv, y1css * syInv, oW, oH, vp.rotation);

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
        style={{ display:"block", transform:"none" }}
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

// ---------- drawing ----------
function drawStyledRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number,
  mode: "ocr" | "hl" | "bound" | "selected"
) {
  ctx.save();
  ctx.setTransform(1,0,0,1,0,0);
  if (mode === "hl") {                 // orange
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(255,140,0,0.95)";
    ctx.fillStyle   = "rgba(255,170,0,0.20)";
  } else if (mode === "bound") {       // green
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(16,158,0,1)";
    ctx.fillStyle   = "rgba(16,158,0,0.18)";
  } else if (mode === "selected") {    // purple
    ctx.lineWidth = 4;
    ctx.setLineDash([8, 5]);
    ctx.strokeStyle = "rgba(145, 66, 255, 1)";
    ctx.fillStyle   = "rgba(145, 66, 255, 0.10)";
  } else {                             // ocr (blue)
    ctx.lineWidth = 1.75;
    ctx.strokeStyle = "rgba(0,120,200,0.9)";
    ctx.fillStyle   = "rgba(0,160,255,0.10)";
  }
  ctx.fillRect(x, y, w, h);
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}
