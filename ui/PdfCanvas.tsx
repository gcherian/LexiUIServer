import { useEffect, useRef, useState } from "react";
import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";

// pdfjs-dist v4 ESM worker (Vite-friendly)
GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

export type Rect = { x0: number; y0: number; x1: number; y1: number };
export type Box = Rect & { page: number; id?: string; label?: string };

type Props = {
  url?: string;
  page: number;
  boxes?: Box[];                         // base layer (blue)
  highlights?: Box[];                    // search results (orange)
  selected?: number[];                   // indices referring to boxes[]
  ocrSize?: { width: number; height: number };
  scale?: number;
  tool?: "select" | "lasso";
  onLasso?: (rect: Rect) => void;        // OCR space
  onSelectBox?: (boxIndex: number) => void;
};

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
  const [vp, setVp] = useState<any>(null);
  const baseCanvas = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);

  // ------- Load PDF -------
  useEffect(() => {
    if (!url) return;
    (async () => {
      pdfRef.current = await getDocument(url).promise;
      const p = await pdfRef.current.getPage(page);
      const viewport = p.getViewport({ scale });
      const c = baseCanvas.current!;
      c.width = viewport.width; c.height = viewport.height;
      const ctx = c.getContext("2d"); if (!ctx) return;
      await p.render({ canvasContext: ctx, viewport }).promise;
      setVp(viewport);
    })();
  }, [url]);

  // ------- Render on page/scale change -------
  useEffect(() => {
    (async () => {
      if (!pdfRef.current) return;
      const p = await pdfRef.current.getPage(page);
      const viewport = p.getViewport({ scale });
      const c = baseCanvas.current!;
      c.width = viewport.width; c.height = viewport.height;
      const ctx = c.getContext("2d"); if (!ctx) return;
      await p.render({ canvasContext: ctx, viewport }).promise;
      setVp(viewport);
    })();
  }, [page, scale]);

  // ------- Draw overlays -------
  useEffect(() => {
    if (!vp) return;
    const ov = overlay.current!;
    const ctx = ov.getContext("2d"); if (!ctx) return;
    ov.width = vp.width; ov.height = vp.height;
    ctx.clearRect(0, 0, ov.width, ov.height);

    if (!ocrSize) return; // cannot scale without OCR dimensions
    const sx = vp.width / ocrSize.width;
    const sy = vp.height / ocrSize.height;

    // Base boxes (blue)
    boxes.forEach((b, i) => {
      if (b.page !== page) return;
      drawBox(ctx, b, sx, sy, selected.includes(i) ? "sel" : "base");
    });

    // Highlights (orange), drawn on top
    highlights.forEach((h) => {
      if (h.page !== page) return;
      drawBox(ctx, h, sx, sy, "hl");
    });
  }, [boxes, highlights, selected, page, vp, ocrSize]);

  // ------- Pointer logic -------
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null);

  const toView = (x: number, y: number) => {
    if (!vp || !ocrSize) return { x, y };
    const sx = vp.width / ocrSize.width;
    const sy = vp.height / ocrSize.height;
    return { x: x * sx, y: y * sy };
  };
  const toOcr = (x: number, y: number) => {
    if (!vp || !ocrSize) return { x, y };
    const sx = ocrSize.width / vp.width;
    const sy = ocrSize.height / vp.height;
    return { x: x * sx, y: y * sy };
  };

  function hitTestClient(clientX: number, clientY: number): number | null {
    if (!overlay.current || !vp || !ocrSize) return null;
    const r = overlay.current.getBoundingClientRect();
    const px = clientX - r.left;
    const py = clientY - r.top;

    // Find the *topmost* box on this page under pointer
    const sx = vp.width / ocrSize.width;
    const sy = vp.height / ocrSize.height;

    // Walk from end -> start to prefer later (likely recently added)
    for (let i = boxes.length - 1; i >= 0; i--) {
      const b = boxes[i];
      if (b.page !== page) continue;
      const x = b.x0 * sx, y = b.y0 * sy, w = (b.x1 - b.x0) * sx, h = (b.y1 - b.y0) * sy;
      if (px >= x && px <= x + w && py >= y && py <= y + h) return i;
    }
    return null;
  }

  return (
    <div className="pdf-stage">
      <canvas ref={baseCanvas} className="pdf-base" />
      <canvas
        ref={overlay}
        className={"pdf-overlay " + tool}
        onMouseDown={(e) => {
          const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
          const ox = e.clientX - r.left, oy = e.clientY - r.top;
          if (tool === "lasso") {
            setDrag({ x: ox, y: oy });
          } else {
            const hit = hitTestClient(e.clientX, e.clientY);
            if (hit != null && onSelectBox) onSelectBox(hit);
          }
        }}
        onMouseUp={(e) => {
          if (tool !== "lasso" || !drag || !onLasso) { setDrag(null); return; }
          const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
          const a = toOcr(drag.x, drag.y);
          const b = toOcr(e.clientX - r.left, e.clientY - r.top);
          onLasso({
            x0: Math.min(a.x, b.x),
            y0: Math.min(a.y, b.y),
            x1: Math.max(a.x, b.x),
            y1: Math.max(a.y, b.y),
          });
          setDrag(null);
        }}
        width={vp?.width || 0}
        height={vp?.height || 0}
      />
    </div>
  );
}

// ---------- helpers ----------
function drawBox(
  ctx: CanvasRenderingContext2D,
  b: Box,
  sx: number,
  sy: number,
  mode: "base" | "sel" | "hl"
) {
  const x = b.x0 * sx, y = b.y0 * sy, w = (b.x1 - b.x0) * sx, h = (b.y1 - b.y0) * sy;

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
}
