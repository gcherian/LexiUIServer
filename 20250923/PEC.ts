// File: src/tsp4/components/lasso/PdfEditCanvas.tsx
import React, { useEffect, useRef } from "react";
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy } from "pdfjs-dist";
import "pdfjs-dist/web/pdf_viewer.css"; // harmless (no SASS)

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

export type TokenBox = {
  page: number; x0: number; y0: number; x1: number; y1: number; text?: string;
};

export type EditRect = { page:number; x0:number; y0:number; x1:number; y1:number };

type Overlay = { label: string; color: string; rect: EditRect | null };

type Props = {
  docUrl: string;
  page: number;
  serverW: number; // page width used by the server (px @ dpi)
  serverH: number; // page height used by the server (px @ dpi)
  zoom?: number;   // 0.5 .. 3 (default 1)
  rect?: EditRect | null;                // pink autolocate
  overlays?: Overlay[];                  // fuzzy/tfidf/minilm/distilbert
  tokens?: TokenBox[];                   // optional orange token boxes
  showTokenBoxes?: boolean;              // toggle for tokens
  // legacy props (ignored to keep FLE compile-clean)
  editable?: boolean;
  onRectChange?: (r: EditRect|null)=>void;
  onRectCommit?: (r: EditRect)=>void;
};

export default function PdfEditCanvas({
  docUrl, page, serverW, serverH, zoom = 1,
  rect, overlays = [], tokens = [], showTokenBoxes = false,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const docRef = useRef<PDFDocumentProxy | null>(null);

  // load pdf once per URL
  useEffect(() => {
    let cancelled = false;
    (async () => {
      docRef.current = null;
      try {
        const loadingTask = getDocument({ url: docUrl, withCredentials: false });
        const pdf = await loadingTask.promise;
        if (!cancelled) docRef.current = pdf;
      } catch (e) {
        console.warn("pdf load failed", e);
      } finally {
        // force a redraw attempt
        paint();
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl]);

  // draw whenever inputs change
  useEffect(() => { paint(); /* eslint-disable-next-line */ }, [page, zoom, rect, overlays, showTokenBoxes, tokens, serverW, serverH]);

  async function paint() {
    const cvs = canvasRef.current;
    const pdf = docRef.current;
    if (!cvs || !pdf) return;

    const pgIdx = Math.max(1, Math.min(page, pdf.numPages));
    const pdfPage = await pdf.getPage(pgIdx);

    // scale the page using server px baseline then apply zoom
    const baseW = serverW || pdfPage.view[2];
    const baseH = serverH || pdfPage.view[3];
    const scale = zoom;

    cvs.width  = Math.floor(baseW * scale);
    cvs.height = Math.floor(baseH * scale);

    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    ctx.save();
    ctx.clearRect(0, 0, cvs.width, cvs.height);

    // render page
    await pdfPage.render({
      canvasContext: ctx,
      viewport: pdfPage.getViewport({ scale: (cvs.width / baseW) }),
      // NOTE: we align the viewport scale so the bitmap fits the canvas
    }).promise;

    // utility to draw rects defined in server coordinates
    const drawRect = (r: EditRect, color: string, dash: number[] = [], width = 2, alpha = 1) => {
      if (!r || r.page !== pgIdx) return;
      const x = Math.min(r.x0, r.x1) * scale;
      const y = Math.min(r.y0, r.y1) * scale;
      const w = Math.abs(r.x1 - r.x0) * scale;
      const h = Math.abs(r.y1 - r.y0) * scale;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.setLineDash(dash);
      ctx.lineWidth = width;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
    };

    // tokens (thin orange)
    if (showTokenBoxes && tokens && tokens.length) {
      ctx.save();
      ctx.strokeStyle = "#fb923c"; // orange-400
      ctx.lineWidth = 1;
      for (const t of tokens) {
        if (t.page !== pgIdx) continue;
        const x = Math.min(t.x0, t.x1) * scale;
        const y = Math.min(t.y0, t.y1) * scale;
        const w = Math.abs(t.x1 - t.x0) * scale;
        const h = Math.abs(t.y1 - t.y0) * scale;
        ctx.strokeRect(x, y, w, h);
      }
      ctx.restore();
    }

    // heuristic overlays first (colored)
    for (const o of overlays) {
      if (!o?.rect) continue;
      drawRect(o.rect, o.color, [6, 4], 2, 0.9);
    }

    // autolocate (pink, on top)
    if (rect) drawRect(rect, "#ec4899", [2, 2], 3, 1);

    ctx.restore();
  }

  return (
    <div style={{ position: "relative", width: "100%", overflow: "auto" }}>
      <canvas ref={canvasRef} style={{ display: "block", maxWidth: "100%" }} />
    </div>
  );
}
