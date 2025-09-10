// File: src/tsp4/components/lasso/PdfEditCanvas.tsx
import React, { useEffect, useRef } from "react";
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
  editable: boolean; // not used here (reserved for future lasso)
  onRectChange: (r: EditRect | null) => void;
  onRectCommit: (r: EditRect) => void;
};

export default function PdfEditCanvas({
  docUrl,
  page,
  serverW,
  serverH,
  tokens,
  rect,
  showTokenBoxes,
}: Props) {
  const baseCanvas = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);

  // Render page exactly at OCR resolution
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!docUrl) return;
      const pdf: PDFDocumentProxy = await getDocument(docUrl).promise;
      if (cancelled) return;
      const pg: PDFPageProxy = await pdf.getPage(page);
      if (cancelled) return;

      // pdf.view: [xMin, yMin, xMax, yMax]; natural width = view[2]
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
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl, page, serverW, serverH]);

  // Repaint overlay when inputs change
  useEffect(() => {
    drawOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokens, rect, showTokenBoxes, page, serverW, serverH]);

  function drawOverlay() {
    const overlay = overlayRef.current;
    if (!overlay) return;
    overlay.innerHTML = "";

    // Orange OCR token boxes
    if (showTokenBoxes) {
      for (const t of tokens) {
        if (t.page !== page) continue;
        const d = document.createElement("div");
        d.className = "tok";
        placeCss(d, t.x0, t.y0, t.x1, t.y1);
        overlay.appendChild(d);
      }
    }

    // Pink union highlight
    if (rect && rect.page === page) {
      const d = document.createElement("div");
      d.className = "pink";
      placeCss(d, rect.x0, rect.y0, rect.x1, rect.y1);
      overlay.appendChild(d);
    }
  }

  // OCR px -> overlay CSS px
  function placeCss(node: HTMLDivElement, x0: number, y0: number, x1: number, y1: number) {
    const overlay = overlayRef.current!;
    const sx = overlay.clientWidth / serverW;
    const sy = overlay.clientHeight / serverH;
    node.style.left = `${Math.min(x0, x1) * sx}px`;
    node.style.top = `${Math.min(y0, y1) * sy}px`;
    node.style.width = `${Math.abs(x1 - x0) * sx}px`;
    node.style.height = `${Math.abs(y1 - y0) * sy}px`;
  }

  return (
    <div
      className="pdf-stage"
      style={{ position: "relative", overflow: "auto", width: "100%", height: "100%" }}
    >
      <canvas ref={baseCanvas} />
      <div
        ref={overlayRef}
        className="overlay"
        style={{
          position: "absolute",
          inset: 0,
          background: "transparent",
          userSelect: "none",
          pointerEvents: "none",
        }}
      />
    </div>
  );
}