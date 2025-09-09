import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy, type PDFPageProxy } from "pdfjs-dist";
import "pdfjs-dist/web/pdf_viewer.css";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.js",
  import.meta.url
).toString();

export type Token = { page: number; x0: number; y0: number; x1: number; y1: number; text?: string };
export type EditRect = { page: number; x0: number; y0: number; x1: number; y1: number };

type Props = {
  docUrl: string;
  page: number;
  serverW: number;
  serverH: number;
  tokens: Token[];
  rect: EditRect | null;                // committed pink rect in OCR px
  showTokenBoxes: boolean;
  editable: boolean;                    // if false: no lasso, overlay ignores pointer
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
  editable,
  onRectChange,
  onRectCommit,
}: Props) {
  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  const pageRef = useRef<PDFPageProxy | null>(null);

  const wrapRef = useRef<HTMLDivElement | null>(null);     // scroll container
  const canvasRef = useRef<HTMLCanvasElement | null>(null);// rendered PDF bitmap

  // live drag state (CSS space)
  const [liveCss, setLiveCss] = useState<{ left: number; top: number; width: number; height: number } | null>(null);
  const dragStartCss = useRef<{ x: number; y: number } | null>(null);
  const pointerIdRef = useRef<number | null>(null);

  // render the page
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!docUrl) return;
      const pdf = await getDocument(docUrl).promise;
      if (cancelled) return;
      pdfRef.current = pdf;
      const pg = await pdf.getPage(page);
      if (cancelled) return;
      pageRef.current = pg;

      const viewport = pg.getViewport({ scale: 1, rotation: pg.rotate || 0 });
      const scale = Math.min(1, 1400 / Math.max(viewport.width, viewport.height));
      const vp = pg.getViewport({ scale, rotation: pg.rotate || 0 });

      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d")!;
      canvas.width = Math.floor(vp.width);
      canvas.height = Math.floor(vp.height);
      canvas.style.width = `${canvas.width}px`;
      canvas.style.height = `${canvas.height}px`;

      await pg.render({ canvasContext: ctx, viewport: vp }).promise;
      // reset live box whenever we re-render the page
      setLiveCss(null);
    })();
    return () => {
      cancelled = true;
    };
  }, [docUrl, page]);

  // keep overlay sized to canvas via layout pass (no flicker)
  useLayoutEffect(() => {
    // nothing else needed: overlay div is absolutely positioned over canvas
  }, [serverW, serverH]);

  // ---------- CSS <-> OCR mapping helpers ----------
  const rot = useMemo(() => (((pageRef.current?.rotate || 0) % 360) + 360) % 360, [page]);
  function cssToOcr(xc: number, yc: number) {
    const r = canvasRef.current!.getBoundingClientRect();
    const cssW = r.width, cssH = r.height;
    const sx = serverW / cssW, sy = serverH / cssH;
    const x = xc * sx, y = yc * sy;
    switch (rot) {
      case 0:   return { x,               y };
      case 90:  return { x: y,            y: serverW - x };
      case 180: return { x: serverW - x,  y: serverH - y };
      case 270: return { x: serverH - y,  y: x };
      default:  return { x,               y };
    }
  }
  function ocrToCss(x: number, y: number) {
    const r = canvasRef.current!.getBoundingClientRect();
    const cssW = r.width, cssH = r.height;
    const sx = cssW / serverW, sy = cssH / serverH;
    switch (rot) {
      case 0:   return { x: x * sx,                    y: y * sy };
      case 90:  return { x: (serverH - y) * sx,        y: x * sy };
      case 180: return { x: (serverW - x) * sx,        y: (serverH - y) * sy };
      case 270: return { x: y * sx,                    y: (serverW - x) * sy };
      default:  return { x: x * sx,                    y: y * sy };
    }
  }

  // ---------- pointer handlers with capture ----------
  function pointerDown(e: React.PointerEvent) {
    if (!editable || !wrapRef.current || !canvasRef.current) return;
    const canvasRect = canvasRef.current.getBoundingClientRect();
    const scx = wrapRef.current.scrollLeft;
    const scy = wrapRef.current.scrollTop;

    const x = e.clientX - canvasRect.left + scx;
    const y = e.clientY - canvasRect.top + scy;

    pointerIdRef.current = e.pointerId;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragStartCss.current = { x, y };
    setLiveCss({ left: x, top: y, width: 0, height: 0 });
  }

  function pointerMove(e: React.PointerEvent) {
    if (!editable || !dragStartCss.current || !wrapRef.current || !canvasRef.current) return;
    if (pointerIdRef.current !== e.pointerId) return;

    const canvasRect = canvasRef.current.getBoundingClientRect();
    const scx = wrapRef.current.scrollLeft;
    const scy = wrapRef.current.scrollTop;

    const x = e.clientX - canvasRect.left + scx;
    const y = e.clientY - canvasRect.top + scy;
    const x0 = Math.max(0, Math.min(dragStartCss.current.x, x));
    const y0 = Math.max(0, Math.min(dragStartCss.current.y, y));
    const x1 = Math.min(canvasRect.width, Math.max(dragStartCss.current.x, x));
    const y1 = Math.min(canvasRect.height, Math.max(dragStartCss.current.y, y));

    setLiveCss({ left: x0, top: y0, width: Math.max(0, x1 - x0), height: Math.max(0, y1 - y0) });
  }

  function pointerUp(e: React.PointerEvent) {
    if (!editable || !dragStartCss.current || !wrapRef.current || !canvasRef.current) return;
    if (pointerIdRef.current !== e.pointerId) return;

    const canvasRect = canvasRef.current.getBoundingClientRect();
    const scx = wrapRef.current.scrollLeft;
    const scy = wrapRef.current.scrollTop;

    const endX = e.clientX - canvasRect.left + scx;
    const endY = e.clientY - canvasRect.top + scy;

    const xCss0 = Math.max(0, Math.min(dragStartCss.current.x, endX));
    const yCss0 = Math.max(0, Math.min(dragStartCss.current.y, endY));
    const xCss1 = Math.min(canvasRect.width, Math.max(dragStartCss.current.x, endX));
    const yCss1 = Math.min(canvasRect.height, Math.max(dragStartCss.current.y, endY));

    // reset drag state first (prevents lingering live rectangles)
    dragStartCss.current = null;
    pointerIdRef.current = null;
    setLiveCss(null);

    // ignore tiny drags
    if ((xCss1 - xCss0) * (yCss1 - yCss0) < 9) return;

    const p0 = cssToOcr(xCss0, yCss0);
    const p1 = cssToOcr(xCss1, yCss1);

    const x0 = Math.max(0, Math.min(Math.floor(Math.min(p0.x, p1.x)), serverW - 1));
    const y0 = Math.max(0, Math.min(Math.floor(Math.min(p0.y, p1.y)), serverH - 1));
    const x1 = Math.max(0, Math.min(Math.ceil(Math.max(p0.x, p1.x)), serverW - 1));
    const y1 = Math.max(0, Math.min(Math.ceil(Math.max(p0.y, p1.y)), serverH - 1));

    const rr: EditRect = { page, x0, y0, x1, y1 };
    onRectChange(rr);        // keep UI in sync instantly
    onRectCommit(rr);        // let parent OCR & persist
  }

  // render helpers (React, not innerHTML)
  const tokenDivs = useMemo(() => {
    if (!showTokenBoxes || !canvasRef.current) return null;
    return tokens.map((t, i) => {
      const a = ocrToCss(t.x0, t.y0);
      const b = ocrToCss(t.x1, t.y1);
      const left = Math.min(a.x, b.x);
      const top = Math.min(a.y, b.y);
      const width = Math.abs(b.x - a.x);
      const height = Math.abs(b.y - a.y);
      return (
        <div
          key={`tok-${i}`}
          className="tok"
          style={{ position: "absolute", left, top, width, height }}
          title={t.text || ""}
        />
      );
    });
  }, [tokens, showTokenBoxes, serverW, serverH, rot]);

  const committedPink = useMemo(() => {
    if (!rect || rect.page !== page || !canvasRef.current) return null;
    const a = ocrToCss(Math.min(rect.x0, rect.x1), Math.min(rect.y0, rect.y1));
    const b = ocrToCss(Math.max(rect.x0, rect.x1), Math.max(rect.y0, rect.y1));
    const left = Math.min(a.x, b.x);
    const top = Math.min(a.y, b.y);
    const width = Math.abs(b.x - a.x);
    const height = Math.abs(b.y - a.y);
    return <div className="pink" style={{ position: "absolute", left, top, width, height }} />;
  }, [rect, page, serverW, serverH, rot]);

  const livePink = useMemo(() => {
    if (!liveCss) return null;
    const { left, top, width, height } = liveCss;
    return <div className="pink live" style={{ position: "absolute", left, top, width, height }} />;
  }, [liveCss]);

  return (
    <div ref={wrapRef} className="pdf-stage" style={{ position: "relative", overflow: "auto" }}>
      <canvas ref={canvasRef} />
      <div
        style={{
          position: "absolute",
          inset: 0,
          zIndex: 2,
          userSelect: "none",
          pointerEvents: editable ? "auto" : "none",
          cursor: editable ? "crosshair" : "default",
        }}
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerUp={pointerUp}
        onPointerCancel={() => {
          dragStartCss.current = null;
          pointerIdRef.current = null;
          setLiveCss(null);
        }}
      >
        {tokenDivs}
        {committedPink}
        {livePink}
      </div>
    </div>
  );
}
