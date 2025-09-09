// File: src/components/lasso/PdfViewer.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import PdfCanvas, { type RectServer } from "./PdfCanvas";

/** Minimal box shape (kept local to avoid deps drift). */
type BoxLike = {
  page: number;
  x0: number; y0: number; x1: number; y1: number;
  id?: string | null;
  label?: string | null;
  text?: string | null;
  confidence?: number | null;
};

export type PageMeta = { w: number; h: number }; // server-space width/height per page

type Props = {
  /** Absolute/relative URL to the PDF (/data/{doc_id}/original.pdf) */
  docUrl: string;

  /** 1-based number of pages OR per-page meta. If you pass meta, we’ll use that for scaling. */
  pages?: number;
  meta?: PageMeta[]; // preferred

  /** Incoming boxes across the whole document (we filter for current page). */
  boxes?: BoxLike[];

  /** Initial page (1-based) */
  initialPage?: number;

  /** Initial UI state */
  initialShowBoxes?: boolean;
  initialLasso?: boolean;

  /** Events */
  onPageChange?: (page: number) => void;
  onBoxClick?: (box: BoxLike) => void;
  onLassoDone?: (rect: RectServer, page: number) => void;

  /** Optional: control the scale factor used by underlying PdfCanvas (visual zoom). */
  zoomSteps?: number[]; // e.g., [0.9, 1.0, 1.25, 1.5, 2.0]
};

export default function PdfViewer({
  docUrl,
  pages,
  meta,
  boxes = [],
  initialPage = 1,
  initialShowBoxes = true,
  initialLasso = false,
  onPageChange,
  onBoxClick,
  onLassoDone,
  zoomSteps = [0.9, 1.0, 1.25, 1.5, 2.0],
}: Props) {
  const hasMeta = Array.isArray(meta) && meta.length > 0;
  const pageCount = hasMeta ? meta!.length : Math.max(1, pages || 1);

  const [page, setPage] = useState<number>(clamp(initialPage, 1, pageCount));
  const [showBoxes, setShowBoxes] = useState<boolean>(initialShowBoxes);
  const [lasso, setLasso] = useState<boolean>(initialLasso);
  const [zoomIdx, setZoomIdx] = useState<number>(() => {
    const i = zoomSteps.findIndex((z) => Math.abs(z - 1.0) < 1e-3);
    return i >= 0 ? i : 1;
  });

  useEffect(() => { setPage(clamp(initialPage, 1, pageCount)); }, [initialPage, pageCount]);

  const currentMeta = useMemo<PageMeta>(() => {
    if (hasMeta) return meta![page - 1] || { w: 1, h: 1 };
    // If no meta, assume 1000x1400 (scaled proportionally). This is only a fallback.
    return { w: 1000, h: 1400 };
  }, [hasMeta, meta, page]);

  // Filter boxes for current page
  const pageBoxes = useMemo(() => (boxes || []).filter((b) => b.page === page), [boxes, page]);

  function go(delta: number) {
    const next = clamp(page + delta, 1, pageCount);
    if (next !== page) {
      setPage(next);
      onPageChange?.(next);
    }
  }
  function jumpTo(input: string) {
    const n = parseInt(input, 10);
    if (Number.isFinite(n)) {
      const next = clamp(n, 1, pageCount);
      setPage(next);
      onPageChange?.(next);
    }
  }

  function zoomIn() { setZoomIdx((i) => clamp(i + 1, 0, zoomSteps.length - 1)); }
  function zoomOut() { setZoomIdx((i) => clamp(i - 1, 0, zoomSteps.length - 1)); }
  function resetZoom() {
    const at = zoomSteps.findIndex((z) => Math.abs(z - 1.0) < 1e-3);
    setZoomIdx(at >= 0 ? at : 1);
  }

  // We apply zoom by CSS transform wrapping PdfCanvas (keeps PdfCanvas render scale constant for quality).
  const zoom = zoomSteps[zoomIdx];
  const zoomWrapStyle: React.CSSProperties = useMemo(
    () => ({ transform: `scale(${zoom})`, transformOrigin: "top left", display: "inline-block" }),
    [zoom]
  );

  return (
    <div className="border pad">
      {/* Toolbar */}
      <div className="toolbar-inline">
        <button onClick={() => go(-1)} disabled={page <= 1}>Prev</button>
        <span className="page-indicator">
          Page{" "}
          <input
            type="number"
            min={1}
            max={pageCount}
            value={page}
            onChange={(e) => jumpTo(e.target.value)}
            style={{ width: 60, margin: "0 6px" }}
          />
          / {pageCount}
        </span>
        <button onClick={() => go(1)} disabled={page >= pageCount}>Next</button>

        <span style={{ width: 12 }} />

        <button onClick={zoomOut} disabled={zoomIdx <= 0}>−</button>
        <span className="small muted" style={{ minWidth: 48, textAlign: "center" }}>
          {(zoom * 100).toFixed(0)}%
        </span>
        <button onClick={zoomIn} disabled={zoomIdx >= zoomSteps.length - 1}>+</button>
        <button onClick={resetZoom} disabled={Math.abs(zoom - 1.0) < 1e-3}>Reset</button>

        <span style={{ width: 12 }} />

        <label className={showBoxes ? "btn toggle active" : "btn toggle"}>
          <input type="checkbox" checked={showBoxes} onChange={() => setShowBoxes((v) => !v)} /> Boxes
        </label>
        <button className={lasso ? "btn toggle active" : "btn toggle"} onClick={() => setLasso((v) => !v)}>
          Lasso
        </button>
      </div>

      {/* Canvas + overlay (zoomed) */}
      <div style={zoomWrapStyle}>
        <PdfCanvas
          docUrl={docUrl}
          page={page}
          serverW={currentMeta.w}
          serverH={currentMeta.h}
          boxes={pageBoxes}
          showBoxes={showBoxes}
          lasso={lasso}
          onBoxClick={onBoxClick}
          onLassoDone={(rect) => onLassoDone?.(rect, page)}
        />
      </div>

      <div className="hint">Tip: Toggle <span className="kbd">Lasso</span> to draw a rectangle and bind it.</div>
    </div>
  );
}

function clamp(n: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, n));
}
