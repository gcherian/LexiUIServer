// src/components/PdfBBoxWorkbench.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  uploadPdf,
  getBoxes,
  listFields,
  saveFieldState,
  type Box as ApiBox,
  type FieldState as ApiFieldState,
} from "../../lib/api";

import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy, type PDFPageProxy } from "pdfjs-dist";
import pdfjsWorker from "pdfjs-dist/build/pdf.worker.min?url";

// Configure pdf.js worker (Vite-friendly)
GlobalWorkerOptions.workerSrc = pdfjsWorker as string;

type Rect = { x0: number; y0: number; x1: number; y1: number };
type Box = ApiBox;
type FieldState = ApiFieldState;

export default function PdfBBoxWorkbench() {
  // Canvas + overlay refs
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<SVGSVGElement | null>(null);

  // PDF state
  const [docId, setDocId] = useState<string>("");
  const [pdfUrl, setPdfUrl] = useState<string>("");
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [pageNum, setPageNum] = useState<number>(1);
  const [viewportH, setViewportH] = useState<number>(0);

  // Data state
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [fields, setFields] = useState<FieldState[]>([]);

  // Selection & UI
  const [selectedBoxId, setSelectedBoxId] = useState<string | undefined>(undefined);
  const [selectedField, setSelectedField] = useState<FieldState | undefined>(undefined);
  const [loading, setLoading] = useState<boolean>(false);
  const [showUrlInput, setShowUrlInput] = useState<boolean>(false);

  // Lasso
  const [isLasso, setIsLasso] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragBox, setDragBox] = useState<Rect | null>(null);

  // Load last used URL (optional convenience)
  useEffect(() => {
    const last = localStorage.getItem("doc_pdf_url") || "";
    if (last) setPdfUrl(last);
  }, []);

  // If URL changes (manual paste), load the PDF
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!pdfUrl) return;
      try {
        setLoading(true);
        const doc = await getDocument(pdfUrl).promise;
        if (cancelled) return;
        setPdf(doc);
        setPageNum(1);
      } catch (e) {
        // ignore; user may paste an invalid URL and upload instead
      } finally {
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [pdfUrl]);

  // Fetch boxes & fields whenever pdf/page changes
  useEffect(() => {
    (async () => {
      if (!pdf || !pdfUrl) return;
      const pageBoxes = await getBoxes({ doc_url: pdfUrl, page: pageNum });
      setBoxes(pageBoxes || []);
      const allFields = await listFields({ doc_url: pdfUrl });
      setFields(allFields || []);
      setSelectedBoxId(undefined);
      setSelectedField(undefined);
    })();
  }, [pdf, pdfUrl, pageNum]);

  // Render current page and overlay the boxes
  useEffect(() => {
    if (!pdf || !canvasRef.current || !overlayRef.current) return;

    (async () => {
      const page: PDFPageProxy = await pdf.getPage(pageNum);
      const viewport = page.getViewport({ scale: 1.5 });

      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d")!;
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      setViewportH(viewport.height);

      const renderTask = page.render({ canvasContext: ctx, viewport });
      await renderTask.promise;

      const svg = overlayRef.current!;
      svg.setAttribute("viewBox", `0 0 ${viewport.width} ${viewport.height}`);
      svg.setAttribute("width", `${viewport.width}`);
      svg.setAttribute("height", `${viewport.height}`);
      svg.innerHTML = ""; // clear

      boxes.forEach((b) => {
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("class", "bbox-group");
        g.addEventListener("mouseenter", () => setSelectedBoxId(b.id || `${b.page}:${b.x0}:${b.y0}`));
        g.addEventListener("mouseleave", () =>
          setSelectedBoxId((prev) => (prev === (b.id || `${b.page}:${b.x0}:${b.y0}`) ? undefined : prev))
        );
        g.addEventListener("click", () => {
          const id = b.id || `${b.page}:${b.x0}:${b.y0}`;
          setSelectedBoxId(id);
          // Try to find an associated field by bbox overlap
          const f = fields.find(
            (f) =>
              f.bbox &&
              f.bbox.page === b.page &&
              !(f.bbox.x1! < b.x0 || f.bbox.x0! > b.x1 || f.bbox.y1! < b.y0 || f.bbox.y0! > b.y1)
          );
          if (f) setSelectedField(f);
          else {
            // Pre-fill a temporary field based on the box
            setSelectedField({
              id,
              name: b.label || "unknown_field",
              value: "",
              page: b.page,
              bbox: { x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1, page: b.page },
              confidence: b.confidence ?? 0,
              source: "bbox",
            });
          }
        });

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", `${b.x0}`);
        rect.setAttribute("y", `${viewport.height - b.y1}`); // flip Y for canvas coords
        rect.setAttribute("width", `${b.x1 - b.x0}`);
        rect.setAttribute("height", `${b.y1 - b.y0}`);
        rect.setAttribute("class", (selectedBoxId && (b.id || `${b.page}:${b.x0}:${b.y0}`) === selectedBoxId) ? "bbox-rect selected" : "bbox-rect");

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", `${b.x0 + 2}`);
        label.setAttribute("y", `${viewport.height - b.y1 - 4}`);
        label.setAttribute("class", "bbox-label");
        label.textContent = b.label ?? "";

        g.appendChild(rect);
        if (b.label) g.appendChild(label);
        svg.appendChild(g);
      });
    })();
  }, [pdf, pageNum, boxes, selectedBoxId, fields]);

  // ===== Upload handling =====
  async function onUpload(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f) return;
    setLoading(true);
    try {
      const res = await uploadPdf(f); // { doc_id, annotated_tokens_url, pages }
      setDocId(res.doc_id);
      setPdfUrl(res.annotated_tokens_url);
      localStorage.setItem("doc_pdf_url", res.annotated_tokens_url);

      // Load the newly uploaded PDF
      const doc = await getDocument(res.annotated_tokens_url).promise;
      setPdf(doc);
      setPageNum(1);
    } catch (e) {
      console.error("Upload failed:", e);
    } finally {
      setLoading(false);
      // Reset file input (so selecting the same file again triggers change)
      (ev.target as HTMLInputElement).value = "";
    }
  }

  // ===== Lasso helpers =====
  function clientToSvg(e: React.MouseEvent<SVGSVGElement, MouseEvent>) {
    const svg = overlayRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const m = svg.getScreenCTM();
    if (!m) return { x: 0, y: 0 };
    const p = pt.matrixTransform(m.inverse());
    return { x: p.x, y: p.y };
  }

  function onOverlayMouseDown(e: React.MouseEvent<SVGSVGElement>) {
    if (!isLasso) return;
    const { x, y } = clientToSvg(e);
    setDragStart({ x, y });
    setDragBox({ x0: x, y0: y, x1: x, y1: y });
  }

  function onOverlayMouseMove(e: React.MouseEvent<SVGSVGElement>) {
    if (!isLasso || !dragStart) return;
    const { x, y } = clientToSvg(e);
    setDragBox({
      x0: Math.min(dragStart.x, x),
      y0: Math.min(dragStart.y, y),
      x1: Math.max(dragStart.x, x),
      y1: Math.max(dragStart.y, y),
    });
  }

  async function onOverlayMouseUp() {
    if (!isLasso || !dragBox) return;
    setIsLasso(false);
    setDragStart(null);

    // Convert SVG (top-left origin) to PDF image coords (bottom-left origin)
    const b = {
      x0: dragBox.x0,
      y0: viewportH - dragBox.y1,
      x1: dragBox.x1,
      y1: viewportH - dragBox.y0,
    };
    const tempId = `bbox_${Date.now()}`;
    const nf: FieldState = {
      id: tempId,
      name: "new_field",
      value: "",
      page: pageNum,
      bbox: { ...b, page: pageNum },
      confidence: 1.0,
      source: "lasso",
    };
    setFields((fs) => [nf, ...fs]);
    setSelectedField(nf);
    setSelectedBoxId(undefined);
    setDragBox(null);
  }

  // ===== Save handler =====
  async function onSaveSelected() {
    if (!selectedField || !pdfUrl) return;
    // normalize: prefer key (server canonical)
    const payload: FieldState = { ...selectedField };
    if (!payload.key && payload.name) payload.key = payload.name;
    await saveFieldState({ doc_url: pdfUrl, field: payload });
    // Refresh field list from server (optional; can be omitted for speed)
    const refreshed = await listFields({ doc_url: pdfUrl });
    setFields(refreshed || []);
  }

  // ===== Derived display =====
  const selectedSummary = useMemo(() => {
    if (!selectedField) return "";
    const p = selectedField.page ?? pageNum;
    const b = selectedField.bbox;
    return `page ${p}${b ? ` @ [${Math.round(b.x0!)},${Math.round(b.y0!)},${Math.round(b.x1!)},${Math.round(b.y1!)}]` : ""}`;
  }, [selectedField, pageNum]);

  return (
    <div className="workbench">
      <div className="wb-toolbar">
        {/* Upload PDF */}
        <label className="btn toggle" style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
          <input type="file" accept="application/pdf" style={{ display: "none" }} onChange={onUpload} />
          Upload PDF
        </label>

        {/* Optional: allow pasting a URL instead */}
        <button className="btn toggle" onClick={() => setShowUrlInput((v) => !v)}>
          {showUrlInput ? "Hide URL" : "Paste URL"}
        </button>
        {showUrlInput && (
          <input
            className="input"
            placeholder="Paste PDF URL…"
            value={pdfUrl}
            onChange={(e) => setPdfUrl(e.target.value)}
            onBlur={() => localStorage.setItem("doc_pdf_url", pdfUrl)}
          />
        )}

        <div className="spacer" />
        {docId && <span style={{ fontSize: 12, color: "#6b7280" }}>doc_id: {docId}</span>}
        <button className={isLasso ? "btn toggle active" : "btn toggle"} onClick={() => setIsLasso((v) => !v)} title="Draw a new box (drag on PDF)">
          {isLasso ? "Lasso: ON" : "Lasso"}
        </button>
        <button disabled={!pdf || pageNum <= 1} onClick={() => setPageNum((p) => p - 1)}>
          Prev
        </button>
        <span className="page-indicator">
          Page {pageNum}
          {pdf ? ` / ${pdf.numPages}` : ""}
        </span>
        <button disabled={!pdf || (pdf && pageNum >= pdf.numPages)} onClick={() => setPageNum((p) => p + 1)}>
          Next
        </button>
      </div>

      <div className="wb-split">
        {/* LEFT: PDF with overlay */}
        <div className="wb-left">
          {loading && <div className="loading">Loading…</div>}
          <div className="pdf-stage">
            <canvas ref={canvasRef} />
            <svg
              ref={overlayRef}
              className={isLasso ? "overlay crosshair" : "overlay"}
              onMouseDown={onOverlayMouseDown}
              onMouseMove={onOverlayMouseMove}
              onMouseUp={onOverlayMouseUp}
            >
              {dragBox && (
                <rect
                  x={dragBox.x0}
                  y={dragBox.y0}
                  width={dragBox.x1 - dragBox.x0}
                  height={dragBox.y1 - dragBox.y0}
                  className="bbox-rect drawing"
                />
              )}
            </svg>
          </div>
        </div>

        {/* RIGHT: Field editor */}
        <div className="wb-right">
          {!selectedField ? (
            <div className="placeholder">Upload a PDF, then click a box (or use Lasso) to edit its field here.</div>
          ) : (
            <div className="field-card">
              <div className="row">
                <label>Field</label>
                <input
                  value={selectedField.name || selectedField.key || ""}
                  onChange={(e) =>
                    setSelectedField((prev) => (prev ? { ...prev, name: e.target.value, key: e.target.value } : prev))
                  }
                />
              </div>
              <div className="row">
                <label>Value</label>
                <input
                  value={selectedField.value ?? ""}
                  onChange={(e) => setSelectedField((prev) => (prev ? { ...prev, value: e.target.value } : prev))}
                />
              </div>
              <div className="row">
                <label>Confidence</label>
                <input
                  type="number"
                  step={0.01}
                  min={0}
                  max={1}
                  value={selectedField.confidence ?? 0}
                  onChange={(e) =>
                    setSelectedField((prev) => (prev ? { ...prev, confidence: parseFloat(e.target.value) } : prev))
                  }
                />
              </div>
              <div className="row">
                <label>Where</label>
                <input disabled value={selectedSummary} />
              </div>
              {!!selectedField.bbox && (
                <div className="grid">
                  <div className="row">
                    <label>x0</label>
                    <input disabled value={Math.round(selectedField.bbox.x0!)} />
                  </div>
                  <div className="row">
                    <label>y0</label>
                    <input disabled value={Math.round(selectedField.bbox.y0!)} />
                  </div>
                  <div className="row">
                    <label>x1</label>
                    <input disabled value={Math.round(selectedField.bbox.x1!)} />
                  </div>
                  <div className="row">
                    <label>y1</label>
                    <input disabled value={Math.round(selectedField.bbox.y1!)} />
                  </div>
                </div>
              )}
              <div className="actions">
                <button className="primary" onClick={onSaveSelected}>
                  Save
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Hidden iframe preview is not used; canvas renders the PDF page */}
    </div>
  );
}
