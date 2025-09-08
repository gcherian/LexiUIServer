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
GlobalWorkerOptions.workerSrc = pdfjsWorker as string;

type Rect = { x: number; y: number; w: number; h: number };
type Box = ApiBox;
type FieldState = ApiFieldState;

export default function PdfBBoxWorkbench() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<SVGSVGElement | null>(null);

  const [docId, setDocId] = useState<string>("");
  const [pdfUrl, setPdfUrl] = useState<string>("");
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [pageNum, setPageNum] = useState<number>(1);
  const [viewportH, setViewportH] = useState<number>(0);

  const [boxes, setBoxes] = useState<Box[]>([]);
  const [fields, setFields] = useState<FieldState[]>([]);

  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [selectedField, setSelectedField] = useState<FieldState | null>(null);

  const [loading, setLoading] = useState<boolean>(false);
  const [showUrlInput, setShowUrlInput] = useState<boolean>(false);

  const [isLasso, setIsLasso] = useState<boolean>(false);
  const [dragBox, setDragBox] = useState<Rect | null>(null);

  // Load last URL
  useEffect(() => {
    const last = localStorage.getItem("doc_pdf_url") || "";
    if (last) setPdfUrl(last);
  }, []);

  // Load PDF when url changes
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!pdfUrl) return;
      try {
        setLoading(true);
        const doc = await getDocument(pdfUrl).promise;
        if (!cancelled) { setPdf(doc); setPageNum(1); }
      } finally {
        setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [pdfUrl]);

  // Fetch boxes/fields
  useEffect(() => {
    (async () => {
      if (!pdf || !pdfUrl) return;
      const pageBoxes = await getBoxes({ doc_url: pdfUrl, page: pageNum });
      setBoxes(Array.isArray(pageBoxes) ? pageBoxes : []);
      const allFields = await listFields({ doc_url: pdfUrl });
      setFields(Array.isArray(allFields) ? allFields : []);
      setSelectedBoxId(null);
      setSelectedField(null);
    })();
  }, [pdf, pdfUrl, pageNum]);

  // Render page + overlay
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

      await page.render({ canvasContext: ctx, viewport }).promise;

      const svg = overlayRef.current!;
      svg.setAttribute("viewBox", `0 0 ${viewport.width} ${viewport.height}`);
      svg.setAttribute("width", `${viewport.width}`);
      svg.setAttribute("height", `${viewport.height}`);
      svg.innerHTML = "";

      const toId = (b: Box) => b.id || `${b.page}:${b.x0}:${b.y0}`;

      boxes.forEach((b) => {
        const id = toId(b);
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("class", "bbox-group");
        g.addEventListener("mouseenter", () => setSelectedBoxId(id));
        g.addEventListener("mouseleave", () => setSelectedBoxId((prev) => (prev === id ? null : prev)));
        g.addEventListener("click", () => {
          setSelectedBoxId(id);
          // try to map to a field via overlap
          const f = fields.find(
            (f) =>
              f?.bbox &&
              f.bbox.page === b.page &&
              !(f.bbox.x1! < b.x0 || f.bbox.x0! > b.x1 || f.bbox.y1! < b.y0 || f.bbox.y0! > b.y1)
          );
          if (f) setSelectedField({ ...f });
          else {
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
        rect.setAttribute("y", `${viewport.height - b.y1}`);
        rect.setAttribute("width", `${b.x1 - b.x0}`);
        rect.setAttribute("height", `${b.y1 - b.y0}`);
        rect.setAttribute("class", id === selectedBoxId ? "bbox-rect selected" : "bbox-rect");

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", `${b.x0 + 2}`);
        label.setAttribute("y", `${viewport.height - b.y1 - 4}`);
        label.setAttribute("class", "bbox-label");
        label.textContent = b.label || "";

        g.appendChild(rect);
        if (b.label) g.appendChild(label);
        svg.appendChild(g);
      });
    })();
  }, [pdf, pageNum, boxes, selectedBoxId, fields]);

  // Upload
  async function onUpload(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f) return;
    setLoading(true);
    try {
      const res = await uploadPdf(f);
      setDocId(res.doc_id);
      setPdfUrl(res.annotated_tokens_url);
      localStorage.setItem("doc_pdf_url", res.annotated_tokens_url);
      const doc = await getDocument(res.annotated_tokens_url).promise;
      setPdf(doc);
      setPageNum(1);
    } finally {
      setLoading(false);
      (ev.target as HTMLInputElement).value = "";
    }
  }

  // Lasso
  function clientToSvg(e: React.MouseEvent<SVGSVGElement>) {
    const svg = overlayRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const m = svg.getScreenCTM();
    const p = m ? pt.matrixTransform(m.inverse()) : ({ x: 0, y: 0 } as any);
    return { x: p.x as number, y: p.y as number };
  }
  function onOverlayMouseDown(e: React.MouseEvent<SVGSVGElement>) {
    if (!isLasso) return;
    const { x, y } = clientToSvg(e);
    setDragBox({ x, y, w: 0, h: 0 });
  }
  function onOverlayMouseMove(e: React.MouseEvent<SVGSVGElement>) {
    if (!isLasso || !dragBox) return;
    const { x, y } = clientToSvg(e);
    setDragBox({ x: Math.min(dragBox.x, x), y: Math.min(dragBox.y, y), w: Math.abs(x - dragBox.x), h: Math.abs(y - dragBox.y) });
  }
  async function onOverlayMouseUp() {
    if (!isLasso || !dragBox) return;
    setIsLasso(false);
    const b = {
      x0: dragBox.x,
      y0: viewportH - (dragBox.y + dragBox.h),
      x1: dragBox.x + dragBox.w,
      y1: viewportH - dragBox.y,
    };
    const nf: FieldState = {
      id: `bbox_${Date.now()}`,
      name: "new_field",
      value: "",
      page: pageNum,
      bbox: { ...b, page: pageNum },
      confidence: 1.0,
      source: "lasso",
    };
    setFields((fs) => [nf, ...fs]);
    setSelectedField(nf);
    setSelectedBoxId(null);
    setDragBox(null);
  }

  // Save
  async function onSaveSelected() {
    if (!selectedField || !pdfUrl) return;
    const payload: FieldState = { ...selectedField };
    if (!payload.key && payload.name) payload.key = payload.name;
    await saveFieldState({ doc_url: pdfUrl, field: payload });
    const refreshed = await listFields({ doc_url: pdfUrl });
    setFields(Array.isArray(refreshed) ? refreshed : []);
  }

  const selSummary = useMemo(() => {
    if (!selectedField) return "";
    const p = selectedField.page ?? pageNum;
    const b = selectedField.bbox;
    return `page ${p}${b ? ` @ [${Math.round(b.x0!)},${Math.round(b.y0!)},${Math.round(b.x1!)},${Math.round(b.y1!)}]` : ""}`;
  }, [selectedField, pageNum]);

  return (
    <div className="workbench">
      <div className="wb-toolbar">
        <label className="btn toggle" style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
          <input type="file" accept="application/pdf" style={{ display: "none" }} onChange={onUpload} />
          Upload PDF
        </label>
        <button className="btn toggle" onClick={() => setShowUrlInput(v => !v)}>{showUrlInput ? "Hide URL" : "Paste URL"}</button>
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
        <button className={isLasso ? "btn toggle active" : "btn toggle"} onClick={() => setIsLasso(v => !v)}>{isLasso ? "Lasso: ON" : "Lasso"}</button>
        <button disabled={!pdf || pageNum <= 1} onClick={() => setPageNum(p => p - 1)}>Prev</button>
        <span className="page-indicator">Page {pageNum}{pdf ? ` / ${pdf.numPages}` : ""}</span>
        <button disabled={!pdf || (pdf && pageNum >= pdf.numPages)} onClick={() => setPageNum(p => p + 1)}>Next</button>
      </div>

      <div className="wb-split">
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
                  x={dragBox.x}
                  y={dragBox.y}
                  width={dragBox.w}
                  height={dragBox.h}
                  className="bbox-rect drawing"
                />
              )}
            </svg>
          </div>
        </div>

        <div className="wb-right">
          {!selectedField ? (
            <div className="placeholder">Hover/click a box or use Lasso to create one, then edit on the right.</div>
          ) : (
            <div className="field-card">
              <div className="row">
                <label>Field</label>
                <input
                  value={selectedField.name || selectedField.key || ""}
                  onChange={(e) => setSelectedField(s => (s ? { ...s, name: e.target.value, key: e.target.value } : s))}
                />
              </div>
              <div className="row">
                <label>Value</label>
                <input
                  value={selectedField.value ?? ""}
                  onChange={(e) => setSelectedField(s => (s ? { ...s, value: e.target.value } : s))}
                />
              </div>
              <div className="row">
                <label>Confidence</label>
                <input
                  type="number" step={0.01} min={0} max={1}
                  value={selectedField.confidence ?? 0}
                  onChange={(e) => setSelectedField(s => (s ? { ...s, confidence: parseFloat(e.target.value) } : s))}
                />
              </div>
              <div className="row">
                <label>Where</label>
                <input disabled value={selSummary} />
              </div>
              {!!selectedField.bbox && (
                <div className="grid">
                  <div className="row"><label>x0</label><input disabled value={Math.round(selectedField.bbox.x0!)} /></div>
                  <div className="row"><label>y0</label><input disabled value={Math.round(selectedField.bbox.y0!)} /></div>
                  <div className="row"><label>x1</label><input disabled value={Math.round(selectedField.bbox.x1!)} /></div>
                  <div className="row"><label>y1</label><input disabled value={Math.round(selectedField.bbox.y1!)} /></div>
                </div>
              )}
              <div className="actions">
                <button className="primary" onClick={onSaveSelected}>Save</button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}