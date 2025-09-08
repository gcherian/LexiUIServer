import { useEffect, useMemo, useRef, useState } from "react";
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy, type PDFPageProxy } from "pdfjs-dist";
import pdfjsWorker from "pdfjs-dist/build/pdf.worker.min?url";
import { getBoxes, listFields, saveFieldState } from "../../lib/api";

GlobalWorkerOptions.workerSrc = pdfjsWorker;

type Rect = { x0:number; y0:number; x1:number; y1:number };
type Box  = Rect & { page:number; id:string; label?:string; confidence?:number };
type FieldState = {
  id: string;
  name: string;
  value?: string | null;
  confidence?: number;
  source?: string;
  page?: number;
  bbox?: (Rect & { page:number }) | null;
};

export default function PdfBBoxWorkbench() {
  const canvasRef   = useRef<HTMLCanvasElement|null>(null);
  const overlayRef  = useRef<SVGSVGElement|null>(null);

  const [pdfUrl, setPdfUrl]       = useState<string>("");
  const [pdf, setPdf]             = useState<PDFDocumentProxy|null>(null);
  const [pageNum, setPageNum]     = useState<number>(1);
  const [viewportH, setViewportH] = useState<number>(0);

  const [boxes, setBoxes]         = useState<Box[]>([]);
  const [fields, setFields]       = useState<FieldState[]>([]);
  const [selectedBoxId, setSelectedBoxId] = useState<string|undefined>(undefined);
  const [selectedField, setSelectedField] = useState<FieldState|undefined>(undefined);

  const [loading, setLoading]     = useState(false);
  const [isLasso, setIsLasso]     = useState(false);
  const [dragStart, setDragStart] = useState<{x:number;y:number}|null>(null);
  const [dragBox, setDragBox]     = useState<{x0:number;y0:number;x1:number;y1:number}|null>(null);

  // Load last PDF URL (or let user paste)
  useEffect(() => {
    const last = localStorage.getItem("doc_pdf_url") || "";
    setPdfUrl(last);
  }, []);

  useEffect(() => {
    if (!pdfUrl) return;
    (async () => {
      setLoading(true);
      try {
        const doc = await getDocument(pdfUrl).promise;
        setPdf(doc);
        setPageNum(1);
      } finally {
        setLoading(false);
      }
    })();
  }, [pdfUrl]);

  // Fetch page boxes + all fields
  useEffect(() => {
    if (!pdf || !pdfUrl) return;
    (async () => {
      const pageBoxes = await getBoxes({ doc_url: pdfUrl, page: pageNum });
      setBoxes(pageBoxes || []);
      const allFields = await listFields({ doc_url: pdfUrl });
      setFields(allFields || []);
    })();
  }, [pdf, pdfUrl, pageNum]);

  // Render page & overlay with boxes
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
      svg.innerHTML = "";

      boxes.forEach((b) => {
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("class", "bbox-group");
        g.addEventListener("mouseenter", () => setSelectedBoxId(b.id));
        g.addEventListener("mouseleave", () => setSelectedBoxId(prev => prev === b.id ? undefined : prev));
        g.addEventListener("click", () => {
          setSelectedBoxId(b.id);
          // try to associate to a field
          const f = fields.find(f =>
            f.bbox && f.bbox.page === b.page &&
            !(f.bbox.x1 < b.x0 || f.bbox.x0 > b.x1 || f.bbox.y1 < b.y0 || f.bbox.y0 > b.y1)
          );
          if (f) setSelectedField(f);
          else {
            setSelectedField({
              id: b.id,
              name: b.label || "unknown_field",
              value: "",
              page: b.page,
              bbox: { ...b, page: b.page },
              confidence: b.confidence ?? 0.0,
              source: "bbox"
            });
          }
        });

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", `${b.x0}`);
        rect.setAttribute("y", `${viewport.height - b.y1}`);
        rect.setAttribute("width", `${b.x1 - b.x0}`);
        rect.setAttribute("height", `${b.y1 - b.y0}`);
        rect.setAttribute("class", b.id === selectedBoxId ? "bbox-rect selected" : "bbox-rect");

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

  // Lasso utilities
  function clientToSvg(e: React.MouseEvent<SVGSVGElement, MouseEvent>) {
    const svg = overlayRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
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
    setFields(fs => [nf, ...fs]);
    setSelectedField(nf);
    setSelectedBoxId(undefined);
    setDragBox(null);
  }

  async function saveEdits() {
    if (!selectedField) return;
    await saveFieldState({ doc_url: pdfUrl, field: selectedField });
    const refreshed = await listFields({ doc_url: pdfUrl });
    setFields(refreshed || []);
  }

  const selectedSummary = useMemo(() => {
    if (!selectedField) return "";
    const p = selectedField.page ?? pageNum;
    const b = selectedField.bbox;
    return `page ${p}${b ? ` @ [${b.x0},${b.y0},${b.x1},${b.y1}]` : ""}`;
  }, [selectedField, pageNum]);

  return (
    <div className="workbench">
      <div className="wb-toolbar">
        <input
          className="input"
          placeholder="Paste PDF URL or choose a doc..."
          value={pdfUrl}
          onChange={(e) => setPdfUrl(e.target.value)}
          onBlur={() => localStorage.setItem("doc_pdf_url", pdfUrl)}
        />
        <div className="spacer" />
        <button className={isLasso ? "btn toggle active" : "btn toggle"} onClick={() => setIsLasso(v => !v)} title="Draw a new box (drag on PDF)">
          {isLasso ? "Lasso: ON" : "Lasso"}
        </button>
        <button disabled={!pdf || pageNum<=1} onClick={() => setPageNum(p => p-1)}>Prev</button>
        <span className="page-indicator">Page {pageNum}{pdf ? ` / ${pdf.numPages}` : ""}</span>
        <button disabled={!pdf || (pdf && pageNum>=pdf.numPages)} onClick={() => setPageNum(p => p+1)}>Next</button>
      </div>

      <div className="wb-split">
        <div className="wb-left">
          {loading && <div className="loading">Loading…</div>}
          <div className="pdf-stage">
            <canvas ref={canvasRef}/>
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

        <div className="wb-right">
          {!selectedField ? (
            <div className="placeholder">Hover/click a box or use Lasso to create one, then edit on the right.</div>
          ) : (
            <div className="field-card">
              <div className="row">
                <label>Field</label>
                <input
                  value={selectedField.name}
                  onChange={(e) => setSelectedField({ ...selectedField, name: e.target.value })}
                />
              </div>
              <div className="row">
                <label>Value</label>
                <input
                  value={selectedField.value ?? ""}
                  onChange={(e) => setSelectedField({ ...selectedField, value: e.target.value })}
                />
              </div>
              <div className="row">
                <label>Confidence</label>
                <input
                  type="number" step="0.01" min={0} max={1}
                  value={selectedField.confidence ?? 0}
                  onChange={(e) => setSelectedField({ ...selectedField, confidence: parseFloat(e.target.value) })}
                />
              </div>
              <div className="row">
                <label>Where</label>
                <input disabled value={selectedSummary}/>
              </div>
              {!!selectedField.bbox && (
                <div className="grid">
                  <div className="row"><label>x0</label><input disabled value={selectedField.bbox.x0}/></div>
                  <div className="row"><label>y0</label><input disabled value={selectedField.bbox.y0}/></div>
                  <div className="row"><label>x1</label><input disabled value={selectedField.bbox.x1}/></div>
                  <div className="row"><label>y1</label><input disabled value={selectedField.bbox.y1}/></div>
                </div>
              )}
              <div className="actions">
                <button className="primary" onClick={saveEdits}>Save</button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
