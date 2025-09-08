import React, { useEffect, useMemo, useState } from "react";
import {
  listFields, saveFieldState, uploadPdf, bindFieldByRect,
  guessDocIdFromUrl, type FieldState, type Box
} from "../../lib/api";
import PdfCanvasWithBoxes, { type RectImg } from "./PdfCanvasWithBoxes";

function useQuery() { const [q] = useState(() => new URLSearchParams(window.location.search)); return q; }

export default function PdfBBoxWorkbench() {
  const query = useQuery();

  const [docUrl, setDocUrl] = useState<string>(query.get("doc_url") || localStorage.getItem("doc_pdf_url") || "");
  const [docId, setDocId] = useState<string>(guessDocIdFromUrl(docUrl) || "");
  const [page, setPage] = useState<number>(1);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [isLasso, setIsLasso] = useState<boolean>(false);

  const [boxes, setBoxes] = useState<Box[]>([]); // optional, not required here (parent can pass empty)
  const [fields, setFields] = useState<FieldState[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number>(-1);
  const selectedField = selectedIdx >= 0 ? fields[selectedIdx] : null;

  useEffect(() => { const id = guessDocIdFromUrl(docUrl) || ""; setDocId(id); }, [docUrl]);

  useEffect(() => { (async () => {
    if (!docUrl) return;
    const fs = await listFields({ doc_url: docUrl });
    setFields(fs || []);
  })(); }, [docUrl]);

  async function onUpload(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0]; if (!f) return;
    const res = await uploadPdf(f);
    localStorage.setItem("doc_pdf_url", res.annotated_tokens_url);
    setDocUrl(res.annotated_tokens_url);
    setDocId(res.doc_id);
    setPage(1);
    (ev.target as HTMLInputElement).value = "";
  }

  // Click a box → if a field is selected, bind; otherwise select/create one
  async function handleBoxClick(b: Box) {
    if (!selectedField) {
      // create a new field stub and select it
      const nf: FieldState = {
        id: b.id || `${b.page}:${b.x0}:${b.y0}`,
        name: (b.label || "field").toLowerCase().replace(/\s+/g, "_"),
        page: b.page,
        bbox: { x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1, page: b.page },
        value: "",
        source: "bbox",
      };
      setFields(prev => [nf, ...prev]);
      setSelectedIdx(0);
      return;
    }
    // bind selection
    const rect = { x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1 };
    await doBind(rect, b.page);
  }

  async function handleLasso(rect: RectImg) {
    if (!selectedField) {
      const nf: FieldState = {
        id: `bbox_${Date.now()}`, name: "new_field", value: "",
        page, bbox: { ...rect, page }, source: "lasso"
      };
      setFields(prev => [nf, ...prev]); setSelectedIdx(0);
      return;
    }
    await doBind(rect, page);
    setIsLasso(false);
  }

  async function doBind(rect: RectImg, pg: number) {
    if (!selectedField || !docId) return;
    const key = selectedField.key || selectedField.name || "";
    if (!key) return;

    // server bind (will OCR inside)
    const state = await bindFieldByRect({ doc_id: docId, key, page: pg, rect });
    setFields(state.fields || []);
  }

  async function onSaveSelected() {
    if (!selectedField || !docUrl) return;
    const f: FieldState = { ...selectedField };
    if (!f.key && f.name) f.key = f.name;
    await saveFieldState({ doc_url: docUrl, field: f });
    const fs = await listFields({ doc_url: docUrl }); setFields(fs || []);
  }

  return (
    <div className="workbench">
      <div className="wb-toolbar">
        <label className="btn toggle">
          <input type="file" accept="application/pdf" style={{ display:"none" }} onChange={onUpload} />
          Upload PDF
        </label>
        <label className={showBoxes ? "btn toggle active" : "btn toggle"}>
          <input type="checkbox" checked={showBoxes} onChange={()=>setShowBoxes(v=>!v)} /> Boxes
        </label>
        <button className={isLasso ? "btn toggle active" : "btn toggle"} onClick={()=>setIsLasso(v=>!v)}>Lasso</button>
        <input className="input" placeholder="Paste PDF URL…" value={docUrl}
          onChange={(e)=>setDocUrl(e.target.value)} onBlur={()=>localStorage.setItem("doc_pdf_url", docUrl)} />
        <div className="spacer" />
        {docId && <span className="muted">doc_id: {docId}</span>}
      </div>

      <div className="wb-split">
        <div className="wb-left">
          {docUrl ? (
            <>
              <div className="toolbar-inline">
                <button disabled={page<=1} onClick={()=>setPage(p=>p-1)}>Prev</button>
                <span className="page-indicator">Page {page}</span>
                <button onClick={()=>setPage(p=>p+1)}>Next</button>
              </div>
              <PdfCanvasWithBoxes
                docUrl={docUrl}
                page={page}
                showBoxes={showBoxes}
                isLasso={isLasso}
                boxes={boxes /* not required; canvas draws boxes from server URL on parent page changes */}
                selectedBoxId={null}
                onBoxClick={handleBoxClick}
                onLasso={handleLasso}
              />
            </>
          ) : (
            <div className="placeholder">Upload or paste a PDF URL.</div>
          )}
        </div>

        <div className="wb-right">
          <div className="section-title">Fields</div>
          {fields.length === 0 ? (
            <div className="placeholder">Select a field (from ECM on /lasso) or click a box to create one, then bind.</div>
          ) : (
            <div className="field-list">
              {fields.map((f,i)=>(
                <div key={(f.key||f.name||"f")+":"+i}
                  className="field-row"
                  style={{ background: i===selectedIdx ? "#eef2ff" : "transparent", borderRadius: 8, padding: 4 }}>
                  <input className="mono" value={f.name || f.key || ""}
                    onChange={(e)=>setFields(prev=>prev.map((x,idx)=>idx===i?{...x, name:e.target.value, key:e.target.value}:x))}
                    onFocus={()=>setSelectedIdx(i)} />
                  <input value={f.value || ""} onChange={(e)=>setFields(prev=>prev.map((x,idx)=>idx===i?{...x, value:e.target.value}:x))}/>
                  <button onClick={onSaveSelected}>Save</button>
                </div>
              ))}
            </div>
          )}
          <div className="hint">Select a field above, then click a box on the PDF (or use Lasso) to bind and OCR.</div>
        </div>
      </div>
    </div>
  );
}
