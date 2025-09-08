import React, { useEffect, useMemo, useState } from "react";
import { uploadPdf, listFields, saveFieldState, type FieldState, type Box } from "../../lib/api";
import PdfCanvasWithBoxes, { type LassoRect } from "./PdfCanvasWithBoxes";

function useQuery() {
  const [q] = useState(() => new URLSearchParams(window.location.search));
  return q;
}

export default function PdfBBoxWorkbench() {
  const query = useQuery();

  const [docUrl, setDocUrl] = useState<string>(query.get("doc_url") || "");
  const [docId, setDocId] = useState<string>("");
  const [page, setPage] = useState<number>(1);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);

  const [fields, setFields] = useState<FieldState[]>([]);
  const [selected, setSelected] = useState<FieldState | null>(null);

  useEffect(() => { (async () => {
    if (!docUrl) return;
    const fs = await listFields({ doc_url: docUrl });
    setFields(fs || []);
  })(); }, [docUrl]);

  async function onUpload(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f) return;
    const res = await uploadPdf(f);
    setDocId(res.doc_id);
    setDocUrl(res.annotated_tokens_url);
    localStorage.setItem("doc_pdf_url", res.annotated_tokens_url);
    (ev.target as HTMLInputElement).value = "";
  }

  function onBoxClick(b: Box) {
    const id = b.id || `${b.page}:${b.x0}:${b.y0}`;
    const f: FieldState = {
      id, name: (b.label || "field").toLowerCase().replace(/\s+/g,"_"),
      value: "", page: b.page,
      bbox: { x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1, page: b.page },
      confidence: b.confidence ?? 0.8, source: "bbox",
    };
    setFields(prev => [f, ...prev]);
    setSelected(f);
  }

  function onLasso(r: LassoRect) {
    const id = `bbox_${Date.now()}`;
    const f: FieldState = {
      id, name: "new_field", value: "",
      page, bbox: { ...r, page }, confidence: 1, source: "lasso",
    };
    setFields(prev => [f, ...prev]);
    setSelected(f);
  }

  async function onSave() {
    if (!selected || !docUrl) return;
    const payload: FieldState = { ...selected };
    if (!payload.key && payload.name) payload.key = payload.name;
    await saveFieldState({ doc_url: docUrl, field: payload });
    const fs = await listFields({ doc_url: docUrl });
    setFields(fs || []);
  }

  const selSummary = useMemo(() => {
    if (!selected) return "";
    const b = selected.bbox;
    return `page ${selected.page ?? page}${b ? ` @ [${Math.round(b.x0!)},${Math.round(b.y0!)},${Math.round(b.x1!)},${Math.round(b.y1!)}]` : ""}`;
  }, [selected, page]);

  return (
    <div className="workbench">
      <div className="wb-toolbar">
        <label className="btn toggle">
          <input type="file" accept="application/pdf" style={{ display:"none" }} onChange={onUpload} />
          Upload PDF
        </label>
        <label className={showBoxes ? "btn toggle active" : "btn toggle"} style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={showBoxes} onChange={()=>setShowBoxes(v=>!v)} /> Boxes
        </label>
        <input
          className="input"
          placeholder="Paste PDF URL…"
          value={docUrl}
          onChange={(e)=>setDocUrl(e.target.value)}
          onBlur={()=>localStorage.setItem("doc_pdf_url", docUrl)}
          style={{ marginLeft: 8 }}
        />
        <div className="spacer" />
        {docId && <span className="muted">doc_id: {docId}</span>}
      </div>

      <div className="wb-split">
        <div className="wb-left">
          {docUrl ? (
            <PdfCanvasWithBoxes
              docUrl={docUrl}
              page={page}
              showBoxes={showBoxes}
              onPageChange={(p)=>setPage(p)}
              onBoxClick={onBoxClick}
              onLasso={onLasso}
            />
          ) : (
            <div className="placeholder">Upload or paste a PDF URL.</div>
          )}
        </div>

        <div className="wb-right">
          {!selected ? (
            <div className="placeholder">Click a box (or lasso) to edit on the right.</div>
          ) : (
            <div className="field-card">
              <div className="row">
                <label>Field</label>
                <input
                  value={selected.name || selected.key || ""}
                  onChange={(e)=>setSelected(s=>s?{...s, name:e.target.value, key:e.target.value}:s)}
                />
              </div>
              <div className="row">
                <label>Value</label>
                <input
                  value={selected.value || ""}
                  onChange={(e)=>setSelected(s=>s?{...s, value:e.target.value}:s)}
                />
              </div>
              <div className="row"><label>Where</label><input disabled value={selSummary} /></div>
              <div className="actions"><button className="primary" onClick={onSave}>Save</button></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}