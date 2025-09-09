// File: src/components/lasso/OcrWorkbench.tsx
import React, { useEffect, useMemo, useState } from "react";
import "../ocr.css";
import {
  API, uploadPdf, getMeta, getBoxes,
  docIdFromUrl, docUrlFromId,
  listProms, getProm, setDoctype, ecmExtract,
  getFields, putFields,
  type Box, type FieldDocState, type PromCatalog
} from "../../lib/api";
import PdfCanvas from "./PdfCanvas";
import BindModal from "./BindModal";

export default function OcrWorkbench() {
  // PDF + page state
  const [docUrl, setDocUrl] = useState<string>("");
  const [docId, setDocId] = useState<string>("");
  const [meta, setMeta] = useState<{ w: number; h: number }[]>([]);
  const [page, setPage] = useState<number>(1);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);

  // PROM / fields state
  const [proms, setProms] = useState<Array<{ doctype: string; file: string }>>([]);
  const [doctype, setDt] = useState<string>("");
  const [catalog, setCatalog] = useState<PromCatalog | null>(null);
  const [state, setState] = useState<FieldDocState | null>(null);

  // Bind modal
  const [modalOpen, setModalOpen] = useState(false);
  const [modalFromBox, setModalFromBox] = useState<Box | null>(null);

  const allKeys = useMemo(() => catalog?.fields?.map((f) => f.key) || [], [catalog]);

  // Upload handler
  async function onUpload(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f) return;
    try {
      const res = await uploadPdf(f);
      const url = res.annotated_tokens_url;
      setDocUrl(url);
      setDocId(res.doc_id);
      const m = await getMeta(res.doc_id);
      setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
      const b = await getBoxes(res.doc_id);
      setBoxes(b);
      setPage(1);
      // reset right panel
      setState(null);
      setDt("");
      setCatalog(null);
    } finally {
      (ev.target as HTMLInputElement).value = "";
    }
  }

  // Load when pasting a /data/{doc_id}/original.pdf URL
  useEffect(() => {
    const id = docIdFromUrl(docUrl);
    if (!id) return;
    (async () => {
      setDocId(id);
      const m = await getMeta(id);
      setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
      const b = await getBoxes(id);
      setBoxes(b);
      setPage(1);
      // reset right panel
      setState(null);
      setDt("");
      setCatalog(null);
    })();
  }, [docUrl]);

  // List doctypes once
  useEffect(() => {
    (async () => {
      try {
        setProms(await listProms());
      } catch {
        setProms([]);
      }
    })();
  }, []);

  // Select doctype -> seed fields from existing state or PROM
  async function onSelectDoctype(dt: string) {
    setDt(dt);
    if (!docId) return;
    await setDoctype(docId, dt);
    try {
      const st = await getFields(docId); // if exists already
      setState(st);
      const cat = await getProm(dt);
      setCatalog(cat);
    } catch {
      const cat = await getProm(dt);
      setCatalog(cat);
      setState({
        doc_id: docId,
        doctype: dt,
        fields: cat.fields.map((f) => ({ key: f.key, value: "", source: "user", confidence: 0 })),
        audit: [],
      });
    }
  }

  // Extract via ECM mock
  async function onExtract() {
    if (!docId || !doctype) return;
    const st = await ecmExtract(docId, doctype);
    setState(st);
  }

  // Save one field on blur
  async function saveField(k: string, val: string) {
    if (!state) return;
    const next: FieldDocState = {
      ...state,
      fields: state.fields.map((f) => (f.key === k ? { ...f, value: val, source: "user" } : f)),
    };
    const saved = await putFields(docId, next);
    setState(saved);
  }

  // Click a drawn bbox -> open modal with box preselected
  function onBoxClick(b: Box) {
    setModalFromBox(b);
    setModalOpen(true);
  }

  // current page meta (server coords)
  const serverW = meta[page - 1]?.w || 1;
  const serverH = meta[page - 1]?.h || 1;

  return (
    <div className="workbench">
      <div className="wb-toolbar">
        <input type="file" accept="application/pdf" onChange={onUpload} />
        <input
          className="input"
          placeholder="Paste /data/{doc_id}/original.pdf"
          value={docUrl}
          onChange={(e) => setDocUrl(e.target.value)}
        />
        <label className={showBoxes ? "btn toggle active" : "btn toggle"} style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={showBoxes} onChange={() => setShowBoxes((v) => !v)} /> Boxes
        </label>
        <span className="spacer" />
        <span className="muted">API: {API}</span>
      </div>

      <div className="wb-split">
        {/* LEFT: PDF with boxes */}
        <div className="wb-left">
          {docUrl ? (
            <>
              <div className="toolbar-inline">
                <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                  Prev
                </button>
                <span className="page-indicator">
                  Page {page} {meta.length ? `/ ${meta.length}` : ""}
                </span>
                <button disabled={meta.length > 0 && page >= meta.length} onClick={() => setPage((p) => p + 1)}>
                  Next
                </button>
              </div>
              <PdfCanvas
                docUrl={docUrl}
                page={page}
                serverW={serverW}
                serverH={serverH}
                boxes={boxes.filter((b) => b.page === page)}
                showBoxes={showBoxes}
                lasso={false}
                onBoxClick={onBoxClick}
              />
              <div className="hint">Click a highlighted box to bind it to a field.</div>
            </>
          ) : (
            <div className="placeholder">Upload or paste a PDF URL to begin.</div>
          )}
        </div>

        {/* RIGHT: Doctype + Extract + Fields Table */}
        <div className="wb-right">
          <div className="row">
            <label>Doctype</label>
            <select value={doctype} onChange={(e) => onSelectDoctype(e.target.value)} disabled={!docId}>
              <option value="">(select)</option>
              {proms.map((p) => (
                <option key={p.doctype} value={p.doctype}>
                  {p.doctype}
                </option>
              ))}
            </select>
          </div>

          <div className="row">
            <label>Actions</label>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="primary" onClick={onExtract} disabled={!doctype || !docId}>
                Extract
              </button>
            </div>
          </div>

          <div style={{ marginTop: 12 }}>
            <div className="section-title">Fields</div>
            {!state || state.fields.length === 0 ? (
              <div className="placeholder">
                Choose a doctype, then click <b>Extract</b> to populate expected keys.
              </div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Key</th>
                    <th>Value</th>
                    <th>Source</th>
                    <th>Conf</th>
                    <th>Bind</th>
                  </tr>
                </thead>
                <tbody>
                  {state.fields.map((f, idx) => {
                    const missing = !f.value;
                    return (
                      <tr key={(f.key || "k") + ":" + idx} className={missing ? "missing" : ""}>
                        <td>
                          <code>{f.key}</code>
                        </td>
                        <td>
                          <input
                            value={f.value || ""}
                            onChange={(e) => {
                              const v = e.target.value;
                              setState((s) =>
                                s ? { ...s, fields: s.fields.map((x) => (x.key === f.key ? { ...x, value: v } : x)) } : s
                              );
                            }}
                            onBlur={(e) => saveField(String(f.key), e.target.value)}
                          />
                        </td>
                        <td>{f.source || ""}</td>
                        <td>{f.confidence ? f.confidence.toFixed(2) : ""}</td>
                        <td>
                          <button
                            onClick={() => {
                              setModalFromBox(null); // start in lasso mode (no preset box)
                              setModalOpen(true);
                            }}
                          >
                            Bind via Lasso
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
          <div className="hint">Orange rows need values. Use <b>Bind via Lasso</b> or click a box on the PDF.</div>
        </div>
      </div>

      {/* Bind modal: for both box-click and lasso workflows */}
      <BindModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        docId={docId}
        docUrl={docUrl}
        page={page}
        serverW={meta[page - 1]?.w || 1}
        serverH={meta[page - 1]?.h || 1}
        allKeys={catalog?.fields?.map((f) => f.key) || []}
        box={modalFromBox}
        initialKey={state?.fields?.find((f) => !f.value)?.key || ""}
        onBound={(st) => setState(st)}
      />
    </div>
  );
}
