// File: src/tsp4/components/lasso/FieldLevelEditor.tsx
import React, { useEffect, useRef, useState } from "react";
import "../ocr.css";
import PdfEditCanvas, { type EditRect, type TokenBox } from "./PdfEditCanvas";
import { API, uploadPdf, getMeta, getBoxes, docIdFromUrl } from "../../../lib/api";

/* ------------ Types ------------ */
type ExtendedField = {
  key: string;
  value: string;
  source?: string;
  confidence?: number;
  bbox?: { page: number; x0: number; y0: number; x1: number; y1: number };
};
type ExtendedFieldDocState = {
  doc_id: string;
  doctype: string;
  fields: ExtendedField[];
};

export default function FieldLevelEditor() {
  // Document/page
  const [docUrl, setDocUrl] = useState("");
  const [docId, setDocId] = useState("");
  const [meta, setMeta] = useState<{ w: number; h: number }[]>([]);
  const [page, setPage] = useState(1);

  const [tokens, setTokens] = useState<TokenBox[]>([]);
  const [rect, setRect] = useState<EditRect | null>(null);

  const [fields, setFields] = useState<ExtendedFieldDocState | null>(null);
  const [focusedKey, setFocusedKey] = useState("");

  // Upload PDF
  async function onUploadPdf(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f) return;
    try {
      const res = await uploadPdf(f);
      setDocId(res.doc_id);
      setDocUrl(res.annotated_tokens_url);
      const m = await getMeta(res.doc_id);
      setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
      const b = await getBoxes(res.doc_id);
      setTokens(b as any);
      setPage(1);
    } finally {
      (ev.target as HTMLInputElement).value = "";
    }
  }

  // Upload JSON extraction output
  async function onUploadJson(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f) return;
    try {
      const text = await f.text();
      const parsed = JSON.parse(text);
      // Wrap JSON into our ExtendedFieldDocState
      const newState: ExtendedFieldDocState = {
        doc_id: docId || "uploaded",
        doctype: parsed["Document Type"] || "unknown",
        fields: Object.entries(parsed).map(([k, v]) => ({
          key: k,
          value: typeof v === "string" ? v : JSON.stringify(v),
          source: "json",
          confidence: 1.0,
          bbox: undefined, // attach if JSON includes bbox
        })),
      };
      setFields(newState);
    } catch (err) {
      alert("Invalid JSON file");
    }
  }

  // Paste URL
  useEffect(() => {
    const id = docIdFromUrl(docUrl);
    if (!id) return;
    (async () => {
      setDocId(id);
      const m = await getMeta(id);
      setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
      const b = await getBoxes(id);
      setTokens(b as any);
      setPage(1);
    })();
  }, [docUrl]);

  // Focus a key → show pink box
  function focusKey(f: ExtendedField) {
    setFocusedKey(f.key);
    if (f.bbox) {
      setPage(f.bbox.page || 1);
      setRect({
        page: f.bbox.page,
        x0: f.bbox.x0,
        y0: f.bbox.y0,
        x1: f.bbox.x1,
        y1: f.bbox.y1,
      });
    } else {
      setRect(null);
    }
  }

  const serverW = meta[page - 1]?.w || 1;
  const serverH = meta[page - 1]?.h || 1;

  return (
    <div className="workbench">
      <div className="wb-toolbar">
        <input type="file" accept="application/pdf" onChange={onUploadPdf} />
        <input type="file" accept="application/json" onChange={onUploadJson} style={{ marginLeft: 8 }} />
        <input
          className="input"
          placeholder="Paste /data/{doc_id}/original.pdf"
          value={docUrl}
          onChange={(e) => setDocUrl(e.target.value || "")}
        />
        <span className="spacer" />
        <span className="muted">API: {API}</span>
      </div>

      <div className="wb-split">
        {/* LEFT: Fields */}
        <div className="wb-left" style={{ flexBasis: "30%" }}>
          <div className="section-title">Fields</div>
          {!fields ? (
            <div className="placeholder">Upload JSON extraction to see fields.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Key</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {fields.fields.map((f, idx) => {
                  const focused = f.key === focusedKey;
                  return (
                    <tr
                      key={f.key + idx}
                      onClick={() => focusKey(f)}
                      style={focused ? { outline: "2px solid #ec4899" } : undefined}
                    >
                      <td>
                        <code>{f.key}</code>
                      </td>
                      <td>{f.value}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* RIGHT: PDF */}
        <div className="wb-right" style={{ flexBasis: "70%" }}>
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

              <PdfEditCanvas
                docUrl={docUrl}
                page={page}
                serverW={serverW}
                serverH={serverH}
                tokens={tokens.filter((t) => t.page === page)}
                rect={rect}
                showTokenBoxes={true}
                editable={false}
                onRectChange={() => {}}
                onRectCommit={() => {}}
              />
            </>
          ) : (
            <div className="placeholder">Upload a PDF to begin.</div>
          )}
        </div>
      </div>
    </div>
  );
}