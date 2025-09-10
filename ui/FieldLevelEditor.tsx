// File: src/tsp4/components/lasso/FieldLevelEditor.tsx
import React, { useEffect, useState } from "react";
import "../ocr.css";
import PdfEditCanvas, { type EditRect, type TokenBox } from "./PdfEditCanvas";
import { API, uploadPdf, getMeta, getBoxes, docIdFromUrl } from "../../../lib/api";

/* ------------ Types ------------ */
type KVRect = { page: number; x0: number; y0: number; x1: number; y1: number };
type AnyJson = Record<string, any>;

type FieldRow = {
  key: string;
  value: string;
  rects?: KVRect[]; // optional: if JSON includes bboxes for that field
};

export default function FieldLevelEditor() {
  // Document/page
  const [docUrl, setDocUrl] = useState("");
  const [docId, setDocId] = useState("");
  const [meta, setMeta] = useState<{ w: number; h: number }[]>([]);
  const [page, setPage] = useState(1);

  // OCR tokens (for orange word boxes; purely visual)
  const [tokens, setTokens] = useState<TokenBox[]>([]);
  const tokensThisPage = tokens.filter((t) => t.page === page);

  // Left table (extraction JSON)
  const [rows, setRows] = useState<FieldRow[]>([]);
  const [focusedKey, setFocusedKey] = useState("");

  // Pink highlight on PDF (single union rect)
  const [rect, setRect] = useState<EditRect | null>(null);

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
      setTokens((b as any) || []);
      setPage(1);
      setRect(null);
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
      const parsed = JSON.parse(text) as AnyJson;
      setRows(parseExtractionToRows(parsed));
      setFocusedKey("");
      setRect(null);
    } catch {
      alert("Invalid JSON file");
    } finally {
      (ev.target as HTMLInputElement).value = "";
    }
  }

  // Paste URL → load meta + tokens
  useEffect(() => {
    const id = docIdFromUrl(docUrl);
    if (!id) return;
    (async () => {
      setDocId(id);
      const m = await getMeta(id);
      setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
      const b = await getBoxes(id);
      setTokens((b as any) || []);
      setPage(1);
      setRect(null);
    })();
  }, [docUrl]);

  // Click a row → show union rect (min/max) of all its boxes
  function onRowClick(r: FieldRow) {
    setFocusedKey(r.key);
    const rects = r.rects || [];
    if (!rects.length) {
      setRect(null);
      return;
    }
    // pick first page (or the page of the first rect) and union only on that page
    // if your fields can span pages, you can adapt this to jump/highlight per page
    const firstPg = rects[0].page;
    const samePage = rects.filter((rr) => rr.page === firstPg);
    const union = samePage.reduce(
      (acc, rr) => ({
        page: acc.page,
        x0: Math.min(acc.x0, rr.x0),
        y0: Math.min(acc.y0, rr.y0),
        x1: Math.max(acc.x1, rr.x1),
        y1: Math.max(acc.y1, rr.y1),
      }),
      { page: firstPg, x0: rects[0].x0, y0: rects[0].y0, x1: rects[0].x1, y1: rects[0].y1 }
    );
    setPage(firstPg);
    setRect(union);
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
          style={{ marginLeft: 8 }}
        />
        <span className="spacer" />
        <span className="muted">API: {API}</span>
      </div>

      <div className="wb-split" style={{ display: "flex", gap: 12 }}>
        {/* LEFT: 30% — fields from JSON */}
        <div className="wb-left" style={{ flexBasis: "30%", flexGrow: 0, flexShrink: 0, overflow: "auto" }}>
          <div className="section-title">Extraction (JSON)</div>
          {!rows.length ? (
            <div className="placeholder">Upload JSON extraction to see fields.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th style={{ width: "38%" }}>Key</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, idx) => {
                  const focused = r.key === focusedKey;
                  return (
                    <tr
                      key={r.key + ":" + idx}
                      onClick={() => onRowClick(r)}
                      style={focused ? { outline: "2px solid #ec4899", outlineOffset: -2 } : undefined}
                    >
                      <td>
                        <code>{r.key}</code>
                      </td>
                      <td style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.value}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* RIGHT: 70% — PDF at stable size */}
        <div className="wb-right" style={{ flexBasis: "70%", overflow: "auto" }}>
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
                tokens={tokensThisPage}
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

/* -------------------------------------------
   JSON → rows[] adapter
   Supports multiple common shapes that may include bboxes.
--------------------------------------------*/
function parseExtractionToRows(obj: AnyJson): FieldRow[] {
  if (!obj) return [];

  // 1) { fields:[{key, value, bboxes:[{page,x0,y0,x1,y1}]}] }
  if (Array.isArray(obj.fields)) {
    return obj.fields.map((f: AnyJson) => ({
      key: String(f.key ?? ""),
      value: f.value != null ? String(f.value) : "",
      rects: normalizeBoxes(f.bboxes || f.boxes || f.bbox || f.rects),
    }));
  }

  const rows: FieldRow[] = [];
  for (const [k, v] of Object.entries(obj)) {
    if (v == null) {
      rows.push({ key: k, value: "" });
      continue;
    }
    // 2) { key: { value, boxes:[...] } }
    if (typeof v === "object" && !Array.isArray(v)) {
      const vv: AnyJson = v as AnyJson;
      rows.push({
        key: k,
        value: vv.value != null ? String(vv.value) : stringy(v),
        rects: normalizeBoxes(vv.bboxes || vv.boxes || vv.bbox || vv.rects),
      });
      continue;
    }
    // 3) { key: [ {page,x0,..}, ... ] }  (value could be separate key like key_value)
    if (Array.isArray(v)) {
      // if array of rects:
      const rects = normalizeBoxes(v);
      if (rects?.length) {
        rows.push({
          key: k,
          value: "", // if you have separate value field, adapt here
          rects,
        });
        continue;
      }
      // else, just join as value
      rows.push({ key: k, value: v.map((x) => String(x)).join(" ") });
      continue;
    }
    // 4) primitive
    rows.push({ key: k, value: String(v) });
  }
  return rows;
}

function normalizeBoxes(input: any): KVRect[] | undefined {
  if (!input) return undefined;
  const arr = Array.isArray(input) ? input : [input];
  const out: KVRect[] = [];
  for (const b of arr) {
    if (b && Number.isFinite(Number(b.page))) {
      out.push({
        page: Number(b.page),
        x0: Number(b.x0),
        y0: Number(b.y0),
        x1: Number(b.x1),
        y1: Number(b.y1),
      });
    }
  }
  return out.length ? out : undefined;
}

function stringy(v: any): string {
  try {
    if (typeof v === "string") return v;
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}