// File: src/tsp4/components/lasso/FieldLevelEditor.tsx
import React, { useEffect, useMemo, useState } from "react";
import "../ocr.css";
import PdfEditCanvas, { type EditRect, type TokenBox } from "./PdfEditCanvas";

import {
  API,
  uploadPdf,
  getMeta,
  getBoxes,
  docIdFromUrl,
  ocrPreview,
  distilExtract,
  type DistilField,
  matchField,
  type MatchResp,
} from "../../../lib/api";

/* ---------------- Types ---------------- */
type KVRect = { page: number; x0: number; y0: number; x1: number; y1: number };
type AnyJson = Record<string, any>;
type FieldRow = { key: string; value: string; rects?: KVRect[] };

export type OverlayRect = {
  page: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  color: string;
  alpha?: number;
  dashed?: boolean;
  label?: string;
};

/* ---------------- Component ---------------- */
export default function FieldLevelEditor() {
  // doc + page
  const [docUrl, setDocUrl] = useState("");
  const [docId, setDocId] = useState("");
  const [meta, setMeta] = useState<{ w: number; h: number }[]>([]);
  const [page, setPage] = useState(1);

  // tokens
  const [tokens, setTokens] = useState<TokenBox[]>([]);
  const tokensPage = useMemo(() => tokens.filter((t) => t.page === page), [tokens, page]);

  // KV table
  const [rows, setRows] = useState<FieldRow[]>([]);
  const [focusedKey, setFocusedKey] = useState("");

  // overlays
  const [rect, setRect] = useState<EditRect | null>(null);
  const [overlays, setOverlays] = useState<OverlayRect[]>([]);

  const [showBoxes, setShowBoxes] = useState(false);
  const [zoom, setZoom] = useState(1);

  const [distil, setDistil] = useState<DistilField[]>([]);
  const DISTIL_OK = 0.45;

  /* -------- Uploads -------- */
  async function onUploadPdf(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const res = await uploadPdf(f);
      setDocId(res.doc_id);
      setDocUrl(res.annotated_tokens_url);
      const m = await getMeta(res.doc_id);
      setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
      setTokens((await getBoxes(res.doc_id)) as any);
      setPage(1);
      setRect(null);
      setOverlays([]);
    } finally {
      (e.target as HTMLInputElement).value = "";
    }
  }

  async function onUploadEcm(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const parsed = JSON.parse(await f.text()) as AnyJson;
      const fr: FieldRow[] = Object.entries(parsed).map(([k, v]) => ({
        key: k,
        value: typeof v === "string" ? v : JSON.stringify(v),
      }));
      setRows(fr);
      setFocusedKey("");
      setRect(null);
      setOverlays([]);
    } catch {
      alert("Invalid ECM JSON");
    } finally {
      (e.target as HTMLInputElement).value = "";
    }
  }

  /* -------- Paste /data/{doc}/original.pdf -------- */
  useEffect(() => {
    const id = docIdFromUrl(docUrl);
    if (!id) return;
    (async () => {
      setDocId(id);
      const m = await getMeta(id);
      setMeta(m.pages.map((p) => ({ w: p.width, h: p.height })));
      setTokens((await getBoxes(id)) as any);
      setPage(1);
      setRect(null);
      setOverlays([]);
    })();
  }, [docUrl]);

  /* -------- Distil Runner -------- */
  async function runDistilNow() {
    if (!docId || !rows.length) return;
    const specs = rows.map((r) => ({ key: r.key, label: r.key, type: "text" }));
    try {
      const res = await distilExtract(docId, specs, 12, 260);
      setDistil(res.fields || []);
    } catch (e) {
      console.warn("distil failed", e);
      setDistil([]);
    }
  }

  useEffect(() => {
    if (docId && rows.length) runDistilNow();
  }, [docId, rows.length]);

  /* -------- Row click -------- */
  async function onRowClick(r: FieldRow) {
    setFocusedKey(r.key);

    const ocrTokens = tokens.map((t) => ({
      text: t.text || "",
      bbox: [t.x0, t.y0, t.x1, t.y1] as [number, number, number, number],
      page: t.page,
    }));

    let resp: MatchResp | null = null;
    try {
      resp = await matchField(ocrTokens, r.key, r.key, r.value, true);
    } catch (e) {
      console.warn("match/field failed", e);
    }

    const ov: OverlayRect[] = [];
    const push = (m: any, color: string, label: string) => {
      if (!m?.bbox || !m.page) return;
      const [x0, y0, x1, y1] = m.bbox;
      ov.push({ page: m.page, x0, y0, x1, y1, color, label, alpha: 0.2 });
    };
    if (resp) {
      push(resp.autolocate, "#1f7ae0", "autolocate"); // blue
      push(resp.bert, "#7a1fe0", "bert"); // purple
      push(resp.tfidf, "#e07a1f", "tfidf"); // orange
    }
    setOverlays(ov);

    // set pink rect to BERT → Auto → TFIDF
    let chosen = resp?.bert || resp?.autolocate || resp?.tfidf;
    if (chosen?.bbox && chosen.page) {
      const [x0, y0, x1, y1] = chosen.bbox;
      setPage(chosen.page);
      setRect({ page: chosen.page, x0, y0, x1, y1 });
    } else {
      setRect(null);
    }
  }

  /* -------- OCR crop commit -------- */
  async function onRectCommitted(rr: EditRect) {
    if (!focusedKey) return;
    try {
      if (docId) {
        const res = await ocrPreview(docId, rr.page, rr);
        const text = (res?.text || "").trim();
        setRows((prev) =>
          prev.map((row) =>
            row.key === focusedKey
              ? { ...row, value: text, rects: [{ page: rr.page, x0: rr.x0, y0: rr.y0, x1: rr.x1, y1: rr.y1 }] }
              : row
          )
        );
      }
    } catch {}
  }

  const serverW = meta[page - 1]?.w || 1;
  const serverH = meta[page - 1]?.h || 1;

  return (
    <div className="workbench">
      <div className="wb-toolbar" style={{ gap: 8 }}>
        <label className="btn">
          <input type="file" accept="application/pdf" onChange={onUploadPdf} style={{ display: "none" }} />
          PDF
        </label>
        <label className="btn">
          <input type="file" accept="application/json" onChange={onUploadEcm} style={{ display: "none" }} />
          ECM JSON
        </label>
        <input
          className="input"
          placeholder="/data/{doc_id}/original.pdf"
          value={docUrl}
          onChange={(e) => setDocUrl(e.target.value || "")}
          style={{ marginLeft: 8, minWidth: 360 }}
        />
      </div>

      <div className="wb-split" style={{ display: "flex", gap: 12 }}>
        <div className="wb-left" style={{ flexBasis: "30%", overflow: "auto" }}>
          <table>
            <thead>
              <tr>
                <th style={{ width: "42%" }}>Key</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr
                  key={r.key + ":" + i}
                  onClick={() => onRowClick(r)}
                  style={r.key === focusedKey ? { outline: "2px solid #ec4899", outlineOffset: -2 } : undefined}
                >
                  <td>
                    <code>{r.key}</code>
                  </td>
                  <td>{r.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="wb-right" style={{ flexBasis: "70%", overflow: "auto" }}>
          {docUrl ? (
            <PdfEditCanvas
              docUrl={docUrl}
              page={page}
              serverW={serverW}
              serverH={serverH}
              tokens={tokensPage}
              rect={rect}
              showTokenBoxes={showBoxes}
              editable={true}
              onRectChange={setRect}
              onRectCommit={onRectCommitted}
              zoom={zoom}
              overlays={overlays} // pass overlays here
            />
          ) : (
            <div className="placeholder">Upload a PDF to begin.</div>
          )}
        </div>
      </div>
    </div>
  );
}