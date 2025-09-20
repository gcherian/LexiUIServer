import React, { useEffect, useMemo, useState } from "react";
import PdfEditCanvas, { type EditRect, type TokenBox, type OverlayRect } from "./PdfEditCanvas";
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
} from "../../../lib/api";

/* ───────── Types / helpers ───────── */
type KVRect = { page: number; x0: number; y0: number; x1: number; y1: number };
type AnyJson = Record<string, any>;
type FieldRow = { key: string; value: string; rects?: KVRect[] };

const COLORS: Record<string, string> = {
  fuzzy: "#22c55e",
  tfidf: "#3b82f6",
  minilm: "#a855f7",
  distilbert: "#eab308", // yellow to avoid confusion with GT pink
  layoutlmv3: "#f97316",
};

const norm = (s: string) =>
  (s || "")
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[\u00A0]/g, " ")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

function unionRect(span: TokenBox[]) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const t of span) {
    x0 = Math.min(x0, t.x0);
    y0 = Math.min(y0, t.y0);
    x1 = Math.max(x1, t.x1);
    y1 = Math.max(y1, t.y1);
  }
  return { x0: Math.floor(x0), y0: Math.floor(y0), x1: Math.ceil(x1), y1: Math.ceil(y1) };
}

function refineWithTokens(
  union: { page: number; x0: number; y0: number; x1: number; y1: number },
  pageTokens: TokenBox[]
) {
  const inBox = (t: TokenBox) => {
    const cx = (t.x0 + t.x1) / 2, cy = (t.y0 + t.y1) / 2;
    const minx = Math.min(union.x0, union.x1), maxx = Math.max(union.x0, union.x1);
    const miny = Math.min(union.y0, union.y1), maxy = Math.max(union.y0, union.y1);
    return cx >= minx && cx <= maxx && cy >= miny && cy <= maxy;
  };
  const pool = pageTokens.filter(inBox);
  if (!pool.length) return union;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const t of pool) {
    x0 = Math.min(x0, t.x0); y0 = Math.min(y0, t.y0);
    x1 = Math.max(x1, t.x1); y1 = Math.max(y1, t.y1);
  }
  return { page: union.page, x0, y0, x1, y1 };
}

/* ECM → table flatten */
function normalizeRects(x: any): KVRect[] | undefined {
  if (!x) return undefined;
  const arr = Array.isArray(x) ? x : [x];
  const out: KVRect[] = [];
  for (const b of arr) {
    if (b && Number.isFinite(+b.page)) {
      if (b.x != null && b.y != null && b.w != null && b.h != null) {
        out.push({ page: +b.page, x0: +b.x, y0: +b.y, x1: +b.x + +b.w, y1: +b.y + +b.h });
      } else {
        out.push({ page: +b.page, x0: +b.x0, y0: +b.y0, x1: +b.x1, y1: +b.y1 });
      }
    }
  }
  return out.length ? out : undefined;
}
function tryStr(v: any) { try { return typeof v === "string" ? v : JSON.stringify(v); } catch { return String(v); } }
function flattenJson(obj: any, prefix = ""): FieldRow[] {
  const rows: FieldRow[] = [];
  const push = (k: string, v: any) => rows.push({ key: k, value: v == null ? "" : String(v) });

  if (obj === null || obj === undefined) return [{ key: prefix || "(null)", value: "" }];

  if (Array.isArray(obj)) {
    obj.forEach((v, i) => {
      const path = prefix ? `${prefix}[${i}]` : `[${i}]`;
      if (typeof v === "object" && v !== null) rows.push(...flattenJson(v, path));
      else push(path, v);
    });
    return rows;
  }
  if (typeof obj !== "object") return [{ key: prefix || "(value)", value: String(obj) }];

  for (const k of Object.keys(obj)) {
    const path = prefix ? `${prefix}.${k}` : k;
    const v = obj[k];
    if (v && typeof v === "object" && !Array.isArray(v)) {
      const rects = normalizeRects(v.bboxes || v.boxes || v.bbox || v.rects);
      if ("value" in v || rects) {
        rows.push({ key: path, value: "value" in v ? String(v.value ?? "") : tryStr(v), rects });
      } else {
        rows.push(...flattenJson(v, path));
      }
    } else if (Array.isArray(v)) {
      rows.push(...flattenJson(v, path));
    } else {
      push(path, v);
    }
  }
  return rows;
}

/* ───────── Component ───────── */
export default function FieldLevelEditor() {
  // document + paging
  const [docUrl, setDocUrl] = useState("");
  const [docId, setDocId] = useState("");
  const [meta, setMeta] = useState<{ w: number; h: number }[]>([]);
  const [page, setPage] = useState(1);

  // tokens & overlays
  const [tokens, setTokens] = useState<TokenBox[]>([]);
  const tokensPage = useMemo(() => tokens.filter((t) => t.page === page), [tokens, page]);

  const [rows, setRows] = useState<FieldRow[]>([]);
  const [focusedKey, setFocusedKey] = useState("");

  const [rect, setRect] = useState<EditRect | null>(null);
  const [overlays, setOverlays] = useState<OverlayRect[]>([]);
  const [showBoxes, setShowBoxes] = useState(false);
  const [loadingBoxes, setLoadingBoxes] = useState(false);

  const [editValue, setEditValue] = useState(false); // if true, OCR preview also updates left-side value
  const [zoom, setZoom] = useState(1);

  // Distil (best-effort)
  const [distil, setDistil] = useState<DistilField[]>([]);
  const DISTIL_OK = 0.45;

  /* Upload handlers */
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
      setRows(flattenJson(parsed));
      setFocusedKey("");
      setRect(null);
      setOverlays([]);
    } catch {
      alert("Invalid ECM JSON");
    } finally {
      (e.target as HTMLInputElement).value = "";
    }
  }

  /* Paste /data/{doc}/original.pdf */
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

  /* Optional: run Distil on current rows */
  async function runDistilNow() {
    if (!docId || !rows.length) return;
    const specs = rows.map((r) => {
      const k = r.key.toLowerCase();
      const type =
        k.includes("zip") ? "zip" :
        k.includes("date") ? "date" :
        k.includes("amount") || k.includes("total") ? "amount" : "text";
      const label = r.key.replace(/\./g, " ").replace(/\[\d+\]/g, " ").replace(/\s+/g, " ").trim();
      return { key: r.key, label, type };
    });
    try {
      const res = await distilExtract(docId, specs, 12, 260);
      setDistil(res.fields || []);
    } catch {
      setDistil([]);
    }
  }
  useEffect(() => {
    if (docId && rows.length) runDistilNow();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docId, rows.length]);

  /* Row click → draw all 4 model overlays + position GT box via ECM/Distil/fallback */
  async function onRowClick(r: FieldRow) {
    setFocusedKey(r.key);
    setLoadingBoxes(true);

    // try ECM rects first to position GT
    if (r.rects?.length) {
      const byPg: Record<number, KVRect[]> = {};
      r.rects.forEach((b) => (byPg[b.page] = byPg[b.page] ? [...byPg[b.page], b] : [b]));
      const pg = Number(Object.keys(byPg).sort((a, b) => byPg[+b].length - byPg[+a].length)[0]);
      const same = byPg[pg];
      const uni = same.reduce(
        (acc, rr) => ({
          page: pg,
          x0: Math.min(acc.x0, rr.x0),
          y0: Math.min(acc.y0, rr.y0),
          x1: Math.max(acc.x1, rr.x1),
          y1: Math.max(acc.y1, rr.y1),
        }),
        { page: pg, x0: same[0].x0, y0: same[0].y0, x1: same[0].x1, y1: same[0].y1 }
      );
      const refined = refineWithTokens(uni, tokens.filter((t) => t.page === pg));
      setPage(pg);
      setRect(refined);
    } else {
      setRect(null);
    }

    // fetch overlays from server (fuzzy/tfidf/minilm/distilbert)
    try {
      if (!docId) return;
      const res = await matchField(docId, r.key, r.value);
      const pick = (m: any): EditRect | null =>
        !m ? null : ({ page: m.page, x0: m.rect.x0, y0: m.rect.y0, x1: m.rect.x1, y1: m.rect.y1 });

      const order: Array<keyof typeof COLORS> = ["fuzzy","tfidf","minilm","distilbert"/*,"layoutlmv3"*/];
      const ovs: OverlayRect[] = order.map((k) => ({
        label: k,
        color: COLORS[k],
        rect: pick((res as any).methods?.[k]),
      }));

      if (!rect) {
        const pg = ovs.find(o => o.rect)?.rect?.page;
        if (pg) setPage(pg!);
      }
      setOverlays(ovs);
    } catch (e) {
      console.warn("matchField failed", e);
      setOverlays([]);
    } finally {
      setLoadingBoxes(false);
    }
  }

  /* Lasso commit → OCR preview; optionally update left value; always update GT rects in row */
  async function onRectCommitted(rr: EditRect) {
    if (!focusedKey) return;
    try {
      if (docId) {
        const res = await ocrPreview(docId, rr.page, rr);
        const text = (res?.text || "").trim();

        setRows((prev) =>
          prev.map((row) =>
            row.key === focusedKey
              ? {
                  ...row,
                  value: editValue ? text : row.value,
                  rects: [{ page: rr.page, x0: rr.x0, y0: rr.y0, x1: rr.x1, y1: rr.y1 }],
                }
              : row
          )
        );
      }
    } catch {
      /* best-effort OCR preview */
    }
  }

  /* Save GT for current key */
  async function saveGTForFocused() {
    if (!docId || !focusedKey) return;
    const row = rows.find(r => r.key === focusedKey);
    if (!row || !row.rects || !row.rects.length) {
      alert("Lasso a ground-truth box first.");
      return;
    }
    const body = {
      doc_id: docId,
      key: row.key,
      value: row.value ?? "",
      rects: row.rects,
    };
    try {
      const r = await fetch(`${API}/lasso/groundtruth/save`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(await r.text());
      const j = await r.json();
      if (j?.iou_report) {
        const lines = Object.entries(j.iou_report)
          .map(([k, v]: any) => `${k}: ${(v * 100).toFixed(1)}%`)
          .join("\n");
        alert(`Saved GT.\nIOU vs models:\n${lines}`);
      } else {
        alert("Saved GT.");
      }
    } catch (e: any) {
      alert(`Save failed: ${e?.message || e}`);
    }
  }

  const serverW = meta[page - 1]?.w || 1;
  const serverH = meta[page - 1]?.h || 1;

  /* ───────── Render ───────── */
  return (
    <div className="workbench">
      <div className="wb-toolbar" style={{ gap: 8 }}>
        <span style={{ fontWeight: 600 }}>Choose:</span>

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
          placeholder="...or paste http://localhost:8080/data/{doc_id}/original.pdf"
          value={docUrl}
          onChange={(e) => setDocUrl(e.target.value || "")}
          style={{ marginLeft: 8, minWidth: 360 }}
        />

        <label className={showBoxes ? "btn toggle active" : "btn toggle"} style={{ marginLeft: 8 }}>
          <input type="checkbox" checked={showBoxes} onChange={() => setShowBoxes((v) => !v)} /> Boxes
        </label>

        <label className={editValue ? "btn toggle active" : "btn toggle"} title="If on, OCR will also update the left-side value.">
          <input type="checkbox" checked={editValue} onChange={() => setEditValue(v => !v)} /> Edit Value
        </label>

        <button onClick={runDistilNow} disabled={!docId || !rows.length}>Refresh Distil</button>
        <button onClick={saveGTForFocused} disabled={!docId || !focusedKey}>Save GT</button>

        <span className="spacer" />

        <div className="toolbar-inline" style={{ gap: 8 }}>
          {(["fuzzy","tfidf","minilm","distilbert"] as const).map((k) => (
            <span key={k} style={{ display:"inline-flex", alignItems:"center", gap:6, fontSize:12 }}>
              <span style={{ width:12, height:12, background:COLORS[k], border:`1px solid ${COLORS[k]}`, display:"inline-block", opacity:0.7 }} />
              {k}
            </span>
          ))}
        </div>

        <span className="spacer" />

        <div className="toolbar-inline" style={{ gap: 4 }}>
          <button onClick={() => setZoom((z) => Math.max(0.5, Math.round((z - 0.1) * 10) / 10))}>–</button>
          <span style={{ width: 44, textAlign: "center" }}>{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom((z) => Math.min(3, Math.round((z + 0.1) * 10) / 10))}>+</button>
          <button onClick={() => setZoom(1)}>Reset</button>
        </div>

        <span className="muted" style={{ marginLeft: 12 }}>API: {API}</span>
      </div>

      <div className="wb-split" style={{ display: "flex", gap: 12 }}>
        {/* LEFT */}
        <div className="wb-left" style={{ flexBasis: "32%", flexGrow: 0, flexShrink: 0, overflow: "auto" }}>
          <div className="section-title">Extraction</div>
          {!rows.length ? (
            <div className="placeholder">Upload ECM JSON to see fields.</div>
          ) : (
            <table>
              <thead>
                <tr><th style={{ width: "42%" }}>Key</th><th>Value</th></tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const focused = r.key === focusedKey;
                  return (
                    <tr
                      key={r.key + ":" + i}
                      onClick={() => onRowClick(r)}
                      style={focused ? { outline: "2px solid #ec4899", outlineOffset: -2 } : undefined}
                    >
                      <td><code>{r.key}</code></td>
                      <td style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {r.value}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* RIGHT */}
        <div className="wb-right" style={{ flexBasis: "68%", overflow: "auto", position:"relative" }}>
          {loadingBoxes && (
            <div style={{
              position:"absolute", right:12, top:8, background:"#0008", color:"#fff",
              padding:"4px 8px", borderRadius:6, fontSize:12, zIndex: 5
            }}>loading boxes…</div>
          )}
          {docUrl ? (
            <>
              <div className="toolbar-inline">
                <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Prev</button>
                <span className="page-indicator">Page {page} {meta.length ? `/ ${meta.length}` : ""}</span>
                <button disabled={meta.length > 0 && page >= meta.length} onClick={() => setPage((p) => p + 1)}>Next</button>
              </div>
              <PdfEditCanvas
                docUrl={docUrl}
                page={page}
                serverW={serverW}
                serverH={serverH}
                tokens={tokensPage}
                rect={rect}
                overlays={overlays}
                showTokenBoxes={showBoxes}
                editable={true}
                onRectChange={setRect}
                onRectCommit={onRectCommitted}
                zoom={zoom}
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