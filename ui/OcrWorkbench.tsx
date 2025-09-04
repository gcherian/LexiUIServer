import { useMemo, useState, type ChangeEvent } from "react";
import PdfCanvas, { Box, Rect } from "./PdfCanvas";
import {
  uploadDoc,
  search,
  lasso,
  audit,
  getMeta,
  getBoxes,            // optional endpoint; handled gracefully if 404
  saveBoxes,           // optional endpoint; no-op if 404
} from "../../lib/api";
import "./ocr.css";

type Match = { page: number; bbox: Rect; text: string; score: number };

const SERVER = "http://localhost:8000";      // root (for /data & /out)
const API    = "http://localhost:8000/ocr";  // router prefix

type Tool = "select" | "lasso";

export default function OcrWorkbench() {
  const [doc, setDoc] = useState<any>(null);
  const [meta, setMeta] = useState<{ pages: { page: number; width: number; height: number }[] } | null>(null);

  const [page, setPage] = useState(1);
  const [scale, setScale] = useState(1.25);

  const [tool, setTool] = useState<Tool>("select");
  const [showBoxes, setShowBoxes] = useState(true);
  const [showHighlights, setShowHighlights] = useState(true);

  // Full set of OCR boxes (tokens/lines)—editable
  const [boxes, setBoxes] = useState<Box[]>([]);
  // Search results drawn as orange overlays (not editable)
  const [highlights, setHighlights] = useState<Box[]>([]);
  // Selected box indices (global indices into `boxes`)
  const [selected, setSelected] = useState<number[]>([]);
  // Inline filter for the list (label or id)
  const [filter, setFilter] = useState("");

  // ---------- Upload ----------
  async function doUpload(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;

    const res = await uploadDoc(API, f, "tesseract");
    const pdfUrl = `${SERVER}${res.annotated_tokens_url}?t=${Date.now()}`;
    setDoc({ ...res, pdfUrl });

    const m = await getMeta(API, res.doc_id);
    setMeta(m);

    setScale(1.25);
    setPage(1);
    setHighlights([]);
    setSelected([]);

    // Try to hydrate all OCR boxes immediately (this is what makes the UX feel “smart”)
    try {
      // Prefer server-provided boxes if present in upload response; else hit /doc/:id/boxes
      let initial: Box[] =
        (Array.isArray(res.boxes) ? res.boxes : []) as Box[];
      if (!initial.length) {
        const got = await getBoxes(API, res.doc_id); // may throw if not implemented
        if (Array.isArray(got)) initial = got as Box[];
      }
      // Ensure shape/page present
      initial = (initial || []).map((b, i) => ({
        id: b.id ?? `b${i}`,
        label: b.label ?? "",
        page: b.page ?? 1,
        x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1,
      }));
      setBoxes(initial);
    } catch {
      setBoxes([]); // fallback: user can search or draw with lasso
    }
  }

  // ---------- Search ----------
  async function doSearch(q: string) {
    if (!doc || !q.trim()) return;
    const r = await search(API, doc.doc_id, q, 40);
    const m: Match[] = r.matches || [];
    const h: Box[] = m.map(({ page, bbox }) => ({ page, ...bbox }));
    setHighlights(h);
    setSelected([]);
    if (h.length) setPage(h[0].page);
  }

  // ---------- Lasso add ----------
  async function onLassoRect(rect: Rect) {
    if (!doc) return;
    const out = await lasso(API, doc.doc_id, page, rect);
    const added: Box = {
      id: `new_${Date.now()}`,
      label: (out?.label ?? out?.text ?? "").slice(0, 40),
      page,
      ...rect,
    };
    setBoxes((prev) => [...prev, added]);
    await audit(API, { event: "lasso", payload: { doc_id: doc.doc_id, page, rect, result: out } });
    setTool("select");
  }

  // ---------- Selection / editing ----------
  function focusBox(idx: number) {
    const b = boxes[idx];
    if (!b) return;
    setPage(b.page);
    setSelected([idx]);
  }

  function updateBox(idx: number, patch: Partial<Box>) {
    setBoxes((prev) => {
      const next = prev.slice();
      next[idx] = { ...prev[idx], ...patch };
      return next;
    });
  }

  function deleteSelected() {
    if (!selected.length) return;
    const keep = new Set(selected);
    setBoxes((prev) => prev.filter((_, i) => !keep.has(i)));
    setSelected([]);
  }

  async function persistBoxes() {
    if (!doc) return;
    try {
      await saveBoxes(API, doc.doc_id, boxes);
      alert("Boxes saved.");
    } catch {
      // If backend doesn’t support it, provide a quick download
      downloadJSON(boxes, `boxes_${doc.doc_id}.json`);
    }
  }

  // ---------- Derived lists ----------
  const pageBoxIndices = useMemo(() => {
    const indices = boxes.map((b, i) => (b.page === page ? i : -1)).filter((i) => i >= 0);
    if (!filter.trim()) return indices;
    const f = filter.toLowerCase();
    return indices.filter((i) => {
      const b = boxes[i];
      return (b.label ?? "").toLowerCase().includes(f) || (b.id ?? "").toLowerCase().includes(f);
    });
  }, [boxes, page, filter]);

  const pageCount = meta?.pages?.length ?? 0;

  return (
    <div className="ocr-app">
      <header className="ocr-header">
        <div className="brand">
          <span className="wf">WELLS FARGO</span>
          <span className="pipe">|</span>
          <span className="app">EDIP Platform Web · OCR Workbench</span>
        </div>
        <div className="toolbar">
          <div className="toolseg">
            <label>Tool</label>
            <div className="seg">
              <button className={tool === "select" ? "active" : ""} onClick={() => setTool("select")}>Select</button>
              <button className={tool === "lasso" ? "active" : ""} onClick={() => setTool("lasso")}>Lasso</button>
            </div>
          </div>

          <div className="toolseg">
            <label>Page</label>
            <div className="pager">
              <button onClick={() => setPage((p) => Math.max(1, p - 1))}>Prev</button>
              <span>{page}{pageCount ? ` / ${pageCount}` : ""}</span>
              <button onClick={() => setPage((p) => p + 1)}>Next</button>
            </div>
          </div>

          <div className="toolseg">
            <label>Zoom</label>
            <div className="seg">
              <button onClick={() => setScale((s) => Math.max(0.5, +(s - 0.25).toFixed(2)))}>-</button>
              <span className="meter">{scale.toFixed(2)}x</span>
              <button onClick={() => setScale((s) => +(s + 0.25).toFixed(2))}>+</button>
            </div>
          </div>

          <div className="toolseg toggles">
            <label>Layers</label>
            <div className="chk">
              <label><input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} /> Boxes</label>
              <label><input type="checkbox" checked={showHighlights} onChange={(e) => setShowHighlights(e.target.checked)} /> Highlights</label>
            </div>
          </div>

          <div className="toolseg">
            <button className="primary" onClick={persistBoxes}>Save</button>
            <button onClick={deleteSelected} disabled={!selected.length}>Delete</button>
          </div>
        </div>
      </header>

      <main className="ocr-main">
        <aside className="left">
          <div className="panel">
            <div className="panel-title">Upload</div>
            <input type="file" accept="application/pdf" onChange={doUpload} />
          </div>

          <div className="panel">
            <div className="panel-title">Search (fuzzy)</div>
            <div className="searchrow">
              <input id="q" placeholder="Amount, name, date, …" onKeyDown={(e) => {
                if (e.key === "Enter") {
                  const val = (e.target as HTMLInputElement).value;
                  doSearch(val);
                }
              }} />
              <button onClick={() => {
                const q = (document.getElementById("q") as HTMLInputElement).value;
                doSearch(q);
              }}>Go</button>
            </div>
          </div>

          <div className="panel">
            <div className="panel-title">Boxes on Page {page}</div>
            <input
              className="filter"
              placeholder="Filter by label or id"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
            <div className="list">
              {pageBoxIndices.length === 0 && <div className="empty">No boxes. Try Search or draw with Lasso.</div>}
              {pageBoxIndices.map((idx) => {
                const b = boxes[idx];
                const active = selected.includes(idx);
                return (
                  <div key={b.id ?? idx} className={"item" + (active ? " active" : "")} onClick={() => focusBox(idx)}>
                    <div className="row">
                      <input
                        className="label"
                        placeholder="Label"
                        value={b.label ?? ""}
                        onChange={(e) => updateBox(idx, { label: e.target.value })}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </div>
                    <div className="meta">
                      <span className="mono">{b.id ?? `#${idx}`}</span>
                      <span className="coords">
                        ({Math.round(b.x0)}, {Math.round(b.y0)}) → ({Math.round(b.x1)}, {Math.round(b.y1)})
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </aside>

        <section className="stage">
          {doc && meta ? (
            <PdfCanvas
              url={doc.pdfUrl}
              page={page}
              scale={scale}
              boxes={showBoxes ? boxes : []}
              highlights={showHighlights ? highlights : []}
              selected={selected}
              ocrSize={meta.pages.find((p) => p.page === page) ? {
                width: meta.pages.find((p) => p.page === page)!.width,
                height: meta.pages.find((p) => p.page === page)!.height
              } : undefined}
              tool={tool}
              onLasso={onLassoRect}
              onSelectBox={focusBox}
            />
          ) : (
            <div className="placeholder">Upload a PDF to begin.</div>
          )}
        </section>
      </main>
    </div>
  );
}

/** Utility: download JSON when backend save is not available */
function downloadJSON(data: unknown, filename = "data.json") {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}
