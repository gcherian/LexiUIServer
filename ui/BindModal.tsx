// File: src/components/lasso/BindModal.tsx
import React, { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import PdfCanvas from "./PdfCanvas";
import type { RectServer } from "./PdfCanvas";
import { ocrPreview, bindField, type FieldDocState } from "../../../lib/api";

/** Minimal box shape (kept local to avoid deps drift). */
type BoxLike = {
  page: number;
  x0: number; y0: number; x1: number; y1: number;
  id?: string | null;
  label?: string | null;
  text?: string | null;
  confidence?: number | null;
};

type Props = {
  open: boolean;
  onClose: () => void;

  // Document context
  docId: string;
  docUrl: string;
  page: number;
  serverW: number;
  serverH: number;

  // Field keys for dropdown (from PROM)
  allKeys: string[];

  // Optional: opening from an existing OCR box
  box?: BoxLike | null;

  // Optional: preselect a key (e.g., first missing key)
  initialKey?: string | null;

  // After successful bind, caller can update its state
  onBound?: (st: FieldDocState) => void;
};

export default function BindModal({
  open,
  onClose,
  docId,
  docUrl,
  page,
  serverW,
  serverH,
  allKeys,
  box,
  initialKey,
  onBound,
}: Props) {
  const [modeLasso, setModeLasso] = useState<boolean>(!box); // start in lasso if no box provided
  const [rect, setRect] = useState<RectServer | null>(box ? { x0: box.x0, y0: box.y0, x1: box.x1, y1: box.y1 } : null);
  const [value, setValue] = useState<string>("");
  const [key, setKey] = useState<string>(initialKey || "");
  const [binding, setBinding] = useState<boolean>(false);

  // Reset per open/box
  useEffect(() => {
    setModeLasso(!box);
    setRect(box ? { x0: box.x0, y0: box.y0, x1: box.x1, y1: box.y1 } : null);
    setValue("");
  }, [box, open]);

  // Sync initialKey when provided
  useEffect(() => {
    if (initialKey) setKey(initialKey);
  }, [initialKey]);

  async function previewOCR() {
    if (!rect) return;
    try {
      const r = await ocrPreview(docId, page, rect);
      setValue(r?.text || "");
    } catch {
      // swallow; UI stays as-is
    }
  }

  async function doBind() {
    if (!rect || !key) return;
    setBinding(true);
    try {
      const st = await bindField(docId, key, page, rect);
      onBound?.(st);
      onClose();
    } finally {
      setBinding(false);
    }
  }

  if (!open) return null;

  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => { /* clicking backdrop = no-op */ }}>
      <div className="modal-card" onMouseDown={(e) => e.stopPropagation()}>
        {/* Sticky header within the card so the top never hides under app header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", position: "sticky", top: 0, background: "inherit", paddingBottom: 6, zIndex: 1 }}>
          <h3 style={{ margin: 0 }}>Bind Field</h3>
          <button onClick={onClose}>✕</button>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 12, marginTop: 8 }}>
          {/* Left: mini viewer with optional lasso and single preselected box */}
          <div className="border pad">
            <div className="toolbar-inline">
              <label className={modeLasso ? "btn toggle active" : "btn toggle"}>
                <input type="checkbox" checked={modeLasso} onChange={() => setModeLasso((v) => !v)} /> Lasso
              </label>
              <button className="btn toggle" onClick={previewOCR} disabled={!rect}>Preview OCR</button>
            </div>

            <PdfCanvas
              docUrl={docUrl}
              page={page}
              serverW={serverW}
              serverH={serverH}
              boxes={box ? [box] : []}
              showBoxes={!!box}
              lasso={modeLasso}
              onLassoDone={(r) => {
                setRect(r);
                setModeLasso(false); // auto-exit lasso after drawing
              }}
            />
          </div>

          {/* Right: form */}
          <div>
            <div className="row">
              <label>Field</label>
              <input
                list="bind-keys"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                placeholder="e.g., invoice_number"
              />
              <datalist id="bind-keys">
                {allKeys.map((k) => (
                  <option key={k} value={k} />
                ))}
              </datalist>
            </div>

            <div className="row">
              <label>Value</label>
              <textarea rows={6} value={value} onChange={(e) => setValue(e.target.value)} />
            </div>

            <div className="row">
              <label>Where</label>
              <input
                disabled
                value={
                  rect
                    ? `x0=${rect.x0}, y0=${rect.y0}, x1=${rect.x1}, y1=${rect.y1}`
                    : "(draw or select a box)"
                }
              />
            </div>

            <div className="flex justify-end" style={{ marginTop: 10 }}>
              <button onClick={onClose}>Cancel</button>
              <button
                className="primary"
                onClick={doBind}
                disabled={!rect || !key || binding}
                style={{ marginLeft: 8 }}
              >
                {binding ? "Binding…" : "Bind"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
