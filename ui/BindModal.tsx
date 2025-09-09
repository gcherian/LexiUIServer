import React, { useEffect, useMemo, useState } from "react";
import PdfEditCanvas, { type EditRect } from "./PdfEditCanvas";
import {
  ocrPreview,
  bindField,
  getBoxes,
  type Box as TokenBox,
  type FieldDocState,
} from "../../../lib/api";

type Props = {
  open: boolean;
  onClose: () => void;
  docId: string;
  docUrl: string;
  page: number;
  serverW: number;
  serverH: number;
  allKeys: string[];
  box: TokenBox | null;          // if opened by clicking a token box
  initialKey: string;            // default field to preselect
  onBound: (state: FieldDocState) => void;
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
  const [keySel, setKeySel] = useState<string>(initialKey || "");
  const [rect, setRect] = useState<EditRect | null>(null);
  const [tokens, setTokens] = useState<TokenBox[]>([]);
  const [ocrText, setOcrText] = useState<string>("");

  useEffect(() => {
    if (!open) return;
    (async () => {
      setTokens((await getBoxes(docId)).filter((b) => b.page === page));
    })();
  }, [open, docId, page]);

  useEffect(() => {
    setKeySel(initialKey || "");
  }, [initialKey, open]);

  // if opened from a token click, pre-populate rect
  useEffect(() => {
    if (box && open) {
      setRect({ page: box.page, x0: box.x0, y0: box.y0, x1: box.x1, y1: box.y1 });
      setOcrText(box.text || "");
    } else if (open) {
      setRect(null);
      setOcrText("");
    }
  }, [box, open]);

  if (!open) return null;

  async function onLassoDone(r: EditRect) {          // ✅ typed
    setRect(r);
    const res = await ocrPreview(docId, r.page, r);
    setOcrText(res?.text || "");
  }

  async function onBind() {
    if (!rect || !keySel) return;
    const st = await bindField(docId, keySel, rect.page, rect);
    onBound(st);
    onClose();
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ fontWeight: 600 }}>Bind value by Lasso</div>
          <button className="icon" onClick={onClose}>×</button>
        </div>

        <div className="modal-body" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div>
            <div style={{ marginBottom: 6 }}>
              <label>Field</label>
              <select value={keySel} onChange={(e) => setKeySel(e.target.value)}>
                <option value="">(choose a field)</option>
                {allKeys.map((k) => (
                  <option key={k} value={k}>{k}</option>
                ))}
              </select>
            </div>
            <div className="pdf-wrap" style={{ border: "1px solid #e5e7eb", borderRadius: 6, padding: 6 }}>
              <PdfEditCanvas
                docUrl={docUrl}
                page={page}
                serverW={serverW}
                serverH={serverH}
                tokens={tokens}
                rect={rect}
                showTokenBoxes={true}
                editable={true}
                onRectChange={setRect}
                onRectCommit={onLassoDone}
              />
            </div>
          </div>

          <div>
            <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 6 }}>OCR preview</div>
            <textarea
              style={{ width: "100%", height: 220 }}
              value={ocrText}
              onChange={(e) => setOcrText(e.target.value)}
            />
            <div style={{ marginTop: 12, display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={onClose}>Cancel</button>
              <button className="primary" disabled={!keySel || !rect} onClick={onBind}>
                Bind
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
