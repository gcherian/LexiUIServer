import React, { useEffect, useMemo, useState } from "react";
import type { Box, Rect, FieldDocState } from "../../lib/api";
import { ocrPreview, bindField } from "../../lib/api";
import PdfCanvas from "./PdfCanvas";

type Props = {
  open: boolean;
  onClose: () => void;

  // doc context
  docId: string;
  docUrl: string;
  page: number;
  serverW: number;
  serverH: number;

  // field keys for dropdown
  allKeys: string[];

  // when opening from box click:
  box?: Box | null;

  // when opening for "Bind via Lasso" on a specific key:
  initialKey?: string | null;

  onBound?: (state: FieldDocState) => void;
};

export default function BindModal({
  open, onClose, docId, docUrl, page, serverW, serverH, allKeys, box, initialKey, onBound
}: Props) {
  const [modeLasso, setModeLasso] = useState<boolean>(!box);
  const [rect, setRect] = useState<Rect | null>(box ? { x0:box.x0, y0:box.y0, x1:box.x1, y1:box.y1 } : null);
  const [value, setValue] = useState<string>("");
  const [key, setKey] = useState<string>(initialKey || "");
  const [binding, setBinding] = useState<boolean>(false);

  useEffect(() => { setModeLasso(!box); setRect(box?{x0:box.x0,y0:box.y0,x1:box.x1,y1:box.y1}:null); setValue(""); }, [box, open]);
  useEffect(() => { if (initialKey) setKey(initialKey); }, [initialKey]);

  async function preview() {
    if (!rect) return;
    const r = await ocrPreview(docId, page, rect);
    setValue(r.text || "");
  }

  async function doBind() {
    if (!rect || !key) return;
    setBinding(true);
    try {
      const st = await bindField(docId, key, page, rect);
      onBound?.(st);
      onClose();
    } finally { setBinding(false); }
  }

  if (!open) return null;

  return (
    <div style={styles.backdrop}>
      <div style={styles.modal}>
        <div style={{display:"flex", alignItems:"center", justifyContent:"space-between"}}>
          <h3 style={{margin:0}}>Bind Field</h3>
          <button onClick={onClose}>✕</button>
        </div>

        <div style={{display:"grid", gridTemplateColumns:"1fr 320px", gap:12, marginTop:10}}>
          {/* left: mini viewer for lasso */}
          <div style={{border:"1px solid #e5e7eb", borderRadius:8, padding:8}}>
            <div style={{display:"flex", gap:8, marginBottom:6}}>
              <label className={modeLasso? "btn toggle active":"btn toggle"}>
                <input type="checkbox" checked={modeLasso} onChange={()=>setModeLasso(v=>!v)} /> Lasso
              </label>
              <button className="btn toggle" onClick={preview} disabled={!rect}>Preview OCR</button>
            </div>
            <PdfCanvas
              docUrl={docUrl}
              page={page}
              serverW={serverW}
              serverH={serverH}
              boxes={box? [box] : []}
              showBoxes={!!box}
              lasso={modeLasso}
              onLassoDone={(r)=>{ setRect(r); setModeLasso(false); }}
            />
          </div>

          {/* right: form */}
          <div>
            <div className="row"><label>Field</label>
              <input list="keys" value={key} onChange={(e)=>setKey(e.target.value)} placeholder="key (e.g., invoice_number)"/>
              <datalist id="keys">{allKeys.map(k=><option key={k} value={k}/>)}</datalist>
            </div>
            <div className="row"><label>Value</label>
              <textarea rows={6} value={value} onChange={(e)=>setValue(e.target.value)}/>
            </div>
            <div className="row"><label>Where</label>
              <input disabled value={rect ? `x0=${rect.x0}, y0=${rect.y0}, x1=${rect.x1}, y1=${rect.y1}` : "(draw a box)"} />
            </div>
            <div style={{display:"flex", gap:8, justifyContent:"flex-end", marginTop:10}}>
              <button onClick={onClose}>Cancel</button>
              <button className="primary" onClick={doBind} disabled={!rect || !key || binding}>
                {binding ? "Binding…" : "Bind"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles:Record<string,React.CSSProperties>={
  backdrop:{ position:"fixed", inset:0, background:"rgba(0,0,0,.35)", display:"flex", alignItems:"center", justifyContent:"center", zIndex:50 },
  modal:{ background:"#fff", width:"min(1100px,96vw)", borderRadius:12, padding:14, boxShadow:"0 10px 30px rgba(0,0,0,.2)" }
};
