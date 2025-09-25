import React, { useRef, useState } from "react";
import PdfPane from "./components/PdfPane";
import KVPane from "./components/KVPane";
import { parseDocAI } from "./lib/docai";

export default function App() {
  const pdfRef = useRef(null);

  const [pdfUrl, setPdfUrl] = useState("");
  const [docai, setDocai] = useState({ header: [], elements: [] });

  async function onChoosePdf(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    const url = URL.createObjectURL(f);   // <-- must be a blob URL; not a disk path
    setPdfUrl(url);
    console.log("[PDF] url set", url);
  }

  async function onChooseDocAI(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const raw = JSON.parse(await f.text());
      const parsed = parseDocAI(raw);     // {header, elements}
      console.log("[DocAI] header keys:", parsed.header.map(h=>h.key));
      console.log("[DocAI] elements:", parsed.elements.length);
      setDocai(parsed);
    } catch (err) {
      console.error("Invalid JSON:", err);
      alert("Invalid DocAI JSON");
    }
  }

  return (
    <div style={{display:"grid", gridTemplateRows:"48px 1fr", height:"100vh", color:"#e5e7eb", background:"#0b1220"}}>
      <div style={{display:"flex", alignItems:"center", gap:8, padding:"8px 12px", background:"#111827"}}>
        <div style={{fontWeight:700}}>DocAI KV Highlighter</div>
        <div style={{flex:1}} />
        <span style={{opacity:.7, marginRight:8}}>
          {docai.elements.length ? `${docai.elements.length} elements` : ""}
        </span>
        <label className="btn">
          <input type="file" accept="application/pdf" onChange={onChoosePdf} style={{display:"none"}} />
          Choose PDF
        </label>
        <label className="btn">
          <input type="file" accept="application/json" onChange={onChooseDocAI} style={{display:"none"}} />
          Choose DocAI JSON
        </label>
        <style>{`.btn{background:#374151;padding:6px 10px;border-radius:6px;cursor:pointer}`}</style>
      </div>

      <div style={{display:"grid", gridTemplateColumns:"360px 1fr", minHeight:0}}>
        <KVPane data={docai} pdfRef={pdfRef} />
        <PdfPane ref={pdfRef} pdfUrl={pdfUrl} />
      </div>
    </div>
  );
}