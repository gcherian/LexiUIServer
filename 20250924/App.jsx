// src/App.jsx
import React, { useRef, useState } from "react";
import KVPane from "./components/KVPane.jsx";
import PdfPane from "./components/PdfPane.jsx";
import { parseDocAI } from "./lib/docai.js";
import "./style.css";

export default function App() {
  const [pdfUrl, setPdfUrl] = useState("");
  const [header, setHeader] = useState({});
  const [elements, setElements] = useState([]);
  const pdfRef = useRef(null);

  async function onUploadPdf(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setPdfUrl(URL.createObjectURL(f));
    e.target.value = "";
  }

  async function onUploadDocAI(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const json = JSON.parse(await f.text());
      const { header, elements } = parseDocAI(json);
      setHeader(header || {});
      setElements(elements || []);
      console.log("[DocAI] header keys:", Object.keys(header || {}));
      console.log("[DocAI] elements:", elements?.length);
    } catch (err) {
      console.error("Invalid JSON:", err);
      alert("Invalid JSON file.");
    } finally {
      e.target.value = "";
    }
  }

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">DocAI KV Highlighter</div>
        <div className="toolbar">
          <span style={{opacity:.8,marginRight:8}}>
            {elements.length ? `${elements.length} elements` : "Upload DocAI JSON"}
          </span>
          <label className="btn">
            <input type="file" accept="application/pdf" onChange={onUploadPdf} />
            Choose PDF
          </label>
          <label className="btn">
            <input type="file" accept="application/json" onChange={onUploadDocAI} />
            Choose DocAI JSON
          </label>
        </div>
      </div>

      <div className="split">
        <KVPane
          header={header}
          elements={elements}
          onHoverDocAI={(el)=> pdfRef.current?.showDocAIBbox(el)}
          onPickValue={(value)=> pdfRef.current?.locateValue(value)}
        />
        <PdfPane ref={pdfRef} pdfUrl={pdfUrl} />
      </div>
    </div>
  );
}