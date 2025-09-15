async function runValidateInline(ocrTokens: OCRToken[], llmFields: Record<string,string>) {
  const res = await fetch("/validate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      ocr_tokens: ocrTokens,
      llm_fields: llmFields,
      // pass FLE autolocate if you already have it:
      // client_autolocate: [{ field: "Customer Legal Name", candidate_value: "...", confidence: 0.97, bbox: [x0,y0,x1,y1] }]
    })
  });
  return await res.json(); // { results: FieldResult[], model_info: {...} }
}

frontend/
  package.json
  vite.config.ts
  tsconfig.json
  index.html
  src/
    main.tsx
    App.tsx
    api/validate.ts
    components/PdfCanvas.tsx
    components/ValidationPanel.tsx
    components/Toolbar.tsx
    state/useDocStore.ts
    types.ts
    styles.css

//p
{
  "name": "docintel-validate-ui",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview --port 5173"
  },
  "dependencies": {
    "pdfjs-dist": "^4.5.136",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "typescript": "^5.5.4",
    "vite": "^5.4.2"
  }
}

//vc
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // proxy API to your FastAPI server (adjust port if needed)
      "/validate": {
        target: "http://localhost:8080",
        changeOrigin: true
      }
    }
  }
});

//tsc

{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2021", "DOM"],
    "jsx": "react-jsx",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "strict": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true
  },
  "include": ["src"]
}

//index.html

<!doctype html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>EDIP Validation</title>
    <link rel="stylesheet" href="/src/styles.css" />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>


//type.ts
export type OCRToken = {
  text: string;
  bbox: [number, number, number, number]; // [x0,y0,x1,y1] in pixel coords of the rendered page image
  page: number;
};

export type FieldResult = {
  field: string;
  llm_value: string;
  winner_method: "autolocate" | "bert" | "minilm" | "layoutlmv3";
  winner_candidate: string;
  winner_confidence: number;
  winner_bbox: [number, number, number, number] | null;
  autolocate_conf: number;
  bert_conf: number;
  minilm_conf: number;
  layoutlmv3_conf: number;
  autolocate_val: string;
  bert_val: string;
  minilm_val: string;
  layoutlmv3_val: string;
};

export type ValidateResponse = {
  results: FieldResult[];
  model_info: Record<string, string>;
};

export type ClientAutoLocate = {
  field: string;
  candidate_value: string;
  confidence: number;
  bbox?: [number, number, number, number];
};

//validate.ts

import type { OCRToken, ValidateResponse, ClientAutoLocate } from "../types";

export async function callValidateAPI(params: {
  ocrTokens: OCRToken[];
  llmFields: Record<string, string>;
  clientAutoLocate?: ClientAutoLocate[];
}): Promise<ValidateResponse> {
  const res = await fetch("/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ocr_tokens: params.ocrTokens,
      llm_fields: params.llmFields,
      client_autolocate: params.clientAutoLocate ?? []
    })
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Validate failed: ${res.status} ${text}`);
  }
  return res.json();
}

//sec/atate/Docstore.ts

import { useState } from "react";
import type { OCRToken, FieldResult } from "../types";

export function useDocStore() {
  const [pdfUrl, setPdfUrl] = useState<string>("");
  const [ocrTokens, setOcrTokens] = useState<OCRToken[]>([]);
  const [llmFields, setLlmFields] = useState<Record<string, string>>({});
  const [results, setResults] = useState<FieldResult[]>([]);
  const [selected, setSelected] = useState<FieldResult | null>(null);
  const [loading, setLoading] = useState(false);

  return {
    pdfUrl, setPdfUrl,
    ocrTokens, setOcrTokens,
    llmFields, setLlmFields,
    results, setResults,
    selected, setSelected,
    loading, setLoading
  };
}

components/PdfCanvas.tax

import { useEffect, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import pdfjsWorker from "pdfjs-dist/build/pdf.worker.min.mjs"; // ✅ fixes the worker issue in Vite
import type { FieldResult } from "../types";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorker;

type Props = {
  fileUrl: string;
  selected?: FieldResult | null;
  onPageReady?: (pageWidth: number, pageHeight: number) => void;
};

export default function PdfCanvas({ fileUrl, selected, onPageReady }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const [pageDims, setPageDims] = useState<{ w: number; h: number } | null>(
    null
  );

  useEffect(() => {
    let cancelled = false;

    async function render() {
      if (!fileUrl || !canvasRef.current) return;

      const loadingTask = pdfjsLib.getDocument(fileUrl);
      const pdf = await loadingTask.promise;
      const page = await pdf.getPage(1); // single-page viewer for demo

      const viewport = page.getViewport({ scale: 1.5 }); // adjust as needed
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width = viewport.width;
      canvas.height = viewport.height;

      await page.render({ canvasContext: ctx, viewport }).promise;
      if (!cancelled) {
        setPageDims({ w: viewport.width, h: viewport.height });
        onPageReady?.(viewport.width, viewport.height);
      }
    }
    render();

    return () => {
      cancelled = true;
    };
  }, [fileUrl, onPageReady]);

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;

    // clear previous
    overlay.innerHTML = "";

    if (!selected?.winner_bbox || !pageDims) return;

    const [x0, y0, x1, y1] = selected.winner_bbox; // expected to be pixel coords for rendered page
    const box = document.createElement("div");
    box.style.position = "absolute";
    box.style.left = `${x0}px`;
    box.style.top = `${y0}px`;
    box.style.width = `${x1 - x0}px`;
    box.style.height = `${y1 - y0}px`;
    box.style.border = "2px solid #1f7ae0";
    box.style.background = "rgba(31,122,224,0.12)";
    overlay.appendChild(box);
  }, [selected, pageDims]);

  return (
    <div className="pdf-stage">
      <div className="pdf-stack">
        <canvas ref={canvasRef} className="pdf-canvas" />
        <div ref={overlayRef} className="pdf-overlay" />
      </div>
    </div>
  );
}

//c/vp.tax

import type { FieldResult } from "../types";

type Props = {
  rows: FieldResult[];
  onSelect: (row: FieldResult) => void;
};

function pct(n: number) {
  return `${Math.round(n * 100)}%`;
}

export default function ValidationPanel({ rows, onSelect }: Props) {
  return (
    <div className="panel">
      <div className="panel-header">
        <h3>Validation Results</h3>
      </div>
      <div className="table-scroll">
        <table className="grid">
          <thead>
            <tr>
              <th>Field</th>
              <th>LLM Value</th>
              <th>Winner</th>
              <th>Conf</th>
              <th>Auto</th>
              <th>BERT</th>
              <th>MiniLM</th>
              <th>LayoutLMv3</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.field} onClick={() => onSelect(r)}>
                <td className="mono">{r.field}</td>
                <td title={r.llm_value}>{r.llm_value}</td>
                <td className="mono">{r.winner_method}</td>
                <td>{pct(r.winner_confidence)}</td>
                <td title={r.autolocate_val}>{pct(r.autolocate_conf)}</td>
                <td title={r.bert_val}>{pct(r.bert_conf)}</td>
                <td title={r.minilm_val}>{pct(r.minilm_conf)}</td>
                <td title={r.layoutlmv3_val}>{pct(r.layoutlmv3_conf)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length === 0 && (
        <div className="empty">Run “Validate” to see results.</div>
      )}
    </div>
  );
}

//tb.tax

type Props = {
  onValidate: () => void;
  onLoadSamples: () => void;
  disabled?: boolean;
};

export default function Toolbar({ onValidate, onLoadSamples, disabled }: Props) {
  return (
    <div className="toolbar">
      <button onClick={onLoadSamples}>Load Sample Doc</button>
      <button onClick={onValidate} disabled={disabled}>
        Validate
      </button>
    </div>
  );
}

//app

import { useCallback } from "react";
import PdfCanvas from "./components/PdfCanvas";
import ValidationPanel from "./components/ValidationPanel";
import Toolbar from "./components/Toolbar";
import { useDocStore } from "./state/useDocStore";
import { callValidateAPI } from "./api/validate";
import type { OCRToken } from "./types";

export default function App() {
  const {
    pdfUrl, setPdfUrl,
    ocrTokens, setOcrTokens,
    llmFields, setLlmFields,
    results, setResults,
    selected, setSelected,
    loading, setLoading
  } = useDocStore();

  // Demo loader (replace with your real doc fetch)
  const loadSamples = useCallback(async () => {
    // You can swap these with your actual blobs/URLs
    setPdfUrl("/sample.pdf");

    // Minimal fake OCR tokens for demo (you already have real ones from backend)
    const tokens: OCRToken[] = [
      { text: "Bobcat", bbox: [60, 50, 180, 80], page: 0 },
      { text: "JOHNSON PLUMBING & HEATING CO", bbox: [70, 210, 470, 235], page: 0 },
      { text: "59897.69", bbox: [440, 680, 520, 705], page: 0 }
    ];
    setOcrTokens(tokens);

    // Minimal LLM fields (replace with your Gemini fields)
    setLlmFields({
      "Partner Name": "Bobcat Enterprises, Inc.",
      "Customer Legal Name": "JOHNSON PLUMBING & HEATING CO",
      "Total Amount Financed": "59897.69"
    });

    setResults([]);
    setSelected(null);
  }, [setPdfUrl, setOcrTokens, setLlmFields, setResults, setSelected]);

  const runValidate = useCallback(async () => {
    if (!ocrTokens.length) return;
    setLoading(true);
    try {
      const resp = await callValidateAPI({
        ocrTokens,
        llmFields
        // clientAutoLocate: [...] // send your FLE autolocate row if you have it
      });
      setResults(resp.results);
      setSelected(null);
      // console.log(resp.model_info); // show which models were used
    } catch (e: any) {
      alert(e.message ?? "Validation error");
    } finally {
      setLoading(false);
    }
  }, [ocrTokens, llmFields, setLoading, setResults, setSelected]);

  return (
    <div className="app">
      <header>
        <h2>EDIP Validation Workbench</h2>
      </header>

      <Toolbar
        onLoadSamples={loadSamples}
        onValidate={runValidate}
        disabled={loading || !ocrTokens.length}
      />

      <div className="split">
        <div className="left">
          {pdfUrl ? (
            <PdfCanvas fileUrl={pdfUrl} selected={selected} />
          ) : (
            <div className="empty">Load a PDF to begin.</div>
          )}
        </div>

        <div className="right">
          <ValidationPanel rows={results} onSelect={setSelected} />
        </div>
      </div>
    </div>
  );
}

//main

import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

//style.css

:root {
  --bg: #0f1115;
  --panel: #161a22;
  --text: #e9eef6;
  --muted: #a8b3c7;
  --accent: #1f7ae0;
  --border: #242b36;
}

* { box-sizing: border-box; }
html, body, #root { height: 100%; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; }

header { padding: 10px 16px; border-bottom: 1px solid var(--border); }

.app { height: 100%; display: flex; flex-direction: column; }

.toolbar { gap: 8px; padding: 8px 16px; border-bottom: 1px solid var(--border); display: flex; align-items: center; }
.toolbar button {
  background: var(--panel); color: var(--text); border: 1px solid var(--border);
  padding: 8px 12px; border-radius: 6px; cursor: pointer;
}
.toolbar button:disabled { opacity: 0.5; cursor: not-allowed; }

.split { display: grid; grid-template-columns: 1fr 1fr; gap: 0; flex: 1; min-height: 0; }
.left, .right { min-height: 0; }

.pdf-stage { width: 100%; height: 100%; display: flex; justify-content: center; align-items: flex-start; padding: 8px; overflow: auto; }
.pdf-stack { position: relative; }
.pdf-canvas { display: block; box-shadow: 0 0 0 1px var(--border); background: #fff; }
.pdf-overlay { position: absolute; inset: 0; pointer-events: none; }

.panel { height: 100%; display: flex; flex-direction: column; }
.panel-header { padding: 8px 12px; border-bottom: 1px solid var(--border); }
.table-scroll { overflow: auto; flex: 1; }

.grid { width: 100%; border-collapse: collapse; font-size: 13px; }
.grid th, .grid td { border-bottom: 1px solid var(--border); padding: 6px 8px; text-align: left; vertical-align: top; }
.grid tbody tr { cursor: pointer; }
.grid tbody tr:hover { background: rgba(255,255,255,0.04); }

.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.empty { padding: 16px; color: var(--muted); }

