import { useState } from "react";
import OcrWorkbench from "./components/OcrWorkbench"; // your existing tab
import PdfBBoxWorkbench from "./components/PdfBBoxWorkbench";
import "./components/ocr.css";

type Tab = "ocr" | "bbox";

export default function App() {
  const [tab, setTab] = useState<Tab>("ocr");

  const tabs: { key: Tab; label: string; el: JSX.Element }[] = [
    { key: "ocr", label: "OCR Workbench", el: <OcrWorkbench /> },
    { key: "bbox", label: "BBox Workbench", el: <PdfBBoxWorkbench /> },
  ];

  return (
    <div className="app-root">
      <header className="tabs">
        {tabs.map(t => (
          <button
            key={t.key}
            className={tab === t.key ? "tab active" : "tab"}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </header>
      <main className="tab-body">
        {tabs.find(t => t.key === tab)?.el}
      </main>
    </div>
  );
}
