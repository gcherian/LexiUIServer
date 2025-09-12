// File: ui/lib/api.ts  (adjust path if your repo differs)
// Minimal, Vite-friendly API helpers used by FieldLevelEditor

/* ---------------- Base ---------------- */
const BASE =
  (import.meta as any).env?.VITE_API_BASE ||
  (typeof window !== "undefined" &&
    (window as any).__API_BASE__) ||
  "http://localhost:8080";

export const API = BASE;

/* ---------------- Types ---------------- */
export type UploadResp = {
  doc_id: string;
  annotated_tokens_url: string;
  pages: number;
};

export type MetaResp = {
  pages: Array<{ page: number; width: number; height: number }>;
};

export type Box = {
  page: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  text?: string;
};

export type OcrPreviewResp = {
  text: string;
  crop_url?: string; // server may or may not return this (best-effort)
};

/* Distil types */
export type DistilBox = { page: number; x0: number; y0: number; x1: number; y1: number };
export type DistilField = {
  key: string;
  label: string;
  type?: string;
  page: number | null;
  key_box?: DistilBox | null;
  value: string;
  value_boxes: DistilBox[];
  value_union?: DistilBox | null;
  confidence: number;
};

export type DistilExtractResp = {
  doc_id: string;
  fields: DistilField[];
  dpi: number;
};

/* ---------------- Helpers ---------------- */
export function docIdFromUrl(url: string): string | null {
  try {
    const m = url.match(/\/data\/([A-Za-z0-9]+)\/original\.pdf/i);
    return m ? m[1] : null;
  } catch {
    return null;
  }
}

export function docUrlFromId(doc_id: string): string {
  return `${API}/data/${doc_id}/original.pdf`;
}

/* ---------------- Calls ---------------- */
export async function uploadPdf(file: File): Promise<UploadResp> {
  const fd = new FormData();
  fd.append("pdf", file);
  // FastAPI endpoint was /lasso/upload
  const r = await fetch(`${API}/lasso/upload`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getMeta(doc_id: string): Promise<MetaResp> {
  const r = await fetch(`${API}/lasso/doc/${doc_id}/meta`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getBoxes(doc_id: string): Promise<Box[]> {
  const r = await fetch(`${API}/lasso/doc/${doc_id}/boxes`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/** Lasso OCR preview (server returns { text, optionally crop_url }) */
export async function ocrPreview(
  doc_id: string,
  page: number,
  rect: { x0: number; y0: number; x1: number; y1: number }
): Promise<OcrPreviewResp> {
  const r = await fetch(`${API}/lasso/lasso`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      doc_id,
      page,
      x0: rect.x0,
      y0: rect.y0,
      x1: rect.x1,
      y1: rect.y1,
    }),
  });
  if (!r.ok) throw new Error(await r.text());
  // Some servers only return {text}; gracefully add undefined crop_url
  const j = await r.json();
  return { text: j?.text || "", crop_url: j?.crop_url };
}

/** Distil hybrid (semantic) extraction; best-effort */
export async function distilExtract(
  doc_id: string,
  fields: { key: string; label: string; type?: string }[],
  max_window = 12,
  dpi = 260
): Promise<DistilExtractResp> {
  const r = await fetch(`${API}/distil/extract`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ doc_id, fields, max_window, dpi }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}