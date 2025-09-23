// File: src/lib/api.ts

/* ---------------- Base ---------------- */
const BASE =
  (import.meta as any).env?.VITE_API_BASE ||
  (typeof window !== "undefined" && (window as any).__API_BASE__) ||
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
  x0: number; y0: number; x1: number; y1: number;
  text?: string;
};

export type OcrPreviewResp = { text: string; crop_url?: string };

export type MatchRect = {
  page: number;
  rect: { x0: number; y0: number; x1: number; y1: number };
  score: number;
};

export type MatchResp = {
  methods: {
    fuzzy?: MatchRect | null;
    tfidf?: MatchRect | null;
    minilm?: MatchRect | null;
    distilbert?: MatchRect | null;
  };
};

export type MatchOptions = {
  max_window?: number;
  fast_only?: boolean;
};

/* ---------------- Helpers ---------------- */
export function docIdFromUrl(url: string): string | null {
  try {
    // supports /data/{docId}/original.pdf or a raw docId
    const m = url.match(/\/data\/([A-Za-z0-9_-]+)\/original\.pdf/i);
    if (m) return m[1];
    if (/^[A-Za-z0-9_-]{6,32}$/.test(url)) return url;
    return null;
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

export async function ocrPreview(
  doc_id: string,
  page: number,
  rect: { x0: number; y0: number; x1: number; y1: number }
): Promise<OcrPreviewResp> {
  const r = await fetch(`${API}/lasso/lasso`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ doc_id, page, ...rect }),
  });
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  return { text: j?.text || "", crop_url: j?.crop_url };
}

/** Fast-first matcher (fuzzy immediately, tfidf next). Heavy models omitted server-side. */
export async function matchField(
  doc_id: string,
  key: string,
  value: string,
  options?: MatchOptions
): Promise<MatchResp> {
  const r = await fetch(`${API}/lasso/match/field`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ doc_id, key, value, options }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
