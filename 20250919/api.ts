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
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  text?: string;
};

export type OcrPreviewResp = {
  text: string;
  crop_url?: string;
};

/* Optional Distil types (best-effort) */
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
export type DistilExtractResp = { doc_id: string; fields: DistilField[]; dpi: number };

/* Matcher types (5 methods) */
export type MatchRect = { page: number; rect: { x0: number; y0: number; x1: number; y1: number }; score: number };
export type MatchResp = {
  doc_id?: string;
  key?: string;
  value?: string;
  methods: {
    fuzzy?: MatchRect | null;
    tfidf?: MatchRect | null;
    minilm?: MatchRect | null;
    distilbert?: MatchRect | null;
    layoutlmv3?: MatchRect | null;
  };
};

/* Ground-truth types */
export type GtRect = { page: number; x0: number; y0: number; x1: number; y1: number };
export type GroundTruthEntry = {
  key: string;
  value: string;
  rect: GtRect;
  text?: string;
  ts?: number;
  by?: string;
};
export type GroundTruthResp = {
  doc_id: string;
  entries: GroundTruthEntry[];
};

/* ---------------- Ground truth ---------------- */
export async function saveGroundTruth(
  doc_id: string,
  key: string,
  rect: { page:number; x0:number; y0:number; x1:number; y1:number },
  value?: string
): Promise<{ ok: boolean }> {
  const r = await fetch(`${API}/lasso/gt/save`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ doc_id, key, rect, value }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/* ---------------- Helpers ---------------- */
export function docIdFromUrl(url: string): string | null {
  try {
    const m = url.match(/\/data\/([A-Za-z0-9_-]+)\/original\.pdf/i);
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
    body: JSON.stringify({ doc_id, page, x0: rect.x0, y0: rect.y0, x1: rect.x1, y1: rect.y1 }),
  });
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  return { text: j?.text || "", crop_url: j?.crop_url };
}

/** Optional Distil (semantic) extraction; best-effort */
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

/** 5-method matcher (fuzzy/tfidf/minilm/distilbert/layoutlmv3) */
export async function matchField(
  doc_id: string,
  key: string,
  value: string,
  max_window = 12,
  models_root?: string
): Promise<MatchResp> {
  const r = await fetch(`${API}/lasso/match/field`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ doc_id, key, value, max_window, models_root }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/* ---------- Ground-truth (human-in-the-loop) ---------- */

/** Save/overwrite ground truth for one field (and get per-model IoU back) */
export async function saveGroundTruth(params: {
  doc_id: string;
  key: string;
  value: string;
  rect: GtRect;
  text?: string;          // optional corrected OCR text
  by?: string;            // optional user id
}): Promise<{ ok: boolean; iou?: Record<string, number> }> {
  const r = await fetch(`${API}/lasso/groundtruth/save`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/** Fetch all stored ground-truth entries for a document */
export async function getGroundTruth(doc_id: string): Promise<GroundTruthResp> {
  const r = await fetch(`${API}/lasso/groundtruth/get?doc_id=${encodeURIComponent(doc_id)}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/* ---------- Warmup (preload local models, no network) ---------- */
export async function warmup(models_root?: string): Promise<{ ok: boolean; loaded: string[] }> {
  const r = await fetch(`${API}/lasso/warmup`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(models_root ? { models_root } : {}),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/* ---------- Optional: /locate for quick debug ---------- */
export async function locateField(params: {
  doc_id: string;
  key: string;
  value: string;
  max_window?: number;
  models_root?: string;
}): Promise<{
  hits: {
    autolocate?: MatchRect | null;
    tfidf?: MatchRect | null;
    minilm?: MatchRect | null;
    distilbert?: MatchRect | null;
    layoutlmv3?: MatchRect | null;
  };
  pages?: any[];
}> {
  const r = await fetch(`${API}/lasso/locate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      doc_id: params.doc_id,
      key: params.key,
      value: params.value,
      max_window: params.max_window ?? 12,
      models_root: params.models_root,
    }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
