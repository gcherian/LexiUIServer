// Unified client for EDIP OCR + Lasso (FastAPI backend).

// ---------- Types (relaxed to avoid setter type errors) ----------
export type Rect = { x0: number; y0: number; x1: number; y1: number };

export type Box = {
  page: number;
  x0: number; y0: number; x1: number; y1: number;
  id?: string | null;
  label?: string | null;
  text?: string | null;
  confidence?: number | null;
};

export type FieldState = {
  id?: string | null;
  key?: string | null;       // canonical on server
  name?: string | null;      // alias accepted; mapped -> key on save
  value?: string | null;
  confidence?: number | null;
  source?: string | null;    // ecm|ocr|user|llm|bbox|lasso
  page?: number | null;
  bbox?: (Rect & { page: number }) | null;
};

export type FieldDocState = {
  doc_id: string;
  doctype: string;
  fields: FieldState[];
  audit?: Array<Record<string, any>>;
};

export type MetaResp = { pages: Array<{ page: number; width: number; height: number }> };
export type UploadResp = { doc_id: string; annotated_tokens_url: string; pages: number };

export type PromField = { key: string; label: string; type?: string; required?: boolean; enum?: string[] };
export type PromCatalog = { doctype: string; version: string; fields: PromField[] };

export type SearchHit = { page: number; bbox: Rect; text: string; score: number };

// ---------- API base ----------
const API_BASE: string =
  (typeof window !== "undefined" && (window as any).__API_BASE__) ||
  (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_BASE) ||
  (typeof process !== "undefined" && (process as any).env?.VITE_API_BASE) ||
  "http://localhost:8000";

export const API = API_BASE;

// ---------- Utils ----------
function qs(obj: Record<string, any>) {
  const u = new URLSearchParams();
  for (const k in obj) {
    const v = (obj as any)[k];
    if (v !== undefined && v !== null) u.set(k, String(v));
  }
  return u.toString();
}
async function jsonFetch<T = any>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    const msg = await r.text().catch(() => `${r.status} ${r.statusText}`);
    throw new Error(`[${r.status}] ${msg}`);
  }
  return (r.status === 204 ? (undefined as any) : await r.json()) as T;
}

export function docUrlFromId(doc_id: string) {
  return `${API_BASE}/data/${doc_id}/original.pdf`;
}
export function guessDocIdFromUrl(doc_url: string): string | null {
  const m = /\/data\/([^/]+)\/original\.pdf$/i.exec(doc_url || "");
  return m ? m[1] : null;
}

// ---------- Upload / Meta ----------
export async function uploadPdf(file: File): Promise<UploadResp> {
  const fd = new FormData();
  fd.append("pdf", file);
  fd.append("backend", "tesseract");
  return jsonFetch<UploadResp>(`${API_BASE}/lasso/upload`, { method: "POST", body: fd });
}
export async function getMetaByDocId(doc_id: string): Promise<MetaResp> {
  return jsonFetch<MetaResp>(`${API_BASE}/lasso/doc/${doc_id}/meta`);
}
export const getMeta = getMetaByDocId;

// ---------- Boxes ----------
export async function getBoxes(params: { doc_url: string; page: number }): Promise<Box[]> {
  return jsonFetch<Box[]>(`${API_BASE}/boxes?${qs(params)}`);
}
export async function getBoxesByDocId(doc_id: string, page?: number): Promise<Box[]> {
  const boxes = await jsonFetch<Box[]>(`${API_BASE}/lasso/doc/${doc_id}/boxes`);
  return page ? boxes.filter(b => Number(b.page) === Number(page)) : boxes;
}

// ---------- Fields ----------
export async function listFields(arg1: { doc_url: string } | string, _doctype?: string): Promise<FieldState[]> {
  const doc_url = typeof arg1 === "string" ? arg1 : arg1.doc_url;
  return jsonFetch<FieldState[]>(`${API_BASE}/fields?${qs({ doc_url })}`);
}
export async function getFieldDocState(doc_id: string): Promise<FieldDocState> {
  return jsonFetch<FieldDocState>(`${API_BASE}/lasso/doc/${doc_id}/fields`);
}
export async function putFieldDocState(doc_id: string, state: FieldDocState): Promise<FieldDocState> {
  return jsonFetch<FieldDocState>(`${API_BASE}/lasso/doc/${doc_id}/fields`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state),
  });
}
export async function saveFieldState(payload: { doc_url: string; field: FieldState }): Promise<boolean> {
  const field = { ...payload.field };
  if (!field.key && field.name) field.key = field.name;
  await jsonFetch(`${API_BASE}/fields`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_url: payload.doc_url, field }),
  });
  return true;
}
export async function saveField(
  doc_url: string,
  key: string,
  value: string,
  bbox?: Rect & { page: number },
  confidence?: number
) {
  const field: FieldState = { key, value, bbox: bbox || null, confidence, source: "user" };
  return saveFieldState({ doc_url, field });
}

// ---------- Doctype / ECM ----------
export async function setDoctype(doc_id: string, doctype: string): Promise<{ ok: boolean; doctype: string }> {
  return jsonFetch(`${API_BASE}/lasso/doc/${doc_id}/doctype`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doctype }),
  });
}
export async function ecmExtract(doc_id: string, doctype: string): Promise<FieldDocState> {
  return jsonFetch<FieldDocState>(`${API_BASE}/lasso/ecm/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id, doctype }),
  });
}

// ---------- Bind / Lasso ----------
export async function bindFieldByRect(params: {
  doc_id: string; key: string; page: number; rect: Rect;
}): Promise<FieldDocState> {
  return jsonFetch<FieldDocState>(`${API_BASE}/lasso/doc/${params.doc_id}/bind`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: params.doc_id, page: params.page, rect: params.rect, key: params.key }),
  });
}
export async function lassoRecognizeText(params: { doc_id: string; page: number; rect: Rect }): Promise<{ text: string }> {
  return jsonFetch<{ text: string }>(`${API_BASE}/lasso/lasso`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: params.doc_id, page: params.page, ...params.rect }),
  });
}

// ---------- Search ----------
export async function tokenSearch(doc_id: string, query: string, topk = 20) {
  const res = await jsonFetch<{ matches: SearchHit[] }>(`${API_BASE}/lasso/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id, query, topk }),
  });
  return res.matches || [];
}
export async function semanticSearch(doc_id: string, q: string, topk = 5) {
  const res = await jsonFetch<{ results: Array<{ page: number; score: number }> }>(
    `${API_BASE}/lasso/semantic_search?${qs({ doc_id, q, topk })}`
  );
  return res.results || [];
}

// ---------- PROM ----------
export async function listPromDoctypes() {
  const res = await jsonFetch<{ doctypes: Array<{ doctype: string; file: string }> }>(`${API_BASE}/lasso/prom`);
  return res.doctypes || [];
}
export async function getPromCatalog(doctype: string) {
  return jsonFetch<PromCatalog>(`${API_BASE}/lasso/prom/${encodeURIComponent(doctype)}`);
}

// ---------- Legacy name shims ----------
export const search = tokenSearch;        // old name
export const listProms = listPromDoctypes;// old name
export const setDocType = setDoctype;     // old name