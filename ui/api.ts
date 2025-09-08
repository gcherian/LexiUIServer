// src/lib/api.ts
// Unified client for EDIP OCR + Lasso flows (FastAPI backend).
// Works with both the new BBox Workbench and your existing OcrWorkbench.

// ---------- Types ----------
export type Rect = { x0: number; y0: number; x1: number; y1: number };
export type Box = Rect & { page: number; id?: string; label?: string; text?: string; confidence?: number };

export type FieldState = {
  id?: string;
  key?: string;           // canonical key used by server
  name?: string;          // alias accepted by server; mapped -> key when posting
  value?: string | null;
  confidence?: number;
  source?: string;        // ecm|ocr|user|llm
  page?: number;
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

// ---------- API base resolution ----------
const API_BASE: string =
  (typeof window !== "undefined" && (window as any).__API_BASE__) ||
  (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_BASE) ||
  (typeof process !== "undefined" && (process as any).env?.VITE_API_BASE) ||
  "http://localhost:8000";

export const API = API_BASE;

// ---------- Utils ----------
function qs(obj: Record<string, any>) {
  const u = new URLSearchParams();
  Object.entries(obj).forEach(([k, v]) => {
    if (v === undefined || v === null) return;
    u.set(k, String(v));
  });
  return u.toString();
}

async function jsonFetch<T = any>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    const msg = await r.text().catch(() => `${r.status} ${r.statusText}`);
    throw new Error(`[${r.status}] ${msg}`);
  }
  if (r.status === 204) return undefined as unknown as T;
  return (await r.json()) as T;
}

// Server stores/serves PDFs under /data/{doc_id}/original.pdf
export function docUrlFromId(doc_id: string) {
  return `${API_BASE}/data/${doc_id}/original.pdf`;
}

// Try to pull {doc_id} back from a /data/{doc_id}/original.pdf URL (best-effort)
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

// Back-compat alias some callers used:
export const getMeta = getMetaByDocId;

// ---------- Boxes (two paths supported) ----------
// Preferred (UI compat): GET /boxes?doc_url=...&page=...
export async function getBoxes(params: { doc_url: string; page: number }): Promise<Box[]> {
  return jsonFetch<Box[]>(`${API_BASE}/boxes?${qs(params)}`);
}

// Canonical (by doc_id): GET /lasso/doc/{doc_id}/boxes then filter by page
export async function getBoxesByDocId(doc_id: string, page?: number): Promise<Box[]> {
  const boxes = await jsonFetch<Box[]>(`${API_BASE}/lasso/doc/${doc_id}/boxes`);
  return page ? boxes.filter(b => Number(b.page) === Number(page)) : boxes;
}

// ---------- Fields (compat + canonical) ----------
// UI compat: GET /fields?doc_url=...
export async function listFields(arg1: { doc_url: string } | string, _doctype?: string): Promise<FieldState[]> {
  const doc_url = typeof arg1 === "string" ? arg1 : arg1.doc_url;
  return jsonFetch<FieldState[]>(`${API_BASE}/fields?${qs({ doc_url })}`);
}

// Canonical: GET /lasso/doc/{doc_id}/fields (entire FieldDocState)
export async function getFieldDocState(doc_id: string): Promise<FieldDocState> {
  return jsonFetch<FieldDocState>(`${API_BASE}/lasso/doc/${doc_id}/fields`);
}

// Canonical: PUT /lasso/doc/{doc_id}/fields
export async function putFieldDocState(doc_id: string, state: FieldDocState): Promise<FieldDocState> {
  return jsonFetch<FieldDocState>(`${API_BASE}/lasso/doc/${doc_id}/fields`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state),
  });
}

// UI compat: POST /fields { doc_url, field }
export async function saveFieldState(payload: { doc_url: string; field: FieldState }): Promise<boolean> {
  // normalize: map name -> key (server accepts either, but keep it tidy)
  const field = { ...payload.field };
  if (!field.key && field.name) field.key = field.name;
  await jsonFetch(`${API_BASE}/fields`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_url: payload.doc_url, field }),
  });
  return true;
}

// Helper: upsert a field directly by doc_id (canonical path)
export async function upsertFieldByDocId(doc_id: string, field: FieldState): Promise<FieldDocState> {
  const state = await getFieldDocState(doc_id);
  const key = field.key || field.name;
  let updated = false;
  if (key) {
    state.fields = state.fields.map(f => (f.key === key ? { ...f, ...field, key } : f));
    if (!state.fields.some(f => f.key === key)) {
      state.fields.push({ ...field, key });
    }
    updated = true;
  }
  if (!updated) state.fields.push(field);
  return putFieldDocState(doc_id, state);
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

// ---------- Bind / Lasso OCR ----------
export async function bindFieldByRect(params: {
  doc_id: string;
  key: string;               // field key (or name)
  page: number;
  rect: Rect;                // coordinates in image pixel space (origin top-left)
}): Promise<FieldDocState> {
  const payload = {
    doc_id: params.doc_id,
    page: params.page,
    rect: params.rect,
    key: params.key,
  };
  return jsonFetch<FieldDocState>(`${API_BASE}/lasso/doc/${params.doc_id}/bind`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function lassoRecognizeText(params: {
  doc_id: string;
  page: number;
  rect: Rect;
}): Promise<{ text: string }> {
  return jsonFetch<{ text: string }>(`${API_BASE}/lasso/lasso`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: params.doc_id, page: params.page, ...params.rect }),
  });
}

// ---------- Search ----------
export async function tokenSearch(doc_id: string, query: string, topk = 20): Promise<SearchHit[]> {
  const res = await jsonFetch<{ matches: SearchHit[] }>(`${API_BASE}/lasso/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id, query, topk }),
  });
  return res.matches || [];
}

export async function semanticSearch(doc_id: string, q: string, topk = 5): Promise<Array<{ page: number; score: number }>> {
  const res = await jsonFetch<{ results: Array<{ page: number; score: number }> }>(
    `${API_BASE}/lasso/semantic_search?${qs({ doc_id, q, topk })}`
  );
  return res.results || [];
}

// ---------- PROM (schema catalog) ----------
export async function listPromDoctypes(): Promise<Array<{ doctype: string; file: string }>> {
  const res = await jsonFetch<{ doctypes: Array<{ doctype: string; file: string }> }>(`${API_BASE}/lasso/prom`);
  return res.doctypes || [];
}

export async function getPromCatalog(doctype: string): Promise<PromCatalog> {
  return jsonFetch<PromCatalog>(`${API_BASE}/lasso/prom/${encodeURIComponent(doctype)}`);
}

// ---------- Convenience helpers for older callers ----------

// Some legacy code calls: listFields(docUrl, doctype)
export async function listFieldsForDocUrl(doc_url: string, _doctype?: string): Promise<FieldState[]> {
  return listFields({ doc_url });
}

// Some legacy code expects: getBoxes({ doc_url, page }) — already provided above

// Some legacy code may expect: saveField(docUrl, key, value, bbox?)
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
